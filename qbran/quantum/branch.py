"""JAX-accelerated fully quantum branch analysis (Dumas et al. 2024).

Tracks eigenstates along the drive resonator ladder using JAX top-k and scan.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import product
from functools import reduce

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from qbran.circuit.components import Transmon, Resonator


def _greedy_assign(ovl, assigned):
    """Assign one eigenstate to each branch.

    ovl      : (n_branch, dim) overlap probabilities |<lambda|psi_b>|^2
    assigned : (dim,) bool mask of eigenstates already used by previous rungs
    """
    n_branch = ovl.shape[0]
    top_val, top_idx = jax.lax.top_k(ovl, 2)
    order = jnp.argsort(-top_val[:, 0])

    def body(k, carry):
        mask, idx = carry
        b = order[k]
        row = jnp.where(mask, -1.0, ovl[b])
        i = jnp.argmax(row)
        return mask.at[i].set(True), idx.at[b].set(i)

    assigned, idx = jax.lax.fori_loop(
        0, n_branch, body, (assigned, jnp.zeros(n_branch, dtype=jnp.int64))
    )
    val = ovl[jnp.arange(n_branch), idx]
    return assigned, idx, val, top_val, top_idx


def _apply_adag(adag, kets, dim_sub, dim_d):
    """Apply drive creation operator a_d^dag to a stack of full-space kets."""
    n = kets.shape[1]
    psi = kets.reshape(dim_sub, dim_d, n)
    psi = jnp.einsum("mn,snb->smb", adag, psi)
    return psi.reshape(dim_sub * dim_d, n)


def _make_quantum_kernel(
    H_static, sweep_diag, mode_diags, adag, weights, label_mat,
    dim_sub, dim_d, n_rungs,
):
    """Builds the JIT-compiled single-frequency quantum solver."""

    def kernel(omega, omega_mod, V_sub):
        H = H_static + jnp.diag(omega * sweep_diag)
        evals, evecs = jnp.linalg.eigh(H)

        dim = H.shape[0]
        assigned = jnp.zeros(dim, dtype=bool)

        # Rung 0: seeds are |b> (x) |n_d = 0>
        zero_slice = evecs.reshape(dim_sub, dim_d, dim)[:, 0, :]
        ovl = jnp.abs(V_sub.conj().T @ zero_slice) ** 2
        assigned, idx0, val0, top0, tidx0 = _greedy_assign(ovl, assigned)

        # Climb the drive ladder
        def step(carry, _):
            mask, cur = carry
            psi = _apply_adag(adag, evecs[:, cur], dim_sub, dim_d)
            psi = psi / jnp.linalg.norm(psi, axis=0, keepdims=True)
            ovl_step = jnp.abs(psi.conj().T @ evecs) ** 2
            mask, idx, val, top, tidx = _greedy_assign(ovl_step, mask)
            return (mask, idx), (idx, val, top, tidx)

        _, (idx_r, val_r, top_r, tidx_r) = jax.lax.scan(
            step, (assigned, idx0), None, length=n_rungs - 1
        )

        dressed = jnp.concatenate([idx0[None], idx_r], axis=0)  # (n_rungs, n_branch)
        ladder = jnp.concatenate([val0[None], val_r], axis=0)
        top = jnp.concatenate([top0[None], top_r], axis=0)
        tidx = jnp.concatenate([tidx0[None], tidx_r], axis=0)

        nb = dressed.shape[1]
        flat = dressed.reshape(-1)
        sel = evecs[:, flat]
        probs = jnp.abs(sel) ** 2
        avg = (mode_diags @ probs).T.reshape(n_rungs, nb, -1)

        c = jnp.tensordot(V_sub.conj().T, sel.reshape(dim_sub, dim_d, -1), axes=(1, 0))
        Pb = jnp.sum(jnp.abs(c) ** 2, axis=1)
        avg_branch = (weights @ Pb).reshape(n_rungs, nb)
        avg_label = (label_mat.T @ Pb).T.reshape(n_rungs, nb, -1)

        e = evals[flat].reshape(dressed.shape)
        quasi = jnp.mod(e + 0.5 * omega_mod, omega_mod) - 0.5 * omega_mod

        return dict(
            dressed_indices=dressed,
            evals=e,
            quasienergies=quasi,
            avg_excitation_modes=avg,
            avg_excitation=avg[..., 0],
            avg_branch=avg_branch,
            avg_label=avg_label,
            hybridization=1.0 - top[..., 0],
            ladder_overlap=ladder,
            partner_index=tidx[..., 1],
            partner_overlap=top[..., 1],
        )

    return jax.jit(kernel)


class QuantumBranchAnalysis:
    """Fully quantum branch analysis engine powered by JAX."""

    def __init__(
        self,
        circuit: Any,
        options: Optional[Any] = None,
        drive: Optional[Any] = None,
    ):
        self.circuit = circuit
        self.options = options

        self._resolve_subsystems(drive)
        self._build_static_matrices()

    def _resolve_subsystems(self, drive: Optional[Any]):
        """Identifies the drive mode and splits the circuit into drive-free and drive parts."""
        comps = self.circuit.components

        # Find drive component
        if drive is None:
            # Default: the last resonator, or the only resonator
            res_list = [c for c in comps if isinstance(c, Resonator)]
            if not res_list:
                raise ValueError("No Resonator found in circuit to act as drive mode.")
            self.drive_component = res_list[-1]
        elif isinstance(drive, (Resonator, Transmon)):
            self.drive_component = drive
        elif isinstance(drive, str):
            # 1. Exact match on component name (case-insensitive)
            matched = [c for c in comps if c.name.lower() == drive.lower()]
            # 2. Generic aliases
            if not matched and drive.lower() in ("res", "resonator", "drive", "readout", "ro"):
                res_list = [c for c in comps if isinstance(c, Resonator)]
                if res_list:
                    matched = [res_list[0]]
            # 3. 'res1', 'res2', etc. (1-indexed)
            if not matched and drive.lower().startswith("res"):
                try:
                    num_str = "".join(filter(str.isdigit, drive))
                    idx = int(num_str) - 1 if num_str else 0
                    res_list = [c for c in comps if isinstance(c, Resonator)]
                    if 0 <= idx < len(res_list):
                        matched = [res_list[idx]]
                except Exception:
                    pass
            if not matched:
                raise ValueError(
                    f"Drive component '{drive}' not found in circuit. Available components: {[c.name for c in comps]}"
                )
            self.drive_component = matched[0]
        elif isinstance(drive, int):
            self.drive_component = comps[drive]
        else:
            raise TypeError(f"Invalid drive specification: {drive}")

        self.drive_idx = comps.index(self.drive_component)
        self.drive_dim = self.drive_component.truncated_dim

        # Drive-free subsystems
        self.sub_components = [c for i, c in enumerate(comps) if i != self.drive_idx]
        self.sub_dims = [c.truncated_dim for c in self.sub_components]
        self.dim_sub = int(np.prod(self.sub_dims)) if self.sub_dims else 1
        self.total_dim = self.dim_sub * self.drive_dim

        # Bare labels for drive-free subsystem
        self.bare_labels = np.array(
            list(product(*[range(d) for d in self.sub_dims])), dtype=int
        )
        self.n_branch = len(self.bare_labels)
        self.branch_weights = np.arange(self.n_branch, dtype=float)

    def _build_static_matrices(self):
        """Constructs the static full Hamiltonian and observable operators in JAX."""
        tmon = None
        for c in self.sub_components:
            if isinstance(c, Transmon):
                tmon = c
                break

        if tmon is None and isinstance(self.sub_components[0], Transmon):
            tmon = self.sub_components[0]

        # 1. Drive-free subsystem Hamiltonian (in eigenbasis)
        # For a single transmon, its bare Hamiltonian in eigenbasis is diagonal
        if len(self.sub_components) == 1 and isinstance(self.sub_components[0], Transmon):
            H_sub = np.diag(self.sub_components[0].evals - self.sub_components[0].evals[0])
            V_sub = np.eye(self.dim_sub, dtype=np.float64)
            n_t = self.sub_components[0].n_operator
        else:
            # Multi-component drive-free subsystem: diagonalize coupled sub-space
            from qbran.circuit.hilbert_space import HilbertSpace
            sub_hs = HilbertSpace(self.sub_components, backend="dense")
            for inter in self.circuit.interactions:
                if inter.component1 != self.drive_component and inter.component2 != self.drive_component:
                    sub_hs.interactions.append(inter)
            evals_s, evecs_s = sub_hs.diagonalize()
            H_sub = np.diag(evals_s - evals_s[0])
            V_sub = evecs_s[:, :self.n_branch]
            # Embed n_t
            tmon_sub_idx = self.sub_components.index(tmon)
            n_t = sub_hs.embed_operator(tmon.n_operator, component_idx=tmon_sub_idx, sparse=False)

        # 2. Drive mode operators
        a_d = np.diag(np.sqrt(np.arange(1, self.drive_dim, dtype=float)), 1)
        x_d = a_d + a_d.T
        n_d = a_d.T @ a_d

        # 3. Coupling to drive
        g_d = 0.0
        for inter in self.circuit.interactions:
            if (inter.component1 == self.drive_component and inter.component2 in self.sub_components) or \
               (inter.component2 == self.drive_component and inter.component1 in self.sub_components):
                g_d = inter.g
                break

        # Full static Hamiltonian (excluding swept drive frequency)
        # H = H_sub (x) I_d + I_sub (x) (w_d * n_d) + g_d * n_t (x) x_d
        H_full_static = np.kron(H_sub, np.eye(self.drive_dim)) + g_d * np.kron(n_t, x_d)

        # Mode diagonals for population tracking across all subsystems
        def _kron_list(ops):
            return reduce(np.kron, ops)

        def _number_diag(k, dims):
            factors = [np.ones(d, dtype=float) for d in dims]
            factors[k] = np.arange(dims[k], dtype=float)
            return _kron_list(factors)

        all_dims = self.sub_dims + [self.drive_dim]
        mode_diags = np.stack([_number_diag(k, all_dims) for k in range(len(all_dims))])
        sweep_diag = _number_diag(len(all_dims) - 1, all_dims)

        self._H_static = jnp.asarray(H_full_static, dtype=jnp.float64)
        self._sweep_diag = jnp.asarray(sweep_diag, dtype=jnp.float64)
        self._adag = jnp.asarray(a_d.T, dtype=jnp.float64)
        self._V_sub = jnp.asarray(V_sub, dtype=jnp.float64)
        self._mode_diags = jnp.asarray(mode_diags, dtype=jnp.float64)

    def run(self, frequencies: Optional[Union[np.ndarray, List[float], float]] = None) -> Any:
        """Executes the quantum branch tracking using JAX."""
        from qbran.analysis.result import Result

        # Determine frequency array
        if frequencies is None:
            w_arr = np.array([self.drive_component.E_osc], dtype=float)
        else:
            w_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))

        n_rungs = getattr(self.options, "n_rungs", None) or max(2, self.drive_dim - 2)
        if n_rungs > self.drive_dim:
            n_rungs = self.drive_dim

        kernel = _make_quantum_kernel(
            self._H_static,
            self._sweep_diag,
            self._mode_diags,
            self._adag,
            jnp.asarray(self.branch_weights),
            jnp.asarray(self.bare_labels, dtype=float),
            self.dim_sub,
            self.drive_dim,
            n_rungs,
        )

        batched_kernel = jax.jit(jax.vmap(kernel, in_axes=(0, 0, None)))

        w_mod = w_arr  # folded modulo drive frequency
        out = batched_kernel(jnp.asarray(w_arr), jnp.asarray(w_mod), self._V_sub)
        res_dict = {k: np.asarray(v) for k, v in out.items()}

        # Squeeze leading frequency dimension if single frequency was evaluated
        single_freq = len(w_arr) == 1
        processed = {}
        for k, v in res_dict.items():
            processed[k] = v[0] if single_freq else v

        # Add physical coordinate axes and metadata
        processed["xi_sq"] = np.arange(n_rungs)
        processed["nr_list"] = np.arange(n_rungs)
        processed["drive_frequency"] = w_arr[0] if single_freq else w_arr
        processed["frequencies"] = w_arr
        processed["n_branches"] = self.n_branch

        # Return as Result container supporting both dict and attribute access
        metadata = {
            "kind": "quantum",
            "drive": self.drive_component.name,
            "dim_sub": self.dim_sub,
            "drive_dim": self.drive_dim,
            "n_rungs": n_rungs,
        }

        # Format quasienergies and hybridization for Result
        quasi_3d = res_dict["quasienergies"]
        hyb_3d = res_dict["hybridization"]

        result = Result(
            metadata=metadata,
            drive_freq=w_arr,
            drive_amp=np.arange(n_rungs),
            quasienergies=quasi_3d,
            hybridization=hyb_3d,
            partner=res_dict["partner_index"],
            partner_overlap=res_dict["partner_overlap"],
            observables={"avg_excitation": processed["avg_excitation"]},
        )

        # Attach raw quantum dictionary fields
        for k, v in processed.items():
            setattr(result, k, v)
            result._data[k] = v

        return result
