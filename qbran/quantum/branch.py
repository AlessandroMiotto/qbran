"""JAX-accelerated fully quantum branch analysis (Dumas et al. 2024).

Tracks eigenstates along the drive resonator ladder using JAX top-k, lax.scan,
and chunked memory-safe batching.
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
        """Identifies drive & swept modes, ordering circuit with drive as last factor."""
        comps = list(self.circuit.components)

        # 1. Resolve drive mode
        if drive is None:
            res_list = [c for c in comps if isinstance(c, Resonator)]
            if not res_list:
                raise ValueError("No Resonator found in circuit to act as drive mode.")
            self.drive_component = res_list[-1]
        elif isinstance(drive, (Resonator, Transmon)):
            self.drive_component = drive
        elif isinstance(drive, str):
            matched = [c for c in comps if c.name.lower() == drive.lower()]
            if not matched and drive.lower() in ("res", "resonator", "drive", "readout", "ro"):
                res_list = [c for c in comps if isinstance(c, Resonator)]
                if res_list:
                    matched = [res_list[-1]]
            if not matched and drive.lower().startswith("res"):
                try:
                    num_str = "".join(filter(str.isdigit, drive))
                    idx = int(num_str) - 1 if num_str else -1
                    res_list = [c for c in comps if isinstance(c, Resonator)]
                    if 0 <= idx < len(res_list):
                        matched = [res_list[idx]]
                except Exception:
                    pass
            if not matched:
                raise ValueError(
                    f"Drive component '{drive}' not found in circuit. Available: {[c.name for c in comps]}"
                )
            self.drive_component = matched[0]
        elif isinstance(drive, int):
            self.drive_component = comps[drive]
        else:
            raise TypeError(f"Invalid drive specification: {drive}")

        # Drive mode is always the last tensor factor
        self.sub_components = [c for c in comps if c != self.drive_component]
        self.ordered_components = self.sub_components + [self.drive_component]

        self.drive_dim = self.drive_component.truncated_dim
        self.sub_dims = [c.truncated_dim for c in self.sub_components]
        self.dim_sub = int(np.prod(self.sub_dims)) if self.sub_dims else 1
        self.total_dim = self.dim_sub * self.drive_dim

        # 2. Resolve swept mode
        sweep_mode = getattr(self.options, "sweep_mode", "drive") if self.options else "drive"
        if sweep_mode in ("drive", None):
            self.sweep_component = self.drive_component
        elif isinstance(sweep_mode, (Resonator, Transmon)):
            self.sweep_component = sweep_mode
        elif isinstance(sweep_mode, str):
            matched = [c for c in comps if c.name.lower() == sweep_mode.lower()]
            if not matched and sweep_mode.lower().startswith("res"):
                try:
                    num_str = "".join(filter(str.isdigit, sweep_mode))
                    idx = int(num_str) - 1 if num_str else 0
                    spec_res = [c for c in self.sub_components if isinstance(c, Resonator)]
                    if 0 <= idx < len(spec_res):
                        matched = [spec_res[idx]]
                except Exception:
                    pass
            self.sweep_component = matched[0] if matched else self.drive_component
        elif isinstance(sweep_mode, int):
            self.sweep_component = self.ordered_components[sweep_mode]
        else:
            self.sweep_component = self.drive_component

        self.sweeps_drive = (self.sweep_component == self.drive_component)
        self.sweep_idx = self.ordered_components.index(self.sweep_component)

        # Bare labels for drive-free subsystem
        self.bare_labels = np.array(
            list(product(*[range(d) for d in self.sub_dims])), dtype=int
        )
        self.n_branch = len(self.bare_labels)
        self.branch_weights = np.arange(self.n_branch, dtype=float)

    def _get_subsystem_basis(self, omega: Optional[float] = None) -> np.ndarray:
        """Returns the branch-labelled dressed basis V_sub for the drive-free subsystem."""
        if len(self.sub_components) == 1 and isinstance(self.sub_components[0], Transmon):
            return np.eye(self.dim_sub, dtype=np.float64)

        from qbran.circuit.hilbert_space import HilbertSpace
        sub_hs = HilbertSpace(self.sub_components, backend="dense")
        for inter in self.circuit.interactions:
            if inter.component1 in self.sub_components and inter.component2 in self.sub_components:
                sub_hs.interactions.append(inter)

        # If a spectator resonator frequency is swept, temporarily set its E_osc to omega
        if not self.sweeps_drive and omega is not None and hasattr(self.sweep_component, "E_osc"):
            orig_e = self.sweep_component.E_osc
            self.sweep_component.E_osc = float(omega)
            evals_s, evecs_s = sub_hs.diagonalize()
            self.sweep_component.E_osc = orig_e
        else:
            evals_s, evecs_s = sub_hs.diagonalize()

        return np.real(evecs_s[:, :self.n_branch])

    def _build_static_matrices(self):
        """Constructs static full Hamiltonian and observable operators in JAX."""
        all_dims = self.sub_dims + [self.drive_dim]

        def _kron_list(ops):
            return reduce(np.kron, ops)

        def _embed(op, idx):
            factors = [np.eye(d, dtype=float) for d in all_dims]
            factors[idx] = op
            return _kron_list(factors)

        def _embed_2body(op1, idx1, op2, idx2):
            factors = [np.eye(d, dtype=float) for d in all_dims]
            factors[idx1] = op1
            factors[idx2] = op2
            return _kron_list(factors)

        def _number_diag(k):
            factors = [np.ones(d, dtype=float) for d in all_dims]
            factors[k] = np.arange(all_dims[k], dtype=float)
            return _kron_list(factors)

        # 1. Bare Hamiltonians for non-swept modes
        H_static = np.zeros((self.total_dim, self.total_dim), dtype=np.float64)
        for idx, comp in enumerate(self.ordered_components):
            if comp == self.sweep_component:
                continue  # Swept mode is added at runtime as omega * sweep_diag
            h_bare = comp.bare_hamiltonian(sparse=False)
            H_static += _embed(h_bare, idx)

        # 2. Add all pairwise interactions
        for inter in self.circuit.interactions:
            idx1 = self.ordered_components.index(inter.component1)
            idx2 = self.ordered_components.index(inter.component2)
            op1, op2, g = inter.get_operator()
            if np.iscomplexobj(op1) and np.max(np.abs(op1.imag)) < 1e-12:
                op1 = op1.real
            if np.iscomplexobj(op2) and np.max(np.abs(op2.imag)) < 1e-12:
                op2 = op2.real
            h_int = _embed_2body(op1, idx1, op2, idx2)
            H_static += g * h_int

        if np.iscomplexobj(H_static) and np.max(np.abs(H_static.imag)) < 1e-12:
            H_static = H_static.real

        # 3. Sweep diagonal and mode diagonals
        self._sweep_diag = jnp.asarray(_number_diag(self.sweep_idx), dtype=jnp.float64)
        self._mode_diags = jnp.asarray(
            np.stack([_number_diag(k) for k in range(len(all_dims))]), dtype=jnp.float64
        )

        # 4. Drive ladder operator (a_d^dag)
        a_d = np.diag(np.sqrt(np.arange(1, self.drive_dim, dtype=float)), 1)
        self._adag = jnp.asarray(a_d.T, dtype=jnp.float64)
        self._H_static = jnp.asarray(H_static, dtype=jnp.float64)

        # 5. Base subsystem basis V_sub
        self._V_sub = jnp.asarray(self._get_subsystem_basis(), dtype=jnp.float64)

    def run(self, frequencies: Optional[Union[np.ndarray, List[float], float]] = None) -> Any:
        """Executes the quantum branch tracking using JAX with chunked memory management."""
        from qbran.analysis.result import Result

        # Determine frequency array
        if frequencies is None:
            w_arr = np.array([self.sweep_component.E_osc], dtype=float)
        else:
            w_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))

        n_rungs = getattr(self.options, "n_rungs", None) or max(2, self.drive_dim - 2)
        if n_rungs > self.drive_dim:
            n_rungs = self.drive_dim

        # Batch size management: chunk frequencies to prevent OOM
        # At dim=5400, each matrix is 233 MB; bs=1 keeps working memory strictly < 700 MB
        bs = getattr(self.options, "batch_size", 1) or 1
        bs = max(1, int(bs))
        if self.total_dim > 2000:
            max_safe = max(1, int(4e9 / (self.total_dim**2 * 8 * 10)))
            bs = min(bs, max_safe)

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

        batched_kernel = jax.jit(jax.vmap(kernel, in_axes=(0, 0, 0)))

        # Subsystem basis V_sub and folding modulus w_mod across frequencies
        if self.sweeps_drive:
            w_mod = w_arr
            V_all = np.repeat(np.asarray(self._V_sub)[None, :, :], len(w_arr), axis=0)
        else:
            w_mod = np.full_like(w_arr, float(self.drive_component.E_osc))
            V_all = np.empty((len(w_arr), self.dim_sub, self.n_branch), dtype=float)
            for i, wi in enumerate(w_arr):
                V_all[i] = self._get_subsystem_basis(wi)

        # Chunked execution loop: keeps memory under 1 GB regardless of frequency count
        out_chunks = []
        for i in range(0, len(w_arr), bs):
            sl = slice(i, i + bs)
            chunk = batched_kernel(
                jnp.asarray(w_arr[sl]),
                jnp.asarray(w_mod[sl]),
                jnp.asarray(V_all[sl]),
            )
            # Transfer chunk to host CPU memory immediately to free accelerator buffer
            out_chunks.append(jax.device_get(chunk))

        res_dict = {
            k: np.concatenate([c[k] for c in out_chunks], axis=0)
            for k in out_chunks[0]
        }

        # Squeeze leading frequency dimension if single frequency was evaluated
        single_freq = len(w_arr) == 1
        processed = {}
        for k, v in res_dict.items():
            processed[k] = v[0] if single_freq else v

        # Add physical coordinate axes and metadata
        processed["xi_sq"] = np.arange(n_rungs)
        processed["nr_list"] = np.arange(n_rungs)
        processed["drive_frequency"] = w_arr[0] if (single_freq and self.sweeps_drive) else (
            float(self.drive_component.E_osc) if not self.sweeps_drive else w_arr
        )
        processed["frequencies"] = w_arr
        processed["n_branches"] = self.n_branch

        metadata = {
            "kind": "quantum",
            "drive": self.drive_component.name,
            "sweep_mode": self.sweep_component.name,
            "dim_sub": self.dim_sub,
            "drive_dim": self.drive_dim,
            "total_dim": self.total_dim,
            "n_rungs": n_rungs,
        }

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

        for k, v in processed.items():
            setattr(result, k, v)
            result._data[k] = v

        return result
