"""Joint Hilbert space of the circuit."""

from typing import Any, List, Optional, Tuple, Union, Dict, TYPE_CHECKING
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from qbran.circuit.components import Component, Transmon, Resonator
from qbran.circuit.interactions import Interaction
from qbran.core.gauge import (
    make_real_gauge,
    apply_gauge_to_evecs,
    apply_gauge_to_operator,
    is_strictly_real,
)

if TYPE_CHECKING:
    from qbran.circuit.fitter import FitResult
    from qbran.analysis.sweep import SweepResult
    from qbran.analysis.analysis import Options


class HilbertSpace:
    """Joint Hilbert space of the circuit.

    Parameters
    ----------
    components : List[Component]
        List of circuit components.
    sparse : bool, optional
        Whether to use sparse matrices for the Hamiltonian by default. Default is True.
    k : int, optional
        Number of lowest energy eigenstates to compute upon diagonalization.
        Default is None (all eigenstates).
    """

    def __init__(self, components: List[Component], sparse: bool = True, k: Optional[int] = None):
        self.components = list(components)
        self.sparse = bool(sparse)
        self.interactions: List[Interaction] = []

        self.dims = [comp.truncated_dim for comp in self.components]
        self.dimension = int(np.prod(self.dims))
        self.k = k if k is not None else self.dimension
        self.k = min(self.k, self.dimension)

        # Precompute strides for flat product index conversions
        self._strides = []
        for i in range(len(self.dims)):
            stride = int(np.prod(self.dims[i + 1 :])) if i + 1 < len(self.dims) else 1
            self._strides.append(stride)

        self._evals: Optional[np.ndarray] = None
        self._evecs: Optional[np.ndarray] = None
        self._is_real: bool = True
        self._gauge_D: Optional[np.ndarray] = None

    def add_interaction(
        self,
        *args,
        **kwargs
    ) -> Interaction:
        """Add a pairwise interaction between two subsystems.

        Accepts flexible signatures:
            add_interaction(component1, component2, g=0.12, kind='capacitive')
            add_interaction(component1, component2, 0.12)
            add_interaction(component1, component2)  # uninitialized g for fitting
            add_interaction(g=0.12, component1=tmon, component2=res)
            add_interaction(0.12, tmon, res)

        Parameters
        ----------
        *args :
            Flexible positional arguments: components (Component instances), coupling g (float),
            or kind (str).
        g : float, optional
            Coupling strength in GHz. If omitted (None), the interaction is created with
            uninitialized coupling to be fitted from target dispersive shifts via
            ``circuit.fit_parameters(...)``. Default is None.
        component1, component2 : Component, optional
            Interacting subsystems.
        kind : str, optional
            Type of interaction: 'capacitive', 'inductive', or 'custom'. Default is 'capacitive'.
        real_gauge : bool, optional
            Whether to express interaction in the real-symmetric gauge. Default is True.
        op1, op2 : np.ndarray, optional
            Custom interaction operators for component1 and component2 if kind is 'custom'.

        Returns
        -------
        Interaction
            The instantiated Interaction object added to this HilbertSpace.
        """
        g_val = kwargs.get('g', None)
        c1 = kwargs.get('component1', kwargs.get('c1', None))
        c2 = kwargs.get('component2', kwargs.get('c2', None))
        kind = kwargs.get('kind', 'capacitive')
        real_gauge = kwargs.get('real_gauge', True)
        op1 = kwargs.get('op1', None)
        op2 = kwargs.get('op2', None)

        remaining = list(args)
        comps = []
        nums = []
        for a in remaining:
            if isinstance(a, Component):
                comps.append(a)
            elif isinstance(a, (int, float, np.floating)):
                nums.append(float(a))
            elif isinstance(a, str):
                kind = a

        if c1 is None and len(comps) > 0:
            c1 = comps.pop(0)
        if c2 is None and len(comps) > 0:
            c2 = comps.pop(0)
        if c1 is None or c2 is None:
            raise ValueError(
                f"add_interaction requires two components. Got: c1={c1}, c2={c2}"
            )

        g = float(g_val) if g_val is not None else None
        inter = Interaction(
            component1=c1,
            component2=c2,
            g=g,
            kind=kind,
            real_gauge=real_gauge,
            op1=op1,
            op2=op2,
        )
        self.interactions.append(inter)
        self._evals = None
        self._evecs = None
        return inter

    def embed_operator(
        self,
        op: Union[np.ndarray, sp.csr_matrix],
        component_idx: int,
        sparse: Optional[bool] = None,
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """Embed a 1-body operator into the joint Hilbert space of the circuit."""
        is_sp = self.sparse if sparse is None else bool(sparse)
        kron_func = sp.kron if is_sp else np.kron
        eye_func = (lambda d: sp.eye(d, format='csr', dtype=op.dtype)) if is_sp else (lambda d: np.eye(d, dtype=op.dtype))

        result = None
        for i, sub in enumerate(self.components):
            curr_op = op if i == component_idx else eye_func(sub.truncated_dim)
            if is_sp and not sp.issparse(curr_op):
                curr_op = sp.csr_matrix(curr_op)
            result = curr_op if result is None else kron_func(result, curr_op)

        if is_sp and not sp.isspmatrix_csr(result):
            result = result.tocsr()
        return result

    def embed_2body(
        self,
        op1: Union[np.ndarray, sp.csr_matrix],
        idx1: int,
        op2: Union[np.ndarray, sp.csr_matrix],
        idx2: int,
        sparse: Optional[bool] = None,
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """Embed a 2-body operator into the joint Hilbert space of the circuit."""
        is_sp = self.sparse if sparse is None else bool(sparse)
        kron_func = sp.kron if is_sp else np.kron
        eye_func = (lambda d: sp.eye(d, format='csr', dtype=op1.dtype)) if is_sp else (lambda d: np.eye(d, dtype=op1.dtype))

        result = None
        for i, sub in enumerate(self.components):
            if i == idx1:
                curr_op = op1
            elif i == idx2:
                curr_op = op2
            else:
                curr_op = eye_func(sub.truncated_dim)
            if is_sp and not sp.issparse(curr_op):
                curr_op = sp.csr_matrix(curr_op)
            result = curr_op if result is None else kron_func(result, curr_op)

        if is_sp and not sp.isspmatrix_csr(result):
            result = result.tocsr()
        return result

    def get_hamiltonian(
        self,
        sparse: Optional[bool] = None,
        enforce_real: bool = True,
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """Build the joint undriven Hamiltonian of the circuit.
        
        Parameters
        ----------
        sparse : bool, optional
            Whether to return sparse CSR or dense matrix. Defaults to self.sparse.
        enforce_real : bool, optional
            Whether to automatically apply unitary gauge transformation to guarantee
            a strictly real symmetric (float64) Hamiltonian. Default is True.
        """
        is_sparse = self.sparse if sparse is None else bool(sparse)

        # Bare hamiltonian
        H_total = None
        for idx, sub in enumerate(self.components):
            h_bare = sub.bare_hamiltonian(sparse=is_sparse)
            h_embedded = self.embed_operator(h_bare, component_idx=idx, sparse=is_sparse)
            H_total = h_embedded if H_total is None else H_total + h_embedded

        # Add interactions
        for inter in self.interactions:
            op1, op2, g = inter.get_operator()
            idx1 = self.components.index(inter.component1)
            idx2 = self.components.index(inter.component2)
            h_interaction = self.embed_2body(op1, idx1, op2, idx2, sparse=is_sparse)
            H_total = H_total + g * h_interaction

        if enforce_real:
            H_real, D, success = make_real_gauge(H_total, sparse=is_sparse)
            self._gauge_D = D
            self._is_real = success
            if success:
                return H_real

        self._gauge_D = np.ones(self.dimension, dtype=np.complex128)
        self._is_real = is_strictly_real(H_total)
        return H_total

    def embed_drive_operator(
        self,
        drive_op: Union[np.ndarray, sp.csr_matrix],
        component_idx: int,
        sparse: Optional[bool] = None,
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """Embed drive operator and align with circuit gauge representation."""
        is_sp = self.sparse if sparse is None else bool(sparse)
        V = self.embed_operator(drive_op, component_idx=component_idx, sparse=is_sp)
        if self._is_real and self._gauge_D is not None:
            V = apply_gauge_to_operator(V, self._gauge_D)
        return V

    def diagonalize(self, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalize the joint Hamiltonian and return lowest k eigenvalues and eigenvectors."""
        k_val = self.k if k is None else min(int(k), self.dimension)
        H = self.get_hamiltonian(sparse=self.sparse, enforce_real=True)

        if self.sparse and k_val < self.dimension:
            # SciPy ARPACK sparse eigensolver
            evals, evecs = spla.eigsh(H, k=k_val, which='SA')
        else:
            # Dense NumPy eigensolver
            H_dense = H.toarray() if sp.issparse(H) else H
            evals, evecs = np.linalg.eigh(H_dense)
            evals = evals[:k_val]
            evecs = evecs[:, :k_val]

        if self._is_real:
            evals = np.real(evals)
            evecs = np.real(evecs)

        idx_sort = np.argsort(evals)
        self._evals = evals[idx_sort]
        self._evecs = evecs[:, idx_sort]

        return self._evals, self._evecs

    def state_to_index(self, *state_indices, **named_indices) -> int:
        """Convert subsystem state numbers (e.g., tmon=4, res=1) to flat product-basis index.

        Examples
        --------
        >>> circuit.state_to_index(4, 1)  # if dims=[13, 3] -> 4 * 3 + 1 = 13
        >>> circuit.state_to_index(tmon=4, res=1)
        """
        if state_indices and named_indices:
            raise ValueError("Provide either positional state indices or keyword arguments, not both.")

        if named_indices:
            indices = []
            for comp in self.components:
                cname = comp.name.lower()
                val = None
                for k, v in named_indices.items():
                    k_low = k.lower()
                    if k_low == cname or k_low in cname or cname in k_low:
                        val = v
                        break
                    if ("tmon" in k_low or "qubit" in k_low or "q" == k_low) and "transmon" in cname:
                        val = v
                        break
                    if ("res" in k_low or "cav" in k_low or "osc" in k_low) and "resonator" in cname:
                        val = v
                        break
                if val is None:
                    raise KeyError(f"Missing state for component '{comp.name}'")
                indices.append(val)
        else:
            indices = list(state_indices)

        if len(indices) != len(self.components):
            raise ValueError(f"Expected {len(self.components)} state indices, got {len(indices)}")

        flat_idx = 0
        for idx, stride, dim in zip(indices, self._strides, self.dims):
            if not (0 <= idx < dim):
                raise IndexError(f"State index {idx} out of range [0, {dim - 1}]")
            flat_idx += idx * stride
        return flat_idx

    def index_to_state(self, flat_index: int) -> Tuple[int, ...]:
        """Convert a flat product-basis index into tuple of subsystem state numbers."""
        if not (0 <= flat_index < self.dimension):
            raise IndexError(f"Flat index {flat_index} out of range [0, {self.dimension - 1}]")

        states = []
        rem = flat_index
        for stride in self._strides:
            states.append(rem // stride)
            rem %= stride
        return tuple(states)

    def state_label(self, flat_index: int) -> str:
        """Return formatted ket label string (e.g. '|4, 1>') for a flat index."""
        states = self.index_to_state(flat_index)
        states_str = ", ".join(str(s) for s in states)
        return f"|{states_str}>"

    def __getitem__(self, key: str) -> np.ndarray:
        """Dictionary access for evals and evecs."""
        if self._evals is None or self._evecs is None:
            self.diagonalize()
        if key == 'evals':
            return self._evals
        elif key == 'evecs':
            return self._evecs
        raise KeyError(f"Invalid key '{key}'. Valid keys are 'evals' and 'evecs'.")

    def branch_analysis(
        self,
        kind: str = "shirley",
        options: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Returns a BranchAnalysis engine configured for this circuit."""
        from qbran.analysis.analysis import BranchAnalysis
        return BranchAnalysis(circuit=self, kind=kind, options=options, **kwargs)

    # Backward compatibility alias
    BranchAnalysis = branch_analysis

    def fit_parameters(
        self,
        target_frequencies: Optional[Dict[Union[str, Component], Union[float, Tuple[float, float]]]] = None,
        target_dispersive_shifts: Optional[Dict[Tuple[Union[str, Component], Union[str, Component]], float]] = None,
        target_anharmonicities: Optional[Dict[Union[str, Component], float]] = None,
        fixed_parameters: Optional[Dict[str, float]] = None,
        in_place: bool = True,
        xtol: float = 1e-8,
        ftol: float = 1e-8,
        max_nfev: int = 150,
    ) -> "FitResult":
        """Fit bare circuit parameters to match target dressed experimental observables.

        Parameters
        ----------
        target_frequencies : dict, optional
            Target dressed frequencies in GHz. Keys can be component names (str) or Component instances.
            Values can be:
            - float: target fundamental frequency (0 -> 1 transition) in GHz.
            - tuple (freq, alpha): for Transmon components, specifies both the 0 -> 1 frequency
              and the anharmonicity.
        target_dispersive_shifts : dict, optional
            Target cross-Kerr dispersive shifts in GHz. Keys are 2-tuples of components (names or instances).
            Dispersive shift is defined as:
                chi_ij = E(1_i, 1_j) - E(1_i, 0_j) - E(0_i, 1_j) + E(0_i, 0_j)
        target_anharmonicities : dict, optional
            Target transmon anharmonicities (self-Kerr) in GHz.
            Anharmonicity is defined as:
                alpha = E_12 - E_01
            (Note: can also be passed directly in ``target_frequencies`` as ``(freq, alpha)``).
        fixed_parameters : dict, optional
            Dictionary of bare parameters to keep fixed during the fit.
        in_place : bool, optional
            Whether to update the bare parameters on this circuit and its components/interactions
            in-place. Default is True.
        xtol : float, optional
            Relative error desired in the approximate solution. Default is 1e-8.
        ftol : float, optional
            Relative error desired in the objective sum of squares. Default is 1e-8.
        max_nfev : int, optional
            Maximum number of function evaluations for the optimizer. Default is 150.

        Returns
        -------
        FitResult
            Container with convergence diagnostics, bare parameters, and target vs fitted observables:
            - ``result.bare_parameters``: converged bare values (EJ, EC, E_osc, g)
            - ``result.summary()``: formatted ASCII diagnostic table
            - ``result.max_residual_mhz``: maximum discrepancy across all observables in MHz
            - ``result.success``: boolean flag indicating optimizer convergence

        Examples
        --------
        >>> import qbran as br
        >>> # Define uninitialized components (no bare parameters needed upfront)
        >>> tmon = br.Transmon(truncated_dim=10, name="tmon")
        >>> res = br.Resonator(truncated_dim=10, name="res")
        >>> circuit = br.HilbertSpace([tmon, res])
        >>> circuit.add_interaction(tmon, res)
        >>>
        >>> # Fit bare parameters to experimental dressed observables
        >>> res = circuit.fit_parameters(
        ...     target_frequencies={
        ...         tmon: (4.50, -0.25),  # 4.5 GHz freq, -250 MHz anharmonicity
        ...         res: 7.20,            # 7.2 GHz resonator freq
        ...     },
        ...     target_dispersive_shifts={
        ...         (tmon, res): -0.002,  # -2 MHz dispersive shift (or 0.002)
        ...     },
        ... )
        >>> print(res.summary())
        >>> print(res.bare_parameters)
        """
        from qbran.circuit.fitter import fit_parameters as _fit_parameters

        return _fit_parameters(
            circuit=self,
            target_frequencies=target_frequencies,
            target_dispersive_shifts=target_dispersive_shifts,
            target_anharmonicities=target_anharmonicities,
            fixed_parameters=fixed_parameters,
            in_place=in_place,
            xtol=xtol,
            ftol=ftol,
            max_nfev=max_nfev,
        )

    def sweep(
        self,
        parameters: Optional[Dict[Any, Union[List[float], np.ndarray]]] = None,
        drive_freq: Optional[Union[float, List[float], np.ndarray]] = None,
        drive_amp: Optional[Union[float, List[float], np.ndarray]] = None,
        kind: str = "shirley",
        options: Optional[Any] = None,
        n_amp_steps: int = 15,
        **kwargs,
    ) -> "SweepResult":
        """Run multidimensional parameter sweep over circuit parameters and drive axes.

        Sweeps arbitrary circuit parameters (e.g. resonator frequencies, transmon charge bias,
        interaction couplings) and drive axes (frequency, amplitude), assembling the resulting
        quasienergies and Xiao hybridization metrics into a multidimensional SweepResult.

        Circuit parameters are snapshotted beforehand and guaranteed to be restored upon completion.

        Parameters
        ----------
        parameters : dict, optional
            Dictionary mapping parameter identifiers to 1D arrays or lists of values.
            Supported key formats:
            - Dot-notation: ``{"buffer.E_osc": np.linspace(7.35, 7.55, 41)}``
            - Transmon bias: ``{"tmon.ng": np.linspace(0.0, 1.0, 51)}``
            - Component tuples: ``{(buffer, "E_osc"): np.linspace(7.3, 7.5, 20)}``
            - Interaction coupling: ``{(tmon, buffer): np.linspace(0.05, 0.15, 10)}``
        drive_freq : float, list, or ndarray
            Drive frequency (or frequencies) in GHz.
        drive_amp : float, list, or ndarray
            Drive amplitude (or amplitudes).
            - Scalar (e.g. ``drive_amp=0.2``): Automatically simulates an internal compact
              amplitude ramp from 0 to drive_amp to ensure valid Xiao diabatic tracking,
              and returns the extracted slice at drive_amp with the amplitude axis squeezed out.
            - 1D Array (e.g. ``drive_amp=np.linspace(0, 0.4, 20)``): Retains the full amplitude axis.
        kind : str, optional
            Floquet simulation framework: 'shirley' (default) or 'propagator'.
        options : Options, optional
            Simulation and parallelization options (e.g. ``Options(n_workers=8, tracked_states=[...])``).
        n_amp_steps : int, optional
            Number of amplitude ramp steps when drive_amp is scalar. Default is 15.

        Returns
        -------
        SweepResult
            Multidimensional dataset container:
            - ``res.hybridization``: N-D array of shape ``(*param_dims, *drive_dims, n_tracked)``
            - ``res.quasienergies``: N-D array of shape ``(*param_dims, *drive_dims, k)``
            - ``res.coords``: Dictionary of 1D coordinate arrays for each dimension
            - ``res.slice(**kwargs)``: Helper method to extract 2D/1D slices by coordinate value
            - ``res.save("sweep.h5")``: Export to HDF5 archive

        Examples
        --------
        >>> # 2D plane sweep: SMPD pump frequency vs buffer frequency at fixed pump amplitude
        >>> sweep_res = circuit.sweep(
        ...     parameters={"buffer.E_osc": np.linspace(7.35, 7.55, 41)},
        ...     drive_freq=np.linspace(6.30, 6.40, 101),
        ...     drive_amp=0.2,       # scalar: auto-ramps & extracts slice at 0.2
        ...     options=ops,
        ... )
        >>> # Plot 2D heatmap directly (tracked state index 2):
        >>> plt.pcolormesh(
        ...     sweep_res.coords["drive_freq"],
        ...     sweep_res.coords["buffer.E_osc"],
        ...     sweep_res.hybridization[:, :, 2],
        ... )
        """
        from qbran.analysis.sweep import ParameterSweep

        sweep_runner = ParameterSweep(
            circuit=self,
            parameters=parameters,
            drive_freq=drive_freq,
            drive_amp=drive_amp,
            kind=kind,
            options=options,
            n_amp_steps=n_amp_steps,
            **kwargs,
        )
        return sweep_runner.run()