"""Joint Hilbert space of the circuit"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from qbran.circuit.components import Component, Transmon, Resonator
from qbran.circuit.interactions import Interaction
from qbran.core.gauge import make_real_gauge, apply_gauge_to_evecs, is_strictly_real

class HilbertSpace:
    """Joint Hilbert space of the circuit.

    Parameters:
    ----------
    components : List[Component]
        List of circuit components.
    backend : str, optional
        Backend for linear algebra representation: 'sparse' (SciPy CSR) or 'dense' (NumPy). Default is 'dense'.
    k : int, optional
        Number of lowest energy eigenstates to compute upon diagonalization. Default is None, which means all eigenstates will be computed.
    """

    def __init__(self, components: List[Component], backend: str = 'dense', k: int = None):
        self.components = list(components)
        self.backend = backend.lower()
        self.interactions: List[Interaction] = []

        self.dims = [comp.truncated_dim for comp in self.components]
        self.dimension = int(np.prod(self.dims))
        self.k = k if k is not None else self.dimension

        self.k = min(self.k, self.dimension)  # Ensure k does not exceed the total dimension

        # Precompute strides for flat product index conversions
        self._strides = []
        for i in range(len(self.dims)):
            stride = int(np.prod(self.dims[i + 1 :])) if i + 1 < len(self.dims) else 1
            self._strides.append(stride)

        self._evals: Optional[np.ndarray] = None
        self._evecs: Optional[np.ndarray] = None
        self._is_real: bool = True
        self._gauge_phases: Optional[np.ndarray] = None

    def add_interaction(
        self,
        *args,
        **kwargs
    ) -> Interaction:
        """Add a pairwise interaction between two subsystems.

        Accepts flexible signatures:
            add_interaction(g, component1, component2, kind='capacitive', ...)
            add_interaction(component1, component2, g=0.12, ...)
            add_interaction(component1, component2, 0.12, ...)
            add_interaction(g=0.12, component1=tmon, component2=res, ...)
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
        if g_val is None and len(nums) > 0:
            g_val = nums.pop(0)

        if c1 is None or c2 is None or g_val is None:
            raise ValueError(
                f"add_interaction requires coupling strength g and two components. Got: c1={c1}, c2={c2}, g={g_val}"
            )

        inter = Interaction(
            g=float(g_val),
            component1=c1,
            component2=c2,
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
        self, op: Union[np.ndarray, sp.csr_matrix], 
        component_idx: int,
        sparse: bool = True,
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """Embed a 1-body operator into the joint Hilbert space of the circuit."""
        kron_func = sp.kron if sparse else np.kron
        eye_func = (lambda d: sp.eye(d, format='csr', dtype=op.dtype)) if sparse else (lambda d: np.eye(d, dtype=op.dtype))

        result = None
        for i, sub in enumerate(self.components):
            curr_op = op if i == component_idx else eye_func(sub.truncated_dim)
            if sparse and not sp.issparse(curr_op):
                curr_op = sp.csr_matrix(curr_op)
            result = curr_op if result is None else kron_func(result, curr_op)

        if sparse and not sp.isspmatrix_csr(result):
            result = result.tocsr()
        return result

    def emmbed_2body(
        self,
        op1: Union[np.ndarray, sp.csr_matrix],
        idx1: int,
        op2: Union[np.ndarray, sp.csr_matrix],
        idx2: int,
        sparse: bool = True,
    ) -> Union[np.ndarray, sp.csr_matrix]:
        """Embed a 2-body operator into the joint Hilbert space of the circuit."""
        kron_func = sp.kron if sparse else np.kron
        eye_func = (lambda d: sp.eye(d, format='csr', dtype=op1.dtype)) if sparse else (lambda d: np.eye(d, dtype=op1.dtype))

        result = None
        for i, sub in enumerate(self.components):
            if i == idx1:
                curr_op = op1
            elif i == idx2:
                curr_op = op2
            else:
                curr_op = eye_func(sub.truncated_dim)
            if sparse and not sp.issparse(curr_op):
                curr_op = sp.csr_matrix(curr_op)
            result = curr_op if result is None else kron_func(result, curr_op)

        if sparse and not sp.isspmatrix_csr(result):
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
            Whether to return sparse CSR or dense matrix.
        enforce_real : bool, optional
            Whether to automatically apply unitary gauge transformation to guarantee
            a strictly real symmetric (float64) Hamiltonian. Default is True.
        """
        is_sparse = (self.backend == 'sparse') if sparse is None else bool(sparse)

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
            h_interaction = self.emmbed_2body(op1, idx1, op2, idx2, sparse=is_sparse)
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

    def diagonalize(self, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalize the joint Hamiltonian of the circuit and return the lowest k eigenvalues and eigenvectors."""
        k_val = self.k if k is None else min(int(k), self.dimension)
        H = self.get_hamiltonian(sparse=(self.backend == 'sparse'), enforce_real=True)

        if self.backend == 'sparse' and k_val < self.dimension:
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

    def __getitem__(self, key: str) -> np.ndarray:
        """Dictionary access for evals and evecs."""
        if self._evals is None or self._evecs is None:
            self.diagonalize()
        if key == 'evals':
            return self._evals
        elif key == 'evecs':
            return self._evecs
        raise KeyError(f"Invalid key '{key}'. Valid keys are 'evals' and 'evecs'.")

    def BranchAnalysis(
        self,
        kind: str = "quantum",
        options: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Returns a BranchAnalysis engine configured for this circuit."""
        from qbran.analysis.analysis import BranchAnalysis
        return BranchAnalysis(circuit=self, kind=kind, options=options, **kwargs)