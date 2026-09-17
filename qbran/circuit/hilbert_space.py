"""Joint Hilbert space of the circuit"""

from typing import List, Optional, Tuple, Tuple, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from qbran.circuit.components import Component, Transmon, Resonator
from qbran.circuit.interactions import Interaction

class HilbertSpace:
    """Joint Hilbert space of the circuit.

    Parameters:
    ----------
    components : List[Component]
        List of circuit components.
    backend : str, optional
        Backend for linear algebra representation: 'sparse' (SciPy CSR) or 'dense' (NumPy/JAX). Default is 'sparse'.
    k : int, optional
        Number of lowest energy eigenstates to compute upon diagonalization. Default is 10.
    """

    def __init__(self, components: List[Component], backend: str = 'sparse', k: int = 10):
        self.components = list(components)
        self.backend = backend.lower()
        self.k = int(k)
        self.interactions: List[Interaction] = []

        self.dims = [comp.truncated_dim for comp in self.components]
        self.dimension = int(np.prod(self.dims))

        self.k = min(self.k, self.dimension)  # Ensure k does not exceed the total dimension

    def add_interaction(
            self,
            g: float,
            component1: Component,
            component2: Component,
            kind: str = 'capacitive',
            op1: Optional[np.ndarray] = None,
            op2: Optional[np.ndarray] = None
    ) -> Interaction:
        """Add a pairwise interaction between two subsystems."""
        inter = Interaction(g=g, component1=component1, component2=component2, kind=kind, op1=op1, op2=op2)
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

    def get_hamiltonian(self, sparse : Optional[bool] = None) -> Union[np.ndarray, 'sp.csr_matrix']:
        """Build the joint undriven Hamiltonian of the circuit."""

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

        # TODO: Add gauge transformation
        
        return H_total

    def diagonalize(self, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalize the joint Hamiltonian of the circuit and return the lowest k eigenvalues and eigenvectors."""
        k_val = self.k if k is None else min(int(k), self.dimension)
        H = self.get_hamiltonian(sparse=(self.backend == 'sparse'))

        if self.backend == 'sparse' and k_val < self.dimension:
            evals, evecs = spla.eigsh(H, k=k_val, which='SA')
        else:
            H_dense = H.toarray() if sp.issparse(H) else H
            evals, evecs = np.linalg.eigh(H_dense)
            evals = evals[:k_val]
            evecs = evecs[:, :k_val]

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