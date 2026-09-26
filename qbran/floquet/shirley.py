from typing import Tuple, Optional, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import linear_sum_assignment

from qbran.circuit.components import Transmon
from qbran.circuit.hilbert_space import HilbertSpace
from qbran.analysis.analysis import Options

def kron(
    A: Union[np.ndarray, sp.csr_matrix], 
    B: Union[np.ndarray, sp.csr_matrix], 
    sparse: bool
) -> Union[np.ndarray, sp.csr_matrix]:
    """Kronecker product of two matrices, supporting both dense and sparse formats."""
    if sparse: 
        return sp.kron(A, B, format='csr')
    else:
        A = A.toarray() if sp.issparse(A) else A
        B = B.toarray() if sp.issparse(B) else B 
        return np.kron(A, B)

def id(D: int, sparse: bool = True) -> Union[np.ndarray, sp.csr_matrix]:
    """Identity matrix of size D x D, supporting both dense and sparse formats."""
    if sparse: return sp.eye(D, format='csr')
    else: return np.eye(D)

def build_shirley_hamiltonian(
    H0: Union[np.ndarray, sp.csr_matrix],
    V: Union[np.ndarray, sp.csr_matrix],
    omega: float,
    amp: float,
    K: int,
    sparse: bool = True
) -> Union[np.ndarray, sp.csr_matrix]:
    r"""Construct the block-tridiagonal Shirley Hamiltonian in frequency domain.
    The Sambe-Shirley Floquet operator is:
    $$\mathcal{H}_{F, mn} = (H_0 + m \omega_d I) \delta_{mn} + \frac{A}{2} H_1 (\delta_{m, n+1} + \delta_{m, n-1})$$
    where Fourier harmonics $m, n \in [-K, K]$.

    Parameters
    ----------
    H0 : np.ndarray
        Drift Hamiltonian (diagonal or square matrix), shape `(D, D)`.
    V : np.ndarray
        Drive operator, shape `(D, D)`.
    omega_d : float
        Drive angular frequency $\omega_d$.
    amp : float
        Drive amplitude $A$.
    K : int, optional
        Fourier harmonic cutoff $K$, (total $2 K + 1$ blocks).
    sparse : bool, optional
        Whether to return a SciPy CSR sparse matrix or dense NumPy array. Default is True.

    Returns
    -------
    np.ndarray or sp.csr_matrix
        Shirley matrix of shape `((2*K+1)*D, (2*K+1)*D)`.
    """

    # Diagonal blocks: H0 + m * omega * I
    I_K = sp.eye(2*K + 1, format='csr')
    N_K = sp.diags(np.arange(-K, K + 1), 0, shape=(2*K + 1, 2*K + 1), format='csr')
    HF = sp.kron(I_K, H0, format='csr') + sp.kron(N_K, sp.eye(H0.shape[0], format='csr'), format='csr') * omega

    # Off-diagonal blocks: A * V / 2
    off_diag = 0.5 * amp * V
    sup_diag = sp.diags([1], [1],  shape=(2*K+1, 2*K+1), format='csr')
    sub_diag = sp.diags([1], [-1], shape=(2*K+1, 2*K+1), format='csr')
    HF = HF + sp.kron(sup_diag, off_diag.conj().T, format='csr')
    HF = HF + sp.kron(sub_diag, off_diag         , format='csr')

    HF = HF if sparse else HF.toarray()
    
    return HF

def init_sorting(
    H0 : Union[np.ndarray, sp.csr_matrix],
    omega : float,
    d : int,
    D: int,
    K: int,
    sparse: bool,
) -> np.ndarray:
    """Construct the ideal folded states at A=0 to lock the sorting"""
    H0_dense = H0.toarray() if sp.issparse(H0) else H0
    eigvals, eigvecs = np.linalg.eigh(H0_dense)

    prev_evecs = np.zeros((D, d), dtype=complex)
    for m in range(d):
        k_m = -int(round(eigvals[m] / omega))
        k_m = max(-K, min(K, k_m))

        # Place the eigenvector in the correct block of the Sambe space
        start_idx = (k_m + K) * d
        end_idx = start_idx + d
        prev_evecs[start_idx:end_idx, m] = eigvecs[:, m]
    return prev_evecs

def diagonalize_shirley(
    HF: Union[np.ndarray, sp.csr_matrix],
    k: int,
    sparse: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize the Shirley Hamiltonian and return the lowest k eigenvalues and eigenvectors."""

    if sparse:
        evals, evecs = spla.eigsh(HF, k=k, sigma=1e-6, which='LM')
    else:
        evals, evecs = np.linalg.eigh(HF)
        idxs = np.argsort(evals)
        evals = evals[idxs]
        evecs = evecs[:, idxs]

    return evals, evecs

def shirley_single_frequency(
    circuit: HilbertSpace,
    freq: float,
    amps: Union[np.ndarray, list],
    options: Options,
):
    d = circuit.dimension   # Hilbert space of the circuit
    K = options.K           # Number of Fourier harmonics
    D = (2 * K + 1) * d     # Total dimension of the expanded Sambe space
    sparse = options.sparse

    k = options.k
    k = d if k is None or options.sparse is False else k

    drive_op = options.drive_op
    drive_idx = options.drive_idx
    if drive_op is None:
        tmon = circuit.components[drive_idx]
        if not isinstance(tmon, Transmon):
            raise ValueError(f"Component at index {drive_idx} is not a Transmon. Please provide a valid drive operator.")
        drive_op = tmon.n_operator

    V = circuit.embed_operator(drive_op, options.drive_idx, sparse=True)
    H0 = circuit.get_hamiltonian(sparse=sparse, enforce_real=True)

    n_op = sp.diags(np.arange(d)) if sparse else np.diag(np.arange(d))
    N = kron(id(2*K + 1, sparse),  n_op, sparse)

    prev_evecs = init_sorting(H0, freq, d, D, K, sparse)
    quasienergies = np.zeros((len(amps), k), dtype=np.float64)
    excitations   = np.zeros((len(amps), k), dtype=np.float64)

    for i, amp in enumerate(amps):
        HF = build_shirley_hamiltonian(H0, V, freq, amp, K, sparse)
        evals, evecs = diagonalize_shirley(HF, k, sparse)

        # Sort the eigenvectors to track the states
        overlap = np.abs(prev_evecs.conj().T @ evecs)
        _, col_ind = linear_sum_assignment(-overlap)

        current_evals = evals[col_ind]
        current_evecs = evecs[:, col_ind]
        prev_evecs = current_evecs
        Nexp = np.diag(current_evecs.conj().T @ (N @ current_evecs))

        quasienergies[i] = current_evals
        excitations[i] = Nexp

    return quasienergies, excitations
