from typing import Tuple, Optional, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eigh
from tqdm import tqdm

from qbran.circuit.components import Transmon
from qbran.circuit.hilbert_space import HilbertSpace
from qbran.analysis.analysis import Options
from qbran.analysis.result import Result

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

class ShirleyCache:
    """Cache frequency-independent quantities used by Shirley analysis."""

    def __init__(self, circuit: HilbertSpace, options: Options):
        self.circuit = circuit
        self.options = options
        self.d = circuit.dimension
        self.K = options.K
        self.D = (2 * self.K + 1) * self.d
        self.sparse = options.sparse

        drive_op = options.drive_op
        if drive_op is None:
            tmon = circuit.components[options.drive_idx]
            if not isinstance(tmon, Transmon):
                raise ValueError(
                    f"Component at index {options.drive_idx} is not a Transmon. "
                    "Please provide a valid drive operator."
                )
            drive_op = tmon.n_operator

        self.H0 = circuit.get_hamiltonian(sparse=self.sparse, enforce_real=True)
        self.V = circuit.embed_operator(drive_op, options.drive_idx, sparse=True)

        n_op = sp.diags(np.arange(self.d)) if self.sparse else np.diag(np.arange(self.d))
        self.N = kron(id(2 * self.K + 1, self.sparse), n_op, self.sparse)
        self.H_drive = build_shirley_off_diagonal(self.V, 1.0, self.K, self.sparse)

        H0_dense = self.H0.toarray() if sp.issparse(self.H0) else self.H0
        self.bare_evals, self.bare_evecs = np.linalg.eigh(H0_dense)

    def initial_states(self, freq: float) -> np.ndarray:
        """Return folded zero-drive states for a drive frequency."""
        prev_evecs = np.zeros((self.D, self.d), dtype=complex)
        for state in range(self.d):
            harmonic = -int(round(self.bare_evals[state] / freq))
            harmonic = max(-self.K, min(self.K, harmonic))
            start_idx = (harmonic + self.K) * self.d
            prev_evecs[start_idx:start_idx + self.d, state] = self.bare_evecs[:, state]
        return prev_evecs

    def diagonal(self, freq: float) -> Union[np.ndarray, sp.csr_matrix]:
        """Return the frequency-dependent diagonal Sambe block."""
        return build_shirley_diagonal(self.H0, freq, self.K, self.sparse)

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

def build_shirley_diagonal(
    H0: Union[np.ndarray, sp.csr_matrix],
    omega: float,
    K: int,
    sparse: bool = True
) -> Union[np.ndarray, sp.csr_matrix]:
    """Construct the diagonal part of the Shirley Hamiltonian."""
    if sparse:
        I_K = sp.eye(2*K + 1, format='csr')
        N_K = sp.diags(np.arange(-K, K + 1), 0, shape=(2*K + 1, 2*K + 1), format='csr')
        H0_s = sp.kron(I_K, H0, format='csr') + sp.kron(N_K, sp.eye(H0.shape[0], format='csr'), format='csr') * omega
        return H0_s
    else:
        H0 = H0.toarray() if sp.issparse(H0) else H0
        I_K = np.eye(2*K + 1)
        N_K = np.diag(np.arange(-K, K + 1))
        H0_s = np.kron(I_K, H0) + np.kron(N_K, np.eye(H0.shape[0])) * omega
        return H0_s

def build_shirley_off_diagonal(
    V: Union[np.ndarray, sp.csr_matrix],
    amp: float,
    K: int,
    sparse: bool = True
) -> Union[np.ndarray, sp.csr_matrix]:
    """Construct the off-diagonal part of the Shirley Hamiltonian."""
    off_diag = 0.5 * amp * V
    sup_diag = sp.diags([1], [1],  shape=(2*K+1, 2*K+1), format='csr')
    sub_diag = sp.diags([1], [-1], shape=(2*K+1, 2*K+1), format='csr')
    HF_off = sp.kron(sup_diag, off_diag.conj().T, format='csr') + sp.kron(sub_diag, off_diag, format='csr')
    HF_off = HF_off if sparse else HF_off.toarray()
    return HF_off

def init_sorting(
    H0 : Union[np.ndarray, sp.csr_matrix],
    omega : float,
    d : int,
    D: int,
    K: int,
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
    v0: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize the Shirley Hamiltonian and return the lowest k eigenvalues and eigenvectors."""

    if sp.issparse(HF):
        if v0 is None: evals, evecs = spla.eigsh(HF, k=k, sigma=1e-6, which='LM')
        else: evals, evecs = spla.eigsh(HF, k=k, sigma=1e-6, which='LM', v0=v0)
    else:
        evals, evecs = eigh(HF, subset_by_index=[0, k-1], check_finite=False)
        idxs = np.argsort(evals)
        evals = evals[idxs]
        evecs = evecs[:, idxs]

    return evals, evecs

def shirley_single_frequency(
    circuit: HilbertSpace,
    freq: float,
    amps: Union[np.ndarray, list],
    options: Options,
    cache: Optional[ShirleyCache] = None,
):
    cache = ShirleyCache(circuit, options) if cache is None else cache
    k = options.k
    k = cache.d if k is None or options.sparse is False else k

    prev_evecs = cache.initial_states(freq)
    evecs = None
    quasienergies = np.zeros((len(amps), k), dtype=np.float64)
    excitations   = np.zeros((len(amps), k), dtype=np.float64)
    hybridization = np.zeros((len(amps), k), dtype=np.float64)

    # Precompute the diagonal part of the Floquet Hamiltonian
    H0_s = cache.diagonal(freq)

    for i, amp in enumerate(amps):
        HF = H0_s + amp * cache.H_drive

        evals, evecs = diagonalize_shirley(HF, k, v0=evecs[:, 0] if evecs is not None else None)

        # Sort the eigenvectors to track the states
        overlap = np.abs(prev_evecs.conj().T @ evecs)
        _, col_ind = linear_sum_assignment(-overlap)

        current_evals = evals[col_ind]
        current_evecs = evecs[:, col_ind]
        prev_evecs = current_evecs

        Npsi = cache.N @ current_evecs
        Nexp = np.sum(current_evecs.conj() * Npsi, axis=0).real

        if options.hybridization == "susceptibility":
            dH_matrix = current_evecs.conj().T @ (cache.H_drive @ current_evecs)
            delta = current_evals[None, :] - current_evals[:, None]
            numerator = np.abs(dH_matrix) ** 2
            np.fill_diagonal(delta, np.inf)
            hybridization[i] = np.sum(numerator / delta**2, axis=1)

        quasienergies[i] = current_evals
        excitations[i] = Nexp

    return quasienergies, excitations, hybridization


def shirley_branch_analysis(
    circuit: HilbertSpace,
    freq: Union[np.ndarray, list],
    amps: Union[np.ndarray, list],
    options: Options,
) -> Result:

    options.k = options.k if options.k is not None else circuit.dimension
    options.k = min(options.k, circuit.dimension)

    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    amps = np.asarray(amps, dtype=float)
    cache = ShirleyCache(circuit, options)

    res = Result(
        metadata   = None,
        drive_freq = freq,
        drive_amp  = amps,
        quasienergies = np.zeros((len(freq), len(amps), options.k), dtype=np.float64),
        hybridization = np.zeros((len(freq), len(amps), options.k), dtype=np.float64),
        observables={
            "avg_excitation" : np.zeros((len(freq), len(amps), options.k), dtype=np.float64),
        },
    )
    
    for i, drive_freq in enumerate(tqdm(freq, desc="Branch Analysis", total=len(freq))):
        quasienergies, excitations, hybridization = shirley_single_frequency(
            circuit, drive_freq, amps, options, cache=cache
        )
        res.quasienergies[i] = quasienergies
        res.observables["avg_excitation"][i] = excitations
        res.hybridization[i] = hybridization

    return res