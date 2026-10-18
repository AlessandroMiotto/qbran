"""Floquet analysis in extended Sambe space using Shirley approach."""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from qbran.floquet.hybridization import HybridizationFit


def _matrix_kron(
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


def _matrix_eye(D: int, sparse: bool = True) -> Union[np.ndarray, sp.csr_matrix]:
    """Identity matrix of size D x D, supporting both dense and sparse formats."""
    if sparse:
        return sp.eye(D, format='csr')
    return np.eye(D)


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
                    "Please provide an explicit drive_op."
                )
            drive_op = tmon.n_operator

        self.H0 = circuit.get_hamiltonian(sparse=self.sparse, enforce_real=True)
        # Embed drive operator and guarantee gauge alignment with H0
        self.V = circuit.embed_drive_operator(drive_op, options.drive_idx, sparse=True)

        n_diag = np.arange(self.d, dtype=np.float64)
        self.diag_N = np.tile(n_diag, 2 * self.K + 1)
        self.H_drive = build_shirley_off_diagonal(self.V, 1.0, self.K, self.sparse)

        H0_dense = self.H0.toarray() if sp.issparse(self.H0) else self.H0
        self.bare_evals, self.bare_evecs = np.linalg.eigh(H0_dense)
        self.Nrt = np.diag(n_diag)
        init_Nrt = np.real(np.einsum('ji,jk,ki->i', self.bare_evecs.conj(), self.Nrt, self.bare_evecs))
        order_Nrt = np.argsort(init_Nrt)
        self.bare_evals = self.bare_evals[order_Nrt]
        self.bare_evecs = self.bare_evecs[:, order_Nrt]

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

    $$\mathcal{H}_{F, mn} = (H_0 + m \omega_d I) \delta_{mn} + \frac{A}{2} V \delta_{m, n+1} + \frac{A}{2} V^\dagger \delta_{m, n-1}$$
    where Fourier harmonics $m, n \in [-K, K]$.
    """
    I_K = sp.eye(2 * K + 1, format='csr')
    N_K = sp.diags(np.arange(-K, K + 1), 0, shape=(2 * K + 1, 2 * K + 1), format='csr')
    HF = sp.kron(I_K, H0, format='csr') + sp.kron(N_K, sp.eye(H0.shape[0], format='csr'), format='csr') * omega

    # Off-diagonal blocks: A * V / 2
    off_diag = 0.5 * amp * V
    sup_diag = sp.diags([1], [1],  shape=(2 * K + 1, 2 * K + 1), format='csr')
    sub_diag = sp.diags([1], [-1], shape=(2 * K + 1, 2 * K + 1), format='csr')
    HF = HF + sp.kron(sup_diag, off_diag.conj().T, format='csr')
    HF = HF + sp.kron(sub_diag, off_diag,          format='csr')

    return HF if sparse else HF.toarray()


def build_shirley_diagonal(
    H0: Union[np.ndarray, sp.csr_matrix],
    omega: float,
    K: int,
    sparse: bool = True
) -> Union[np.ndarray, sp.csr_matrix]:
    """Construct the diagonal part of the Shirley Hamiltonian."""
    if sparse:
        I_K = sp.eye(2 * K + 1, format='csr')
        N_K = sp.diags(np.arange(-K, K + 1), 0, shape=(2 * K + 1, 2 * K + 1), format='csr')
        return sp.kron(I_K, H0, format='csr') + sp.kron(N_K, sp.eye(H0.shape[0], format='csr'), format='csr') * omega
    else:
        H0 = H0.toarray() if sp.issparse(H0) else H0
        I_K = np.eye(2 * K + 1)
        N_K = np.diag(np.arange(-K, K + 1))
        return np.kron(I_K, H0) + np.kron(N_K, np.eye(H0.shape[0])) * omega


def build_shirley_off_diagonal(
    V: Union[np.ndarray, sp.csr_matrix],
    amp: float,
    K: int,
    sparse: bool = True
) -> Union[np.ndarray, sp.csr_matrix]:
    """Construct the off-diagonal part of the Shirley Hamiltonian."""
    off_diag = 0.5 * amp * V
    sup_diag = sp.diags([1], [1],  shape=(2 * K + 1, 2 * K + 1), format='csr')
    sub_diag = sp.diags([1], [-1], shape=(2 * K + 1, 2 * K + 1), format='csr')
    HF_off = sp.kron(sup_diag, off_diag.conj().T, format='csr') + sp.kron(sub_diag, off_diag, format='csr')
    return HF_off if sparse else HF_off.toarray()


def init_sorting(
    H0: Union[np.ndarray, sp.csr_matrix],
    omega: float,
    d: int,
    D: int,
    K: int,
) -> np.ndarray:
    """Construct the ideal folded states at A=0 to lock the sorting."""
    H0_dense = H0.toarray() if sp.issparse(H0) else H0
    eigvals, eigvecs = np.linalg.eigh(H0_dense)

    prev_evecs = np.zeros((D, d), dtype=complex)
    for m in range(d):
        k_m = -int(round(eigvals[m] / omega))
        k_m = max(-K, min(K, k_m))

        start_idx = (k_m + K) * d
        end_idx = start_idx + d
        prev_evecs[start_idx:end_idx, m] = eigvecs[:, m]
    return prev_evecs


def diagonalize_shirley(
    HF: Union[np.ndarray, sp.csr_matrix],
    k: int,
    v0: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize the Shirley Hamiltonian and return lowest k eigenvalues and eigenvectors."""
    if sp.issparse(HF):
        if v0 is None:
            evals, evecs = spla.eigsh(HF, k=k, sigma=1e-6, which='LM')
        else:
            evals, evecs = spla.eigsh(HF, k=k, sigma=1e-6, which='LM', v0=v0)
    else:
        evals, evecs = eigh(HF, subset_by_index=[0, k - 1], check_finite=False)
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
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Compute quasienergies, excitations, and Sambe modes for a single frequency across drive amplitudes."""
    cache = ShirleyCache(circuit, options) if cache is None else cache
    k = options.k
    k = cache.d if k is None or options.sparse is False else k

    # Bare folded states in full Sambe space: shape (D, d)
    init_sambe_states = cache.initial_states(freq)
    prev_evecs = init_sambe_states.copy()

    quasienergies = np.zeros((len(amps), k), dtype=np.float64)
    excitations = np.zeros((len(amps), k), dtype=np.float64)

    has_tracked = options.tracked_states is not None
    if has_tracked:
        tracked_arr = np.asarray(options.tracked_states, dtype=int)
        bare_to_floquet = np.empty(cache.d, dtype=int)
        tracked_sambe_states = np.zeros((len(amps), cache.D, len(options.tracked_states)), dtype=np.complex128)
    else:
        tracked_sambe_states = None

    H0_s = cache.diagonal(freq)
    evecs = None
    diag_N = cache.diag_N

    for i, amp in enumerate(amps):
        HF = H0_s + amp * cache.H_drive
        evals, evecs = diagonalize_shirley(HF, k, v0=evecs[:, 0] if evecs is not None else None)

        # Adiabatic state tracking against previous amplitude
        overlap = np.abs(prev_evecs.conj().T @ evecs)
        _, col_ind = linear_sum_assignment(-overlap)

        current_evals = evals[col_ind]
        current_evecs = evecs[:, col_ind]
        prev_evecs = current_evecs

        # Fast diagonal expectation value
        Nexp = diag_N @ (current_evecs.real**2 + current_evecs.imag**2)

        quasienergies[i] = current_evals
        # En_folded = (current_evals + 0.5 * freq) % freq - 0.5 * freq
        excitations[i] = Nexp

        # Diabatic tracking to uncoupled bare states
        if has_tracked:
            overlap_diab = np.abs(init_sambe_states.conj().T @ evecs)
            row_diab, col_diab = linear_sum_assignment(-overlap_diab)

            bare_to_floquet[row_diab] = col_diab
            tracked_sambe_states[i] = evecs[:, bare_to_floquet[tracked_arr]]

    return quasienergies, excitations, tracked_sambe_states


def _worker_init():
    """Pin threads in subprocesses to avoid BLAS thread oversubscription."""
    import glob
    import ctypes

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["SCIPY_OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        import site
        search_dirs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
        for sdir in search_dirs:
            for so in glob.glob(os.path.join(sdir, "*libs", "*openblas*.so")):
                try:
                    dll = ctypes.CDLL(so)
                    for fn in ["scipy_openblas_set_num_threads", "scipy_openblas_set_num_threads64_", "openblas_set_num_threads"]:
                        if hasattr(dll, fn):
                            getattr(dll, fn)(1)
                except Exception:
                    pass
    except Exception:
        pass


def _shirley_worker_task(args):
    """Worker task for multi-core Shirley frequency sweeping."""
    f_idx, circuit, drive_freq, amps, options, cache = args
    quasienergies, excitations, tracked_sambe_states = shirley_single_frequency(
        circuit, drive_freq, amps, options, cache=cache
    )
    bare_init = cache.initial_states(drive_freq)[:, options.tracked_states] if options.tracked_states is not None else None
    return f_idx, quasienergies, excitations, tracked_sambe_states, bare_init


def _warn_memory_headroom(n_workers: int):
    """Inform user if available system memory is low relative to requested worker count."""
    if n_workers <= 1:
        return
    try:
        import os
        import warnings
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        avail_gb = (page_size * avail_pages) / (1024**3)
        est_per_worker = 0.3
        req_gb = n_workers * est_per_worker
        if req_gb > avail_gb:
            warnings.warn(
                f"Requested {n_workers} workers for Shirley analysis may require ~{req_gb:.1f} GB RAM, "
                f"but only ~{avail_gb:.1f} GB is currently free. "
                "If system freezes or experiences memory pressure, consider reducing n_workers.",
                ResourceWarning,
                stacklevel=2,
            )
    except Exception:
        pass


def shirley_branch_analysis(
    circuit: HilbertSpace,
    freq: Union[np.ndarray, list],
    amps: Union[np.ndarray, list],
    options: Options,
) -> Result:
    """Executes Floquet branch analysis using extended Sambe-Shirley space."""
    options.k = options.k if options.k is not None else circuit.dimension
    options.k = min(options.k, circuit.dimension)

    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    amps = np.asarray(amps, dtype=float)
    cache = ShirleyCache(circuit, options)

    show_progress = getattr(options, "progress_bar", True)
    has_tracked = options.tracked_states is not None
    num_tracked = len(options.tracked_states) if has_tracked else 0
    n_workers = getattr(options, "n_workers", 1)

    num_workers = os.cpu_count() or 1 if n_workers == -1 else int(n_workers)
    _warn_memory_headroom(num_workers)

    res = Result(
        metadata=None,  # Populated by BranchAnalysis.run
        drive_freq=freq,
        drive_amp=amps,
        quasienergies=np.zeros((len(freq), len(amps), options.k), dtype=np.float64),
        hybridization=np.zeros((len(freq), len(amps), num_tracked), dtype=np.float64) if has_tracked else None,
        observables={
            "avg_excitation": np.zeros((len(freq), len(amps), options.k), dtype=np.float64),
        },
    )

    all_states = (
        np.zeros((len(freq), len(amps), cache.D, num_tracked), dtype=np.complex128)
        if has_tracked
        else None
    )
    all_bare = (
        np.zeros((len(freq), cache.D, num_tracked), dtype=np.complex128)
        if has_tracked
        else None
    )

    if num_workers <= 1 or len(freq) <= 1:
        for i, drive_freq in enumerate(tqdm(freq, desc="Branch Analysis", total=len(freq), disable=not show_progress)):
            quasienergies, excitations, tracked_sambe_states = shirley_single_frequency(
                circuit, drive_freq, amps, options, cache=cache
            )
            res.quasienergies[i] = quasienergies
            res.observables["avg_excitation"][i] = excitations
            if has_tracked and tracked_sambe_states is not None:
                all_states[i] = tracked_sambe_states
                all_bare[i] = cache.initial_states(drive_freq)[:, options.tracked_states]
    else:
        max_workers = min(num_workers, len(freq))
        tasks = [
            (i, circuit, drive_freq, amps, options, cache)
            for i, drive_freq in enumerate(freq)
        ]
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"), initializer=_worker_init) as executor:
            futures = {executor.submit(_shirley_worker_task, task): task[0] for task in tasks}
            with tqdm(total=len(freq), desc=f"Branch Analysis ({max_workers} workers)", disable=not show_progress) as pbar:
                for future in as_completed(futures):
                    f_idx, quasienergies, excitations, tracked_sambe_states, bare_init = future.result()
                    res.quasienergies[f_idx] = quasienergies
                    res.observables["avg_excitation"][f_idx] = excitations
                    if has_tracked and tracked_sambe_states is not None:
                        all_states[f_idx] = tracked_sambe_states
                        all_bare[f_idx] = bare_init
                    pbar.update(1)

    if has_tracked and all_states is not None:
        fitter = HybridizationFit(
            fit_cutoff=options.fit_cutoff,
            overlap_cutoff=options.overlap_cutoff,
            fit_range_fraction=getattr(options, "fit_range_fraction", 1.0),
        )
        res.hybridization = fitter.compute_hybridization(freq, amps, all_states, all_bare)

    return res