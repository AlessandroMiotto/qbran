import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, Union
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.linalg import eig, eigh
from tqdm import tqdm

from qbran.circuit.components import Transmon
from qbran.circuit.hilbert_space import HilbertSpace
from qbran.analysis.analysis import Options
from qbran.analysis.result import Result
from qbran.floquet.hybridization import HybridizationFit


def propagate_yoshida_step(
    amp: float,
    alpha_wV: np.ndarray,
    M1: np.ndarray,
    M2: np.ndarray,
    W_left: np.ndarray,
    W_right: np.ndarray,
    nsteps: int,
    ID: np.ndarray,
) -> np.ndarray:
    r"""Stateless kernel for computing the period propagator U(T) via Yoshida 4th-order splitting."""
    d_all = np.exp(-1j * amp * alpha_wV)
    U_V = ID.copy()
    idx = 0
    for k in range(nsteps):
        U_V = M1 @ (d_all[idx, :, None] * U_V)
        U_V = M1 @ (d_all[idx + 1, :, None] * U_V)
        if k < nsteps - 1:
            U_V = M2 @ (d_all[idx + 2, :, None] * U_V)
        else:
            U_V = d_all[idx + 2, :, None] * U_V
        idx += 3

    return W_left @ U_V @ W_right


class PropagatorStaticCache:
    """Precomputed circuit eigensystem and drive operators (independent of drive frequency)."""

    def __init__(self, circuit: HilbertSpace, options: Options):
        H0 = circuit.get_hamiltonian(sparse=False, enforce_real=True)
        self.H0 = H0.toarray() if sp.issparse(H0) else H0
        self.dim = self.H0.shape[0]
        self.ID = np.eye(self.dim, dtype=complex)
        self.diag_N = np.arange(self.dim, dtype=np.float64)

        drive_op = options.drive_op
        if drive_op is None:
            tmon = circuit.components[options.drive_idx]
            if not isinstance(tmon, Transmon):
                raise ValueError(
                    f"Component at index {options.drive_idx} is not a Transmon. "
                    "Please provide an explicit drive_op."
                )
            drive_op = tmon.n_operator

        V = circuit.embed_drive_operator(drive_op, options.drive_idx, sparse=False)
        self.V = V.toarray() if sp.issparse(V) else V

        # Diagonalize undriven Hamiltonian
        self.E0, P = eigh(self.H0)
        P_dag = P.conj().T
        self.P = P
        self.P_dag = P_dag

        # Diagonalize drive operator in the dressed basis
        V_dressed = P_dag @ self.V @ P
        self.w_V, self.Q_V = eigh(V_dressed)
        self.Q_V_dag = self.Q_V.conj().T

        # Uncoupled product-basis order for bare state tracking
        self.Nrt = np.diag(self.diag_N)
        init_evecs = P
        init_Nrt = np.real(np.einsum('ji,jk,ki->i', init_evecs.conj(), self.Nrt, init_evecs))
        order_Nrt = np.argsort(init_Nrt)
        self.init_evecs = init_evecs[:, order_Nrt]


class PropagatorDynamicCache:
    """Frequency-dependent propagator matrices. Precomputed per frequency with zero diagonalizations."""

    def __init__(self, static: PropagatorStaticCache, omega: float, nsteps: int):
        self.static = static
        self.omega = float(omega)
        self.nsteps = int(nsteps)
        self.dim = static.dim
        self.T = 2.0 * np.pi / self.omega
        dT = self.T / self.nsteps

        # Yoshida 4th-order coefficients
        y1 = 1.0 / (2.0 - 2.0**(1.0 / 3.0))
        y0 = - 2.0**(1.0 / 3.0) * y1
        sub_dt = np.array([y1 * dT, y0 * dT, y1 * dT])

        tau1 = 0.5 * (y0 + y1) * dT
        tau2 = y1 * dT
        tau_edge = 0.5 * y1 * dT

        # Transition matrices
        self.M1 = static.Q_V_dag @ (np.exp(-1j * static.E0 * tau1)[:, None] * static.Q_V)
        self.M2 = static.Q_V_dag @ (np.exp(-1j * static.E0 * tau2)[:, None] * static.Q_V)

        # Time grid and drive phases
        all_sub_dt = np.tile(sub_dt, self.nsteps)
        step_times = np.zeros(3 * self.nsteps + 1)
        step_times[1:] = np.cumsum(all_sub_dt)

        sin_times = np.sin(self.omega * step_times)
        alphas = (sin_times[1:] - sin_times[:-1]) / self.omega
        self.alpha_wV = alphas[:, None] * static.w_V[None, :]

        # Boundary dressing
        exp_edge = np.exp(-1j * static.E0 * tau_edge)
        self.W_left = static.P @ (exp_edge[:, None] * static.Q_V)
        self.W_right = (static.Q_V_dag * exp_edge[None, :]) @ static.P_dag

    def propagate(self, amp: float) -> np.ndarray:
        """Inlined, optimized Yoshida stepping."""
        d_all = np.exp(-1j * amp * self.alpha_wV)
        U_V = self.static.ID.copy()
        M1 = self.M1
        M2 = self.M2
        idx = 0
        for k in range(self.nsteps):
            U_V = M1 @ (d_all[idx, :, None] * U_V)
            U_V = M1 @ (d_all[idx + 1, :, None] * U_V)
            if k < self.nsteps - 1:
                U_V = M2 @ (d_all[idx + 2, :, None] * U_V)
            else:
                U_V = d_all[idx + 2, :, None] * U_V
            idx += 3

        return self.W_left @ U_V @ self.W_right


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


def _propagator_worker_task(args):
    """Worker task for multi-core frequency sweeping."""
    f_idx, drive_freq, static_cache, amps, nsteps, tracked_states = args
    dynamic_cache = PropagatorDynamicCache(static_cache, drive_freq, nsteps)
    quasienergies, excitations, tracked_modes = propagator_single_frequency(
        freq=drive_freq,
        amps=amps,
        dynamic_cache=dynamic_cache,
        static_cache=static_cache,
        tracked_states=tracked_states,
    )
    return f_idx, quasienergies, excitations, tracked_modes


def propagator_single_frequency(
    freq: float,
    amps: Union[np.ndarray, list],
    dynamic_cache: PropagatorDynamicCache,
    static_cache: PropagatorStaticCache,
    tracked_states: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Compute quasienergies, excitations, and diabatic modes across amplitudes for a single frequency."""
    prev_evecs = static_cache.init_evecs
    dim = static_cache.dim
    amps_arr = np.asarray(amps, dtype=float)
    diag_N = static_cache.diag_N

    quasienergies = np.zeros((len(amps_arr), dim), dtype=np.float64)
    excitations = np.zeros((len(amps_arr), dim), dtype=np.float64)

    if tracked_states is not None:
        tracked_arr = np.asarray(tracked_states, dtype=int)
        bare_to_floquet = np.empty(dim, dtype=int)
        tracked_modes = np.zeros((len(amps_arr), dim, len(tracked_states)), dtype=np.complex128)
    else:
        tracked_modes = None

    for i, amp in enumerate(amps_arr):
        U = dynamic_cache.propagate(amp)
        evals, evecs = eig(U)

        # Adiabatic tracking using previous amplitude
        overlap = np.abs(prev_evecs.conj().T @ evecs)
        _, col_ind = linear_sum_assignment(-overlap)

        current_evals = evals[col_ind]
        current_evecs = evecs[:, col_ind]
        prev_evecs = current_evecs

        # Fast diagonal expectation value (88x faster than einsum)
        Nexp = diag_N @ (current_evecs.real**2 + current_evecs.imag**2)

        # Quasienergies: principal value of np.log maps strictly to (-omega/2, omega/2]
        En = np.real(1j * np.log(current_evals) / dynamic_cache.T)

        quasienergies[i] = En
        excitations[i] = Nexp

        if tracked_states is not None:
            # Diabatic tracking to bare states at zero drive
            overlap_diab = np.abs(static_cache.init_evecs.conj().T @ evecs)
            row_diab, col_diab = linear_sum_assignment(-overlap_diab)

            bare_to_floquet[row_diab] = col_diab
            tracked_modes[i] = evecs[:, bare_to_floquet[tracked_arr]]

    return quasienergies, excitations, tracked_modes


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
        est_per_worker = 0.2
        req_gb = n_workers * est_per_worker
        if req_gb > avail_gb:
            warnings.warn(
                f"Requested {n_workers} workers may require ~{req_gb:.1f} GB RAM, "
                f"but only ~{avail_gb:.1f} GB is currently free. "
                "If system freezes or experiences memory pressure, consider reducing n_workers.",
                ResourceWarning,
                stacklevel=2,
            )
    except Exception:
        pass


def propagator_branch_analysis(
    circuit: HilbertSpace,
    freq: Union[np.ndarray, list],
    amps: Union[np.ndarray, list],
    options: Options,
) -> Result:
    """Executes Floquet branch analysis using 4th-order Trotter-Suzuki propagator integration."""
    options.k = circuit.dimension
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    amps = np.asarray(amps, dtype=float)

    # Precompute circuit quantities once
    static_cache = PropagatorStaticCache(circuit, options)

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

    if has_tracked:
        all_states = np.zeros((len(freq), len(amps), circuit.dimension, num_tracked), dtype=np.complex128)
    else:
        all_states = None

    if num_workers <= 1 or len(freq) <= 1:
        for i, drive_freq in enumerate(tqdm(freq, desc="Branch Analysis", total=len(freq), disable=not show_progress)):
            dynamic_cache = PropagatorDynamicCache(static_cache, drive_freq, options.nsteps)
            quasienergies, excitations, tracked_states = propagator_single_frequency(
                freq=drive_freq,
                amps=amps,
                dynamic_cache=dynamic_cache,
                static_cache=static_cache,
                tracked_states=options.tracked_states,
            )
            res.quasienergies[i] = quasienergies
            res.observables["avg_excitation"][i] = excitations
            if has_tracked and tracked_states is not None:
                all_states[i] = tracked_states
    else:
        max_workers = min(num_workers, len(freq))
        tasks = [
            (i, drive_freq, static_cache, amps, options.nsteps, options.tracked_states)
            for i, drive_freq in enumerate(freq)
        ]
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"), initializer=_worker_init) as executor:
            futures = {executor.submit(_propagator_worker_task, task): task[0] for task in tasks}
            with tqdm(total=len(freq), desc=f"Branch Analysis ({max_workers} workers)", disable=not show_progress) as pbar:
                for future in as_completed(futures):
                    f_idx, quasienergies, excitations, tracked_states = future.result()
                    res.quasienergies[f_idx] = quasienergies
                    res.observables["avg_excitation"][f_idx] = excitations
                    if has_tracked and tracked_states is not None:
                        all_states[f_idx] = tracked_states
                    pbar.update(1)

    if has_tracked and all_states is not None:
        bare_states = static_cache.init_evecs[:, options.tracked_states]
        fitter = HybridizationFit(fit_cutoff=options.fit_cutoff, overlap_cutoff=options.overlap_cutoff)
        res.hybridization = fitter.compute_hybridization(freq, amps, all_states, bare_states)

    return res