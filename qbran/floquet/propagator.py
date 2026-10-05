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


class PropagatorCache:
    def __init__(self, circuit: HilbertSpace, options: Options, omega: float):
        self.nsteps = options.nsteps
        self.omega = omega

        H0 = circuit.get_hamiltonian(sparse=False)
        self.H0 = H0.toarray() if sp.issparse(H0) else H0
        self.dim = self.H0.shape[0]
        self.ID = np.eye(self.dim, dtype=complex)

        drive_op = options.drive_op
        if drive_op is None:
            tmon = circuit.components[options.drive_idx]
            if not isinstance(tmon, Transmon):
                raise ValueError(
                    f"Component at index {options.drive_idx} is not a Transmon. "
                    "Please provide a valid drive operator."
                )
            drive_op = tmon.n_operator
        self.V = circuit.embed_operator(drive_op, options.drive_idx, sparse=False)
        self.V = self.V.toarray() if sp.issparse(self.V) else self.V

        T = 2 * np.pi / omega
        dT = T / self.nsteps
        self.T = T

        # Dressed basis of coupled H0
        self.E0, P = eigh(self.H0)
        P_dag = P.conj().T

        # Drive operator in the dressed basis
        V_dressed = P_dag @ self.V @ P
        self.w_V, self.Q_V = eigh(V_dressed)
        self.Q_V_dag = self.Q_V.conj().T

        # Yoshida 4th-order coefficients
        y1 = 1.0 / (2.0 - 2.0**(1.0 / 3.0))
        y0 = - 2.0**(1.0 / 3.0) * y1
        sub_dt = np.array([y1 * dT, y0 * dT, y1 * dT])

        tau1 = 0.5 * (y0 + y1) * dT
        tau2 = y1 * dT
        tau_edge = 0.5 * y1 * dT

        # Static transition variables
        self.M1 = self.Q_V_dag @ (np.exp(-1j * self.E0 * tau1)[:, None] * self.Q_V)
        self.M2 = self.Q_V_dag @ (np.exp(-1j * self.E0 * tau2)[:, None] * self.Q_V)

        # Time grid and drive matrix
        all_sub_dt = np.tile(sub_dt, self.nsteps)
        step_times = np.zeros(3 * self.nsteps + 1)
        step_times[1:] = np.cumsum(all_sub_dt)

        sin_times = np.sin(omega * step_times)
        alphas = (sin_times[1:] - sin_times[:-1]) / omega
        self.alpha_wV = alphas[:, None] * self.w_V[None, :]

        # Boundary dressing
        exp_edge = np.exp(-1j * self.E0 * tau_edge)
        self.W_left = P @ (exp_edge[:, None] * self.Q_V)
        self.W_right = (self.Q_V_dag * exp_edge[None, :]) @ P_dag

        self.Nrt = np.diag(np.arange(self.dim))
        init_evecs = P
        init_Nrt = np.real(np.einsum('ji,jk,ki->i', init_evecs.conj(), self.Nrt, init_evecs))
        order_Nrt = np.argsort(init_Nrt)
        self.init_evecs = init_evecs[:, order_Nrt]

        # Parameters for fitting hybridization metric
        self.fit_cutoff = options.fit_cutoff
        self.overlap_cutoff = options.overlap_cutoff
        self.tracked_states = options.tracked_states

    def propagator_yoshida(self, amp: float) -> np.ndarray:
        r"""Compute the ropagastor by solving $i dU(t)/dt = H(t)U(t)$ with $U(0) = I$
        using the Trotter-Suzuki decomposition with Yoshida 4th-order correction.
        """
        d_all = np.exp(-1j * amp * self.alpha_wV)

        U_V = self.ID.copy()
        idx = 0
        for k in range(self.nsteps):
            U_V = self.M1 @ (d_all[idx, :, None] * U_V)
            U_V = self.M1 @ (d_all[idx + 1, :, None] * U_V)
            if k < self.nsteps - 1:
                U_V = self.M2 @ (d_all[idx + 2, :, None] * U_V)
            else:
                U_V = d_all[idx + 2, :, None] * U_V
            idx += 3
    
        return self.W_left @ U_V @ self.W_right


def propagrator_single_frequency(
    freq: float,
    amps: Union[np.ndarray, list],
    cache: PropagatorCache,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    cache.omega = freq
    prev_evecs = cache.init_evecs

    quasienergies = np.zeros((len(amps), cache.dim), dtype=np.float64)
    excitations   = np.zeros((len(amps), cache.dim), dtype=np.float64)

    if cache.tracked_states is not None:
        tracked_states = np.zeros((len(amps), cache.dim, len(cache.tracked_states)), dtype=np.complex128)
    else:
        tracked_states = None

    for i, amp in enumerate(amps):
        U = cache.propagator_yoshida(amp)
        evals, evecs = eig(U)

        # Adiabatic tracking using previous amplitude
        overlap = np.abs(prev_evecs.conj().T @ evecs)
        _, col_ind = linear_sum_assignment(-overlap)

        current_evals = evals[col_ind]
        current_evecs = evecs[:, col_ind]
        prev_evecs = current_evecs

        Nexp = np.real(np.einsum('ji,jk,ki->i', current_evecs.conj(), cache.Nrt, current_evecs))
        En = np.real(1j * np.log(current_evals) / cache.T)
        #En = (En + 0.5 * cache.omega) % cache.omega - 0.5 * cache.omega

        quasienergies[i] = En
        excitations[i]   = Nexp

        if cache.tracked_states is not None:
            # Diabatic tracking
            overlap_diab = np.abs(cache.init_evecs.conj().T @ evecs)
            _, col_diab = linear_sum_assignment(-overlap_diab)
            diabatic_evecs = evecs[:, col_diab]
            tracked_states[i] = diabatic_evecs[:, cache.tracked_states]

    return quasienergies, excitations, tracked_states


def propagator_branch_analysis(
    circuit: HilbertSpace,
    freq: Union[np.ndarray, list],
    amps: Union[np.ndarray, list],
    options: Options,
) -> Result:

    options.k = circuit.dimension
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    amps = np.asarray(amps, dtype=float)

    res = Result(
        metadata   = None,
        drive_freq = freq,
        drive_amp  = amps,
        quasienergies = np.zeros((len(freq), len(amps), options.k), dtype=np.float64),
        hybridization = np.zeros((len(freq), len(amps), len(options.tracked_states)), dtype=np.float64) if options.tracked_states is not None else None,
        observables={
            "avg_excitation" : np.zeros((len(freq), len(amps), options.k), dtype=np.float64),
        },
    )

    # TODO: add function to update frequency without rebuilding cache for each frequency

    all_states = np.zeros((len(freq), len(amps), circuit.dimension, len(options.tracked_states)), dtype=np.complex128)

    for i, drive_freq in enumerate(tqdm(freq, desc="Branch Analysis", total=len(freq))):
        cache = PropagatorCache(circuit, options, drive_freq)
        quasienergies, excitations, tracked_states = propagrator_single_frequency(drive_freq, amps, cache=cache)
        res.quasienergies[i] = quasienergies
        res.observables["avg_excitation"][i] = excitations
        if tracked_states is not None: all_states[i] = tracked_states

    if options.tracked_states is not None:
        bare_states = cache.init_evecs[:, options.tracked_states]
        fitter = HybridizationFit(fit_cutoff=options.fit_cutoff, overlap_cutoff=options.overlap_cutoff)
        res.hybridization = fitter.compute_hybridization(freq, amps, all_states, bare_states)

    return res