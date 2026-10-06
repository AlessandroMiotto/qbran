"""Xiao diabatic tracking, polynomial fit, and hybridization metric."""

from typing import Union
import numpy as np


class HybridizationFit:
    """Xiao diabatic tracking, polynomial fit, and hybridization metric.

    Based on the methodology from Xiao et al. (2023) and validated in Dai et al. (2024).

    Parameters
    ----------
    fit_cutoff : int, optional
        Maximum polynomial order (m + n <= fit_cutoff) for dispersive background fitting.
        Default is 4.
    overlap_cutoff : float, optional
        Minimum overlap threshold with bare state. Points with overlap <= cutoff are
        flagged as in-resonance and masked out of the polynomial fit. Default is 0.8.
    """

    def __init__(
        self,
        fit_cutoff: int = 4,
        overlap_cutoff: float = 0.8,
    ):
        self.fit_cutoff = int(fit_cutoff)
        self.overlap_cutoff = float(overlap_cutoff)

    def compute_hybridization(
        self,
        freqs: np.ndarray,
        amps: np.ndarray,
        all_modes: np.ndarray,
        bare_states: np.ndarray,
    ) -> np.ndarray:
        """Computes hybridization metric Theta across the 2D sweep.

        Parameters
        ----------
        freqs : np.ndarray
            1D array of drive frequencies of length W.
        amps : np.ndarray
            1D array of drive amplitudes of length A. Must have len(amps) >= 2.
        all_modes : np.ndarray
            Tracked mode vectors of shape (W, A, D, k), where D is the space dimension
            (D = d for propagator, D = (2K+1)*d for Shirley) and k is the number of tracked states.
        bare_states : np.ndarray
            Zero-drive reference states of shape (D, k) or (W, D, k).

        Returns
        -------
        hybridization : np.ndarray
            Array of shape (W, A, k) containing the hybridization metric Theta in [0, 1].
        """
        W = len(freqs)
        A = len(amps)

        if A < 2:
            raise ValueError(f"Hybridization fitting requires at least 2 amplitude points, got {A}")

        D = all_modes.shape[2]
        k = all_modes.shape[3]

        span_w = 0.5 * (np.max(freqs) - np.min(freqs))
        w_norm = (freqs - np.mean(freqs)) / (span_w if span_w > 1e-12 else 1.0)
        max_a = np.max(amps)
        a_norm = amps / (max_a if max_a > 1e-12 else 1.0)

        W_grid, A_grid = np.meshgrid(w_norm, a_norm, indexing="ij")
        w_flat = W_grid.ravel()
        a_flat = A_grid.ravel()

        max_m = 0 if W <= 1 else min(W - 1, self.fit_cutoff)
        max_n = min(A - 1, self.fit_cutoff)

        terms = [
            (m, n)
            for m in range(max_m + 1)
            for n in range(1, max_n + 1)
            if m + n <= self.fit_cutoff
        ]
        n_terms = len(terms)

        if n_terms == 0:
            raise ValueError("No polynomial terms could be constructed for hybridization fitting.")

        X = np.column_stack([(w_flat**m) * (a_flat**n) for m, n in terms])
        hybridization = np.zeros((W, A, k), dtype=np.float64)

        is_bare_3d = bare_states.ndim == 3

        for j in range(k):
            modes_j = all_modes[:, :, :, j].reshape(W * A, D)

            # Construct bare state grid across (W * A, D)
            if is_bare_3d:
                v0_grid = np.repeat(
                    bare_states[:, None, :, j], A, axis=1
                ).reshape(W * A, D)
            else:
                v0_grid = np.broadcast_to(bare_states[None, :, j], (W * A, D))

            # Inner product gauge fixing: align global phase to bare state
            proj = np.sum(modes_j * v0_grid.conj(), axis=-1)
            abs_proj = np.abs(proj)
            phase = np.where(abs_proj > 1e-12, proj / abs_proj, 1.0)
            modes_aligned = modes_j / phase[:, None]

            # Mask out resonances where overlap drops below threshold
            keep = abs_proj > self.overlap_cutoff
            pert = modes_aligned - v0_grid

            if np.count_nonzero(keep) >= n_terms:
                coeffs, *_ = np.linalg.lstsq(X[keep], pert[keep], rcond=None)
                ideal_unnorm = v0_grid + X @ coeffs
            else:
                ideal_unnorm = v0_grid

            norms = np.linalg.norm(ideal_unnorm, axis=-1, keepdims=True)
            ideal_states = ideal_unnorm / np.where(norms > 1e-12, norms, 1.0)

            # Overlap between ideal dispersive state and actual Floquet mode
            ov = np.abs(np.sum(ideal_states.conj() * modes_aligned, axis=-1))
            theta = np.clip(1.0 - ov**2, 0.0, 1.0)
            hybridization[:, :, j] = theta.reshape(W, A)

        return hybridization