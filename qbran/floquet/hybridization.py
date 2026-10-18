"""Xiao diabatic tracking, polynomial fit, and hybridization metric."""

from typing import Union, List, Tuple
import numpy as np


class HybridizationFit:
    """Xiao diabatic tracking, polynomial fit, and hybridization metric.

    Based on the methodology from Xiao et al. (2023) and validated in Dai et al. (2024),
    with iterative amplitude windowing (Weiss Floquet framework).

    Parameters
    ----------
    fit_cutoff : int, optional
        Maximum polynomial order (m + n <= fit_cutoff) for dispersive background fitting.
        Default is 4.
    overlap_cutoff : float, optional
        Minimum overlap threshold with reference state. Points with overlap <= cutoff are
        flagged as in-resonance and masked out of the polynomial fit. Default is 0.8.
    fit_range_fraction : float, optional
        Fraction of the amplitude range to sweep over before updating the reference state
        to that of the fitted state from the previous window (iterative fitting window).
        Default is 1.0 (indicating a single global window). Must satisfy 0 < fit_range_fraction <= 1.0.
    """

    def __init__(
        self,
        fit_cutoff: int = 4,
        overlap_cutoff: float = 0.8,
        fit_range_fraction: float = 1.0,
    ):
        self.fit_cutoff = int(fit_cutoff)
        self.overlap_cutoff = float(overlap_cutoff)
        frac = float(fit_range_fraction)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"fit_range_fraction must be in (0, 1], got {frac}")
        self.fit_range_fraction = frac

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

        # Determine amplitude windows
        num_windows = int(np.ceil(1.0 / self.fit_range_fraction)) if self.fit_range_fraction < 1.0 else 1
        num_windows = max(1, min(num_windows, A // 2))

        window_slices: List[Tuple[int, int]] = []
        if num_windows > 1:
            pts_per_win = int(np.floor(A / num_windows))
            for win_idx in range(num_windows):
                start_i = win_idx * pts_per_win
                end_i = (win_idx + 1) * pts_per_win if win_idx < num_windows - 1 else A
                window_slices.append((start_i, end_i))
        else:
            window_slices.append((0, A))

        for j in range(k):
            modes_j = all_modes[:, :, :, j]  # shape (W, A, D)

            # Construct bare state grid across (W, A, D)
            if is_bare_3d:
                v0_grid_3d = np.repeat(bare_states[:, None, :, j], A, axis=1)
            else:
                v0_grid_3d = np.broadcast_to(bare_states[None, None, :, j], (W, A, D)).copy()

            v0_grid = v0_grid_3d.reshape(W * A, D)
            modes_flat = modes_j.reshape(W * A, D)

            # Inner product gauge fixing: align global phase to bare state
            proj = np.sum(modes_flat * v0_grid.conj(), axis=-1)
            abs_proj = np.abs(proj)
            phase = np.where(abs_proj > 1e-12, proj / abs_proj, 1.0)
            modes_aligned = (modes_flat / phase[:, None]).reshape(W, A, D)

            # Determine resonance mask (windowed vs single)
            if len(window_slices) == 1:
                # Single window: direct comparison with bare state
                keep = (abs_proj > self.overlap_cutoff)
            else:
                # Piecewise iterative windowing
                keep_grid = np.zeros((W, A), dtype=bool)
                ideal_prev_win = None

                for win_idx, (s_idx, e_idx) in enumerate(window_slices):
                    win_len = e_idx - s_idx
                    modes_win = modes_aligned[:, s_idx:e_idx, :]  # shape (W, win_len, D)

                    if win_idx == 0:
                        # Reference is bare state
                        ref_win = v0_grid_3d[:, s_idx:e_idx, :]
                    else:
                        # Reference is ideal displaced state evaluated at the start of this window
                        ref_at_boundary = ideal_prev_win[:, -1:, :]  # shape (W, 1, D)
                        ref_win = np.repeat(ref_at_boundary, win_len, axis=1)

                    # Compute overlap with window reference
                    proj_win = np.sum(modes_win * ref_win.conj(), axis=-1)
                    keep_win = np.abs(proj_win) > self.overlap_cutoff
                    keep_grid[:, s_idx:e_idx] = keep_win

                    # Fit local window polynomial to construct reference for next window
                    if win_idx < len(window_slices) - 1:
                        pert_win = (modes_win - ref_win).reshape(W * win_len, D)
                        w_win = W_grid[:, s_idx:e_idx].ravel()
                        a_win = A_grid[:, s_idx:e_idx].ravel()

                        max_m_win = 0 if W <= 1 else min(W - 1, self.fit_cutoff)
                        max_n_win = min(win_len - 1, self.fit_cutoff)
                        win_terms = [
                            (m, n)
                            for m in range(max_m_win + 1)
                            for n in range(1, max_n_win + 1)
                            if m + n <= self.fit_cutoff
                        ]
                        if len(win_terms) > 0 and np.count_nonzero(keep_win) >= len(win_terms):
                            X_loc = np.column_stack([(w_win**m) * (a_win**n) for m, n in win_terms])
                            coeffs_loc, *_ = np.linalg.lstsq(X_loc[keep_win.ravel()], pert_win[keep_win.ravel()], rcond=None)
                            unnorm_loc = ref_win.reshape(W * win_len, D) + X_loc @ coeffs_loc
                        else:
                            unnorm_loc = ref_win.reshape(W * win_len, D)

                        norms_loc = np.linalg.norm(unnorm_loc, axis=-1, keepdims=True)
                        ideal_prev_win = (unnorm_loc / np.where(norms_loc > 1e-12, norms_loc, 1.0)).reshape(W, win_len, D)

                keep = keep_grid.ravel()

            # Global polynomial fit over unmasked points
            pert_global = (modes_aligned.reshape(W * A, D) - v0_grid)
            if np.count_nonzero(keep) >= n_terms:
                coeffs, *_ = np.linalg.lstsq(X[keep], pert_global[keep], rcond=None)
                ideal_unnorm = v0_grid + X @ coeffs
            else:
                ideal_unnorm = v0_grid

            norms = np.linalg.norm(ideal_unnorm, axis=-1, keepdims=True)
            ideal_states = ideal_unnorm / np.where(norms > 1e-12, norms, 1.0)

            # Overlap between ideal dispersive state and actual Floquet mode
            ov = np.abs(np.sum(ideal_states.conj() * modes_aligned.reshape(W * A, D), axis=-1))
            theta = np.clip(1.0 - ov**2, 0.0, 1.0)
            hybridization[:, :, j] = theta.reshape(W, A)

        return hybridization