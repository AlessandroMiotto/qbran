from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

class HybridizationFit:
    """Xiao diabatic tracking, polynomial fit, and hybridization metric."""

    def __init__(
        self,
        fit_cutoff: int = 4,
        overlap_cutoff: float = 0.8,    
    ):
        self.fit_cutoff = fit_cutoff
        self.overlap_cutoff = overlap_cutoff
    
    def compute_hybridization(
        self,
        freqs: np.ndarray,
        amps: np.ndarray,
        all_modes: np.ndarray,
        bare_states: np.ndarray,
    ) -> np.ndarray:
        W = len(freqs)
        A = len(amps)
        d, k = bare_states.shape

        # Normalize frequency & amplitude grids for fitting
        span_w = 0.5 * (np.max(freqs) - np.min(freqs))
        w_norm = (freqs - np.mean(freqs)) / (span_w if span_w > 1e-12 else 1.0)
        max_a = np.max(amps)
        a_norm = amps / (max_a if max_a > 1e-12 else 1.0)

        W_grid, A_grid = np.meshgrid(w_norm, a_norm, indexing="ij")
        w_flat = W_grid.ravel()
        a_flat = A_grid.ravel()

        # Guard against rank deficiency when W=1 or small W
        max_m = 0 if W <= 1 else min(W - 1, self.fit_cutoff)
        max_n = min(A - 1, self.fit_cutoff)

        # 2D polynomial terms
        terms = [
            (m, n) 
            for m in range(self.fit_cutoff + 1) 
            for n in range(1, self.fit_cutoff + 1 - m) 
            if m + n <= self.fit_cutoff
        ]
        n_terms = len(terms)

        X = np.column_stack([(w_flat**m) * (a_flat**n) for m, n in terms])
        hybridization = np.zeros((W, A, k), dtype=np.float64)

        for j in range(k):
            modes_j = all_modes[:, :, :, j].reshape(W * A, d)
            v0 = bare_states[:, j]

            # Phase alignment: ensure inner product with bare state is real and positive
            proj = np.sum(modes_j * v0.conj(), axis=-1)
            abs_proj = np.abs(proj)
            phase = np.where(abs_proj > 1e-12, proj / abs_proj, 1.0)
            modes_aligned = modes_j / phase[:, None]

            # Mask out resonant avoided crossings
            keep = abs_proj > self.overlap_cutoff
            pert = modes_aligned - v0[None, :]

            # Solve linear least-squares on all non-resonant points across the 2D grid
            if np.count_nonzero(keep) >= n_terms:
                coeffs, *_ = np.linalg.lstsq(X[keep], pert[keep], rcond=None)
                ideal_unnorm = v0[None, :] + X @ coeffs
            else:
                ideal_unnorm = np.tile(v0, (W * A, 1))

            norms = np.linalg.norm(ideal_unnorm, axis=-1, keepdims=True)
            ideal_states = ideal_unnorm / np.where(norms > 1e-12, norms, 1.0)

            # Compute overlap and hybridization
            ov = np.abs(np.sum(ideal_states.conj() * modes_aligned, axis=-1))
            theta = 1.0 - ov**2
            hybridization[:, :, j] = theta.reshape(W, A)

        return hybridization