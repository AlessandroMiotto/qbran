"""BranchAnalysis runner and Options container for 2D Floquet parameter sweeps."""

from typing import List, Optional, Any, Union
import numpy as np

from qbran.analysis.result import Result


class Options:
    """Configuration options for Floquet, branch analysis, and parameter sweeps.
    
    Parameters
    ----------
    hybridization : str, optional
        Metric: 'pr', 'entropy', or 'overlap', default 'pr'.
    fit_degree : int, optional
        Degree of 2D polynomial when hybridization='overlap', default 4.
    overlap_cutoff : float, optional
        Overlap threshold to exclude resonant outliers from polynomial fit, default 0.8.
    num_cpu : int, optional
        Number of CPU worker processes for parallel sweeps, default 4.
    adaptive : bool, optional
        Whether to use 2D quadtree adaptive mesh refinement, default False.
    n_replicas : int, optional
        Harmonic replica truncation order M (total replicas 2M+1), default 5.
    target_states : List[int], optional
        Indices of computational states to track (default [0, 1]).
    average_ng : bool, optional
        Whether to perform gate-charge noise averaging, default False.
    n_samples_ng : int, optional
        Number of ng samples over [0, 0.5] if average_ng is True, default 20.
    """

    def __init__(
        self,
        hybridization: str = "pr",
        fit_degree: int = 4,
        overlap_cutoff: float = 0.8,
        num_cpu: int = 4,
        adaptive: bool = False,
        n_replicas: int = 5,
        target_states: Optional[List[int]] = None,
        average_ng: bool = False,
        n_samples_ng: int = 20,
    ):
        self.hybridization = hybridization.lower()
        self.fit_degree = int(fit_degree)
        self.overlap_cutoff = float(overlap_cutoff)
        self.num_cpu = int(num_cpu)
        self.adaptive = bool(adaptive)
        self.n_replicas = int(n_replicas)
        self.target_states = [0, 1] if target_states is None else list(target_states)
        self.average_ng = bool(average_ng)
        self.n_samples_ng = int(n_samples_ng)


class BranchAnalysis:
    """Executes branch analysis
    
    Parameters
    ----------
    circuit : Any
        Circuit Hilbert space (transmon + resonators).
    kind : str, optional
        Simulation kind: 'floquet', 'quantum'.
    options : Options, optional
        Configuration options for the analysis.
    """

    def __init__(self, circuit: Any, kind: str, options: Optional[Options] = None):
        self.circuit = circuit
        self.kind = kind.lower()
        self.options = options if options is not None else Options()
        if self.kind not in ["floquet", "quantum"]:
            raise ValueError(f"Invalid kind '{self.kind}'. Must be 'floquet' or 'quantum'.")

    def run(
        self,
        drive_freq: Union[np.ndarray, List[float]],
        drive_amp: Union[np.ndarray, List[float]] = None,
        save: bool = False,
        filename: str = "out.h5",
    ) -> Result:
        """Runs the 2D sweep across drive frequencies and amplitudes."""
        
        freq_arr = np.asarray(drive_freq, dtype=np.float64) 
        amp_arr = np.asarray(drive_amp, dtype=np.float64)

        if self.options.adaptive:
            # TODO: Add adaptive mesh refinement logic
            raise NotImplementedError("Adaptive mesh refinement is not yet implemented.")

        if self.kind == "floquet":
            # TODO: Add Floquet analysis logic
            raise NotImplementedError("Floquet analysis is not yet implemented.")
        if self.kind == "quantum":
            # TODO: Add quantum analysis logic
            raise NotImplementedError("Quantum analysis is not yet implemented.")

        # if save:
        #     #result.save(filename)

        return None #result