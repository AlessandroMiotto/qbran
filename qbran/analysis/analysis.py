"""BranchAnalysis runner and Options container for quantum and Floquet simulations."""

from typing import List, Optional, Any, Union
import numpy as np

from qbran.analysis.result import Result


class Options:
    """Configuration options for quantum and Floquet branch analysis.

    Parameters
    ----------
    hybridization : str, optional
        Metric: 'fit', 'pr'. Default 'fit'.
    num_cpu : int, optional
        Number of CPU worker processes for Floquet parallel sweeps, default 4.
    """

    def __init__(
        self,
        hybridization: str = "fit",
        num_cpu: int = 4,
        **kwargs,
    ):
        self.hybridization = hybridization.lower()
        self.num_cpu = int(num_cpu)

        for k, v in kwargs.items():
            setattr(self, k, v)


class BranchAnalysis:
    """Unified engine for quantum and Floquet branch analysis.

    Parameters
    ----------
    circuit : Any
        Circuit Hilbert space (transmon + resonators).
    kind : str, optional
        Simulation framework: 'shirley' (default) or 'propagator'.
    options : Options, optional
        Configuration options for the simulation.
    """

    def __init__(
        self,
        circuit: Any,
        kind: str = "shirley",
        options: Optional[Options] = None,
        **kwargs,
    ):
        self.circuit = circuit
        self.kind = kind.lower()
        self.options = options if options is not None else Options()
        self.kwargs = kwargs

        if self.kind not in ["shirley", "propagator"]:
            raise ValueError(f"Invalid kind '{self.kind}'. Must be 'shirley' or 'propagator'.")

    def run(
        self,
        drive_freq: Optional[Union[np.ndarray, List[float], float]] = None,
        drive_amp: Optional[Union[np.ndarray, List[float]]] = None,
        save: bool = False,
        filename: str = "out.h5",
    ) -> Result:
        """Executes the branch analysis.

        Parameters
        ----------
        drive_freq : float, list, or ndarray, optional
            Drive frequency (or frequencies) in GHz. For quantum analysis, if omitted,
            defaults to the fundamental frequency of the drive resonator.
        drive_amp : list or ndarray, optional
            Drive amplitudes (for Floquet sweeps).
        save : bool, optional
            Whether to save the result to an HDF5 archive. Default False.
        filename : str, optional
            Target filename if save is True. Default 'out.h5'.
        """
        if self.kind == "shirley":
            raise NotImplementedError("Shirley's Floquet is not implemented yet.")

        elif self.kind == "propagator":
            raise NotImplementedError("Floquet analysis with time-evolution prpagator is not implemented yet.")

        return None