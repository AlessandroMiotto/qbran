"""BranchAnalysis runner and Options container for quantum and Floquet simulations."""

from typing import List, Optional, Any, Union
import numpy as np

from qbran.analysis.result import Result


class Options:
    """Configuration options for quantum and Floquet branch analysis.

    Parameters
    ----------
    n_rungs : int, optional
        Number of drive resonator rungs to track in quantum branch analysis.
        Defaults to drive.truncated_dim - 2.
    batch_size : int, optional
        Number of frequencies evaluated simultaneously via JAX vmap. Default 1.
    drive : Any, optional
        Drive mode specification (name, index, or Component).
    hybridization : str, optional
        Metric: 'theta' / 'dai', 'pr', 'entropy', default 'theta'.
    num_cpu : int, optional
        Number of CPU worker processes for Floquet parallel sweeps, default 4.
    target_states : List[int], optional
        Indices of computational branches to track, default [0, 1].
    """

    def __init__(
        self,
        n_rungs: Optional[int] = None,
        batch_size: int = 1,
        drive: Optional[Any] = None,
        hybridization: str = "theta",
        num_cpu: int = 4,
        target_states: Optional[List[int]] = None,
        **kwargs,
    ):
        self.n_rungs = n_rungs
        self.batch_size = int(batch_size)
        self.drive = drive
        self.hybridization = hybridization.lower()
        self.num_cpu = int(num_cpu)
        self.target_states = [0, 1] if target_states is None else list(target_states)

        for k, v in kwargs.items():
            setattr(self, k, v)


class BranchAnalysis:
    """Unified engine for quantum and Floquet branch analysis.

    Parameters
    ----------
    circuit : Any
        Circuit Hilbert space (transmon + resonators).
    kind : str, optional
        Simulation framework: 'quantum' (default) or 'floquet'.
    options : Options, optional
        Configuration options for the simulation.
    drive : Any, optional
        Measurement / drive resonator component (e.g. 'res1', Resonator object, or index).
    """

    def __init__(
        self,
        circuit: Any,
        kind: str = "quantum",
        options: Optional[Options] = None,
        drive: Optional[Any] = None,
        **kwargs,
    ):
        self.circuit = circuit
        self.kind = kind.lower()
        self.options = options if options is not None else Options()
        self.drive = drive if drive is not None else getattr(self.options, "drive", None)
        self.kwargs = kwargs

        if self.kind not in ["floquet", "quantum"]:
            raise ValueError(f"Invalid kind '{self.kind}'. Must be 'quantum' or 'floquet'.")

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
        if self.kind == "quantum":
            from qbran.quantum.branch import QuantumBranchAnalysis

            engine = QuantumBranchAnalysis(
                circuit=self.circuit,
                options=self.options,
                drive=self.drive,
            )
            result = engine.run(frequencies=drive_freq)

            if save:
                result.save(filename)
            return result

        elif self.kind == "floquet":
            raise NotImplementedError("Floquet analysis will be configured next.")

        return None