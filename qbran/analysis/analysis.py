"""BranchAnalysis runner and Options container for quantum and Floquet simulations."""

from datetime import datetime
from typing import List, Optional, Any, Union, Dict
import numpy as np
import scipy.sparse as sp

import qbran
from qbran.analysis.result import Result


class Options:
    r"""Configuration options for quantum and Floquet branch analysis.

    Parameters
    ----------
    sparse : bool, optional
        Whether to use sparse matrix representation for the Shirley Hamiltonian. Default True.
    K : int, optional
        Fourier harmonic cutoff for the Shirley Hamiltonian. Default 8.
    k : int, optional
        Number of lowest energy eigenstates to compute upon diagonalization.
        Default None (all eigenstates in the bare dimension).
    nsteps : int, optional
        Number of time steps per drive period for 4th-order propagator method. Default 20.
    drive_idx : int, optional
        Index of the circuit component to which the drive is applied. Default 0.
    drive_op : np.ndarray or sp.csr_matrix, optional
        Drive operator for the component at drive_idx. If None, defaults to the charge
        operator for a Transmon component.
    hybridization : str, optional
        Method for computing hybridization between states. Default 'fit' (Xiao polynomial fit).
    fit_cutoff : int, optional
        Maximum polynomial order for hybridization fitting. Default 4.
    overlap_cutoff : float, optional
        Minimum overlap threshold for resonance masking in hybridization fitting. Default 0.8.
    tracked_states : list of int, optional
        Indices of bare states to track for hybridization analysis. If None, hybridization
        metric is not computed. Default None.
    backend : str, optional
        Computation backend: 'numpy' (default) or 'auto'.
    n_workers : int, optional
        Number of parallel worker processes for frequency sweeps. Default 1 (sequential).
        Set to -1 to automatically use all available CPU cores.
    progress_bar : bool, optional
        Whether to display a tqdm progress bar during frequency sweeps. Default True.
    """

    def __init__(
        self,
        sparse: bool = True,
        K: int = 8,
        k: Optional[int] = None,
        nsteps: int = 20,
        drive_idx: int = 0,
        drive_op: Optional[Union[np.ndarray, sp.csr_matrix]] = None,
        hybridization: str = "fit",
        fit_cutoff: int = 4,
        overlap_cutoff: float = 0.8,
        tracked_states: Optional[List[int]] = None,
        backend: str = "numpy",
        n_workers: int = 1,
        progress_bar: bool = True,
        **kwargs,
    ):
        self.sparse = bool(sparse)
        self.K = int(K)
        self.k = int(k) if k is not None else None
        self.nsteps = int(nsteps)
        self.drive_idx = int(drive_idx)
        self.drive_op = drive_op
        self.hybridization = str(hybridization).lower()
        self.fit_cutoff = int(fit_cutoff)
        self.overlap_cutoff = float(overlap_cutoff)
        self.tracked_states = list(tracked_states) if tracked_states is not None else None

        valid_backends = {"auto", "numpy"}
        backend_str = str(backend).lower()
        if backend_str not in valid_backends:
            raise ValueError(
                f"Invalid backend '{backend}'. QBRAN uses optimized multi-core NumPy/SciPy parallelization (choose 'numpy' or 'auto')."
            )
        self.backend = "numpy"
        self.n_workers = int(n_workers)
        self.progress_bar = bool(progress_bar)

        for k_arg, v_arg in kwargs.items():
            setattr(self, k_arg, v_arg)


class BranchAnalysis:
    """Unified engine for Floquet branch analysis.

    Parameters
    ----------
    circuit : Any
        Circuit Hilbert space.
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

        if self.options.drive_idx < 0 or self.options.drive_idx >= len(circuit.components):
            raise IndexError(
                f"drive_idx {self.options.drive_idx} out of range for circuit with "
                f"{len(circuit.components)} components."
            )

    def _build_metadata(self) -> Dict[str, Any]:
        """Construct comprehensive metadata dictionary describing circuit and simulation parameters."""
        components_meta = []
        for c in self.circuit.components:
            info = {
                "name": c.name,
                "type": type(c).__name__,
                "truncated_dim": c.truncated_dim,
            }
            if hasattr(c, "params_dict"):
                info.update(c.params_dict())
            components_meta.append(info)

        interactions_meta = []
        for inter in self.circuit.interactions:
            interactions_meta.append({
                "component1": inter.component1.name,
                "component2": inter.component2.name,
                "g": inter.g,
                "kind": inter.kind,
                "real_gauge": inter.real_gauge,
            })

        return {
            "qbran_version": getattr(qbran, "__version__", "0.1.0"),
            "timestamp": datetime.now().isoformat(),
            "solver": self.kind,
            "circuit_dimension": self.circuit.dimension,
            "components": components_meta,
            "interactions": interactions_meta,
            "options": {
                "sparse": self.options.sparse,
                "K": self.options.K,
                "k": self.options.k,
                "nsteps": self.options.nsteps,
                "drive_idx": self.options.drive_idx,
                "fit_cutoff": self.options.fit_cutoff,
                "overlap_cutoff": self.options.overlap_cutoff,
                "tracked_states": self.options.tracked_states,
                "backend": self.options.backend,
                "n_workers": self.options.n_workers,
            },
        }

    def run(
        self,
        drive_freq: Optional[Union[np.ndarray, List[float], float]] = None,
        drive_amp: Optional[Union[np.ndarray, List[float], float]] = None,
        save: bool = False,
        filename: str = "out.h5",
    ) -> Result:
        """Executes the branch analysis.

        Parameters
        ----------
        drive_freq : float, list, or ndarray
            Drive frequency (or frequencies) in GHz.
        drive_amp : float, list, or ndarray
            Drive amplitudes.
        save : bool, optional
            Whether to save the result to an HDF5 archive. Default False.
        filename : str, optional
            Target filename if save is True. Default 'out.h5'.
        """
        if drive_freq is None:
            raise ValueError("drive_freq must be provided.")
        if drive_amp is None:
            raise ValueError("drive_amp must be provided.")

        if self.kind == "shirley":
            from qbran.floquet.shirley import shirley_branch_analysis
            result = shirley_branch_analysis(self.circuit, drive_freq, drive_amp, self.options)

        elif self.kind == "propagator":
            from qbran.floquet.propagator import propagator_branch_analysis
            result = propagator_branch_analysis(self.circuit, drive_freq, drive_amp, self.options)

        # Attach metadata
        result.metadata = self._build_metadata()
        result._data["metadata"] = result.metadata

        if save:
            result.save(filename)
        return result