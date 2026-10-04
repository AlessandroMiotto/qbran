"""BranchAnalysis runner and Options container for quantum and Floquet simulations."""

from typing import List, Optional, Any, Union
import numpy as np
import scipy.sparse as sp

from qbran.analysis.result import Result

class Options:
    r"""Configuration options for quantum and Floquet branch analysis.

    Parameters
    ----------
    sparse : bool, optional
            Whether to use sparse matrix representation for the Shirley Hamiltonian, default True.
    K : int, optional
        Fourier harmonic cutoff for the Shirley Hamiltonian, default 8.
    k : int, optional
        Number of lowest energy eigenstates to compute upon diagonalization, default None (all eigenstates).
    drive_idx : int, optional
        Index of the component to which the drive is applied, default 0.
    drive_op : np.ndarray or sp.csr_matrix, optional
        Drive operator for the component at drive_idx. If None, defaults to the number operator for a Transmon component.  
        $$\hat{H}_{\text{drive}} = \hat{V} \cos{\omega_d t}$$
    hybridization : str, optional
        Method for computing hybridization between states: TODO.
    nsteps : int, optional
        Number of time steps for the 4th order propagator method, default 20.
    """

    def __init__(
        self,
        sparse: bool = True,
        K: int = 8,
        k: int = None,
        drive_idx: int = 0,
        drive_op: Optional[Union[np.ndarray, sp.csr_matrix]] = None,
        hybridization: str = "susceptibility",
        nsteps: int = 20,
        **kwargs,
    ):
        self.sparse = bool(sparse)

        self.K = int(K)
        self.k = k
        self.drive_idx = int(drive_idx)
        self.drive_op = drive_op
        self.hybridization = hybridization.lower()
        self.nsteps = int(nsteps)

        #TODO: check that drive_op is valid for the component at drive_idx

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
            from qbran.floquet.shirley import shirley_branch_analysis
            result = shirley_branch_analysis(self.circuit, drive_freq, drive_amp, self.options)

        elif self.kind == "propagator":
            from qbran.floquet.propagator import propagator_branch_analysis
            result = propagator_branch_analysis(self.circuit, drive_freq, drive_amp, self.options)

        if save: result.save(filename)
        return result