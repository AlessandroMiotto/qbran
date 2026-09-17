"""Circuit subsystem entities: Transmon and Resonator."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import scipy.sparse as sp

from qbran.core.operators import (
    charge_operator,
    cos_phi_operator,
    annihilation_operator,
    creation_operator,
    number_operator,
    quadrature_x,
    quadrature_p,
)

class Component(ABC):
    """Abstract class for circuit components."""

    def __init__(self, name: str, truncated_dim: int):
        self.name = name
        self.truncated_dim = truncated_dim

    @abstractmethod
    def bare_hamiltonian(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        """Return the bare Hamiltonian of the component."""
        pass

    def identity(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        """Return the identity matrix of the component."""
        if sparse: return sp.eye(self.truncated_dim, format='csr')
        else: return np.eye(self.truncated_dim)

class Transmon(Component):
    """Transmon qubit.

    Hamiltonian in charge basis:
        H = 4 * E_C * (n - n_g)^2 - E_J * cos(phi)
    
    Parameters:
    ----------
    EJ : float
        Josephson energy in GHz.
    EC : float
        Charging energy in GHz.
    ng : float
        Offset charge in units of Cooper pairs. Default is 0.
    ncut : int
        Charge cutoff (-ncut <= n <= ncut). Default is 31.
    truncated_dim : int
        Number of lowest energy levels to keep in the truncated Hilbert space. Default is 13.
    name : str, optional
        Name of the component. Default is "Transmon".
    """

    def __init__(
            self,
            EJ: float,
            EC: float,
            ng: float = 0.0,
            ncut: int = 31,
            truncated_dim: int = 13,
            name: str = "Transmon"
    ):
        super().__init__(name=name, truncated_dim=truncated_dim)
        self.EJ = EJ
        self.EC = EC
        self.ng = ng
        self.ncut = ncut
        self.charge_dim = 2 * self.ncut + 1

        self._diagonalize()

    def _diagonalize(self):
        """Diagonalize the transmon Hamiltonian in the charge basis."""
        n_op = charge_operator(self.ncut, sparse=False)
        cos_op = cos_phi_operator(self.ncut, sparse=False)

        H_charge = 4.0 * self.EC * (n_op - self.ng * np.eye(self.charge_dim)) ** 2 - self.EJ * cos_op
        evals, evecs = np.linalg.eigh(H_charge)

        self._all_evals = evals
        self._evecs = evecs[:, :self.truncated_dim]
        self.evals = evals[:self.truncated_dim]

        # Project operators to the truncated eigenbasis: O_eig = evecs^T @ O @ evecs
        self._n_eig = self._evecs.T @ n_op @ self._evecs
        self._cos_phi_eig = self._evecs.T @ cos_op @ self._evecs

    @property
    def wp(self) -> float:
        """Plasma frequency of the transmon wp = sqrt(8 * E_J * E_C)."""
        return np.sqrt(8.0 * self.EJ * self.EC)

    @property
    def anharmonicity(self) -> float:
        """Qubit anharmonicity alpha = (E2 - E1) - (E1 - E0)."""
        return (self.evals[2] - self.evals[1]) - (self.evals[1] - self.evals[0])

    @property
    def E01(self) -> float:
        """Transition energy E01 = E1 - E0."""
        return self.evals[1] - self.evals[0]

    @property
    def n_operator(self) -> np.ndarray:
        """Charge number operator in the truncated eigenbasis."""
        return self._n_eig

    def bare_hamiltonian(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        """Return the bare Hamiltonian of the transmon in the truncated eigenbasis."""
        diag = self.evals - self.evals[0]
        if sparse:
            return sp.diags(diag, 0, format='csr', dtype=np.float64)
        return np.diag(diag)

class Resonator(Component):
    """Linear harmonic oscillator (LC resonator).

    Hamiltonian in Fock basis:
        H = E_osc * a^dagger * a
    
    Parameters:
    ----------
    E_osc : float
        Resonator frequency in GHz.
    truncated_dim : int, optional
       Fock space cutoff, default is 3.
    name : str, optional
        Name of the component. Default is "Resonator".
    """

    def __init__(self, E_osc: float, truncated_dim: int = 13, name: str = "Resonator"):
        super().__init__(name=name, truncated_dim=truncated_dim)
        self.E_osc = float(E_osc)

    def bare_hamiltonian(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        diag = self.E_osc * np.arange(self.truncated_dim, dtype=np.float64)
        if sparse:
            return sp.diags(diag, 0, format='csr', dtype=np.float64)
        return np.diag(diag)

    def annihilation_operator(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        """Return the annihilation operator in the truncated Fock basis."""
        return annihilation_operator(self.truncated_dim, sparse=sparse)

    def creation_operator(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        """Return the creation operator in the truncated Fock basis."""
        return creation_operator(self.truncated_dim, sparse=sparse)

    @property
    def a(self) -> np.ndarray:
        """Annihilation operator in the truncated Fock basis."""
        return self.annihilation_operator(sparse=False)

    @property
    def adag(self) -> np.ndarray:
        """Creation operator in the truncated Fock basis."""
        return self.creation_operator(sparse=False)

    @property
    def n(self) -> np.ndarray:
        """Number operator in the truncated Fock basis."""
        return number_operator(self.truncated_dim, sparse=False)

    @property
    def x(self) -> np.ndarray:
        """Quadrature operator x = (a + a^dagger) / sqrt(2) in the truncated Fock basis."""
        return quadrature_x(self.truncated_dim, sparse=False)

    @property
    def p(self) -> np.ndarray:
        """Quadrature operator p = (a - a^dagger) / (i * sqrt(2)) in the truncated Fock basis."""
        return quadrature_p(self.truncated_dim, sparse=False)

Oscillator = Resonator  # Alias