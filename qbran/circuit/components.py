"""Circuit subsystem entities: Transmon and Resonator."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np
import scipy.sparse as sp

from qbran.core.operators import (
    charge_operator,
    cos_phi_operator,
    sin_phi_operator,
    annihilation_operator,
    creation_operator,
    number_operator,
    quadrature_x,
    quadrature_p,
)


class Component(ABC):
    """Abstract base class for circuit components."""

    def __init__(self, name: str, truncated_dim: int):
        self.name = name
        self.truncated_dim = int(truncated_dim)

    @abstractmethod
    def bare_hamiltonian(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        """Return the bare Hamiltonian of the component."""
        pass

    def identity(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        """Return the identity matrix of the component."""
        if sparse:
            return sp.eye(self.truncated_dim, format='csr')
        return np.eye(self.truncated_dim)

    @abstractmethod
    def params_dict(self) -> Dict[str, Any]:
        """Return dictionary of physical parameters for experiment metadata."""
        pass


class Transmon(Component):
    """Transmon qubit.

    Hamiltonian in charge basis:
        H = 4 * E_C * (n - n_g)^2 - E_J * cos(phi)

    Parameters
    ----------
    EJ : float
        Josephson energy in GHz.
    EC : float
        Charging energy in GHz.
    ng : float, optional
        Offset charge in units of Cooper pairs. Default is 0.
    ncut : int, optional
        Charge cutoff (-ncut <= n <= ncut). Default is 31.
    truncated_dim : int, optional
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
        name: str = "Transmon",
    ):
        super().__init__(name=name, truncated_dim=truncated_dim)
        self.EJ = float(EJ)
        self.EC = float(EC)
        self.ng = float(ng)
        self.ncut = int(ncut)
        self.charge_dim = 2 * self.ncut + 1

        self._diagonalize()

    def _diagonalize(self):
        """Diagonalize the transmon Hamiltonian in the charge basis."""
        n_diag = np.arange(-self.ncut, self.ncut + 1, dtype=np.float64)
        cos_op = cos_phi_operator(self.ncut, sparse=False)
        sin_op = sin_phi_operator(self.ncut, sparse=False)

        # Kinetic term is strictly diagonal: 4 * EC * (n - ng)^2
        T_diag = 4.0 * self.EC * (n_diag - self.ng) ** 2
        H_charge = np.diag(T_diag) - self.EJ * cos_op

        evals, evecs = np.linalg.eigh(H_charge)

        # Enforce canonical sign convention (element with largest magnitude is positive)
        for i in range(self.charge_dim):
            idx_max = np.argmax(np.abs(evecs[:, i]))
            if evecs[idx_max, i] < 0:
                evecs[:, i] *= -1.0

        self._all_evals = evals
        self._evecs = evecs[:, :self.truncated_dim]
        self.evals = evals[:self.truncated_dim]

        # Project operators to the truncated eigenbasis: O_eig = evecs^\dagger @ O @ evecs
        n_dense = np.diag(n_diag)
        self._n_eig = self._evecs.conj().T @ n_dense @ self._evecs
        self._cos_phi_eig = self._evecs.conj().T @ cos_op @ self._evecs
        self._sin_phi_eig = self._evecs.conj().T @ sin_op @ self._evecs

    @property
    def wp(self) -> float:
        """Plasma frequency of the transmon wp = sqrt(8 * E_J * E_C)."""
        return float(np.sqrt(8.0 * self.EJ * self.EC))

    @property
    def anharmonicity(self) -> float:
        """Qubit anharmonicity alpha = (E2 - E1) - (E1 - E0)."""
        return float((self.evals[2] - self.evals[1]) - (self.evals[1] - self.evals[0]))

    @property
    def E01(self) -> float:
        """Transition energy E01 = E1 - E0."""
        return float(self.evals[1] - self.evals[0])

    @property
    def n_operator(self) -> np.ndarray:
        """Charge number operator in the truncated eigenbasis."""
        return self._n_eig

    @property
    def cos_phi_operator(self) -> np.ndarray:
        """Cosine of phase operator in the truncated eigenbasis."""
        return self._cos_phi_eig

    @property
    def sin_phi_operator(self) -> np.ndarray:
        """Sine of phase operator in the truncated eigenbasis."""
        return self._sin_phi_eig

    def bare_hamiltonian(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        """Return the bare Hamiltonian of the transmon in the truncated eigenbasis."""
        diag = self.evals - self.evals[0]
        if sparse:
            return sp.diags(diag, 0, format='csr', dtype=np.float64)
        return np.diag(diag)

    def params_dict(self) -> Dict[str, Any]:
        return {
            "EJ": self.EJ,
            "EC": self.EC,
            "ng": self.ng,
            "ncut": self.ncut,
            "wp": self.wp,
            "E01": self.E01,
            "anharmonicity": self.anharmonicity,
        }


class Resonator(Component):
    """Linear harmonic oscillator (LC resonator).

    Hamiltonian in Fock basis:
        H = E_osc * a^dagger * a

    Parameters
    ----------
    E_osc : float
        Resonator fundamental frequency in GHz.
    truncated_dim : int, optional
        Fock space cutoff, default is 13.
    name : str, optional
        Name of the component. Default is "Resonator".
    """

    def __init__(self, E_osc: float, truncated_dim: int = 13, name: str = "Resonator"):
        super().__init__(name=name, truncated_dim=truncated_dim)
        self.E_osc = float(E_osc)

    def bare_hamiltonian(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        diag = self.E_osc * np.arange(self.truncated_dim, dtype=np.float64)
        if sparse:
            return sp.diags(diag, 0, format="csr", dtype=np.float64)
        return np.diag(diag)

    def annihilation_operator(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        """Annihilation operator in Fock basis."""
        return annihilation_operator(self.truncated_dim, sparse=sparse)

    def creation_operator(self, sparse: bool = False) -> Union[np.ndarray, sp.csr_matrix]:
        """Creation operator in Fock basis."""
        return creation_operator(self.truncated_dim, sparse=sparse)

    @property
    def a(self) -> np.ndarray:
        """Annihilation operator in Fock basis."""
        return annihilation_operator(self.truncated_dim, sparse=False)

    @property
    def adag(self) -> np.ndarray:
        """Creation operator in Fock basis."""
        return creation_operator(self.truncated_dim, sparse=False)

    @property
    def n_photons_operator(self) -> np.ndarray:
        """Photon number operator in Fock basis."""
        return number_operator(self.truncated_dim, sparse=False)

    @property
    def x_operator(self) -> np.ndarray:
        """Position quadrature (a + adag) / sqrt(2)."""
        return quadrature_x(self.truncated_dim, sparse=False)

    @property
    def p_operator(self) -> np.ndarray:
        """Momentum quadrature -i * (a - adag) / sqrt(2)."""
        return quadrature_p(self.truncated_dim, sparse=False)

    def params_dict(self) -> Dict[str, Any]:
        return {
            "E_osc": self.E_osc,
        }


Oscillator = Resonator  # Alias