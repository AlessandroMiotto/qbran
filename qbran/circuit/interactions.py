"""Pairwise interactions between circuit components."""

from typing import Tuple, Optional

import numpy as np

from qbran.circuit.components import Component, Transmon, Resonator

class Interaction:
    """Pairwise interaction between two circuit components.

    Parameters:
    ----------
    g : float
        Coupling strength in GHz.
    component1 : Component
        First circuit component.
    component2 : Component
        Second circuit component.
    kind : str, optional
        Coupling type: 'capacitive', 'inductive' or 'custom'. Default is 'capacitive'.
    op1: np.ndarray, optional
        Custom operator for component1. Required if kind is 'custom'.
    op2: np.ndarray, optional
        Custom operator for component2. Required if kind is 'custom'.
    """

    def __init__(
        self,
        g: float,
        component1: Component,
        component2: Component,
        kind: str = 'capacitive',
        op1: Optional[np.ndarray] = None,
        op2: Optional[np.ndarray] = None
    ):
        self.g = float(g)
        self.component1 = component1
        self.component2 = component2
        self.kind = kind.lower()
        self.op1 = op1
        self.op2 = op2

    def get_operator(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return the interaction operator for the two components."""
        if self.kind == 'custom':
            if self.op1 is None or self.op2 is None:
                raise ValueError("kind='custom' requires both op1 and op2 to be provided.")
            return self.op1, self.op2, self.g

        # Transmon <-> Resonator
        if isinstance(self.component1, Transmon) and isinstance(self.component2, Resonator):
            if self.kind == 'capacitive':
                self.op1 = self.component1.n_operator
                self.op2 = -1j * (self.component2.a - self.component2.adag)
                return self.op1, self.op2, self.g
            elif self.kind == 'inductive':
                self.op1 = self.component1.sin_phi_operator
                self.op2 = (self.component2.a + self.component2.adag)
                return self.op1, self.op2, self.g

        if isinstance(self.component1, Resonator) and isinstance(self.component2, Transmon):
            self.op2, self.op1, self.g = Interaction(self.g, self.component2, self.component1, self.kind).get_operator()
            return self.op1, self.op2, self.g

        # Resonator <-> Resonator
        if isinstance(self.component1, Resonator) and isinstance(self.component2, Resonator):
            if self.kind == 'capacitive':
                self.op1 = (self.component1.a - self.component1.adag)
                self.op2 = (self.component2.a - self.component2.adag)
                return self.op1, self.op2, self.g
            if self.kind == 'inductive':
                self.op1 = (self.component1.a + self.component1.adag)
                self.op2 = (self.component2.a + self.component2.adag)
                return self.op1, self.op2, self.g

        # Transmon <-> Transmon
        elif isinstance(self.component1, Transmon) and isinstance(self.component2, Transmon):
            if self.kind == 'capacitive':
                self.op1 = self.component1.n_operator
                self.op2 = self.component2.n_operator
                return self.op1, self.op2, 4.0 * self.g

        raise NotImplementedError(
            f"Coupling kind '{self.kind}' between {type(self.component1).__name__} "
            f"and {type(self.component2).__name__} is not implemented. Use kind='custom'"
        )