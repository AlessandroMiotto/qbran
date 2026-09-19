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
        real_gauge: bool = True,
        op1: Optional[np.ndarray] = None,
        op2: Optional[np.ndarray] = None
    ):
        self.g = float(g)
        self.component1 = component1
        self.component2 = component2
        self.kind = kind.lower()
        self.real_gauge = real_gauge
        self.custom_op1 = op1
        self.custom_op2 = op2

    def get_operator(self, real_gauge: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return the interaction operator for the two components."""
        use_real = self.real_gauge if real_gauge is None else bool(real_gauge)

        if self.kind == "custom":
            if self.custom_op1 is None or self.custom_op2 is None:
                raise ValueError("kind='custom' requires op1 and op2 to be provided.")
            return self.custom_op1, self.custom_op2, self.g

        # Transmon <-> Resonator
        if isinstance(self.component1, Transmon) and isinstance(self.component2, Resonator):
            if self.kind == "capacitive":
                op1 = self.component1.n_operator
                if use_real:
                    # Real-symmetric gauge: H_int = g * n_t * (a + a^dagger) = sqrt(2) * g * n_t * x_r
                    op2 = self.component2.a + self.component2.adag
                else:
                    # Canonical charge-dipole gauge: H_int = -i * g * n_t * (a - a^dagger)
                    op2 = -1j * (self.component2.a - self.component2.adag)
                return op1, op2, self.g
            elif self.kind == "inductive":
                op1 = self.component1.sin_phi_operator
                op2 = (self.component2.a + self.component2.adag)
                return op1, op2, self.g

        elif isinstance(self.component1, Resonator) and isinstance(self.component2, Transmon):
            op2, op1, g = Interaction(
                self.g, self.component2, self.component1, kind=self.kind, real_gauge=use_real
            ).get_operator(real_gauge=use_real)
            return op1, op2, g

        # Resonator <-> Resonator
        elif isinstance(self.component1, Resonator) and isinstance(self.component2, Resonator):
            if self.kind == "capacitive":
                if use_real:
                    # Real-symmetric gauge: H_int = g * (a1 + a1^dagger) * (a2 + a2^dagger)
                    op1 = (self.component1.a + self.component1.adag)
                    op2 = (self.component2.a + self.component2.adag)
                else:
                    # Canonical laboratory gauge: H_int = g * (a1 - a1^dagger) * (a2 - a2^dagger)
                    op1 = (self.component1.a - self.component1.adag)
                    op2 = (self.component2.a - self.component2.adag)
                return op1, op2, self.g
            elif self.kind == "inductive":
                # H_int = g * (a1 + a1^dagger) * (a2 + a2^dagger)
                op1 = (self.component1.a + self.component1.adag)
                op2 = (self.component2.a + self.component2.adag)
                return op1, op2, self.g

        # Transmon <-> Transmon
        elif isinstance(self.component1, Transmon) and isinstance(self.component2, Transmon):
            if self.kind == "capacitive":
                # H_int = 4 * g * (n1 - ng1) * (n2 - ng2) (purely real and symmetric)
                op1 = self.component1.n_operator
                op2 = self.component2.n_operator
                return op1, op2, 4.0 * self.g

        raise NotImplementedError(
            f"Coupling kind '{self.kind}' between {type(self.component1).__name__} "
            f"and {type(self.component2).__name__} is not implemented. Use kind='custom'"
        )