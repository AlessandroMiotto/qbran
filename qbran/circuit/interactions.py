"""Pairwise interactions between circuit components."""

from typing import Tuple, Optional
import numpy as np

from qbran.circuit.components import Component, Transmon, Resonator


class Interaction:
    """Pairwise interaction between two circuit components.

    Parameters
    ----------
    *args :
        Flexible positional arguments: can pass components and coupling strength
        in any intuitive order, e.g. ``Interaction(tmon, res, 0.12)`` or ``Interaction(tmon, res)``.
    g : float, optional
        Coupling strength in GHz. Can be omitted (None) when creating an uninitialized
        interaction to be fitted from target dispersive shifts via ``circuit.fit_parameters(...)``.
        Default is None.
    component1 : Component, optional
        First circuit component.
    component2 : Component, optional
        Second circuit component.
    kind : str, optional
        Coupling type: 'capacitive', 'inductive' or 'custom'. Default is 'capacitive'.
    real_gauge : bool, optional
        Whether to express interaction in the real-symmetric gauge. Default is True.
    op1 : np.ndarray, optional
        Custom operator for component1. Required if kind is 'custom'.
    op2 : np.ndarray, optional
        Custom operator for component2. Required if kind is 'custom'.
    """

    def __init__(
        self,
        *args,
        g: Optional[float] = None,
        component1: Optional[Component] = None,
        component2: Optional[Component] = None,
        kind: str = 'capacitive',
        real_gauge: bool = True,
        op1: Optional[np.ndarray] = None,
        op2: Optional[np.ndarray] = None,
        **kwargs,
    ):
        # Flexible positional argument handling
        comps = []
        nums = []
        for a in args:
            if isinstance(a, Component):
                comps.append(a)
            elif isinstance(a, (int, float, np.floating)):
                nums.append(float(a))
            elif isinstance(a, str):
                kind = a

        c1 = component1 if component1 is not None else (comps.pop(0) if comps else None)
        c2 = component2 if component2 is not None else (comps.pop(0) if comps else None)
        g_val = g if g is not None else (nums.pop(0) if nums else None)

        if c1 is None or c2 is None:
            raise ValueError(f"Interaction requires two components. Got: component1={c1}, component2={c2}")

        self.component1 = c1
        self.component2 = c2
        self.g = float(g_val) if g_val is not None else None
        self.kind = kind.lower()
        self.real_gauge = bool(real_gauge)
        self.custom_op1 = op1
        self.custom_op2 = op2

    @property
    def is_initialized(self) -> bool:
        """Whether coupling strength g is initialized."""
        return self.g is not None

    def set_parameters(self, g: float):
        """Update coupling strength g."""
        self.g = float(g)

    def get_operator(self, real_gauge: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return the interaction operators (op1, op2) and effective coupling strength."""
        if not self.is_initialized:
            raise ValueError(
                f"Interaction between '{self.component1.name}' and '{self.component2.name}' has uninitialized coupling g=None. "
                "Run fit_parameters() or provide g before simulating."
            )
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
                    # Real-symmetric gauge: H_int = g * n_t * (a + a^dagger)
                    op2 = self.component2.a + self.component2.adag
                else:
                    # Canonical dipole gauge: H_int = -i * g * n_t * (a - a^dagger)
                    op2 = -1j * (self.component2.a - self.component2.adag)
                return op1, op2, self.g
            elif self.kind == "inductive":
                op1 = self.component1.sin_phi_operator
                op2 = self.component2.a + self.component2.adag
                return op1, op2, self.g

        # Resonator <-> Transmon (swapped)
        elif isinstance(self.component1, Resonator) and isinstance(self.component2, Transmon):
            if self.kind == "capacitive":
                op_t = self.component2.n_operator
                op_r = (self.component1.a + self.component1.adag) if use_real else -1j * (self.component1.a - self.component1.adag)
                return op_r, op_t, self.g
            elif self.kind == "inductive":
                op_t = self.component2.sin_phi_operator
                op_r = self.component1.a + self.component1.adag
                return op_r, op_t, self.g

        # Resonator <-> Resonator
        elif isinstance(self.component1, Resonator) and isinstance(self.component2, Resonator):
            if self.kind == "capacitive":
                if use_real:
                    op1 = self.component1.a + self.component1.adag
                    op2 = self.component2.a + self.component2.adag
                else:
                    op1 = self.component1.a - self.component1.adag
                    op2 = self.component2.a - self.component2.adag
                return op1, op2, self.g
            elif self.kind == "inductive":
                op1 = self.component1.a + self.component1.adag
                op2 = self.component2.a + self.component2.adag
                return op1, op2, self.g

        # Transmon <-> Transmon
        elif isinstance(self.component1, Transmon) and isinstance(self.component2, Transmon):
            if self.kind == "capacitive":
                # H_int = 4 * g * n1 * n2 (standard mutual charging energy form)
                op1 = self.component1.n_operator
                op2 = self.component2.n_operator
                return op1, op2, 4.0 * self.g

        raise NotImplementedError(
            f"Coupling kind '{self.kind}' between {type(self.component1).__name__} "
            f"and {type(self.component2).__name__} is not implemented. Use kind='custom'"
        )