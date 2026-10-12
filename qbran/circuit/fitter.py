"""Fitting utilities for matching circuit bare parameters to experimental dressed observables."""

from typing import Dict, Tuple, Optional, Union, List, Any
import numpy as np
import scipy.optimize as opt

from qbran.circuit.components import Component, Transmon, Resonator
from qbran.circuit.interactions import Interaction
from qbran.circuit.hilbert_space import HilbertSpace


class FitResult:
    """Container for circuit parameter fitting results and diagnostics.

    Attributes
    ----------
    circuit : HilbertSpace
        The fitted circuit containing the converged bare physical parameters.
    bare_parameters : Dict[str, float]
        Converged bare physical parameters (e.g., 'tmon.EJ', 'buffer.E_osc', 'g_tmon_buffer').
    target_observables : Dict[str, float]
        Target experimental dressed observables in GHz.
    fitted_observables : Dict[str, float]
        Fitted model dressed observables in GHz.
    residuals : Dict[str, float]
        Difference (fitted - target) in GHz.
    residuals_mhz : Dict[str, float]
        Difference (fitted - target) in MHz.
    max_residual_mhz : float
        Maximum absolute residual across all observables in MHz.
    success : bool
        Whether the numerical optimizer converged successfully.
    message : str
        Termination status message from the numerical optimizer.
    iterations : int
        Number of function evaluations performed.
    cost : float
        Final objective sum of squared residuals.
    """

    def __init__(
        self,
        circuit: HilbertSpace,
        bare_parameters: Dict[str, float],
        target_observables: Dict[str, float],
        fitted_observables: Dict[str, float],
        success: bool,
        message: str,
        iterations: int,
        cost: float,
    ):
        self.circuit = circuit
        self.bare_parameters = bare_parameters
        self.target_observables = target_observables
        self.fitted_observables = fitted_observables
        self.residuals = {
            k: fitted_observables[k] - target_observables[k] for k in target_observables
        }
        self.residuals_mhz = {k: v * 1e3 for k, v in self.residuals.items()}
        self.max_residual_mhz = (
            max(abs(v) for v in self.residuals_mhz.values()) if self.residuals_mhz else 0.0
        )
        self.success = bool(success)
        self.message = str(message)
        self.iterations = int(iterations)
        self.cost = float(cost)

    def summary(self) -> str:
        """Return formatted summary table comparing target and fitted observables."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            "=" * 120,
            f"QBRAN Circuit Parameter Fit Results",
            f"Status: {status} ({self.message})",
            f"Iterations: {self.iterations} | Max Residual: {self.max_residual_mhz:.4f} MHz ({self.max_residual_mhz * 1e3:.2f} kHz)",
            "=" * 120,
            f"{'Observable':<26} {'Target (GHz)':<16} {'Fitted (GHz)':<16} {'Residual (kHz)':<16} {'Fitted Bare Parameter'}",
            "-" * 120,
        ]

        # Associate bare parameters to observables for display
        obs_to_param = {}
        for obs_name in self.target_observables:
            if obs_name.endswith(".frequency"):
                comp_name = obs_name[:-10]
                if f"{comp_name}.EJ" in self.bare_parameters:
                    obs_to_param[obs_name] = f"{comp_name}.EJ"
                elif f"{comp_name}.E_osc" in self.bare_parameters:
                    obs_to_param[obs_name] = f"{comp_name}.E_osc"
            elif obs_name.endswith(".anharmonicity"):
                comp_name = obs_name[:-14]
                if f"{comp_name}.EC" in self.bare_parameters:
                    obs_to_param[obs_name] = f"{comp_name}.EC"
            elif obs_name.startswith("chi_"):
                pair_suffix = obs_name[4:]
                for p_name in self.bare_parameters:
                    if p_name.startswith("g_"):
                        g_pair = p_name[2:]
                        if g_pair == pair_suffix or g_pair == "_".join(reversed(pair_suffix.split("_"))):
                            obs_to_param[obs_name] = p_name
                            break

        for obs_name, target_val in self.target_observables.items():
            fitted_val = self.fitted_observables[obs_name]
            res_khz = (fitted_val - target_val) * 1e6
            p_name = obs_to_param.get(obs_name, None)
            if p_name and p_name in self.bare_parameters:
                bare_str = f"{p_name:<18} = {self.bare_parameters[p_name]:>11.6f} GHz"
            else:
                bare_str = ""
            lines.append(
                f"{obs_name:<26} {target_val:<16.6f} {fitted_val:<16.6f} {res_khz:<+16.3f} {bare_str}"
            )

        lines.append("=" * 120)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<FitResult success={self.success} max_residual={self.max_residual_mhz:.4f} MHz>"


def fit_parameters(
    circuit: HilbertSpace,
    target_frequencies: Optional[Dict[Union[str, Component], Union[float, Tuple[float, float]]]] = None,
    target_dispersive_shifts: Optional[Dict[Tuple[Union[str, Component], Union[str, Component]], float]] = None,
    target_anharmonicities: Optional[Dict[Union[str, Component], float]] = None,
    fixed_parameters: Optional[Dict[str, float]] = None,
    in_place: bool = True,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
    max_nfev: int = 150,
) -> FitResult:
    """Fit bare circuit parameters to match target dressed experimental observables.

        Parameters
        ----------
        target_frequencies : dict, optional
            Target dressed frequencies in GHz. Keys can be component names (str) or Component instances.
            Values can be:
            - float: target fundamental frequency (0 -> 1 transition) in GHz.
            - tuple (freq, alpha): for Transmon components, specifies both the 0 -> 1 frequency
                and the anharmonicity.
        target_dispersive_shifts : dict, optional
            Target cross-Kerr dispersive shifts in GHz. Keys are 2-tuples of components (names or instances).
            Dispersive shift is defined as:
                chi_ij = E(1_i, 1_j) - E(1_i, 0_j) - E(0_i, 1_j) + E(0_i, 0_j)
        target_anharmonicities : dict, optional
            Target transmon anharmonicities (self-Kerr) in GHz.
            Anharmonicity is defined as:
                alpha = E_12 - E_01
            (Note: can also be passed directly in ``target_frequencies`` as ``(freq, alpha)``).
        fixed_parameters : dict, optional
            Dictionary of bare parameters to keep fixed during the fit.
        in_place : bool, optional
            Whether to update the bare parameters on this circuit and its components/interactions
            in-place. Default is True.
        xtol : float, optional
            Relative error desired in the approximate solution. Default is 1e-8.
        ftol : float, optional
            Relative error desired in the objective sum of squares. Default is 1e-8.
        max_nfev : int, optional
            Maximum number of function evaluations for the optimizer. Default is 150.

        Returns
        -------
        FitResult
            Container with convergence diagnostics, bare parameters, and target vs fitted observables:
            - ``result.bare_parameters``: converged bare values (EJ, EC, E_osc, g)
            - ``result.summary()``: formatted ASCII diagnostic table
            - ``result.max_residual_mhz``: maximum discrepancy across all observables in MHz
            - ``result.success``: boolean flag indicating optimizer convergence

        Examples
        --------
        >>> import qbran as br
        >>> # Define uninitialized components (no bare parameters needed upfront)
        >>> tmon = br.Transmon(truncated_dim=10, name="tmon")
        >>> res = br.Resonator(truncated_dim=10, name="res")
        >>> circuit = br.HilbertSpace([tmon, res])
        >>> circuit.add_interaction(tmon, res)
        >>>
        >>> # Fit bare parameters to experimental dressed observables
        >>> res = circuit.fit_parameters(
        ...     target_frequencies={
        ...         tmon: (4.50, -0.25),  # 4.5 GHz freq, -250 MHz anharmonicity
        ...         res: 7.20,            # 7.2 GHz resonator freq
        ...     },
        ...     target_dispersive_shifts={
        ...         (tmon, res): -0.002,  # -2 MHz dispersive shift (or 0.002)
        ...     },
        ... )
        >>> print(res.summary())
        >>> print(res.bare_parameters)
        """
    comp_by_name = {c.name: c for c in circuit.components}
    fixed_params = fixed_parameters or {}

    def resolve_comp(c: Union[str, Component]) -> Component:
        if isinstance(c, Component):
            return c
        if str(c) in comp_by_name:
            return comp_by_name[str(c)]
        raise ValueError(f"Unknown circuit component '{c}'. Available components: {list(comp_by_name.keys())}")

    # 1. Parse target frequencies & anharmonicities
    targets_w: Dict[Component, float] = {}
    targets_alpha: Dict[Component, float] = {}

    if target_frequencies:
        for c_key, val in target_frequencies.items():
            comp = resolve_comp(c_key)
            if isinstance(val, (tuple, list)):
                if len(val) != 2:
                    raise ValueError(f"Expected (frequency, anharmonicity) tuple for '{comp.name}', got {val}")
                targets_w[comp] = float(val[0])
                targets_alpha[comp] = -abs(float(val[1]))
            else:
                targets_w[comp] = float(val)

    if target_anharmonicities:
        for c_key, val in target_anharmonicities.items():
            comp = resolve_comp(c_key)
            targets_alpha[comp] = -abs(float(val))

    # 2. Parse target dispersive shifts
    targets_chi: Dict[Tuple[Component, Component], float] = {}
    if target_dispersive_shifts:
        for pair, chi_val in target_dispersive_shifts.items():
            if len(pair) != 2:
                raise ValueError(f"Dispersive shift key must be a pair of components, got {pair}")
            c1 = resolve_comp(pair[0])
            c2 = resolve_comp(pair[1])
            targets_chi[(c1, c2)] = -abs(float(chi_val))

    # 3. Determine parameter vector theta to fit
    param_names: List[str] = []
    x0: List[float] = []
    bounds_low: List[float] = []
    bounds_high: List[float] = []

    # Map interactions
    inter_by_pair: Dict[frozenset, Interaction] = {}
    for inter in circuit.interactions:
        key = frozenset([inter.component1, inter.component2])
        inter_by_pair[key] = inter

    # For each component:
    for comp in circuit.components:
        if isinstance(comp, Transmon):
            # Check EC
            ec_fixed = None
            if f"{comp.name}.EC" in fixed_params:
                ec_fixed = float(fixed_params[f"{comp.name}.EC"])
            elif comp.EC is not None:
                ec_fixed = float(comp.EC)

            w_target = targets_w.get(comp, 5.0)

            if ec_fixed is not None:
                # Only fit EJ
                p_name = f"{comp.name}.EJ"
                param_names.append(p_name)
                ej_guess = comp.EJ if comp.EJ is not None else ((w_target + ec_fixed) ** 2) / (8.0 * ec_fixed)
                x0.append(max(1.0, float(ej_guess)))
                bounds_low.append(0.1)
                bounds_high.append(250.0)
            else:
                # Fit both EJ and EC
                if comp not in targets_alpha:
                    raise ValueError(
                        f"Transmon '{comp.name}' has uninitialized EC and no target anharmonicity was provided. "
                        "Please provide anharmonicity as a tuple (freq, alpha) in target_frequencies, "
                        "via target_anharmonicities, or by initializing Transmon(EC=...)."
                    )
                alpha_target = targets_alpha[comp]
                ec_guess = abs(alpha_target)
                ej_guess = comp.EJ if comp.EJ is not None else ((w_target + ec_guess) ** 2) / (8.0 * ec_guess)

                param_names.append(f"{comp.name}.EJ")
                x0.append(max(1.0, float(ej_guess)))
                bounds_low.append(0.1)
                bounds_high.append(250.0)

                param_names.append(f"{comp.name}.EC")
                x0.append(max(0.01, float(ec_guess)))
                bounds_low.append(0.005)
                bounds_high.append(2.0)

        elif isinstance(comp, Resonator):
            p_name = f"{comp.name}.E_osc"
            if p_name not in fixed_params:
                param_names.append(p_name)
                w_target = targets_w.get(comp, 7.0)
                e_osc_guess = comp.E_osc if comp.E_osc is not None else w_target
                x0.append(max(0.1, float(e_osc_guess)))
                bounds_low.append(0.1)
                bounds_high.append(50.0)

    # For each interaction:
    for inter in circuit.interactions:
        p_name = f"g_{inter.component1.name}_{inter.component2.name}"
        if p_name not in fixed_params:
            param_names.append(p_name)
            # Find target chi if available
            chi_target = None
            for (c1, c2), val in targets_chi.items():
                if set([c1, c2]) == set([inter.component1, inter.component2]):
                    chi_target = val
                    break

            if inter.g is not None:
                g_guess = float(inter.g)
            elif chi_target is not None:
                w1 = targets_w.get(inter.component1, 6.0)
                w2 = targets_w.get(inter.component2, 7.5)
                delta = abs(w1 - w2)
                alpha = targets_alpha.get(inter.component1, targets_alpha.get(inter.component2, -0.25))
                # Dispersive estimate: chi ≈ g^2 * alpha / (Delta * (Delta + alpha))
                g_est = np.sqrt(np.abs(chi_target * delta * (delta + alpha) / alpha))
                g_guess = float(np.clip(g_est, 0.01, 0.35))
            else:
                g_guess = 0.05

            x0.append(g_guess)
            bounds_low.append(1e-6)
            bounds_high.append(2.0)

    # 4. State indices in the product basis
    state_0 = tuple(0 for _ in circuit.components)
    idx_000 = circuit.state_to_index(*state_0)

    idx_single = {}
    for comp in circuit.components:
        state_e = [0] * len(circuit.components)
        state_e[circuit.components.index(comp)] = 1
        idx_single[comp] = circuit.state_to_index(*state_e)

    idx_double = {}
    for comp in circuit.components:
        if isinstance(comp, Transmon):
            state_2 = [0] * len(circuit.components)
            state_2[circuit.components.index(comp)] = 2
            idx_double[comp] = circuit.state_to_index(*state_2)

    idx_pair = {}
    for (c1, c2) in targets_chi:
        state_p = [0] * len(circuit.components)
        state_p[circuit.components.index(c1)] = 1
        state_p[circuit.components.index(c2)] = 1
        idx_pair[(c1, c2)] = circuit.state_to_index(*state_p)

    k_diag = min(max(35, len(circuit.components) * 6), circuit.dimension)

    # 5. Objective evaluation
    def apply_params(p_vec: np.ndarray):
        p_dict = dict(zip(param_names, p_vec))
        p_dict.update(fixed_params)

        for comp in circuit.components:
            if isinstance(comp, Transmon):
                ej_val = p_dict.get(f"{comp.name}.EJ", comp.EJ)
                ec_val = p_dict.get(f"{comp.name}.EC", comp.EC)
                comp.set_parameters(EJ=ej_val, EC=ec_val)
            elif isinstance(comp, Resonator):
                e_osc_val = p_dict.get(f"{comp.name}.E_osc", comp.E_osc)
                comp.set_parameters(E_osc=e_osc_val)

        for inter in circuit.interactions:
            p_key = f"g_{inter.component1.name}_{inter.component2.name}"
            p_key_rev = f"g_{inter.component2.name}_{inter.component1.name}"
            g_val = p_dict.get(p_key, p_dict.get(p_key_rev, inter.g))
            if g_val is not None:
                inter.set_parameters(g=g_val)

    def compute_observables(p_vec: np.ndarray):
        apply_params(p_vec)
        evals, evecs = circuit.diagonalize(k=k_diag)

        def get_E(target_idx: int) -> float:
            overlaps = np.abs(evecs[target_idx, :]) ** 2
            return evals[np.argmax(overlaps)]

        e_ground = get_E(idx_000)

        # Frequencies
        freqs_calc = {}
        for comp in targets_w:
            freqs_calc[comp] = get_E(idx_single[comp]) - e_ground

        # Anharmonicities
        alphas_calc = {}
        for comp in targets_alpha:
            e1 = get_E(idx_single[comp])
            e2 = get_E(idx_double[comp])
            alphas_calc[comp] = (e2 - e1) - (e1 - e_ground)

        # Dispersive shifts
        chis_calc = {}
        for pair in targets_chi:
            c1, c2 = pair
            e_pair = get_E(idx_pair[pair])
            e1 = get_E(idx_single[c1])
            e2 = get_E(idx_single[c2])
            chis_calc[pair] = (e_pair - e1) - (e2 - e_ground)

        return freqs_calc, alphas_calc, chis_calc

    def residual_fn(p_vec: np.ndarray) -> np.ndarray:
        freqs_calc, alphas_calc, chis_calc = compute_observables(p_vec)
        res = []

        for comp, target_val in targets_w.items():
            # Frequency residual normalized to 1.0 GHz
            res.append((freqs_calc[comp] - target_val) / 1.0)

        for comp, target_val in targets_alpha.items():
            # Anharmonicity residual normalized to 0.05 GHz
            res.append((alphas_calc[comp] - target_val) / 0.05)

        for pair, target_val in targets_chi.items():
            # Dispersive shift residual normalized to 0.002 GHz for high MHz precision
            res.append((chis_calc[pair] - target_val) / 0.002)

        return np.asarray(res, dtype=np.float64)

    # 6. Run optimization
    opt_res = opt.least_squares(
        residual_fn,
        x0=x0,
        bounds=(bounds_low, bounds_high),
        xtol=xtol,
        ftol=ftol,
        max_nfev=max_nfev,
        method="trf",
    )

    # Apply optimal bare parameters
    final_params = dict(zip(param_names, opt_res.x))
    final_params.update(fixed_params)

    apply_params(opt_res.x)

    # Populate all bare parameters on the circuit
    for comp in circuit.components:
        if isinstance(comp, Transmon):
            if comp.EJ is not None:
                final_params.setdefault(f"{comp.name}.EJ", float(comp.EJ))
            if comp.EC is not None:
                final_params.setdefault(f"{comp.name}.EC", float(comp.EC))
        elif isinstance(comp, Resonator):
            if comp.E_osc is not None:
                final_params.setdefault(f"{comp.name}.E_osc", float(comp.E_osc))
    for inter in circuit.interactions:
        if inter.g is not None:
            final_params.setdefault(f"g_{inter.component1.name}_{inter.component2.name}", float(inter.g))

    # 7. Collect final target vs fitted observables
    freqs_final, alphas_final, chis_final = compute_observables(opt_res.x)

    target_obs: Dict[str, float] = {}
    fitted_obs: Dict[str, float] = {}

    for comp, target_val in targets_w.items():
        key = f"{comp.name}.frequency"
        target_obs[key] = target_val
        fitted_obs[key] = freqs_final[comp]

    for comp, target_val in targets_alpha.items():
        key = f"{comp.name}.anharmonicity"
        target_obs[key] = target_val
        fitted_obs[key] = alphas_final[comp]

    for (c1, c2), target_val in targets_chi.items():
        key = f"chi_{c1.name}_{c2.name}"
        target_obs[key] = target_val
        fitted_obs[key] = chis_final[(c1, c2)]

    return FitResult(
        circuit=circuit,
        bare_parameters=final_params,
        target_observables=target_obs,
        fitted_observables=fitted_obs,
        success=opt_res.success,
        message=opt_res.message,
        iterations=opt_res.nfev,
        cost=opt_res.cost,
    )
