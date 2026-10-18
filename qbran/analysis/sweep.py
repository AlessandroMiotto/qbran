"""Multidimensional parameter sweep engine and results container."""

import os
import json
import copy
import itertools
from typing import Dict, List, Tuple, Optional, Union, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import h5py
from tqdm import tqdm

from qbran.circuit.components import Component, Transmon, Resonator
from qbran.circuit.interactions import Interaction
from qbran.circuit.hilbert_space import HilbertSpace
from qbran.analysis.analysis import Options
from qbran.floquet.shirley import _worker_init


def _resolve_parameter_target(
    circuit: HilbertSpace, key: Any
) -> Tuple[str, Any, str]:
    """Resolve a user-provided parameter key to (canonical_name, target_object, attr_name).

    Supported formats:
    - Dot-notation: "buffer.E_osc", "tmon.ng", "tmon.EJ", "tmon.EC"
    - Interaction string: "g_tmon_buffer" or "g_buffer_tmon"
    - Component tuple: (buffer, "E_osc") or ("buffer", "E_osc")
    - Interaction tuple: (tmon, buffer) or ("tmon", "buffer")
    """
    comp_by_name = {c.name: c for c in circuit.components}

    # Case 1: Tuple format
    if isinstance(key, (tuple, list)):
        if len(key) != 2:
            raise ValueError(f"Parameter tuple must have length 2, got {key}")
        first, second = key

        # Sub-case 1A: (comp, attr_str) e.g. (buffer, "E_osc") or ("buffer", "E_osc")
        if isinstance(second, str) and (
            isinstance(first, Component) or (isinstance(first, str) and first in comp_by_name)
        ):
            comp = first if isinstance(first, Component) else comp_by_name[first]
            attr = str(second)
            canonical = f"{comp.name}.{attr}"
            return canonical, comp, attr

        # Sub-case 1B: (c1, c2) representing an interaction coupling
        c1 = first if isinstance(first, Component) else comp_by_name.get(str(first))
        c2 = second if isinstance(second, Component) else comp_by_name.get(str(second))
        if c1 is not None and c2 is not None:
            for inter in circuit.interactions:
                if (inter.component1 == c1 and inter.component2 == c2) or (
                    inter.component1 == c2 and inter.component2 == c1
                ):
                    canonical = f"g_{inter.component1.name}_{inter.component2.name}"
                    return canonical, inter, "g"
            raise ValueError(f"No interaction found between '{c1.name}' and '{c2.name}'.")

        raise ValueError(f"Could not resolve parameter tuple {key}.")

    # Case 2: String format
    if isinstance(key, str):
        # Sub-case 2A: "comp_name.attr"
        if "." in key:
            parts = key.split(".", 1)
            comp_name, attr = parts[0], parts[1]
            if comp_name in comp_by_name:
                return f"{comp_name}.{attr}", comp_by_name[comp_name], attr
            raise ValueError(
                f"Component '{comp_name}' not found. Available components: {list(comp_by_name.keys())}"
            )

        # Sub-case 2B: "g_c1_c2"
        if key.startswith("g_"):
            suffix = key[2:]
            for inter in circuit.interactions:
                pair1 = f"{inter.component1.name}_{inter.component2.name}"
                pair2 = f"{inter.component2.name}_{inter.component1.name}"
                if suffix == pair1 or suffix == pair2:
                    canonical = f"g_{inter.component1.name}_{inter.component2.name}"
                    return canonical, inter, "g"
            raise ValueError(
                f"No interaction matching '{key}'. Available interactions: "
                f"{[f'g_{i.component1.name}_{i.component2.name}' for i in circuit.interactions]}"
            )

    raise ValueError(
        f"Unrecognized parameter key format: {key!r}. "
        "Use 'component.attribute', (component, 'attribute'), or (c1, c2) for interaction coupling."
    )


def _set_target_param(obj: Any, attr: str, val: float) -> None:
    """Set a physical parameter on a component or interaction, triggering re-diagonalization if needed."""
    val = float(val)
    if isinstance(obj, Transmon):
        obj.set_parameters(**{attr: val})
    elif isinstance(obj, Resonator):
        if attr == "E_osc":
            obj.set_parameters(E_osc=val)
        else:
            setattr(obj, attr, val)
    elif isinstance(obj, Interaction):
        if attr == "g":
            obj.set_parameters(g=val)
        else:
            setattr(obj, attr, val)
    else:
        setattr(obj, attr, val)


def _backup_circuit_state(circuit: HilbertSpace) -> Dict[str, Any]:
    """Snapshot the current bare physical parameters of all components and interactions."""
    backup = {}
    for comp in circuit.components:
        if isinstance(comp, Transmon):
            backup[f"{comp.name}.EJ"] = comp.EJ
            backup[f"{comp.name}.EC"] = comp.EC
            backup[f"{comp.name}.ng"] = comp.ng
        elif isinstance(comp, Resonator):
            backup[f"{comp.name}.E_osc"] = comp.E_osc
        elif hasattr(comp, "params_dict"):
            for k, v in comp.params_dict().items():
                backup[f"{comp.name}.{k}"] = v
    for inter in circuit.interactions:
        key = f"g_{inter.component1.name}_{inter.component2.name}"
        backup[key] = inter.g
    return backup


def _restore_circuit_state(circuit: HilbertSpace, backup: Dict[str, Any]) -> None:
    """Restore circuit bare physical parameters from a snapshot."""
    comp_by_name = {c.name: c for c in circuit.components}
    for key, val in backup.items():
        if "." in key:
            comp_name, attr = key.split(".", 1)
            if comp_name in comp_by_name and val is not None:
                _set_target_param(comp_by_name[comp_name], attr, val)
        elif key.startswith("g_"):
            suffix = key[2:]
            for inter in circuit.interactions:
                pair1 = f"{inter.component1.name}_{inter.component2.name}"
                pair2 = f"{inter.component2.name}_{inter.component1.name}"
                if (suffix == pair1 or suffix == pair2) and val is not None:
                    inter.set_parameters(g=val)


def _apply_circuit_parameters(
    circuit: HilbertSpace, param_dict: Dict[str, float]
) -> None:
    """Apply a dictionary of parameter updates to the circuit."""
    for key, val in param_dict.items():
        _, obj, attr = _resolve_parameter_target(circuit, key)
        _set_target_param(obj, attr, val)


def _match_coord_name(query_name: str, coords_keys: List[str]) -> Optional[str]:
    """Helper to match a user query name against canonical coordinate names."""
    # 1. Exact match
    if query_name in coords_keys:
        return query_name

    # 2. Check candidate dot substitutions (e.g. buffer_E_osc -> buffer.E_osc)
    parts = query_name.split("_")
    for i in range(1, len(parts)):
        cand = "_".join(parts[:i]) + "." + "_".join(parts[i:])
        if cand in coords_keys:
            return cand

    # 3. Check normalized equality ignoring delimiters and case
    q_norm = query_name.replace("_", "").replace(".", "").lower()
    for k in coords_keys:
        if k.replace("_", "").replace(".", "").lower() == q_norm:
            return k

    # 4. Match suffix (e.g. "ng" -> "tmon.ng", "E_osc" -> "buffer.E_osc")
    matches = [k for k in coords_keys if k.endswith(f".{query_name}") or k.endswith(f"_{query_name}")]
    if len(matches) == 1:
        return matches[0]

    return None


class SweepResult:
    """Multidimensional container for Floquet parameter sweep results.

    Parameters
    ----------
    metadata : dict, optional
        Simulation metadata and execution options.
    coords : dict, optional
        Mapping of coordinate names to 1D coordinate arrays.
    dims : tuple of str, optional
        Names of the swept coordinate dimensions.
    quasienergies : np.ndarray, optional
        N-D quasienergy spectrum array.
    hybridization : np.ndarray, optional
        N-D hybridization metric array.
    observables : dict, optional
        Dictionary of auxiliary N-D observable arrays (e.g. 'avg_excitation').
    """

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        coords: Optional[Dict[str, np.ndarray]] = None,
        dims: Optional[Tuple[str, ...]] = None,
        quasienergies: Optional[np.ndarray] = None,
        hybridization: Optional[np.ndarray] = None,
        observables: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.metadata = metadata or {}
        self.coords = coords or {}
        self.dims = tuple(dims) if dims is not None else ()
        self.quasienergies = quasienergies
        self.hybridization = hybridization
        self.observables = observables or {}

        self._data: Dict[str, Any] = {
            "metadata": self.metadata,
            "coords": self.coords,
            "dims": self.dims,
            "quasienergies": self.quasienergies,
            "hybridization": self.hybridization,
        }
        for k, v in self.observables.items():
            self._data[k] = v

    def __getitem__(self, key: str) -> Any:
        if key in self._data:
            return self._data[key]
        if key in self.coords:
            return self.coords[key]
        if key in self.observables:
            return self.observables[key]
        raise KeyError(f"Key '{key}' not found in SweepResult. Available keys: {list(self._data.keys())}")

    def __contains__(self, key: str) -> bool:
        return key in self._data or key in self.coords or key in self.observables

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, self.coords.get(key, self.observables.get(key, default)))

    def slice(self, **kwargs) -> "SweepResult":
        """Extract a sub-slice along one or more coordinates by value or index.

        Parameters
        ----------
        **kwargs : float or int or slice
            Coordinate name and value to slice at. Nearest matching coordinate value is used.
            Names can be full canonical names (e.g. ``buffer_E_osc=7.46``) or suffixes (e.g. ``ng=0.5``).

        Returns
        -------
        SweepResult
            New SweepResult with sliced dimensions removed or sub-sampled.

        Examples
        --------
        >>> sliced = res.slice(buffer_E_osc=7.46)
        >>> sliced_2d = res.slice(ng=0.5, drive_amp=0.2)
        """
        new_coords = dict(self.coords)
        new_dims = list(self.dims)
        q = self.quasienergies
        h = self.hybridization
        obs = {k: v.copy() for k, v in self.observables.items()}

        for raw_k, val in kwargs.items():
            target_coord = _match_coord_name(raw_k, list(new_coords.keys()))
            if target_coord is None:
                raise KeyError(
                    f"Coordinate '{raw_k}' not found in SweepResult coordinates: {list(new_coords.keys())}"
                )
            if target_coord not in new_dims:
                raise ValueError(f"Coordinate '{target_coord}' has already been sliced or is not in dims.")

            axis_idx = new_dims.index(target_coord)
            coord_arr = new_coords[target_coord]

            if isinstance(val, (int, float, np.floating)):
                # Nearest index lookup
                idx = int(np.argmin(np.abs(coord_arr - float(val))))
                if q is not None:
                    q = np.take(q, idx, axis=axis_idx)
                if h is not None:
                    h = np.take(h, idx, axis=axis_idx)
                for obs_name in obs:
                    obs[obs_name] = np.take(obs[obs_name], idx, axis=axis_idx)
                del new_coords[target_coord]
                new_dims.pop(axis_idx)

            elif isinstance(val, slice):
                # Slice range
                sl = val
                if q is not None:
                    q = np.take(q, range(*sl.indices(len(coord_arr))), axis=axis_idx)
                if h is not None:
                    h = np.take(h, range(*sl.indices(len(coord_arr))), axis=axis_idx)
                for obs_name in obs:
                    obs[obs_name] = np.take(obs[obs_name], range(*sl.indices(len(coord_arr))), axis=axis_idx)
                new_coords[target_coord] = coord_arr[sl]
            else:
                raise TypeError(f"Slice value for '{raw_k}' must be float, int, or slice, got {type(val)}.")

        return SweepResult(
            metadata=dict(self.metadata),
            coords=new_coords,
            dims=tuple(new_dims),
            quasienergies=q,
            hybridization=h,
            observables=obs,
        )

    def save(self, filename: str) -> None:
        """Export the multidimensional sweep dataset to a compressed HDF5 archive (.h5)."""
        if not filename.endswith(".h5"):
            filename += ".h5"

        with h5py.File(filename, "w") as f:
            meta_grp = f.create_group("metadata")
            for k, v in self.metadata.items():
                try:
                    if isinstance(v, (dict, list)):
                        meta_grp.attrs[k] = json.dumps(v)
                    else:
                        meta_grp.attrs[k] = v
                except Exception:
                    meta_grp.attrs[k] = str(v)

            f.attrs["dims"] = json.dumps(list(self.dims))

            coords_grp = f.create_group("coords")
            for c_name, c_arr in self.coords.items():
                coords_grp.create_dataset(c_name, data=c_arr, compression="gzip")

            if self.quasienergies is not None:
                f.create_dataset("quasienergies", data=self.quasienergies, compression="gzip")

            if self.hybridization is not None:
                f.create_dataset("hybridization", data=self.hybridization, compression="gzip")

            if self.observables:
                obs_grp = f.create_group("observables")
                for obs_name, arr in self.observables.items():
                    obs_grp.create_dataset(obs_name, data=arr, compression="gzip")

    @classmethod
    def load(cls, filename: str) -> "SweepResult":
        """Load a multidimensional sweep dataset from an HDF5 archive."""
        if not filename.endswith(".h5"):
            filename += ".h5"

        with h5py.File(filename, "r") as f:
            metadata = {}
            if "metadata" in f:
                for k, v in f["metadata"].attrs.items():
                    if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
                        try:
                            metadata[k] = json.loads(v)
                            continue
                        except Exception:
                            pass
                    metadata[k] = v

            dims = tuple(json.loads(f.attrs["dims"])) if "dims" in f.attrs else ()

            coords = {}
            if "coords" in f:
                for c_name in f["coords"]:
                    coords[c_name] = f["coords"][c_name][:]

            quasienergies = f["quasienergies"][:] if "quasienergies" in f else None
            hybridization = f["hybridization"][:] if "hybridization" in f else None

            observables = {}
            if "observables" in f:
                for obs_name in f["observables"]:
                    observables[obs_name] = f["observables"][obs_name][:]

        return cls(
            metadata=metadata,
            coords=coords,
            dims=dims,
            quasienergies=quasienergies,
            hybridization=hybridization,
            observables=observables,
        )

    def summary(self) -> str:
        """Return formatted summary of the sweep dimensions and dataset shapes."""
        lines = [
            "=" * 70,
            "QBRAN Multidimensional Sweep Result",
            "=" * 70,
            f"Dimensions: {self.dims}",
            "-" * 70,
            "Coordinates:",
        ]
        for name in self.dims:
            if name in self.coords:
                arr = self.coords[name]
                lines.append(
                    f"  - {name:<22}: {len(arr)} points in [{arr[0]:.4f}, {arr[-1]:.4f}]"
                )
        lines.append("-" * 70)
        if self.quasienergies is not None:
            lines.append(f"Quasienergies shape : {self.quasienergies.shape}")
        if self.hybridization is not None:
            lines.append(f"Hybridization shape : {self.hybridization.shape}")
        if self.observables:
            lines.append(f"Observables         : {list(self.observables.keys())}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        hyb_shape = self.hybridization.shape if self.hybridization is not None else None
        return f"<SweepResult dims={self.dims} hybridization_shape={hyb_shape}>"


def _sweep_worker_task(args: Tuple) -> Tuple[int, np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Worker task executed in subprocess for a single circuit parameter configuration."""
    idx, circuit, config_dict, freq_arr, actual_amps, is_amp_scalar, kind, options = args

    # Apply configuration to worker's local circuit copy
    _apply_circuit_parameters(circuit, config_dict)

    # Disable sub-process spawning and inner progress bars in workers
    worker_options = copy.copy(options)
    worker_options.n_workers = 1
    worker_options.progress_bar = False

    analysis = circuit.BranchAnalysis(kind=kind, options=worker_options)
    res = analysis.run(drive_freq=freq_arr, drive_amp=actual_amps)

    if is_amp_scalar:
        q_out = res.quasienergies[:, -1, :]
        h_out = res.hybridization[:, -1, :] if res.hybridization is not None else None
        obs_out = {k: v[:, -1, :] for k, v in res.observables.items()}
    else:
        q_out = res.quasienergies
        h_out = res.hybridization
        obs_out = res.observables

    return idx, q_out, h_out, obs_out


class ParameterSweep:
    """Orchestrator for multidimensional parameter sweeping across quantum circuits.

    Parameters
    ----------
    circuit : HilbertSpace
        Circuit layout with components and interactions.
    parameters : dict, optional
        Dictionary mapping parameter identifiers to 1D arrays or lists of values.
        Supports:
        - "buffer.E_osc": array
        - "tmon.ng": array
        - (buffer, "E_osc"): array
        - (tmon, buffer): array  (interaction coupling g)
    drive_freq : float, list, or ndarray
        Drive frequency (or frequencies) in GHz.
    drive_amp : float, list, or ndarray
        Drive amplitude (or amplitudes). If a scalar float is provided, an internal
        compact amplitude ramp from 0 to drive_amp is simulated to ensure valid
        Xiao diabatic tracking, and the slice at drive_amp is extracted.
    kind : str, optional
        Simulation method: 'shirley' (default) or 'propagator'.
    options : Options, optional
        Configuration options for simulation and parallel execution.
    n_amp_steps : int, optional
        Number of amplitude ramp steps when drive_amp is a scalar. Default is 15.
    """

    def __init__(
        self,
        circuit: HilbertSpace,
        parameters: Optional[Dict[Any, Union[List[float], np.ndarray]]] = None,
        drive_freq: Optional[Union[float, List[float], np.ndarray]] = None,
        drive_amp: Optional[Union[float, List[float], np.ndarray]] = None,
        kind: str = "shirley",
        options: Optional[Options] = None,
        n_amp_steps: int = 15,
        **kwargs,
    ):
        self.circuit = circuit
        self.parameters = parameters or {}
        if drive_freq is None:
            raise ValueError("drive_freq must be provided.")
        if drive_amp is None:
            raise ValueError("drive_amp must be provided.")

        self.raw_drive_freq = drive_freq
        self.raw_drive_amp = drive_amp
        self.kind = str(kind).lower()
        self.options = options if options is not None else Options()
        self.n_amp_steps = max(2, int(n_amp_steps))
        self.kwargs = kwargs

        # Parse drive frequency
        self.freq_arr = np.atleast_1d(np.asarray(drive_freq, dtype=float))

        # Parse drive amplitude: scalar vs vector
        if np.isscalar(drive_amp) or (
            isinstance(drive_amp, (list, np.ndarray)) and len(np.asarray(drive_amp).shape) == 0
        ) or (isinstance(drive_amp, (list, np.ndarray)) and len(drive_amp) == 1):
            self.is_amp_scalar = True
            target_amp = float(drive_amp if np.isscalar(drive_amp) else drive_amp[0])
            self.actual_amps = np.linspace(0.0, target_amp, self.n_amp_steps)
            self.target_amp = target_amp
        else:
            self.is_amp_scalar = False
            self.actual_amps = np.asarray(drive_amp, dtype=float)
            self.target_amp = None

        # Resolve parameter targets and coordinates
        self.param_names: List[str] = []
        self.param_arrays: List[np.ndarray] = []
        self.resolved_targets: List[Tuple[str, Any, str]] = []

        for k, v in self.parameters.items():
            canonical, obj, attr = _resolve_parameter_target(self.circuit, k)
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            self.param_names.append(canonical)
            self.param_arrays.append(arr)
            self.resolved_targets.append((canonical, obj, attr))

        self.param_shapes = tuple(len(a) for a in self.param_arrays)
        self.total_configs = int(np.prod(self.param_shapes)) if self.param_shapes else 1

    def run(self) -> SweepResult:
        """Execute the multidimensional sweep and assemble the results."""
        # Backup original circuit state
        backup = _backup_circuit_state(self.circuit)

        try:
            return self._run_internal()
        finally:
            # Guarantees circuit state is restored even on exception
            _restore_circuit_state(self.circuit, backup)

    def _run_internal(self) -> SweepResult:
        coords: Dict[str, np.ndarray] = {}
        for name, arr in zip(self.param_names, self.param_arrays):
            coords[name] = arr
        coords["drive_freq"] = self.freq_arr
        if not self.is_amp_scalar:
            coords["drive_amp"] = self.actual_amps

        dims: List[str] = list(self.param_names) + ["drive_freq"]
        if not self.is_amp_scalar:
            dims.append("drive_amp")

        # Generate all parameter combinations
        if self.param_arrays:
            combinations = list(itertools.product(*self.param_arrays))
        else:
            combinations = [()]

        config_dicts: List[Dict[str, float]] = []
        for combo in combinations:
            cfg = {name: val for name, val in zip(self.param_names, combo)}
            config_dicts.append(cfg)

        show_progress = getattr(self.options, "progress_bar", True)
        n_workers = getattr(self.options, "n_workers", 1)
        num_workers = os.cpu_count() or 1 if n_workers == -1 else int(n_workers)

        results_by_idx: Dict[int, Tuple] = {}

        # Path 1: Multi-worker parallel execution across parameter configurations
        if num_workers > 1 and self.total_configs > 1:
            max_workers = min(num_workers, self.total_configs)
            tasks = [
                (
                    i,
                    self.circuit,
                    cfg,
                    self.freq_arr,
                    self.actual_amps,
                    self.is_amp_scalar,
                    self.kind,
                    self.options,
                )
                for i, cfg in enumerate(config_dicts)
            ]

            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_worker_init,
            ) as executor:
                futures = {executor.submit(_sweep_worker_task, task): task[0] for task in tasks}
                with tqdm(
                    total=self.total_configs,
                    desc=f"Parameter Sweep ({max_workers} workers)",
                    disable=not show_progress,
                ) as pbar:
                    for future in as_completed(futures):
                        idx, q_out, h_out, obs_out = future.result()
                        results_by_idx[idx] = (q_out, h_out, obs_out)
                        pbar.update(1)

        # Path 2: Sequential execution
        else:
            pbar = tqdm(
                total=self.total_configs,
                desc="Parameter Sweep",
                disable=not show_progress or self.total_configs <= 1,
            )
            inner_options = copy.copy(self.options)
            if self.total_configs > 1:
                inner_options.progress_bar = False

            for idx, cfg in enumerate(config_dicts):
                _apply_circuit_parameters(self.circuit, cfg)
                analysis = self.circuit.BranchAnalysis(kind=self.kind, options=inner_options)
                raw_res = analysis.run(drive_freq=self.freq_arr, drive_amp=self.actual_amps)

                if self.is_amp_scalar:
                    q_out = raw_res.quasienergies[:, -1, :]
                    h_out = raw_res.hybridization[:, -1, :] if raw_res.hybridization is not None else None
                    obs_out = {k: v[:, -1, :] for k, v in raw_res.observables.items()}
                else:
                    q_out = raw_res.quasienergies
                    h_out = raw_res.hybridization
                    obs_out = raw_res.observables

                results_by_idx[idx] = (q_out, h_out, obs_out)
                pbar.update(1)
            pbar.close()

        # Assemble N-D arrays
        sample_q, sample_h, sample_obs = results_by_idx[0]
        full_q_shape = self.param_shapes + sample_q.shape
        full_quasienergies = np.zeros(full_q_shape, dtype=np.float64)

        has_hybridization = sample_h is not None
        if has_hybridization:
            full_h_shape = self.param_shapes + sample_h.shape
            full_hybridization = np.zeros(full_h_shape, dtype=np.float64)
        else:
            full_hybridization = None

        full_observables: Dict[str, np.ndarray] = {}
        for obs_name, arr in sample_obs.items():
            full_observables[obs_name] = np.zeros(self.param_shapes + arr.shape, dtype=np.float64)

        for idx in range(self.total_configs):
            q_out, h_out, obs_out = results_by_idx[idx]
            if self.param_shapes:
                multi_idx = np.unravel_index(idx, self.param_shapes)
            else:
                multi_idx = ()

            full_quasienergies[multi_idx] = q_out
            if has_hybridization and h_out is not None:
                full_hybridization[multi_idx] = h_out
            for obs_name, arr in obs_out.items():
                full_observables[obs_name][multi_idx] = arr

        metadata = {
            "solver": self.kind,
            "swept_parameters": self.param_names,
            "total_configurations": self.total_configs,
            "drive_amp_fixed": self.is_amp_scalar,
            "target_drive_amp": self.target_amp,
        }

        return SweepResult(
            metadata=metadata,
            coords=coords,
            dims=tuple(dims),
            quasienergies=full_quasienergies,
            hybridization=full_hybridization,
            observables=full_observables,
        )
