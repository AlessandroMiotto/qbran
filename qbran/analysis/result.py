"""Result container for parameter sweeps and branch analyses with HDF5 persistence."""

import json
from typing import Dict, Any, Optional
import numpy as np
import h5py


class Result:
    """Stores the full physical dataset of a 2D Floquet parameter sweep.

    Supports both dictionary indexing (`results['avg_excitation']`) and attribute
    access (`results.avg_excitation`).

    Attributes
    ----------
    metadata : dict
        Circuit parameters, solver options, subsystem properties, timestamp.
    drive_freq : np.ndarray
        1D array of swept drive frequencies (GHz).
    drive_amp : np.ndarray
        1D array of swept drive amplitudes.
    quasienergies : np.ndarray
        Array of quasienergies of shape (len(drive_freq), len(drive_amp), k).
    hybridization : np.ndarray or None
        Array of hybridization metric values of shape (len(drive_freq), len(drive_amp), num_tracked),
        or None if tracked_states was not specified.
    observables : dict
        Mapping of observable name -> array.
    """

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]],
        drive_freq: np.ndarray,
        drive_amp: np.ndarray,
        quasienergies: np.ndarray,
        hybridization: Optional[np.ndarray],
        observables: Dict[str, np.ndarray],
        **extra_data,
    ):
        self.metadata = dict(metadata) if metadata is not None else {}
        self.drive_freq = np.asarray(drive_freq)
        self.drive_amp = np.asarray(drive_amp)
        self.quasienergies = np.asarray(quasienergies)
        self.hybridization = np.asarray(hybridization) if hybridization is not None else None
        self.observables = {k: np.asarray(v) for k, v in observables.items()}

        self._data: Dict[str, Any] = {
            "metadata": self.metadata,
            "drive_freq": self.drive_freq,
            "drive_amp": self.drive_amp,
            "quasienergies": self.quasienergies,
            "hybridization": self.hybridization,
            "observables": self.observables,
        }

        # Expose observables as attributes if not conflicting with existing methods
        protected_attrs = set(dir(self))
        for k, v in self.observables.items():
            self._data[k] = v
            if k not in protected_attrs:
                setattr(self, k, v)

        for k, v in extra_data.items():
            self._data[k] = v
            if k not in protected_attrs:
                setattr(self, k, v)

    def __getitem__(self, key: str) -> Any:
        if key in self._data:
            return self._data[key]
        if key in self.observables:
            return self.observables[key]
        raise KeyError(f"Key '{key}' not found in Result. Available keys: {list(self._data.keys())}")

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
        if key not in dir(self):
            setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._data or key in self.observables

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, self.observables.get(key, default))

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def save(self, filename: str) -> None:
        """Exports the simulation results to a compressed HDF5 archive (.h5)."""
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

            axes_grp = f.create_group("axes")
            axes_grp.create_dataset("drive_freq", data=self.drive_freq, compression="gzip")
            axes_grp.create_dataset("drive_amp", data=self.drive_amp, compression="gzip")

            f.create_dataset("quasienergies", data=self.quasienergies, compression="gzip")

            if self.hybridization is not None:
                f.create_dataset("hybridization", data=self.hybridization, compression="gzip")

            obs_grp = f.create_group("observables")
            for obs_name, arr in self.observables.items():
                obs_grp.create_dataset(obs_name, data=arr, compression="gzip")

    @classmethod
    def load(cls, filename: str) -> "Result":
        """Loads a saved simulation from an HDF5 file."""
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

            drive_freq = f["axes/drive_freq"][:]
            drive_amp = f["axes/drive_amp"][:]
            quasienergies = f["quasienergies"][:]

            hybridization = f["hybridization"][:] if "hybridization" in f else None

            observables = {}
            if "observables" in f:
                for k in f["observables"]:
                    observables[k] = f["observables"][k][:]

        return cls(
            metadata=metadata,
            drive_freq=drive_freq,
            drive_amp=drive_amp,
            quasienergies=quasienergies,
            hybridization=hybridization,
            observables=observables,
        )

    def __repr__(self) -> str:
        keys_str = ", ".join(list(self._data.keys())[:8])
        return f"Result(keys=[{keys_str}, ...])"