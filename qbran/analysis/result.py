"""Result container for parameter sweeps and branch analyses with HDF5 persistence."""

from typing import Dict, Any, Optional
import numpy as np
import h5py

class Result:
    """Stores the full physical dataset of a 2D Floquet or Branch parameter sweep.

    Attributes
    ----------
    metadata : dict
        Chip parameters, options, subsystem properties, timestamps.
    drive_freq : np.ndarray
        1D array of swept drive frequencies (GHz).
    drive_amp : np.ndarray
        1D array of swept drive amplitudes (n_photons or xi).
    quasienergies : np.ndarray
        3D array of shape (n_freq, n_amp, n_levels).
    hybridization : np.ndarray
        3D array of shape (n_freq, n_amp, n_target_states).
    partner : np.ndarray
        3D integer array of shape (n_freq, n_amp, n_target_states).
    partner_overlap : np.ndarray
        3D array of shape (n_freq, n_amp, n_target_states).
    observables : dict
        Mapping of observable name -> 3D array (n_freq, n_amp, n_target_states).
    """

    def __init__(
        self,
        metadata: Dict[str, Any],
        drive_freq: np.ndarray,
        drive_amp: np.ndarray,
        quasienergies: np.ndarray,
        hybridization: np.ndarray,
        partner: np.ndarray,
        partner_overlap: np.ndarray,
        observables: Dict[str, np.ndarray],
    ):
        self.metadata = metadata
        self.drive_freq = np.asarray(drive_freq)
        self.drive_amp = np.asarray(drive_amp)
        self.quasienergies = np.asarray(quasienergies)
        self.hybridization = np.asarray(hybridization)
        self.partner = np.asarray(partner)
        self.partner_overlap = np.asarray(partner_overlap)
        self.observables = {k: np.asarray(v) for k, v in observables.items()}

    def save(self, filename: str) -> None:
        """Exports the simulation results to a compressed HDF5 archive (.h5)."""
        if not filename.endswith(".h5"):
            filename += ".h5"

        with h5py.File(filename, "w") as f:
            # Metadata as attributes
            meta_grp = f.create_group("metadata")
            for k, v in self.metadata.items():
                try:
                    meta_grp.attrs[k] = v
                except Exception:
                    meta_grp.attrs[k] = str(v)

            # Axes
            axes_grp = f.create_group("axes")
            axes_grp.create_dataset("drive_freq", data=self.drive_freq, compression="gzip")
            axes_grp.create_dataset("drive_amp", data=self.drive_amp, compression="gzip")

            # Metrics and spectrum
            f.create_dataset("quasienergies", data=self.quasienergies, compression="gzip")
            f.create_dataset("hybridization", data=self.hybridization, compression="gzip")
            f.create_dataset("partner", data=self.partner, compression="gzip")
            f.create_dataset("partner_overlap", data=self.partner_overlap, compression="gzip")

            # Observables
            obs_grp = f.create_group("observables")
            for obs_name, arr in self.observables.items():
                obs_grp.create_dataset(obs_name, data=arr, compression="gzip")

    @classmethod
    def load(cls, filename: str) -> "Result":
        """Loads a saved simulation from an HDF5 file."""
        if not filename.endswith(".h5"):
            filename += ".h5"

        with h5py.File(filename, "r") as f:
            metadata = dict(f["metadata"].attrs)
            drive_freq = f["axes/drive_freq"][:]
            drive_amp = f["axes/drive_amp"][:]
            quasienergies = f["quasienergies"][:]
            hybridization = f["hybridization"][:]
            partner = f["partner"][:]
            partner_overlap = f["partner_overlap"][:]

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
            partner=partner,
            partner_overlap=partner_overlap,
            observables=observables,
        )