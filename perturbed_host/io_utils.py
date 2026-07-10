"""HDF5 snapshot I/O for the perturbed-host pipeline.

One file per run. Each science snapshot is a group ``/snap_XXXX`` holding the full
phase-space state plus a per-particle ``is_perturber`` boolean, so a downstream BFE-fitting
pipeline can cleanly separate the host's own density from the perturber's known trajectory.
Storing the flag per snapshot (rather than once at the top level) handles the Phase A -> B
change in particle count without special cases.
"""

from __future__ import annotations

import numpy as np
import h5py


class SnapshotWriter:
    """Incrementally append science snapshots to a single HDF5 file."""

    def __init__(self, path, config_yaml=None):
        self.path = str(path)
        self._count = 0
        with h5py.File(self.path, "w") as f:
            f.attrs["format"] = "perturbed_host.snapshots.v1"
            if config_yaml is not None:
                f.attrs["config_yaml"] = config_yaml

    def write(self, step, time, pos, vel, masses, is_perturber):
        """Write one snapshot group. ``step`` is the global integrator step index."""
        with h5py.File(self.path, "a") as f:
            g = f.create_group(f"snap_{self._count:04d}")
            g.attrs["step"] = int(step)
            g.attrs["time"] = float(time)
            g.create_dataset("pos", data=np.asarray(pos, dtype=np.float64))
            g.create_dataset("vel", data=np.asarray(vel, dtype=np.float64))
            g.create_dataset("mass", data=np.asarray(masses, dtype=np.float64))
            g.create_dataset("is_perturber", data=np.asarray(is_perturber, dtype=bool))
        self._count += 1

    @property
    def n_snapshots(self):
        return self._count


def read_snapshots(path):
    """Read all snapshots back, ordered by write index.

    Returns a list of dicts with keys ``step, time, pos, vel, mass, is_perturber``.
    """
    out = []
    with h5py.File(str(path), "r") as f:
        names = sorted(k for k in f.keys() if k.startswith("snap_"))
        for name in names:
            g = f[name]
            out.append(dict(
                step=int(g.attrs["step"]),
                time=float(g.attrs["time"]),
                pos=g["pos"][:],
                vel=g["vel"][:],
                mass=g["mass"][:],
                is_perturber=g["is_perturber"][:],
            ))
    return out
