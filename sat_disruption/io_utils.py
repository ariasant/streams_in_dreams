"""HDF5 snapshot I/O.

One file per snapshot: ``{out_dir}/{run_name}/snapshot_{k:04d}.hdf5``.
Independent files are convenient to feed into a downstream BFE-fitting
pipeline and are the simplest thing to write first.

Each file stores plain arrays with units documented in attributes:
    t     : scalar,  Myr
    pos   : (N, 3),  kpc
    vel   : (N, 3),  km/s
    mass  : (N,),    Msun
File-level attrs additionally record dt, dt_int, host potential type/params
(as a JSON string), the run name and the random seed.
"""

import json
import os

import h5py
import numpy as np


def run_dir(out_dir, run_name):
    """Directory holding all snapshots for a run."""
    return os.path.join(out_dir, run_name)


def snapshot_path(out_dir, run_name, index):
    return os.path.join(run_dir(out_dir, run_name), f"snapshot_{index:04d}.hdf5")


def write_snapshot(path, t, pos, vel, mass, attrs):
    """Write one snapshot to ``path``.

    Parameters
    ----------
    path : str
    t : float
        Time in Myr.
    pos : (N, 3) ndarray, kpc
    vel : (N, 3) ndarray, km/s
    mass : (N,) ndarray, Msun
    attrs : dict
        Run-level metadata (dt, dt_int, host_type, host_params, run_name,
        seed, ...). Values that are not plain scalars/strings are JSON-encoded.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        d_t = f.create_dataset("t", data=float(t))
        d_t.attrs["units"] = "Myr"
        d_pos = f.create_dataset("pos", data=np.asarray(pos, dtype=np.float64))
        d_pos.attrs["units"] = "kpc"
        d_vel = f.create_dataset("vel", data=np.asarray(vel, dtype=np.float64))
        d_vel.attrs["units"] = "km/s"
        d_mass = f.create_dataset("mass", data=np.asarray(mass, dtype=np.float64))
        d_mass.attrs["units"] = "Msun"

        for key, val in attrs.items():
            if isinstance(val, (str, int, float, np.integer, np.floating)):
                f.attrs[key] = val
            else:
                f.attrs[key] = json.dumps(val)


def read_snapshot(path):
    """Read a snapshot file into a dict with keys t, pos, vel, mass, attrs."""
    with h5py.File(path, "r") as f:
        out = {
            "t": float(f["t"][()]),
            "pos": f["pos"][()],
            "vel": f["vel"][()],
            "mass": f["mass"][()],
            "attrs": dict(f.attrs),
        }
    return out


def list_snapshots(out_dir, run_name):
    """Return the sorted list of snapshot file paths for a run."""
    d = run_dir(out_dir, run_name)
    if not os.path.isdir(d):
        return []
    files = [
        os.path.join(d, fn)
        for fn in os.listdir(d)
        if fn.startswith("snapshot_") and fn.endswith(".hdf5")
    ]
    return sorted(files)
