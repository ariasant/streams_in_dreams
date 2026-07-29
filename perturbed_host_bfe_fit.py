#!/usr/bin/env python3
"""Fit EXP spherical BFE coefficients to the perturbed_host point-mass run at several cadences.

Builds one "sphereSL" basis from the host's relaxed, pre-perturbation state, then computes
per-snapshot coefficients at each requested cadence by subsampling the run's HDF5 snapshots.
The leapfrog integration runs at a fixed dt_int independent of the science snapshot cadence
dt (see perturbed_host/simulation.py), so subsampling a fine-cadence run's snapshots is
identical to rerunning the pipeline at a coarser dt -- just instant, and without regenerating
the (tens-of-GB) snapshot file. Requested cadences must be exact multiples of the file's base
cadence.

This does the expensive pyEXP work (createFromArray on up to N_host=1e6 particles, per
snapshot, per cadence) and writes results to disk; perturbed_host_bfe_cadence.ipynb only reads
them back for plotting -- it never calls createFromArray itself. Meant to run as a SLURM batch
job (see prospero_submit_bfe_fit.sh), not interactively.

Usage:
    python perturbed_host_bfe_fit.py --snapshots runs/point_mass/point_mass_run_snapshots.h5 \
        --output-dir runs/point_mass/bfe --base-dt 0.1 --cadences 0.2,10,30
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import yaml
from tqdm import tqdm

import pyEXP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import DREAMS_utils


def compute_host_com(snapshot_path, snap_names):
    """Host (is_perturber==False) mass-weighted center of mass at each of the given snapshots.

    The perturber (M_pert=0.2, a fifth of the host's mass) exerts a real gravitational recoil on
    the host once injected -- host |COM| grows from ~0.003 at injection to ~4 (four scale radii)
    by the end of the run. That's unlike the isolated, non-perturbed toy Plummer sphere
    plummer_bfe_report.ipynb / plummer_sphere_stream_sim.ipynb were built for, which never needed
    this. The static sphereSL basis assumes an origin-centered density, so positions must be
    recentered on this per-snapshot COM before fitting coefficients (matching those notebooks'
    `pynbody.analysis.center(..., with_velocity=False)` step) -- otherwise the reconstructed
    potential's center systematically lags the host's real one, and every downstream force/orbit
    comparison is contaminated by pure bulk translation on top of the cadence effect being studied.
    """
    com = np.empty((len(snap_names), 3))
    times = np.empty(len(snap_names))
    with h5py.File(snapshot_path, "r") as f:
        for i, name in enumerate(tqdm(snap_names, desc="host COM")):
            g = f[name]
            flag = g["is_perturber"][:]
            pos = g["pos"][:][~flag]
            mass = g["mass"][:][~flag]
            com[i] = np.average(pos, axis=0, weights=mass)
            times[i] = float(g.attrs["time"])
    return times, com


def read_host_snapshot(path, name, com=None):
    """Read one science snapshot's host-only phase-space data (masks out the perturber).

    If `com` is given, positions are recentered on it (position-only, matching
    pynbody.analysis.center(..., with_velocity=False) in the reference notebooks).
    """
    with h5py.File(path, "r") as f:
        g = f[name]
        flag = g["is_perturber"][:]
        pos = g["pos"][:][~flag]
        mass = g["mass"][:][~flag]
        t = float(g.attrs["time"])
    if com is not None:
        pos = pos - com
    return pos, mass, t


def build_basis(snapshot_path, first_snap_name, com0, output_dir, lmax, nmax, numr=4000):
    """Build a static 'sphereSL' basis from the host's relaxed (pre-perturbation) state."""
    pos, mass, t0 = read_host_snapshot(snapshot_path, first_snap_name, com=com0)
    r = np.linalg.norm(pos, axis=1)

    rbins, dvals = DREAMS_utils.return_density(
        r=r, weights=mass, rangevals=[r[r > 0].min() * 0.5, r.max() * 1.1],
        bins=500, log_bins=True, smooth=True,
    )

    model_file = os.path.join(output_dir, "basis_empirical_model.txt")
    cache_file = os.path.join(output_dir, "basis_sphereSL.cache.run0")
    for f in (model_file, cache_file):
        if os.path.exists(f):
            os.remove(f)

    DREAMS_utils.makemodel_empirical(rvals=rbins, dvals=dvals, pfile=model_file)

    config = {
        "id": "sphereSL",
        "parameters": {
            "numr": numr,
            "rmin": 0.01,
            "rmax": 10,
            "Lmax": lmax,
            "nmax": nmax,
            "rmapping": 0.1,  
            "modelname": model_file,
            "cachename": cache_file,
            "pcavar": False,  # no SNR/covariance analysis needed for the force-vs-time comparison
        },
        "runtag": "run0",
    }
    yaml_file = os.path.join(output_dir, "basis_sphereSL.yml")
    with open(yaml_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    with open(yaml_file, "r") as f:
        yaml_config = f.read()
    basis = pyEXP.basis.Basis.factory(yaml_config)

    print(f"Basis built from {first_snap_name} (t={t0:.3f}), "
          f"r range [{rbins.min():.4g}, {rbins.max():.4g}]", flush=True)
    return basis, yaml_file


def fit_cadence(basis, snapshot_path, snap_names, com, stride, output_dir, label):
    """Accumulate BFE coefficients over a strided subset of snapshots."""
    selected_names = snap_names[::stride]
    selected_com = com[::stride]
    coefs_container = None
    start = time.time()
    for name, c in zip(tqdm(selected_names, desc=f"dt={label}"), selected_com):
        pos, mass, t = read_host_snapshot(snapshot_path, name, com=c)
        coefs = basis.createFromArray(mass, pos, time=t)
        if coefs_container is None:
            coefs_container = pyEXP.coefs.Coefs.makecoefs(coefs)
        coefs_container.add(coefs)

    # System (Henon) units: G=M_host=a_host=1, all dimensionless -- WriteH5Coefs requires these
    # to be set explicitly (pyEXP raises otherwise).
    for unit_type in ("G", "length", "mass", "time"):
        coefs_container.setUnits(unit_type, "none", 1.0)

    coefs_file = os.path.join(output_dir, f"coefs_dt{label}.h5")
    if os.path.exists(coefs_file):
        os.remove(coefs_file)
    coefs_container.WriteH5Coefs(coefs_file)

    elapsed = (time.time() - start) / 60
    print(f"dt={label}: {len(selected_names)} snapshots, {elapsed:.2f} min -> {coefs_file}",
          flush=True)
    return coefs_file


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", required=True, help="path to <run>_snapshots.h5")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-dt", type=float, required=True,
                     help="science snapshot cadence the file was written at")
    ap.add_argument("--cadences", required=True,
                     help="comma-separated list of cadences to fit, e.g. 0.2,10,30")
    ap.add_argument("--lmax", type=int, default=6)
    ap.add_argument("--nmax", type=int, default=20)
    args = ap.parse_args()

    # Absolute, so the basis YAML's modelname/cachename fields resolve the same way regardless
    # of the caller's cwd -- gala's EXPPotential doesn't resolve relative paths the way
    # pyEXP.basis.Basis.factory does, so a relative --output-dir breaks it downstream.
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.snapshots, "r") as f:
        snap_names = sorted(k for k in f if k.startswith("snap_"))
    print(f"{len(snap_names)} snapshots found in {args.snapshots}", flush=True)

    com_times, com = compute_host_com(args.snapshots, snap_names)
    np.savez(os.path.join(args.output_dir, "host_com.npz"), time=com_times, com=com)
    com_norm = np.linalg.norm(com, axis=1)
    print(f"Host |COM| range: [{com_norm.min():.4f}, {com_norm.max():.4f}] "
          f"(a_host=1) -- saved to host_com.npz", flush=True)

    basis, basis_yaml = build_basis(
        args.snapshots, snap_names[0], com[0], args.output_dir, args.lmax, args.nmax)

    for cadence in args.cadences.split(","):
        cadence = cadence.strip()
        exact_stride = float(cadence) / args.base_dt
        stride = round(exact_stride)
        if not np.isclose(stride, exact_stride) or stride < 1:
            raise ValueError(
                f"cadence {cadence} is not an exact positive multiple of base-dt {args.base_dt}")
        fit_cadence(basis, args.snapshots, snap_names, com, stride, args.output_dir, cadence)


if __name__ == "__main__":
    main()
