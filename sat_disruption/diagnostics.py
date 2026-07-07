"""Sanity-check diagnostics for a completed run.

Reads the HDF5 snapshots of a run and produces:
  1. fractional total-energy error vs time (self-gravity + KE + host PE);
  2. bound mass fraction and 10/50/90% Lagrange radii vs time;
  3. x-y and x-z position scatters at a few snapshots (tidal tails).

All energies are computed in (Msun, kpc, km/s), consistent with the softened
Plummer per-particle potentials used by the integrator.
"""

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from initial_conditions import G_KPC_KMS2_MSUN
from io_utils import list_snapshots, read_snapshot, run_dir
from potentials import build_host_potential


# --- energy -----------------------------------------------------------------

def self_potential_energy(pos, mass, eps):
    """Self-gravity PE of N softened Plummer particles [Msun (km/s)^2].

    Consistent with the integrator: particle j sources a Plummer potential
    phi_j(r) = -G m_j / sqrt(|r - r_j|^2 + eps^2). Total PE is
    -0.5 G sum_{i != j} m_i m_j / sqrt(r_ij^2 + eps^2).
    """
    # pairwise squared distances (N, N)
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    inv = 1.0 / np.sqrt(r2 + eps * eps)
    np.fill_diagonal(inv, 0.0)  # drop self term
    mm = np.outer(mass, mass)
    return -0.5 * G_KPC_KMS2_MSUN * np.sum(mm * inv)


def total_energy(snap, host, eps):
    """Total energy of a snapshot [Msun (km/s)^2]."""
    pos = snap["pos"]          # kpc
    vel = snap["vel"]          # km/s
    mass = snap["mass"]        # Msun

    ke = 0.5 * np.sum(mass * np.sum(vel * vel, axis=1))

    # host specific potential -> (km/s)^2
    q = pos.T * u.kpc
    phi = host.energy(q).to_value((u.km / u.s) ** 2)
    ext_pe = np.sum(mass * phi)

    self_pe = self_potential_energy(pos, mass, eps)
    return ke + ext_pe + self_pe


# --- bound mass / Lagrange radii --------------------------------------------

def _shrinking_sphere_com(pos, mass, shrink=0.9, min_frac=0.1, min_n=20):
    """Density-center COM position via a shrinking-sphere iteration."""
    idx = np.arange(pos.shape[0])
    com = np.average(pos, axis=0, weights=mass)
    r_cut = np.max(np.linalg.norm(pos - com, axis=1))
    n_min = max(min_n, int(min_frac * pos.shape[0]))
    while idx.size > n_min:
        r = np.linalg.norm(pos[idx] - com, axis=1)
        r_cut *= shrink
        keep = idx[r < r_cut]
        if keep.size < n_min:
            break
        idx = keep
        com = np.average(pos[idx], axis=0, weights=mass[idx])
    return com, idx


def bound_analysis(snap, r_s, n_iter=5):
    """Return (bound_fraction, lagrange_radii dict, com_pos, com_vel).

    A particle is bound if its specific KE relative to the satellite COM is
    below the depth of the satellite's own Plummer potential (built from the
    currently-bound mass) at that particle's radius. Iterated a few times to
    converge the bound set.
    """
    pos, vel, mass = snap["pos"], snap["vel"], snap["mass"]
    com_pos, core = _shrinking_sphere_com(pos, mass)
    com_vel = np.average(vel[core], axis=0, weights=mass[core])

    rel_pos = pos - com_pos
    r = np.linalg.norm(rel_pos, axis=1)

    bound = np.ones(pos.shape[0], dtype=bool)
    for _ in range(n_iter):
        com_vel = np.average(vel[bound], axis=0, weights=mass[bound])
        M_b = np.sum(mass[bound])
        v_rel2 = np.sum((vel - com_vel) ** 2, axis=1)
        phi = G_KPC_KMS2_MSUN * M_b / np.sqrt(r ** 2 + r_s ** 2)  # |Phi_sat(r)|
        new_bound = 0.5 * v_rel2 < phi
        if np.array_equal(new_bound, bound):
            bound = new_bound
            break
        bound = new_bound

    bound_fraction = np.sum(mass[bound]) / np.sum(mass)

    lagrange = {}
    if bound.any():
        rb = r[bound]
        mb = mass[bound]
        order = np.argsort(rb)
        cum = np.cumsum(mb[order]) / np.sum(mb)
        for frac in (0.1, 0.5, 0.9):
            j = np.searchsorted(cum, frac)
            j = min(j, rb.size - 1)
            lagrange[frac] = rb[order][j]
    else:
        for frac in (0.1, 0.5, 0.9):
            lagrange[frac] = np.nan

    return bound_fraction, lagrange, com_pos, com_vel


# --- plots ------------------------------------------------------------------

def plot_energy(out_dir, run_name, cfg):
    snaps = list_snapshots(out_dir, run_name)
    host = build_host_potential(cfg)
    eps = float(cfg["satellite"].get("eps") or
                cfg["satellite"]["r_s"] / np.sqrt(cfg["satellite"]["N"]))

    ts, es = [], []
    for p in snaps:
        s = read_snapshot(p)
        ts.append(s["t"])
        es.append(total_energy(s, host, eps))
    ts, es = np.array(ts), np.array(es)
    frac = (es - es[0]) / np.abs(es[0])

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(ts, frac)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$(E - E_0)/|E_0|$")
    ax.set_title(f"Energy conservation ({run_name})\nmax |frac err| = "
                 f"{np.max(np.abs(frac)):.2e}")
    path = f"{run_dir(out_dir, run_name)}/diag_energy.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, float(np.max(np.abs(frac)))


def plot_bound_and_lagrange(out_dir, run_name, cfg):
    snaps = list_snapshots(out_dir, run_name)
    r_s = float(cfg["satellite"]["r_s"])

    ts, fbound = [], []
    lag = {0.1: [], 0.5: [], 0.9: []}
    for p in snaps:
        s = read_snapshot(p)
        bf, lr, _, _ = bound_analysis(s, r_s)
        ts.append(s["t"])
        fbound.append(bf)
        for f in lag:
            lag[f].append(lr[f])
    ts = np.array(ts)

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True,
                             layout="constrained")
    axes[0].plot(ts, fbound)
    axes[0].set_ylabel("bound mass fraction")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Disruption diagnostics ({run_name})")

    for f, style in zip((0.1, 0.5, 0.9), ("-", "--", ":")):
        axes[1].plot(ts, lag[f], style, label=f"{int(f*100)}%")
    axes[1].set_xlabel("t [Myr]")
    axes[1].set_ylabel("Lagrange radius [kpc]")
    axes[1].legend(title="enclosed bound mass")

    path = f"{run_dir(out_dir, run_name)}/diag_bound_lagrange.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_scatters(out_dir, run_name, n_panels=4):
    snaps = list_snapshots(out_dir, run_name)
    if not snaps:
        return None
    idxs = np.linspace(0, len(snaps) - 1, min(n_panels, len(snaps))).astype(int)

    fig, axes = plt.subplots(2, len(idxs), figsize=(3.2 * len(idxs), 6.4),
                             layout="constrained", squeeze=False)
    for col, i in enumerate(idxs):
        s = read_snapshot(snaps[i])
        x, y, z = s["pos"][:, 0], s["pos"][:, 1], s["pos"][:, 2]
        axes[0][col].scatter(x, y, s=1, alpha=0.4)
        axes[0][col].set_title(f"t = {s['t']:.0f} Myr")
        axes[0][col].set_xlabel("x [kpc]"); axes[0][col].set_ylabel("y [kpc]")
        axes[0][col].set_aspect("equal")
        axes[1][col].scatter(x, z, s=1, alpha=0.4)
        axes[1][col].set_xlabel("x [kpc]"); axes[1][col].set_ylabel("z [kpc]")
        axes[1][col].set_aspect("equal")

    path = f"{run_dir(out_dir, run_name)}/diag_scatter.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_all(out_dir, run_name, cfg):
    """Generate all diagnostic plots for a run into its run directory."""
    epath, max_err = plot_energy(out_dir, run_name, cfg)
    bpath = plot_bound_and_lagrange(out_dir, run_name, cfg)
    spath = plot_scatters(out_dir, run_name)
    print(f"[diag] max |fractional energy error| = {max_err:.3e}")
    return {"energy": epath, "bound_lagrange": bpath, "scatter": spath,
            "max_energy_error": max_err}
