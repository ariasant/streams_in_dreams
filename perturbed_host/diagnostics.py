"""Diagnostics for a perturbed-host run.

Extends the energy and density checks in the repo's ``nbody_sim.py`` to span both phases and
the perturbation event:

1. energy conservation vs. time (flagging the Phase A/B seam);
2. host density + velocity-dispersion profiles at a few times through the encounter;
3. perturber orbit -- separation from the host COM, with the realized pericenter passage
   compared to the target ``r_peri`` and the two-body time estimate ``T/2``;
4. host Lagrange radii (10/50/90% mass) vs. time.

The radial-density binning mirrors ``DREAMS_utils.return_density`` (log-spaced shells,
mass / shell-volume) but is reimplemented here so the pipeline stays self-contained and does
not pull in that module's ``pynbody``/``astropy`` dependencies.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from nbody import plum_rho, plum_sig


# --------------------------------------------------------------------------------------
# Radial profile helpers
# --------------------------------------------------------------------------------------
def radial_density(r, weights, rangevals, bins=60, smooth=True):
    """Log-binned density: mass per spherical shell / shell volume (mirrors return_density)."""
    rbins = np.logspace(np.log10(rangevals[0]), np.log10(rangevals[1]), bins)
    V = 4.0 / 3.0 * np.pi * (rbins[1:] ** 3 - rbins[:-1] ** 3)
    M, _ = np.histogram(r, bins=rbins, weights=weights)
    density = M / V
    rcentre = 0.5 * (rbins[:-1] + rbins[1:])
    if smooth:
        density = gaussian_filter1d(density, 2.0)
    return rcentre, density


def radial_dispersion(r, vel, rangevals, bins=30):
    """1-D velocity dispersion in log-spaced radial bins (mean-subtracted, isotropic)."""
    rbins = np.logspace(np.log10(rangevals[0]), np.log10(rangevals[1]), bins)
    idx = np.digitize(r, rbins)
    rcentre = 0.5 * (rbins[:-1] + rbins[1:])
    sigma = np.full(len(rcentre), np.nan)
    for i in range(1, len(rbins)):
        sel = idx == i
        if sel.sum() > 3:
            v = vel[sel]
            v = v - v.mean(axis=0)
            sigma[i - 1] = np.sqrt(np.mean(np.sum(v ** 2, axis=1)) / 3.0)
    return rcentre, sigma


def _com(pos, mass):
    return np.sum(pos * mass[:, None], axis=0) / mass.sum()


# --------------------------------------------------------------------------------------
# Individual diagnostics
# --------------------------------------------------------------------------------------
def energy_conservation(captures, t_relax, out_path):
    """Plot total energy vs. time; fractional drift is measured within each phase.

    The step at ``t_relax`` is the perturber injection (energy legitimately added), not a
    numerical discontinuity -- it is annotated, and conservation is judged by the flatness of
    each phase separately.
    """
    # Only captures where the energy was actually evaluated (others store nan).
    have = [c for c in captures if np.isfinite(c["energy"])]
    t = np.array([c["t"] for c in have])
    E = np.array([c["energy"] for c in have])
    a_mask = t < t_relax
    b_mask = ~a_mask

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    summary = {}
    for mask, label, color in [(a_mask, "Phase A (host)", "C0"),
                               (b_mask, "Phase B (perturbed)", "C1")]:
        if mask.sum() >= 2:
            E0 = E[mask][0]
            frac = (E[mask] - E0) / abs(E0)
            ax.plot(t[mask], frac, ".-", color=color, label=label)
            summary[label] = float(np.max(np.abs(frac)))
    ax.axvline(t_relax, ls="--", color="k", lw=1, label="perturber injection")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$(E - E_0)/|E_0|$ within phase")
    ax.legend()
    ax.set_title("Energy conservation")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return summary


def perturber_orbit(captures_B, out_path, target_r_peri, target_t_peri, t_relax):
    """Separation between perturber COM and host COM vs. time; report realized pericenter."""
    times, seps = [], []
    for c in captures_B:
        flag = c["is_perturber"]
        if not flag.any():
            continue
        host_com = _com(c["pos"][~flag], c["mass"][~flag])
        pert_com = _com(c["pos"][flag], c["mass"][flag])
        times.append(c["t"])
        seps.append(np.linalg.norm(pert_com - host_com))
    times = np.array(times)
    seps = np.array(seps)

    i_min = int(np.argmin(seps))
    realized_r_peri = float(seps[i_min])
    realized_t_peri = float(times[i_min] - t_relax)  # measured from start of Phase B

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(times - t_relax, seps, "-", color="C3")
    ax.axhline(target_r_peri, ls="--", color="k", lw=1, label=f"target r_peri={target_r_peri:g}")
    ax.axvline(target_t_peri, ls=":", color="k", lw=1,
               label=f"two-body est. t_peri={target_t_peri:.2f}")
    ax.plot(realized_t_peri, realized_r_peri, "o", color="C3",
            label=f"realized ({realized_t_peri:.2f}, {realized_r_peri:.2f})")
    ax.set_xlabel("time since perturber injection")
    ax.set_ylabel("perturber-host separation")
    ax.legend()
    ax.set_title("Perturber orbit")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return dict(realized_r_peri=realized_r_peri, realized_t_peri=realized_t_peri,
                target_r_peri=target_r_peri, target_t_peri=target_t_peri)


def host_profiles(captures, cfg, out_path, t_peri_estimate, n_times=3):
    """Host density and dispersion profiles at a few times spanning the encounter."""
    # choose times: first, near the estimated pericenter, last
    times = np.array([c["t"] for c in captures])
    t_peri_abs = cfg["t_relax"] + t_peri_estimate
    pick = sorted({0, int(np.argmin(np.abs(times - t_peri_abs))), len(captures) - 1})
    rangevals = [0.2 * cfg["a_host"], 20.0 * cfg["a_host"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(pick)))
    for color, k in zip(colors, pick):
        c = captures[k]
        flag = c["is_perturber"]
        pos = c["pos"][~flag]
        vel = c["vel"][~flag]
        mass = c["mass"][~flag]
        com = _com(pos, mass)
        r = np.linalg.norm(pos - com, axis=1)
        rc, dens = radial_density(r, mass, rangevals)
        axes[0].plot(rc, dens, color=color, label=f"t={c['t']:.1f}")
        rc2, sig = radial_dispersion(r, vel, rangevals)
        axes[1].plot(rc2, sig, color=color, label=f"t={c['t']:.1f}")

    r_an = np.logspace(np.log10(rangevals[0]), np.log10(rangevals[1]), 100)
    axes[0].plot(r_an, plum_rho(r_an, cfg["M_host"], cfg["a_host"]), "k--", label="analytic (t=0)")
    axes[1].plot(r_an, plum_sig(r_an, cfg["M_host"], cfg["a_host"], cfg["G"]), "k--",
                 label="analytic (t=0)")
    for ax, ylab in [(axes[0], "density"), (axes[1], r"$\sigma_{1D}$")]:
        ax.set_xscale("log")
        ax.set_xlabel("r (from host COM)")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_title("Host density profile")
    axes[1].set_title("Host velocity dispersion")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def lagrange_radii(captures, out_path, fractions=(0.1, 0.5, 0.9)):
    """Host Lagrange radii (mass fractions) vs. time, excluding the perturber."""
    times, radii = [], []
    for c in captures:
        flag = c["is_perturber"]
        pos = c["pos"][~flag]
        mass = c["mass"][~flag]
        com = _com(pos, mass)
        r = np.linalg.norm(pos - com, axis=1)
        order = np.argsort(r)
        cum = np.cumsum(mass[order]) / mass.sum()
        radii.append([r[order][np.searchsorted(cum, f)] for f in fractions])
        times.append(c["t"])
    times = np.array(times)
    radii = np.array(radii)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for j, f in enumerate(fractions):
        ax.plot(times, radii[:, j], label=f"{int(f*100)}% mass")
    ax.set_xlabel("time")
    ax.set_ylabel("Lagrange radius (host)")
    ax.legend()
    ax.set_title("Host Lagrange radii")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def make_all(results, out_dir):
    """Run every diagnostic; return a printable summary dict."""
    cfg = results["config"]
    tag = cfg["run_name"]
    t_peri_est = results["placement"]["t_peri_estimate"]
    energy = energy_conservation(
        results["captures"], results["t_relax"],
        os.path.join(out_dir, f"{tag}_energy.pdf"))
    orbit = perturber_orbit(
        results["captures_B"], os.path.join(out_dir, f"{tag}_orbit.pdf"),
        cfg["r_peri"], t_peri_est, results["t_relax"])
    host_profiles(results["captures"], cfg, os.path.join(out_dir, f"{tag}_profiles.pdf"),
                  t_peri_est)
    lagrange_radii(results["captures"], os.path.join(out_dir, f"{tag}_lagrange.pdf"))
    return dict(energy=energy, orbit=orbit)
