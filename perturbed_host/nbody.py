"""Globals-free Plummer sampling and leapfrog N-body integration.

Refactored from the repo's ``nbody_sim.py`` (which runs as a top-level script with
module-level globals like ``M``, ``a``, ``G``, ``theta``). The physics is unchanged:
inverse-transform Plummer positions with Aarseth-Henon-Wielen velocity sampling, and a
kick-drift-kick leapfrog whose forces come from ``pytreegrav``. Everything here is a plain
function taking explicit parameters, so Phase A (host only) and Phase B (host + perturber)
can drive the *same* integrator with different particle sets.

All quantities are in system units; ``G`` is passed explicitly (default 1.0).
"""

from __future__ import annotations
from tqdm import tqdm

import numpy as np
from pytreegrav import Accel, Potential


# --------------------------------------------------------------------------------------
# Initial conditions
# --------------------------------------------------------------------------------------
def generate_plummer(num_particles, total_mass, scale_radius, rng, G=1.0):
    """Sample an isotropic Plummer sphere (positions and velocities).

    Adapted from ``nbody_sim.generate_plummer`` (``option='original'``): radial positions by
    inverse-transform sampling and speeds by the Aarseth, Henon & Wielen rejection method.

    Parameters
    ----------
    num_particles : int
    total_mass, scale_radius : float
    rng : numpy.random.Generator
        Explicit generator for reproducibility (replaces the old global ``np.random``).
    G : float, optional

    Returns
    -------
    positions, velocities : ndarray, shape (num_particles, 3)
    """
    n = int(num_particles)

    # Radial positions via inverse of the Plummer mass profile.
    u = rng.random(n)
    r = scale_radius * (u ** (-2.0 / 3.0) - 1.0) ** (-0.5)

    # Speeds via the Aarseth-Henon-Wielen rejection sampler (q = v / v_esc).
    v_esc = np.sqrt(2.0 * G * total_mass / scale_radius) * (
        1.0 + r ** 2 / scale_radius ** 2
    ) ** (-0.25)
    q = np.empty(n)
    remaining = np.ones(n, dtype=bool)
    while remaining.any():
        m = int(remaining.sum())
        x = rng.random(m)
        y = 0.1 * rng.random(m)
        accept = y <= x ** 2 * (1.0 - x ** 2) ** 3.5
        idx = np.flatnonzero(remaining)[accept]
        q[idx] = x[accept]
        remaining[idx] = False
    speed = q * v_esc

    positions = _isotropic_vectors(r, rng)
    velocities = _isotropic_vectors(speed, rng)
    return positions, velocities


def _isotropic_vectors(magnitude, rng):
    """Random isotropic 3-vectors with the given magnitudes."""
    n = len(magnitude)
    phi = 2.0 * np.pi * rng.random(n)
    costheta = 2.0 * rng.random(n) - 1.0
    sintheta = np.sqrt(1.0 - costheta ** 2)
    x = magnitude * sintheta * np.cos(phi)
    y = magnitude * sintheta * np.sin(phi)
    z = magnitude * costheta
    return np.stack([x, y, z], axis=1)


# --------------------------------------------------------------------------------------
# Forces and integration
# --------------------------------------------------------------------------------------
def compute_accel(pos, masses, softening, G, method="direct", theta=0.7, parallel=True):
    """Gravitational acceleration on every particle via ``pytreegrav``.

    ``method='direct'`` is exact O(N^2) brute force (default; best for a few thousand
    particles). ``method='tree'`` is Barnes-Hut with opening angle ``theta``.
    """
    if method == "direct":
        return Accel(pos, masses, softening, G=G, parallel=parallel, method="bruteforce")
    elif method == "tree":
        return Accel(pos, masses, softening, G=G, parallel=parallel, theta=theta, method="tree")
    raise ValueError(f"unknown force method: {method!r}")


def compute_potential(pos, masses, softening, G, method="direct", theta=0.7, parallel=True):
    """Per-particle gravitational potential via ``pytreegrav`` (for energy diagnostics)."""
    if method == "direct":
        return Potential(pos, masses, softening, G=G, parallel=parallel, method="bruteforce")
    elif method == "tree":
        return Potential(pos, masses, softening, G=G, parallel=parallel, theta=theta, method="tree")
    raise ValueError(f"unknown force method: {method!r}")


def leapfrog_step(dt, pos, vel, accel, masses, softening, G, method="direct", theta=0.7,
                  parallel=True):
    """One kick-drift-kick leapfrog step (from ``nbody_sim.frog_step``, globals removed).

    ``accel`` is passed in and returned so the force at the end of a step is reused as the
    first half-kick of the next, keeping the integrator symplectic with one solve per step.
    """
    vel = vel + 0.5 * dt * accel
    pos = pos + dt * vel
    accel = compute_accel(pos, masses, softening, G, method, theta, parallel)
    vel = vel + 0.5 * dt * accel
    return pos, vel, accel


def total_energy(pos, vel, masses, softening, G, method="direct", theta=0.7, parallel=True):
    """Total (kinetic + potential) energy of the system and its components."""
    ke = 0.5 * np.sum(masses * np.sum(vel ** 2, axis=1))
    phi = compute_potential(pos, masses, softening, G, method, theta, parallel)
    pe = 0.5 * np.sum(masses * phi)
    return ke + pe, ke, pe


def run_integration(pos, vel, masses, softening, is_perturber, *, dt_int, n_steps,
                    G, method="direct", theta=0.7, parallel=True, step0=0, t0=0.0,
                    snapshot_steps=frozenset(), capture_steps=frozenset(),
                    energy_steps=frozenset(), writer=None):
    """Advance one phase with fixed internal step ``dt_int``.

    Snapshot/capture scheduling is keyed on the *global* step index ``g = step0 + i`` so a
    two-phase run (Phase A then Phase B) shares one continuous schedule.

    Position captures are cheap and taken at every ``capture_steps`` index (used by the GIF
    and the orbit/profile/Lagrange diagnostics). The total energy requires an O(N^2)
    potential solve, so it is computed only on the smaller ``energy_steps`` subset; other
    captures store ``energy = nan``.

    Parameters
    ----------
    pos, vel, masses, softening : ndarray
        Phase-space state and per-particle parameters.
    is_perturber : ndarray of bool
        Marks perturber particles (vs. host); recorded in snapshots/captures.
    dt_int : float
        Fixed internal integrator step (independent of the science cadence ``dt``).
    n_steps : int
        Number of leapfrog steps to advance in this phase.
    step0, t0 : int, float
        Global step index and absolute time at the first recorded state.
    snapshot_steps, capture_steps, energy_steps : set of int
        Global step indices at which to write an HDF5 science snapshot / capture an
        in-memory fine-cadence frame / additionally evaluate the total energy.
    writer : io_utils.SnapshotWriter or None
        Receives ``.write(...)`` calls for science snapshots.

    Returns
    -------
    pos, vel : ndarray
        Final phase-space state (for the Phase A -> B handoff).
    captures : list of dict
        Fine-cadence frames with keys ``t, pos, vel, mass, is_perturber, energy, ke, pe``.
    """
    accel = compute_accel(pos, masses, softening, G, method, theta, parallel)
    captures = []

    for i in tqdm(range(n_steps + 1)):
        g = step0 + i
        t = t0 + i * dt_int

        if g in snapshot_steps and writer is not None:
            writer.write(g, t, pos, vel, masses, is_perturber)
        if g in capture_steps:
            if g in energy_steps:
                e, ke, pe = total_energy(pos, vel, masses, softening, G, method, theta,
                                         parallel)
            else:
                e = ke = pe = np.nan
            captures.append(dict(t=t, pos=pos.copy(), vel=vel.copy(), mass=masses.copy(),
                                 is_perturber=is_perturber.copy(), energy=e, ke=ke, pe=pe))

        if i < n_steps:
            pos, vel, accel = leapfrog_step(dt_int, pos, vel, accel, masses, softening,
                                            G, method, theta, parallel)

    return pos, vel, captures


# --------------------------------------------------------------------------------------
# Analytic Plummer profiles (for diagnostics; from nbody_sim.py)
# --------------------------------------------------------------------------------------
def plum_rho(r, M, a):
    """Analytic Plummer density."""
    return (3.0 * M / (4.0 * np.pi * a ** 3)) / (1.0 + (r / a) ** 2) ** 2.5


def plum_sig(r, M, a, G=1.0):
    """Analytic Plummer 1-D velocity dispersion."""
    return np.sqrt(G * M / 6.0 / np.sqrt(r * r + a * a))
