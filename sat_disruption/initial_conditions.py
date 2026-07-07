"""Plummer-sphere initial conditions for the satellite.

The sampling math is adapted (reimplemented independently) from
``nbody_sim.py:generate_plummer`` in the parent directory, but uses the
physically-correct Aarseth, Henon & Wielen (1974) rejection sampler for the
speed distribution rather than the simpler closed-form approximation.

Everything here is plain NumPy; the only ``gala`` dependency is in
``build_satellite_w0``, which wraps the sampled arrays in a
``PhaseSpacePosition``. Positions are returned in kpc, velocities in km/s,
masses in Msun.
"""

import numpy as np

# Gravitational constant in galactic-ish units: kpc * (km/s)^2 / Msun.
# With this G, sqrt(G * M[Msun] / r[kpc]) comes out in km/s.
G_KPC_KMS2_MSUN = 4.300917270e-6


def _sample_radii(n, rng):
    """Draw dimensionless radii (scale radius = 1) from the Plummer profile.

    Inverse-transform sampling of the Plummer cumulative mass profile
    M(<r)/M = r^3 / (r^2 + 1)^{3/2}, i.e. r = (u^{-2/3} - 1)^{-1/2}.
    """
    u = rng.uniform(size=n)
    return (u ** (-2.0 / 3.0) - 1.0) ** (-0.5)


def _sample_speed_fractions(n, rng):
    """Draw q = v / v_esc from the Plummer isotropic DF via rejection sampling.

    The relevant density is g(q) = q^2 (1 - q^2)^{7/2} on q in [0, 1]
    (Aarseth, Henon & Wielen 1974). Its maximum is ~0.092, so 0.1 is a safe
    rejection envelope.
    """
    out = np.empty(n)
    filled = 0
    while filled < n:
        m = n - filled
        # draw a generous batch to cut down on Python-level loop iterations
        q = rng.uniform(size=int(m * 2.5) + 1)
        y = rng.uniform(0.0, 0.1, size=q.size)
        accepted = q[y < q * q * (1.0 - q * q) ** 3.5]
        take = min(accepted.size, m)
        out[filled:filled + take] = accepted[:take]
        filled += take
    return out


def _isotropic_directions(mag, rng):
    """Return (N, 3) vectors with the given magnitudes, isotropic in direction."""
    n = mag.size
    phi = 2.0 * np.pi * rng.uniform(size=n)
    cos_theta = 2.0 * rng.uniform(size=n) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    x = mag * sin_theta * np.cos(phi)
    y = mag * sin_theta * np.sin(phi)
    z = mag * cos_theta
    return np.stack([x, y, z], axis=1)


def sample_plummer_sphere(N, M_sat, r_s, rng, G=G_KPC_KMS2_MSUN):
    """Sample an isotropic Plummer sphere of N equal-mass particles.

    Parameters
    ----------
    N : int
        Number of particles.
    M_sat : float
        Total satellite mass in Msun.
    r_s : float
        Plummer scale radius in kpc.
    rng : numpy.random.Generator
        Random generator (for reproducibility).
    G : float
        Gravitational constant in kpc (km/s)^2 / Msun.

    Returns
    -------
    pos : (N, 3) ndarray, kpc
    vel : (N, 3) ndarray, km/s
    mass : (N,) ndarray, Msun  (all equal to M_sat / N)

    The sphere is recentered to zero mass-weighted COM position and velocity
    in its own frame.
    """
    # Dimensionless radii (scale radius = 1) and speed fractions q = v/v_esc.
    r_nat = _sample_radii(N, rng)
    q = _sample_speed_fractions(N, rng)

    # Dimensionless escape speed for the Plummer model (with G=M=a=1):
    # v_esc(r) = sqrt(2) * (1 + r^2)^{-1/4}.
    v_esc_nat = np.sqrt(2.0) * (1.0 + r_nat ** 2) ** (-0.25)
    v_nat = q * v_esc_nat

    # Physical scales.
    v_scale = np.sqrt(G * M_sat / r_s)  # km/s
    pos = _isotropic_directions(r_nat, rng) * r_s          # kpc
    vel = _isotropic_directions(v_nat, rng) * v_scale      # km/s

    mass = np.full(N, M_sat / N)

    # Recenter so the sphere is at rest in its own frame (mass-weighted COM).
    pos -= np.average(pos, axis=0, weights=mass)
    vel -= np.average(vel, axis=0, weights=mass)

    return pos, vel, mass


def build_satellite_w0(cfg, rng):
    """Build the initial gala PhaseSpacePosition for the satellite.

    Reads the ``satellite`` block of the config, samples a Plummer sphere,
    offsets it to the requested COM position/velocity, and returns the
    packaged phase-space state along with per-particle masses and the
    softening length.

    Returns
    -------
    w0 : gala.dynamics.PhaseSpacePosition
    mass : (N,) ndarray, Msun
    eps : float, kpc
    """
    import astropy.units as u
    import gala.dynamics as gd

    sat = cfg["satellite"]
    N = int(sat["N"])
    M_sat = float(sat["M_sat"])
    r_s = float(sat["r_s"])

    pos, vel, mass = sample_plummer_sphere(N, M_sat, r_s, rng)

    pos = pos + np.asarray(sat["pos"], dtype=float)  # kpc
    vel = vel + np.asarray(sat["vel"], dtype=float)  # km/s

    eps = sat.get("eps")
    if eps is None:
        eps = r_s / np.sqrt(N)
    eps = float(eps)

    # gala PhaseSpacePosition expects pos/vel of shape (3, N).
    w0 = gd.PhaseSpacePosition(
        pos=pos.T * u.kpc,
        vel=vel.T * u.km / u.s,
    )
    return w0, mass, eps


# --- analytic Plummer profiles (used by tests/diagnostics) ------------------

def plummer_density(r, M_sat, r_s):
    """Analytic Plummer mass density rho(r) [Msun / kpc^3]."""
    return (3.0 * M_sat / (4.0 * np.pi * r_s ** 3)) * (1.0 + (r / r_s) ** 2) ** (-2.5)


def plummer_sigma(r, M_sat, r_s, G=G_KPC_KMS2_MSUN):
    """Analytic Plummer 1D velocity dispersion sigma(r) [km/s].

    For the isotropic Plummer model, sigma_1d^2 = G M / (6 sqrt(r^2 + r_s^2)).
    """
    return np.sqrt(G * M_sat / (6.0 * np.sqrt(r ** 2 + r_s ** 2)))
