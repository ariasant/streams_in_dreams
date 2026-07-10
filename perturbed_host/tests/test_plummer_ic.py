"""The refactored Plummer sampler reproduces the analytic density and dispersion."""

import numpy as np

import nbody
from diagnostics import radial_density, radial_dispersion


def test_plummer_density_matches_analytic():
    rng = np.random.default_rng(0)
    N, M, a, G = 100_000, 1.0, 1.0, 1.0
    pos, _ = nbody.generate_plummer(N, M, a, rng, G=G)
    r = np.linalg.norm(pos, axis=1)
    mass = np.full(N, M / N)

    rc, dens = radial_density(r, mass, [0.3, 3.0], bins=25, smooth=False)
    analytic = nbody.plum_rho(rc, M, a)
    sel = (rc > 0.4) & (rc < 2.5)
    rel = np.abs(dens[sel] - analytic[sel]) / analytic[sel]
    assert np.median(rel) < 0.08


def test_plummer_dispersion_matches_analytic():
    rng = np.random.default_rng(1)
    N, M, a, G = 100_000, 1.0, 1.0, 1.0
    pos, vel = nbody.generate_plummer(N, M, a, rng, G=G)
    r = np.linalg.norm(pos, axis=1)

    rc, sig = radial_dispersion(r, vel, [0.3, 3.0], bins=15)
    analytic = nbody.plum_sig(rc, M, a, G)
    sel = np.isfinite(sig) & (rc > 0.4) & (rc < 2.5)
    rel = np.abs(sig[sel] - analytic[sel]) / analytic[sel]
    assert np.median(rel) < 0.06
