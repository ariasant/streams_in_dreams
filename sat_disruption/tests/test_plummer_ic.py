"""Check the sampled Plummer ICs reproduce the analytic profiles.

Pure-NumPy: does not require gala.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from initial_conditions import (  # noqa: E402
    G_KPC_KMS2_MSUN,
    plummer_density,
    plummer_sigma,
    sample_plummer_sphere,
)


M_SAT = 1.0e8
R_S = 0.3
N = 40000


@pytest.fixture(scope="module")
def sphere():
    rng = np.random.default_rng(12345)
    pos, vel, mass = sample_plummer_sphere(N, M_SAT, R_S, rng)
    return pos, vel, mass


def test_com_is_zero(sphere):
    pos, vel, mass = sphere
    # After recentering, mass-weighted COM position/velocity must vanish.
    assert np.allclose(np.average(pos, axis=0, weights=mass), 0.0, atol=1e-9)
    assert np.allclose(np.average(vel, axis=0, weights=mass), 0.0, atol=1e-9)


def test_total_mass(sphere):
    _, _, mass = sphere
    assert np.isclose(np.sum(mass), M_SAT)
    assert np.allclose(mass, M_SAT / N)


def test_density_profile(sphere):
    pos, _, mass = sphere
    r = np.linalg.norm(pos, axis=1)

    # Use log-spaced shells over the well-sampled range (avoid the sparse tails).
    edges = np.logspace(np.log10(0.1 * R_S), np.log10(5 * R_S), 12)
    counts, _ = np.histogram(r, bins=edges)
    shell_mass = counts * (M_SAT / N)
    vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho_num = shell_mass / vol
    r_mid = np.sqrt(edges[1:] * edges[:-1])
    rho_ana = plummer_density(r_mid, M_SAT, R_S)

    # Compare in log space; require reasonable agreement where counts are decent.
    good = counts > 50
    assert good.sum() >= 6
    log_ratio = np.log10(rho_num[good] / rho_ana[good])
    assert np.median(np.abs(log_ratio)) < 0.1          # ~25% typical
    assert np.max(np.abs(log_ratio)) < 0.25            # ~<80% worst bin


def test_velocity_dispersion(sphere):
    pos, vel, _ = sphere
    r = np.linalg.norm(pos, axis=1)
    speed2 = np.sum(vel * vel, axis=1)

    edges = np.logspace(np.log10(0.2 * R_S), np.log10(3 * R_S), 8)
    idx = np.digitize(r, edges)
    for b in range(1, len(edges)):
        sel = idx == b
        if sel.sum() < 200:
            continue
        # 3D dispersion; sigma_3d^2 = 3 * sigma_1d^2.
        sigma3d_num = np.sqrt(np.mean(speed2[sel]))
        r_mid = np.sqrt(edges[b - 1] * edges[b])
        sigma3d_ana = np.sqrt(3.0) * plummer_sigma(r_mid, M_SAT, R_S)
        assert abs(sigma3d_num / sigma3d_ana - 1.0) < 0.15


def test_reproducible():
    p1, v1, _ = sample_plummer_sphere(1000, M_SAT, R_S, np.random.default_rng(7))
    p2, v2, _ = sample_plummer_sphere(1000, M_SAT, R_S, np.random.default_rng(7))
    assert np.array_equal(p1, p2)
    assert np.array_equal(v1, v2)


def test_virial_energy_scale(sphere):
    # Sanity: mean kinetic energy per unit mass should be of order G M / r_s.
    _, vel, _ = sphere
    ke_specific = 0.5 * np.mean(np.sum(vel * vel, axis=1))
    scale = G_KPC_KMS2_MSUN * M_SAT / R_S
    assert 0.05 < ke_specific / scale < 0.5
