"""Two-body Kepler sanity test for the DirectNBody + chunked-loop machinery.

Two equal point masses (no external potential) on a bound eccentric orbit.
Integrating through the same chunked ``integrate_orbit`` pattern the pipeline
uses, we check the relative orbit reproduces the analytic Kepler ellipse:
apocenter/pericenter separations, conserved energy, and the semi-major axis
recovered from vis-viva.

Requires gala.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

gala = pytest.importorskip("gala")

import astropy.units as u  # noqa: E402
import gala.dynamics as gd  # noqa: E402
import gala.potential as gp  # noqa: E402
from gala.dynamics import Orbit  # noqa: E402
from gala.dynamics.nbody import DirectNBody  # noqa: E402
from gala.units import galactic  # noqa: E402

from initial_conditions import G_KPC_KMS2_MSUN  # noqa: E402


def _final_state(result):
    """Version-robust final PhaseSpacePosition (see nbody_runner._final_state)."""
    return result[-1] if isinstance(result, Orbit) else result

# Orbit setup (galactic units): two m = 1e9 Msun masses, a = 1 kpc, e = 0.5.
M_PART = 1.0e9
A = 1.0          # kpc, semi-major axis of the relative orbit
E = 0.5
B_SOFT = 1.0e-3  # kpc, softening (<< separation, so ~point-mass Kepler)

GM = G_KPC_KMS2_MSUN * (2.0 * M_PART)   # relative orbit uses total mass
R_APO = A * (1.0 + E)
R_PERI = A * (1.0 - E)
V_APO = np.sqrt(GM / A * (1.0 - E) / (1.0 + E))     # relative speed at apo
# analytic period, kpc/(km/s) -> Myr conversion via astropy
T_KPC_PER_KMS = 2.0 * np.pi * np.sqrt(A ** 3 / GM)
T_PERIOD = (T_KPC_PER_KMS * u.kpc / (u.km / u.s)).to_value(u.Myr)


def _make_w0():
    # Symmetric about the COM at the origin; start at apocenter.
    pos = np.array([[+R_APO / 2, 0.0, 0.0],
                    [-R_APO / 2, 0.0, 0.0]])          # kpc
    vel = np.array([[0.0, +V_APO / 2, 0.0],
                    [0.0, -V_APO / 2, 0.0]])          # km/s
    return gd.PhaseSpacePosition(pos=pos.T * u.kpc, vel=vel.T * u.km / u.s)


def _chunked_run(w0, particle_pot, dt, dt_int, n_snap):
    """Mirror nbody_runner's chunked loop; return list of (t, pos, vel)."""
    out = []
    w = w0
    out.append((0.0,
                w.xyz.to_value(u.kpc).T,
                w.v_xyz.to_value(u.km / u.s).T))
    for k in range(1, n_snap + 1):
        nbody = DirectNBody(w, particle_pot, external_potential=None,
                            units=galactic, save_all=False)
        result = nbody.integrate_orbit(dt=dt_int * u.Myr, t1=0.0 * u.Myr,
                                       t2=dt * u.Myr)
        w = _final_state(result)
        out.append((k * dt,
                    w.xyz.to_value(u.kpc).T,
                    w.v_xyz.to_value(u.km / u.s).T))
    return out


def _two_body_energy(pos, vel):
    """Specific energy of the relative orbit [ (km/s)^2 ]."""
    rel_r = np.linalg.norm(pos[0] - pos[1])
    rel_v2 = np.sum((vel[0] - vel[1]) ** 2)
    return 0.5 * rel_v2 - GM / np.sqrt(rel_r ** 2 + B_SOFT ** 2)


@pytest.fixture(scope="module")
def trajectory():
    w0 = _make_w0()
    particle_pot = [
        gp.PlummerPotential(m=M_PART * u.Msun, b=B_SOFT * u.kpc, units=galactic)
        for _ in range(2)
    ]
    dt = T_PERIOD / 60.0        # snapshot cadence: 60 snapshots per period
    dt_int = 0.01               # fixed small internal step (Myr)
    n_snap = int(round(2 * T_PERIOD / dt))   # two full periods
    return _chunked_run(w0, particle_pot, dt, dt_int, n_snap)


def test_apo_peri_separation(trajectory):
    seps = np.array([np.linalg.norm(pos[0] - pos[1]) for _, pos, _ in trajectory])
    assert abs(seps.max() / R_APO - 1.0) < 0.03
    assert abs(seps.min() / R_PERI - 1.0) < 0.05


def test_energy_conserved(trajectory):
    es = np.array([_two_body_energy(pos, vel) for _, pos, vel in trajectory])
    frac = np.abs((es - es[0]) / es[0])
    assert frac.max() < 1e-3


def test_semimajor_axis_from_visviva(trajectory):
    # a = -GM / (2 E) from the conserved specific energy.
    _, pos0, vel0 = trajectory[0]
    e_spec = _two_body_energy(pos0, vel0)
    a_meas = -GM / (2.0 * e_spec)
    assert abs(a_meas / A - 1.0) < 0.02


def test_period(trajectory):
    # Recover the period from successive pericenter passages.
    ts = np.array([t for t, _, _ in trajectory])
    seps = np.array([np.linalg.norm(pos[0] - pos[1]) for _, pos, _ in trajectory])
    # local minima of separation
    mins = [i for i in range(1, len(seps) - 1)
            if seps[i] < seps[i - 1] and seps[i] < seps[i + 1]]
    assert len(mins) >= 2
    period_meas = ts[mins[1]] - ts[mins[0]]
    assert abs(period_meas / T_PERIOD - 1.0) < 0.05
