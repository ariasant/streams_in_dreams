"""The apocenter drop reaches the requested pericenter at the estimated time (~T/2).

Place a point-mass perturber (dropped from apocenter r_start) around a single-particle host
and integrate the resulting two-body system directly; the realized pericenter distance should
match r_peri and its time should match the derived T/2 estimate, since the relative orbit is a
clean Kepler ellipse.
"""

import numpy as np

import nbody
from perturber import place_perturber


def test_two_body_pericenter():
    G = 1.0
    M_host, M_pert = 1.0, 0.1
    r_start, r_peri = 3.0, 0.7

    host_pos = np.zeros((1, 3))
    host_vel = np.zeros((1, 3))
    host_mass = np.array([M_host])

    com_pos, com_vel, info = place_perturber(
        host_pos, host_vel, host_mass,
        M_pert=M_pert, r_start=r_start, r_peri=r_peri, G=G)

    # Start radius is the apocenter; eccentricity is derived from (r_start, r_peri).
    assert abs(info["r_start"] - r_start) < 1e-9
    assert abs(info["ecc"] - (r_start - r_peri) / (r_start + r_peri)) < 1e-9

    # Two-body state: free host + perturber, no softening -> clean Kepler.
    pos = np.vstack([host_pos, com_pos])
    vel = np.vstack([host_vel, com_vel])
    mass = np.array([M_host, M_pert])
    soft = np.zeros(2)

    t_peri_est = info["t_peri_estimate"]
    dt = 0.002
    n = int((t_peri_est * 1.3) / dt)
    accel = nbody.compute_accel(pos, mass, soft, G, method="direct", parallel=False)
    seps, times = [], []
    for i in range(n + 1):
        seps.append(np.linalg.norm(pos[1] - pos[0]))
        times.append(i * dt)
        pos, vel, accel = nbody.leapfrog_step(dt, pos, vel, accel, mass, soft, G,
                                              method="direct", parallel=False)
    seps = np.array(seps)
    times = np.array(times)
    i_min = int(np.argmin(seps))

    assert abs(seps[i_min] - r_peri) / r_peri < 0.02
    assert abs(times[i_min] - t_peri_est) / t_peri_est < 0.02
