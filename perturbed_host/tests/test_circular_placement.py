"""A circular=True placement stays at r_start and completes one period on schedule.

Place a point-mass perturber on a circular orbit at r_start around a single-particle host and
integrate the resulting two-body system directly; the separation should stay ~constant at
r_start throughout, and the perturber should return close to its starting position after one
orbital period.
"""

import numpy as np

import nbody
from perturber import place_perturber


def test_two_body_circular_orbit():
    G = 1.0
    M_host, M_pert = 1.0, 0.1
    r_start = 3.0

    host_pos = np.zeros((1, 3))
    host_vel = np.zeros((1, 3))
    host_mass = np.array([M_host])

    com_pos, com_vel, info = place_perturber(
        host_pos, host_vel, host_mass, M_pert=M_pert, r_start=r_start, circular=True, G=G)

    M_tot = M_host + M_pert
    v_circ_expected = np.sqrt(G * M_tot / r_start)
    assert info["circular"] is True
    assert abs(info["v_circ"] - v_circ_expected) < 1e-9
    assert np.allclose(com_pos, [r_start, 0.0, 0.0])
    assert np.allclose(com_vel, [0.0, v_circ_expected, 0.0])

    # Two-body state: free host + perturber, no softening -> clean circular Kepler orbit.
    pos = np.vstack([host_pos, com_pos])
    vel = np.vstack([host_vel, com_vel])
    mass = np.array([M_host, M_pert])
    soft = np.zeros(2)

    period = info["period"]
    dt = 0.005
    n = int(period / dt)
    accel = nbody.compute_accel(pos, mass, soft, G, method="direct", parallel=False)
    seps = []
    for _ in range(n + 1):
        seps.append(np.linalg.norm(pos[1] - pos[0]))
        pos, vel, accel = nbody.leapfrog_step(dt, pos, vel, accel, mass, soft, G,
                                              method="direct", parallel=False)
    seps = np.array(seps)

    # Separation stays close to r_start throughout the orbit.
    assert np.max(np.abs(seps - r_start)) / r_start < 0.01

    # After one period, the perturber is back near its starting relative position.
    rel_pos_final = pos[1] - pos[0]
    assert np.linalg.norm(rel_pos_final - np.array([r_start, 0.0, 0.0])) / r_start < 0.02
