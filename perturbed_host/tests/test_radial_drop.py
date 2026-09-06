"""The zero-velocity radial drop reaches the host at the estimated free-fall time.

Place a point-mass perturber at rest at separation r_start around a single-particle host and
integrate the resulting two-body system directly; the closest approach should occur near the
analytic two-body free-fall time, since the relative orbit is a clean radial (zero-angular-
momentum) Kepler infall.
"""

import numpy as np

import nbody
from perturber import place_perturber


def test_two_body_radial_freefall():
    G = 1.0
    M_host, M_pert = 1.0, 0.1
    r_start = 3.0
    eps = 0.05  # softening, needed since a radial orbit passes through zero separation

    host_pos = np.zeros((1, 3))
    host_vel = np.zeros((1, 3))
    host_mass = np.array([M_host])

    com_pos, com_vel, info = place_perturber(
        host_pos, host_vel, host_mass, M_pert=M_pert, r_start=r_start, G=G)

    # Placed at r_start on the +x axis, at rest relative to the host.
    assert abs(info["r_start"] - r_start) < 1e-9
    assert np.allclose(com_vel, 0.0)
    assert np.allclose(com_pos, [r_start, 0.0, 0.0])

    # Two-body state: free host + perturber, softened -> radial infall through closest approach.
    pos = np.vstack([host_pos, com_pos])
    vel = np.vstack([host_vel, com_vel])
    mass = np.array([M_host, M_pert])
    soft = np.array([eps, eps])

    t_ff_est = info["t_freefall_estimate"]
    dt = 0.001
    n = int((t_ff_est * 1.3) / dt)
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

    assert seps[i_min] < 2 * eps
    assert abs(times[i_min] - t_ff_est) / t_ff_est < 0.05
