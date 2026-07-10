"""The Phase A -> B concatenation preserves the host state (no energy discontinuity).

The perturber legitimately adds its own energy, but the host particles handed from Phase A
must be byte-identical in Phase B, so the host subset's total energy is continuous.
"""

import numpy as np

import nbody
from perturber import place_perturber, build_perturber


def test_host_subset_energy_continuous():
    G = 1.0
    N_host, M_host, a_host = 300, 1.0, 1.0
    rng = np.random.default_rng(3)

    pos, vel = nbody.generate_plummer(N_host, M_host, a_host, rng, G=G)
    mass = np.full(N_host, M_host / N_host)
    soft = np.full(N_host, 0.1)
    flag = np.zeros(N_host, dtype=bool)

    # Short relax to obtain a realistic Phase A final state.
    pos, vel, _ = nbody.run_integration(
        pos, vel, mass, soft, flag, dt_int=0.02, n_steps=25, G=G,
        method="direct", parallel=False)

    E_host_A, _, _ = nbody.total_energy(pos, vel, mass, soft, G, method="direct",
                                        parallel=False)

    # Build Phase B by concatenating a point-mass perturber.
    com_pos, com_vel, _ = place_perturber(pos, vel, mass, M_pert=0.2, r_start=2.0,
                                          r_peri=0.7, G=G)
    p_pos, p_vel, p_mass, p_soft, p_flag = build_perturber(
        com_pos, com_vel, perturber_type="point_mass", M_pert=0.2, eps_pert=0.1, G=G)

    all_pos = np.concatenate([pos, p_pos])
    all_vel = np.concatenate([vel, p_vel])
    all_mass = np.concatenate([mass, p_mass])
    all_soft = np.concatenate([soft, p_soft])

    # Host particles must be untouched by the handoff.
    assert np.array_equal(all_pos[:N_host], pos)
    assert np.array_equal(all_vel[:N_host], vel)

    # Host-only energy recomputed from the Phase B arrays is identical.
    E_host_B0, _, _ = nbody.total_energy(all_pos[:N_host], all_vel[:N_host],
                                         all_mass[:N_host], all_soft[:N_host], G,
                                         method="direct", parallel=False)
    assert abs(E_host_B0 - E_host_A) < 1e-10 * abs(E_host_A)
