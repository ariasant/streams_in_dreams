"""Perturber initial conditions: a bound Kepler flyby of the host, dropped from apocenter.

The user sets the starting separation ``r_start`` and the target pericenter ``r_peri``; the
code treats ``r_start`` as the orbit's apocenter (turnaround) and *infers* the velocity --
the tangential speed at apocenter needed to reach ``r_peri``. Starting at apocenter makes the
placement unambiguous (zero radial velocity, purely tangential), and the derived eccentricity
and time-to-pericenter (half the orbital period) are reported. This is only a two-body
*estimate* -- the live host and dynamical friction perturb the real trajectory -- so the
pipeline integrates it self-consistently and checks the realized pericenter passage afterward.

Two perturber flavours share this placement (both are "just more particles" downstream):
a single softened point mass, or a small live self-gravitating Plummer satellite.
"""

from __future__ import annotations

import numpy as np

from nbody import generate_plummer


def com_state(pos, vel, masses):
    """Mass-weighted centre-of-mass position and velocity."""
    m = np.asarray(masses)
    total = m.sum()
    r = np.sum(pos * m[:, None], axis=0) / total
    v = np.sum(vel * m[:, None], axis=0) / total
    return r, v


def place_perturber(host_pos, host_vel, host_mass, *, M_pert, r_start, r_peri, G=1.0,
                    direction=1.0):
    """Compute the perturber COM position/velocity for a bound orbit dropped from apocenter.

    ``r_start`` is the apocenter (starting separation) and ``r_peri`` the target pericenter.
    The perturber begins on the +x axis at ``r_start`` with a purely tangential velocity
    (sense set by ``direction``, +/-1); the speed is inferred from vis-viva. Returns
    ``(com_pos, com_vel, info)`` with absolute (host-COM-referenced) 3-vectors and the derived
    orbit parameters, including ``t_peri_estimate = T/2`` (apocenter -> pericenter).
    """
    if r_start <= r_peri:
        raise ValueError(f"r_start ({r_start}) must exceed r_peri ({r_peri})")

    R_com, V_com = com_state(host_pos, host_vel, host_mass)
    M_host = float(np.sum(host_mass))
    M_tot = M_host + M_pert

    r_apo = r_start
    a = 0.5 * (r_apo + r_peri)                       # semi-major axis
    ecc = (r_apo - r_peri) / (r_apo + r_peri)         # derived eccentricity
    T = 2.0 * np.pi * np.sqrt(a ** 3 / (G * M_tot))   # orbital period
    v_apo = np.sqrt(G * M_tot * (2.0 / r_apo - 1.0 / a))  # vis-viva at apocenter (tangential)

    # Apocenter on +x, tangential velocity along +/-y; orbit lies in the x-y plane.
    rel_pos = np.array([r_apo, 0.0, 0.0])
    rel_vel = np.array([0.0, direction * v_apo, 0.0])

    com_pos = R_com + rel_pos
    com_vel = V_com + rel_vel

    info = dict(a=a, ecc=ecc, period=T, r_peri=r_peri, r_apo=r_apo, r_start=r_apo,
                v_start=float(v_apo), t_peri_estimate=0.5 * T, M_tot=M_tot,
                host_com_pos=R_com, host_com_vel=V_com)
    return com_pos, com_vel, info


def build_perturber(com_pos, com_vel, *, perturber_type, M_pert, eps_pert, G=1.0,
                    N_pert=None, a_pert=None, rng=None):
    """Build the perturber's particle arrays to concatenate onto the host in Phase B.

    Returns ``pos, vel, mass, softening, is_perturber`` (the last all-True).
    """
    if perturber_type == "point_mass":
        pos = com_pos[None, :].copy()
        vel = com_vel[None, :].copy()
        mass = np.array([M_pert], dtype=float)
        softening = np.array([eps_pert], dtype=float)

    elif perturber_type == "small_satellite":
        if N_pert is None or a_pert is None:
            raise ValueError("small_satellite requires N_pert and a_pert")
        if rng is None:
            rng = np.random.default_rng()
        p_int, v_int = generate_plummer(N_pert, M_pert, a_pert, rng, G=G)
        pos = p_int + com_pos[None, :]        # bulk-shift to the perturber COM
        vel = v_int + com_vel[None, :]        # internal velocities on top of COM velocity
        mass = np.full(N_pert, M_pert / N_pert, dtype=float)
        softening = np.full(N_pert, eps_pert, dtype=float)

    else:
        raise ValueError(f"unknown perturber_type: {perturber_type!r}")

    is_perturber = np.ones(len(mass), dtype=bool)
    return pos, vel, mass, softening, is_perturber
