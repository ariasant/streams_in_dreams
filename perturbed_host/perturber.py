"""Perturber initial conditions: placed at a starting separation ``r_start``.

By default the perturber is placed on the +x axis at distance ``r_start`` from the host COM
with zero velocity (in the host's frame) -- a pure radial infall, no orbit parameters to derive
or tune. Set ``circular=True`` to instead place it on a circular orbit at ``r_start`` (tangential
velocity from the point-mass ``v_circ = sqrt(G*M_tot/r_start)``). Both cases are two-body,
point-mass estimates -- the live host and dynamical friction perturb the real trajectory -- so
the reported timescales (free-fall time, circular period) are only a ballpark for sizing
``t_end``.

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


def place_perturber(host_pos, host_vel, host_mass, *, M_pert, r_start, circular=False, G=1.0):
    """Compute the perturber COM position/velocity at separation ``r_start``.

    By default (``circular=False``) the perturber begins on the +x axis at rest relative to the
    host (``com_vel`` equals the host's COM velocity) -- a radial drop. If ``circular=True`` it
    instead gets a tangential velocity equal to the point-mass circular speed at ``r_start``.
    Returns ``(com_pos, com_vel, info)`` with absolute 3-vectors and the two-body point-mass
    estimates ``t_freefall_estimate`` (radial infall time) and ``period``/``v_circ`` (circular
    orbit), any of which may be used to size ``t_end`` depending on ``circular``.
    """
    R_com, V_com = com_state(host_pos, host_vel, host_mass)
    M_host = float(np.sum(host_mass))
    M_tot = M_host + M_pert

    v_circ = np.sqrt(G * M_tot / r_start)
    period = 2.0 * np.pi * r_start / v_circ
    # Radial two-body free fall from rest: time to reach the centre is half the period of the
    # degenerate ellipse with semi-major axis a = r_start / 2.
    t_ff = (np.pi / (2.0 * np.sqrt(2.0))) * np.sqrt(r_start ** 3 / (G * M_tot))

    rel_pos = np.array([r_start, 0.0, 0.0])
    rel_vel = np.array([0.0, v_circ, 0.0]) if circular else np.zeros(3)

    com_pos = R_com + rel_pos
    com_vel = V_com + rel_vel

    info = dict(r_start=r_start, M_tot=M_tot, circular=circular,
                t_freefall_estimate=float(t_ff), period=float(period), v_circ=float(v_circ),
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
