"""Chunked ``DirectNBody`` integration loop -- the heart of the pipeline.

Why the chunked loop exists
---------------------------
The whole point of this pipeline is that the *snapshot cadence* ``dt`` is
decoupled from the *internal integration step* ``dt_int``. ``gala``'s
``integrate_orbit`` takes a single fixed step size and integrates from ``t1``
to ``t2``; that step size is the internal accuracy-setting step, NOT a snapshot
cadence.

So to save state exactly every ``dt`` Myr while integrating with a much finer
fixed step ``dt_int``, we loop: each iteration integrates one chunk of wall
length ``dt`` using internal step ``dt_int`` (with ``save_all=False`` so gala
only keeps the endpoint, not every sub-step), writes a snapshot, and then
re-instantiates ``DirectNBody`` with the chunk's final ``PhaseSpacePosition``
as the new ``w0``. Changing ``dt`` therefore changes only how often we snapshot,
never the integration accuracy -- exactly analogous to how cosmological
simulations write snapshots every ~100 Myr despite a much finer timestep.
"""

import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import numpy as np
from gala.dynamics import Orbit
from gala.dynamics.nbody import DirectNBody
from gala.units import galactic

from initial_conditions import build_satellite_w0
from io_utils import snapshot_path
from potentials import build_host_potential


def _final_state(result):
    """Return the final PhaseSpacePosition from an integrate_orbit result.

    gala versions differ in what ``integrate_orbit(save_all=False)`` returns:
    newer gala returns an ``Orbit`` with a (length-1) time axis, so we take the
    last timestep; gala 1.4.x returns the final ``PhaseSpacePosition`` directly
    (shape ``(N,)``), which must NOT be indexed (that would pick a particle).
    """
    if isinstance(result, Orbit):
        return result[-1]
    return result


def _particle_potentials(mass, eps):
    """One softened Plummer kernel per particle (its own self-gravity source)."""
    return [
        gp.PlummerPotential(m=float(m) * u.Msun, b=float(eps) * u.kpc, units=galactic)
        for m in mass
    ]


def _write_state(out_dir, run_name, index, t, w, mass, attrs):
    """Extract kpc / km/s arrays from a PhaseSpacePosition and write a snapshot."""
    from io_utils import write_snapshot

    pos = w.xyz.to_value(u.kpc).T           # (N, 3)
    vel = w.v_xyz.to_value(u.km / u.s).T     # (N, 3)
    write_snapshot(
        snapshot_path(out_dir, run_name, index),
        t=t, pos=pos, vel=vel, mass=mass, attrs=attrs,
    )


def run(cfg):
    """Run a satellite-disruption simulation and write one snapshot per ``dt``.

    Returns the list of snapshot file paths.
    """
    out_dir = cfg["out_dir"]
    run_name = cfg["run_name"]
    seed = int(cfg["seed"])
    dt = float(cfg["dt"])
    t_end = float(cfg["t_end"])
    dt_int = float(cfg.get("dt_int", 0.01))  # fixed, independent of dt

    rng = np.random.default_rng(seed)

    host = build_host_potential(cfg)
    w0, mass, eps = build_satellite_w0(cfg, rng)
    particle_pot = _particle_potentials(mass, eps)

    n_snap = int(round(t_end / dt))

    attrs = {
        "dt": dt,
        "dt_int": dt_int,
        "t_end": t_end,
        "eps": eps,
        "run_name": run_name,
        "seed": seed,
        "host_type": cfg["host_potential"]["type"],
        "host_params": cfg["host_potential"].get("params") or {},
    }

    paths = []

    # Snapshot 0: initial conditions.
    _write_state(out_dir, run_name, 0, 0.0, w0, mass, attrs)
    paths.append(snapshot_path(out_dir, run_name, 0))

    w = w0
    for k in range(1, n_snap + 1):
        nbody = DirectNBody(
            w, particle_pot,
            external_potential=host,
            units=galactic,
            save_all=False,
        )
        result = nbody.integrate_orbit(
            dt=dt_int * u.Myr, t1=0.0 * u.Myr, t2=dt * u.Myr,
        )
        # Final state seeds the next chunk (version-robust; see _final_state).
        w = _final_state(result)

        t = k * dt
        _write_state(out_dir, run_name, k, t, w, mass, attrs)
        paths.append(snapshot_path(out_dir, run_name, k))

    return paths
