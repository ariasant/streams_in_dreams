"""Factory that turns a config block into a ``gala.potential`` object.

Example config block::

    host_potential:
      type: HernquistPotential
      params: {m: 1.0e12, c: 15.0}
      units: galactic

The factory looks the class up by name in ``gala.potential`` so any potential
that package exposes works with no code change -- only the config. Numeric
params are given in the native galactic unit system (mass in Msun, lengths in
kpc, velocities in km/s) and the appropriate astropy unit is attached here.
"""

import astropy.units as u
import gala.potential as gp
from gala.units import galactic

# Unit system lookup, extend as needed.
_UNIT_SYSTEMS = {"galactic": galactic}

# Per-parameter units for the common host-potential parameters. Anything not
# listed is passed through as a bare float (gala treats it as being in the
# potential's native unit system).
_PARAM_UNITS = {
    "m": u.Msun,     # mass
    "c": u.kpc,      # Hernquist / NFW scale radius
    "b": u.kpc,      # Plummer / Isochrone scale radius
    "a": u.kpc,      # scale length
    "r_s": u.kpc,    # NFW scale radius
    "r_h": u.kpc,
    "v_c": u.km / u.s,   # circular / logarithmic velocity scale
}


def _attach_units(params):
    """Attach astropy units to known numeric params; pass others through.

    Values are coerced to float first: YAML 1.1 parses ``1.0e12`` (no signed
    exponent) as a *string*, so ``float(...)`` normalises it before we build a
    Quantity.
    """
    out = {}
    for key, val in params.items():
        unit = _PARAM_UNITS.get(key)
        out[key] = (float(val) * unit) if unit is not None else val
    return out


def build_host_potential(cfg):
    """Build the external host potential from ``cfg['host_potential']``.

    Returns
    -------
    gala.potential.PotentialBase
    """
    block = cfg["host_potential"]
    cls_name = block["type"]
    params = block.get("params") or {}
    units = _UNIT_SYSTEMS[block.get("units", "galactic")]

    try:
        cls = getattr(gp, cls_name)
    except AttributeError as exc:
        raise ValueError(
            f"Unknown gala.potential class {cls_name!r}. "
            f"Pick any class exposed by gala.potential."
        ) from exc

    # MilkyWayPotential / MilkyWayPotential2022 are prebuilt composites that
    # take no m/c; calling with empty params gives the default MW model.
    if cls_name.startswith("MilkyWayPotential"):
        return cls(units=units, **params)

    return cls(units=units, **_attach_units(params))
