# `sat_disruption` — satellite disruption in a static `gala` host

A small, deliberately simple pipeline that integrates a self-gravitating
Plummer satellite (N equal-mass particles) as it disrupts inside a **static,
analytic host potential** from `gala.potential`, using
`gala.dynamics.nbody.DirectNBody`. It writes one HDF5 snapshot every `dt` Myr.

It exists as a controlled test bed for a downstream study of how a
basis-function-expansion (BFE) potential degrades when its coefficients are
only evaluated every `dt` Myr and interpolated in between.

This module is **self-contained** and does not depend on or modify the
`nbody_sim.py` / `sampling_utils.py` code in the parent directory (the Plummer
sampling math was adapted from `nbody_sim.py:generate_plummer` but
reimplemented here independently).

## The one design idea: `dt` vs `dt_int`

There are two different time steps, and keeping them separate is the whole
point of this pipeline:

| knob     | meaning                                   | when to change it |
|----------|-------------------------------------------|-------------------|
| `dt`     | **snapshot cadence** — time between saved snapshots (Myr) | This is the parameter you sweep (1, 10, 25, 100, 200 …). |
| `dt_int` | **internal integrator step** (Myr), fixed & small (default 0.01) | Only if you deliberately want to change integration accuracy. |

`gala`'s `integrate_orbit(dt=…)` takes a *single fixed integration step*, not a
snapshot cadence. To save state exactly every `dt` while integrating with the
much finer `dt_int`, the runner **loops**: each iteration integrates one chunk
of length `dt` (internally sub-stepping at `dt_int`, with `save_all=False` so
gala keeps only the endpoint), writes a snapshot, then re-instantiates
`DirectNBody` from the chunk's final state. So changing `dt` changes only how
often you snapshot — never the integration accuracy. This mirrors how
cosmological simulations write snapshots every ~100 Myr despite a much finer
internal timestep.

## Install

`gala` is a compiled C-extension package that links against GSL, so how you
install it depends on your platform.

### Linux / macOS / cluster (recommended for production)

conda-forge ships binaries for a current gala:

```bash
conda create -n satdis -c conda-forge python=3.11 gala astropy numpy scipy h5py matplotlib pyyaml pytest -y
conda activate satdis
```

(Or `pip install -r requirements.txt`, which needs a C compiler + GSL.)

### Windows with `uv` (what this repo is set up for)

gala publishes **no Windows wheels** for modern versions (a source build needs
MSVC + GSL), and `uv`/`pip` can't provide GSL. The reproducible workaround used
here is a `uv` venv running the last gala that *did* ship a Windows wheel
(**1.4.1**, cp38), with GSL's runtime DLLs staged into the venv from
conda-forge. Just run the setup script:

```bash
bash setup_windows_uv.sh          # creates .venv/, installs everything
./.venv/Scripts/python.exe -m pytest tests -q
./.venv/Scripts/python.exe run_simulation.py --config config.yaml
```

See the top of `setup_windows_uv.sh` for exactly why each step is needed. Note
that gala 1.4.1 predates `MilkyWayPotential2022` (use `MilkyWayPotential`); the
`DirectNBody` API this pipeline relies on is otherwise identical.

> Version note: `integrate_orbit(save_all=False)` returns the final
> `PhaseSpacePosition` directly in gala 1.4.x, but an `Orbit` in newer gala.
> `nbody_runner._final_state()` handles both, so the code runs unchanged on a
> modern gala too.

## Run

```bash
# baseline run from the example config
python run_simulation.py --config config.yaml

# sweep the snapshot cadence without editing the config; dt_int is unchanged
python run_simulation.py --config config.yaml --dt 100 --name dt100_run

# short run + diagnostic plots
python run_simulation.py --config config.yaml --t-end 500 --diagnostics
```

CLI flags (`--dt`, `--dt-int`, `--t-end`, `--N`, `--seed`, `--out`, `--name`)
override the YAML. Snapshots are written to `{out_dir}/{run_name}/`.

## Config

See `config.yaml` for a documented example. The host potential is chosen by
class name:

```yaml
host_potential:
  type: HernquistPotential      # any gala.potential class works by name
  params: {m: 1.0e12, c: 15.0}  # native galactic units (Msun, kpc)
  units: galactic
```

`HernquistPotential`, `NFWPotential`, `LeeSutoTriaxialNFWPotential`,
`MilkyWayPotential`, `MilkyWayPotential2022`, `IsochronePotential`,
`LogarithmicPotential`, … all work out of the box — extend to any other
`gala.potential` class purely through config.

## Snapshot format

One HDF5 file per snapshot: `{out_dir}/{run_name}/snapshot_{k:04d}.hdf5`, with
datasets `t` (Myr), `pos` (N,3 kpc), `vel` (N,3 km/s), `mass` (N, Msun) and
file attributes recording `dt`, `dt_int`, host type/params, run name and seed.
Read them back with `io_utils.read_snapshot(path)`.

## Diagnostics

`diagnostics.make_all(out_dir, run_name, cfg)` (or `--diagnostics`) writes:

- `diag_energy.png` — fractional total-energy error vs time (self-gravity + KE
  + host PE); should stay well under ~1%.
- `diag_bound_lagrange.png` — bound mass fraction and 10/50/90% Lagrange radii
  vs time (the disruption signature).
- `diag_scatter.png` — x-y and x-z position scatters at several snapshots
  (tidal-tail formation).

## Tests

```bash
python -m pytest tests -q
```

- `test_plummer_ic.py` — sampled ICs reproduce the analytic Plummer density and
  velocity-dispersion profiles (no gala needed).
- `test_two_body.py` — two masses, no host, reproduce the analytic Kepler
  ellipse (apo/peri, energy, semi-major axis, period) through the chunked loop.

## Limitations

- Force evaluation is direct softened point-mass summation — O(N²) per step.
  Fine for N ~ 10³–10⁴, slow beyond. No tree code in v1 (deliberate: this is a
  controlled test, not a production run).
- Everything internal is in `gala.units.galactic`; units are converted only at
  the I/O boundary.
