# perturbed_host

A self-gravitating N-body pipeline that **relaxes a Plummer host, then perturbs it with an
infalling body at a known time**, producing a genuinely time-varying potential sampled at a
configurable snapshot cadence. It is the controlled ground truth for a downstream study of
how a basis-function-expansion (BFE) potential degrades when its coefficients are evaluated
only every `dt` and interpolated (cf. Lowing et al. 2011, Sanders et al. 2020, Arora et al.
2024; see `../bfe_validation_literature_review.md`). Fitting BFE coefficients is a **separate,
later step** — this pipeline only produces the simulation, snapshots, diagnostics and movie.

Built on the repo's `pytreegrav` leapfrog (`../nbody_sim.py`), refactored here into
globals-free, importable functions.

## Design

Two phases with an explicit array-concatenation handoff (no array resizing mid-loop):

- **Phase A — relax the host.** Sample an isotropic Plummer sphere and integrate with
  self-gravity only for `t_relax`, so sampling transients settle. Save the final state.
- **Phase B — inject the perturber.** Concatenate the perturber's phase-space state onto the
  host arrays and continue the same leapfrog. The perturber is a config switch:
  `point_mass` (one softened particle — the clean Keplerian baseline) or `small_satellite`
  (its own live self-gravitating Plummer sphere that can tidally disrupt).

By default the perturber is **dropped from rest**: you set the starting separation `r_start`
and the perturber is placed there with zero velocity relative to the host, a pure radial
infall. An analytic two-body free-fall time (point mass, rest-to-rest) is reported as a rough
estimate of when it reaches the host. Set `circular: true` to instead place it on a circular
orbit at `r_start` (tangential velocity from the point-mass `v_circ`); the analytic circular
period is reported instead. Either way this is only a two-body estimate — the live host and
dynamical friction perturb the real trajectory — so the pipeline integrates it self-consistently
and reports the *realized* closest approach against the estimate (radial drop only; a circular
placement has no pericenter to compare against).

The **science cadence `dt`** (the variable to vary later) is decoupled from the **fixed
internal step `dt_int`** (holds integration accuracy constant). HDF5 snapshots are written
every `dt`; the GIF and diagnostics use a separate fine capture cadence.

All quantities are in **system units** (`G` configurable, default 1, with `M_host = a_host =
1` in the shipped config). Map `dt` to physical Myr externally for the cadence experiment.

## Layout

| file | role |
|------|------|
| `nbody.py` | globals-free Plummer sampling + leapfrog (`generate_plummer`, `leapfrog_step`, `run_integration`) |
| `perturber.py` | zero-velocity radial-drop placement + `point_mass`/`small_satellite` builder |
| `simulation.py` | Phase A/B orchestration, step schedules, concatenation handoff |
| `io_utils.py` | HDF5 snapshot writer/reader with a per-snapshot `is_perturber` flag |
| `diagnostics.py` | energy, host density/dispersion, perturber orbit, Lagrange radii |
| `visualize.py` | two-panel (x-y, x-z) datashader GIF |
| `run_simulation.py` | CLI entry point |
| `config.yaml` / `config_smoke.yaml` | representative run / fast end-to-end smoke test |

## Usage

```bash
# Windows (this machine): use the dedicated venv
./.venv/Scripts/python.exe run_simulation.py --config config_smoke.yaml   # ~seconds
./.venv/Scripts/python.exe run_simulation.py --config config.yaml         # representative run
```

Outputs land in `output_dir`: `<run>_snapshots.h5`, `<run>_energy.pdf`,
`<run>_orbit.pdf`, `<run>_profiles.pdf`, `<run>_lagrange.pdf`, and `<run>.gif`.
Flags: `--no-gif`, `--no-diagnostics`.

### HDF5 format

One group `snap_XXXX` per science snapshot with datasets `pos (N,3)`, `vel (N,3)`,
`mass (N,)`, `is_perturber (N,) bool`, and attrs `step`, `time`. The per-snapshot flag lets a
downstream BFE fitter separate the host density from the perturber's known trajectory, and
handles the Phase A/B change in particle count. Read back with `io_utils.read_snapshots`.

## Diagnostics

1. **Energy conservation** across both phases (fractional drift measured *within* each phase;
   the step at `t_relax` is the perturber injection, annotated).
2. **Host density + velocity dispersion** at a few times through the encounter vs. the
   analytic Plummer profiles — confirms the host structurally responds.
3. **Perturber orbit**: separation from the host COM vs. time, realized closest approach vs.
   the two-body free-fall time estimate.
4. **Host Lagrange radii** (10/50/90% mass, host-only) vs. time.

The radial-density binning mirrors `../DREAMS_utils.py:return_density` (log shells,
mass/volume) but is reimplemented locally so the venv stays free of its
`pynbody`/`astropy` dependencies.

## Environment

Dedicated `perturbed_host/.venv` (Python 3.10). Recreate with:

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Tests

```bash
./.venv/Scripts/python.exe -m pytest tests -q
```

- `test_plummer_ic` — sampled density/dispersion vs. analytic Plummer.
- `test_radial_drop` — two-body integration of the zero-velocity drop reaches closest approach
  at ~the estimated free-fall time.
- `test_circular_placement` — two-body integration of a `circular=True` placement stays at
  `r_start` and returns to its starting position after one period.
- `test_handoff` — host state (and host-subset energy) preserved across the concatenation.
- `test_io` — HDF5 round-trip including the Phase A/B particle-count change.
