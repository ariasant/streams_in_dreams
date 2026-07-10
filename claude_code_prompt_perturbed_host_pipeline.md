# Prompt for Claude Code: perturbed self-gravitating host N-body pipeline

Paste everything below into Claude Code (CLI), run from the root of this repo.

---

## Context

I build basis function expansion (BFE) approximations of the gravitational potential of
cosmological simulations. In my production pipeline, BFE coefficients are computed at
each simulation snapshot (typically ~100 Myr apart) and linearly interpolated between
snapshots for orbit integration. I'm concerned this coarse cadence breaks down for stars
with short dynamical times (~10 Myr) near the galactic center, since a lot can happen
dynamically between two 100-Myr-apart coefficient evaluations.

To test this in a controlled setting (rather than on a full cosmological simulation, where
I wouldn't have a clean ground truth to compare against), I need a **self-gravitating
N-body simulation of a host system that gets perturbed by an infalling body at a known
time**, producing a genuinely time-varying potential. This pipeline's only job is to
produce that N-body simulation — the "ground truth" snapshots. Fitting BFE coefficients
to it at different cadences and comparing reconstructed orbits/forces against this ground
truth is a separate, later step, not part of this task.

Before starting, read `bfe_validation_literature_review.md` (also saved as a PDF) in this
repository — it's a literature review I put together surveying how basis function
expansions are validated in the astrophysics literature, organized by what aspect of the
approximation each method targets. The most relevant category for this pipeline is
"temporal cadence and coefficient interpolation" (covering Lowing et al. 2011, MNRAS 416,
2697; Sanders et al. 2020, MNRAS 499, 4793; Arora et al. 2024, ApJ 977, 23) — all of these
BFE-fit a genuinely evolving N-body system snapshot-by-snapshot and measure how
reconstructed orbits/forces degrade as a function of snapshot spacing. This pipeline
should produce a system suited to the same kind of test: a live, evolving, self-gravitating
host with a known, controllable perturbation event, sampled at a configurable cadence.

This repo already has relevant building blocks I want you to build on, not replace:

- `nbody_sim.py`: a `pytreegrav`-based leapfrog N-body integrator (`frog_step`/`run_sim`)
  for an isolated, self-gravitating Plummer sphere, with energy-conservation and
  density-profile diagnostics already implemented (currently runs as a top-level script
  with module-level globals — this needs refactoring into reusable functions, see below).
- `sampling_utils.py`: Plummer-sphere and Eddington-inversion initial-condition sampling
  utilities.

Keep the implementation as simple as possible — correctness and readability over
performance. This doesn't need to scale beyond a few thousand host particles for now.

## Design: two-phase simulation (relax, then perturb)

**Phase A — relax the host.** Sample the host as an isotropic Plummer sphere (reuse
`generate_plummer()` from `nbody_sim.py`) and integrate it with self-gravity only (no
perturber) for a burn-in period, so sampling-noise transients settle out before the actual
experiment starts. Save the final phase-space state (positions, velocities, masses).

**Phase B — inject the perturber and run the experiment.** Take Phase A's final state,
concatenate the perturber's phase-space state onto the position/velocity/mass arrays, and
continue the same self-gravitating leapfrog integration with this larger particle set.
Place the perturber far enough from the host with an inbound velocity that it reaches a
chosen pericenter distance at a chosen time within Phase B (approximate two-body Kepler
estimate for the starting radius/velocity — this doesn't need to be exact, since the
host's own gravity will perturb the real trajectory anyway; just check afterward how close
the realized pericenter passage was to the target). This two-call design (Phase A then
Phase B, with an explicit state handoff) avoids ever having to resize particle arrays
inside a running integration loop.

**Perturber type — make this a config switch, not a fixed choice**: support both
(a) a single point-mass perturber (softened like every other particle, not special-cased)
and (b) a small live, self-gravitating satellite built the same way as the host (its own
`generate_plummer()` call, lower mass and/or particle count, its own softening). Both are
"just more particles" to the same `pytreegrav` force solver, so this should be a small
config-driven difference in Phase B's setup, not two separate code paths. The point-mass
case is the fast, easy-to-interpret baseline (single, well-understood Keplerian flyby,
useful as a first sanity-checked run); the small-live-satellite case adds a second,
independently evolving self-gravitating body — its own internal structure and possible
tidal disruption during the encounter mean it doesn't perturb the host as a single clean
frequency the way a point mass does. I plan to run both and compare.

## Target run configuration (use this as the first real production run, after the smoke tests)

```yaml
dt_int: 0.01          # internal integration step (accurate orbits are the priority here)
dt: 0.1                # snapshot cadence
t_relax: 5              # Phase A burn-in duration
t_end: 15                # Phase B duration -> total simulation time T = t_relax + t_end = 20 (a few dynamical times)
N_host: 1_000_000
M_host: 1.0             # Henon units, matching nbody_sim.py's existing convention (G=1)
a_host: 1.0
perturber_type: point_mass
M_pert: 0.1
r_start: 10.0             # perturber's separation from the host at the start of Phase B (i.e. at absolute t=5)
```

Notes on this configuration, carried over from earlier discussion so you don't "fix" them
as if they were mistakes:

- `M_pert=0.5` is half the host's total mass — a near-equal-mass interaction, not a minor
  perturbation. This is a deliberate, accepted choice for this run, not an oversight.
- `r_start=1.0` equals `a_host`, i.e. the perturber starts already partway inside the
  host's mass distribution (about 35% of a Plummer sphere's mass is enclosed within
  `r=a`) rather than genuinely falling in from far outside. This means the two-body
  Kepler estimate for translating a target pericenter distance/time into a starting
  velocity (described in the Design section above) isn't really applicable here — instead,
  just place the perturber directly at `r_start` (any reasonable direction) with an
  inbound velocity of your choosing (e.g. a modest infall velocity, or even starting
  near-radially inward), since the precise approach trajectory matters less than usual
  given how close-in it already starts. Confirm via the perturber-orbit diagnostic where
  and when the actual pericenter ends up happening, rather than assuming it lands exactly
  at the nominal injection point.
- At `N_host=1,000,000`, this is a real production-scale run: use the tree-code
  (`pytreegrav`) acceleration path, not direct summation, and run it on the same kind of
  multi-core/cluster hardware you've used for `nbody_sim.py` before (its own benchmark
  comment: N=10⁶, `dt=0.01`, 10,000 steps ≈ "a couple minutes" on that hardware; this run
  is 2,000 steps, so proportionally faster, but confirm rather than assume). Before
  launching the full `t_end=15` run, time just 50-100 steps at the full `N_host` on the
  actual target machine and extrapolate — don't guess.
- At this `N_host`, snapshot output at `dt=0.1` over `T=20` is ~200 snapshots; write
  positions/velocities in float32 rather than float64 in the HDF5 files unless you need
  the extra precision, to keep total snapshot storage (~10 GB at float64, ~5 GB at
  float32) and write time reasonable.
- For the GIF at this `N_host`, cap `gif_max_particles` to something datashader can
  animate at ~1200-1800 frames without that step dominating total wall-clock time (a few
  hundred thousand is likely still comfortable for datashader — this is more about total
  frame-rendering time than datashader's per-frame capacity).

## Required inputs (config file, e.g. YAML, and/or CLI flags)

1. **Snapshot cadence `dt`** and **internal integrator step `dt_int`**, independent of each
   other — `dt` is the parameter I'll be varying later (e.g. equivalent to 1, 10, 25, 100,
   200 Myr); `dt_int` should stay small and fixed regardless of what `dt` is, since I need
   to hold integration accuracy fixed while varying snapshot cadence.
2. **Phase A (host) parameters**: `N_host`, `M_host`, `a_host` (Plummer scale radius),
   softening, burn-in duration `t_relax`, random seed.
3. **Phase B (perturber) parameters**: `perturber_type` (`point_mass` or
   `small_satellite`), perturber mass `M_pert`, desired pericenter distance `r_peri`,
   desired pericenter time `t_peri` (measured from the start of Phase B), total Phase B
   duration `t_end`, perturber softening; if `small_satellite`, also `N_pert` and `a_pert`
   (its own Plummer scale radius).
4. **Visualization parameters**: `gif_duration` (wall-clock length of the output GIF,
   default 60 s), `gif_fps` (default ~20-30), and optionally `gif_max_particles` (cap on
   how many particles to plot, for speed/file-size, subsampled randomly if `N_host` is
   larger).
5. Output directory / run name.

## What to implement

- Refactor `nbody_sim.py`'s Plummer sampling and leapfrog integration into importable,
  parameterized functions (no module-level globals like the current `M`, `a`, `G`,
  `theta`) without changing their physics — needed so Phase A and Phase B (and the
  point-mass/small-satellite variants) can call the same integration function with
  different particle sets.
- Implement Phase A and Phase B as described, with the array-concatenation handoff.
- Implement perturber IC placement (approximate two-body Kepler estimate for starting
  radius/velocity from `M_pert`, `r_peri`, `t_peri`, and the host's total mass), for
  both perturber types.
- Implement `dt`/`dt_int`-driven snapshot selection. `nbody_sim.py`'s current `run_sim()`
  takes a `t_outputs` list of step indices to save; compute this from `dt`/`dt_int`
  (roughly `t_outputs = [i for i in range(nsteps) if i % round(dt/dt_int) == 0]`) rather
  than saving every step (the current script's `t_outputs=[i for i in range(0,nsteps)]`
  saves every single step, which won't scale — fix this).
- Implement HDF5 snapshot output: positions, velocities, masses, time, and a flag marking
  which particle(s) are the perturber (vs. host), so a downstream BFE-fitting pipeline can
  easily separate the host's own density from the perturber's known trajectory.

## Visualization: simulation GIF

The pipeline should produce a ~1-minute animated GIF of each run with **two side-by-side
projections, x-y and x-z**, both showing host particles and the perturber in
distinguishable colors, spannin