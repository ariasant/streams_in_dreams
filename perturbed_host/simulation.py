"""Two-phase orchestration: relax the host (Phase A), then inject the perturber (Phase B).

Phase A integrates the isolated self-gravitating host so sampling transients settle. Its
final phase-space state is handed to Phase B by *concatenating* the perturber's particle
arrays onto it -- never resizing arrays inside a running loop. Both phases drive the same
``nbody.run_integration`` with a single continuous global step schedule, so science
snapshots (cadence ``dt``) and fine-cadence GIF/diagnostic captures line up across the seam.
"""

from __future__ import annotations

import os
import numpy as np

import nbody
import perturber as pert
from io_utils import SnapshotWriter


def _global_schedules(nsteps_A, nsteps_B, snapshot_stride, num_frames, num_energy):
    """Compute the global step indices for science snapshots, captures and energy samples.

    Science snapshots start at the beginning of Phase B (``g = nsteps_A``) and repeat every
    ``snapshot_stride`` steps. Captures are ``num_frames`` indices evenly spaced over the
    whole run ``[0, total]`` (independent of the science cadence ``dt``). Energy samples are a
    smaller evenly-spaced subset of the capture steps (each needs an O(N^2) potential solve).
    """
    total = nsteps_A + nsteps_B
    snap = set(range(nsteps_A, total + 1, snapshot_stride))
    snap.add(total)  # always capture the final state
    cap = set(int(round(x)) for x in np.linspace(0, total, num_frames))
    # Energy steps must be a subset of capture steps: snap each energy tick to a capture.
    cap_sorted = np.array(sorted(cap))
    ener = set(int(cap_sorted[np.argmin(np.abs(cap_sorted - x))])
               for x in np.linspace(0, total, num_energy))
    return total, snap, cap, ener


def run_pipeline(cfg):
    """Run Phase A and Phase B from a config dict; return a results dict.

    The results feed diagnostics and visualization: ``captures`` (fine frames spanning both
    phases), ``placement`` (perturber orbit info), ``snapshot_path`` (HDF5), and the step
    bookkeeping needed to mark the Phase A/B seam.
    """
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    snapshot_path = os.path.join(out_dir, f"{cfg['run_name']}_snapshots.h5")

    G = cfg["G"]
    dt_int = cfg["dt_int"]
    method = cfg.get("force_method", "direct")
    theta = cfg.get("theta", 0.7)
    parallel = cfg.get("parallel", True)
    rng = np.random.default_rng(cfg["seed"])

    # --- Step / cadence bookkeeping -------------------------------------------------
    nsteps_A = int(round(cfg["t_relax"] / dt_int))
    nsteps_B = int(round(cfg["t_end"] / dt_int))
    snapshot_stride = max(1, int(round(cfg["dt"] / dt_int)))
    num_frames = max(2, int(round(cfg["gif_duration"] * cfg["gif_fps"])))
    num_energy = min(num_frames, cfg.get("num_energy_samples", 120))
    total, snap_steps, cap_steps, energy_steps = _global_schedules(
        nsteps_A, nsteps_B, snapshot_stride, num_frames, num_energy)

    snap_A = {g for g in snap_steps if g < nsteps_A} if cfg.get("snapshot_phase_a") else set()
    snap_B = {g for g in snap_steps if g >= nsteps_A}
    cap_A = {g for g in cap_steps if g < nsteps_A}
    cap_B = {g for g in cap_steps if g >= nsteps_A}
    ener_A = {g for g in energy_steps if g < nsteps_A}
    ener_B = {g for g in energy_steps if g >= nsteps_A}

    writer = SnapshotWriter(snapshot_path, config_yaml=cfg.get("_yaml_text"))

    # --- Phase A: relax the host ----------------------------------------------------
    host_pos, host_vel = nbody.generate_plummer(
        cfg["N_host"], cfg["M_host"], cfg["a_host"], rng, G=G)
    host_mass = np.full(cfg["N_host"], cfg["M_host"] / cfg["N_host"])
    host_soft = np.full(cfg["N_host"], cfg["eps_host"])
    host_flag = np.zeros(cfg["N_host"], dtype=bool)

    host_pos, host_vel, captures_A = nbody.run_integration(
        host_pos, host_vel, host_mass, host_soft, host_flag,
        dt_int=dt_int, n_steps=nsteps_A, G=G, method=method, theta=theta,
        parallel=parallel, step0=0, t0=0.0,
        snapshot_steps=snap_A, capture_steps=cap_A, energy_steps=ener_A, writer=writer)

    # --- Phase B: inject the perturber and continue ---------------------------------
    com_pos, com_vel, placement = pert.place_perturber(
        host_pos, host_vel, host_mass,
        M_pert=cfg["M_pert"], r_start=cfg["r_start"], r_peri=cfg["r_peri"], G=G)

    T_orb = placement["period"]
    print(f"[Phase B] Kepler orbital period T = {T_orb:.4g}  "
          f"(a={placement['a']:.4g}, ecc={placement['ecc']:.4f}, "
          f"t_peri_estimate={placement['t_peri_estimate']:.4g})")
    if cfg["t_end"] < T_orb:
        import warnings
        warnings.warn(
            f"Phase B duration t_end={cfg['t_end']:.4g} is shorter than the estimated "
            f"Kepler orbital period T={T_orb:.4g}. The perturber may not complete a full "
            f"orbit; consider increasing t_end.",
            stacklevel=2,
        )

    p_pos, p_vel, p_mass, p_soft, p_flag = pert.build_perturber(
        com_pos, com_vel, perturber_type=cfg["perturber_type"],
        M_pert=cfg["M_pert"], eps_pert=cfg["eps_pert"], G=G,
        N_pert=cfg.get("N_pert"), a_pert=cfg.get("a_pert"), rng=rng)

    all_pos = np.concatenate([host_pos, p_pos], axis=0)
    all_vel = np.concatenate([host_vel, p_vel], axis=0)
    all_mass = np.concatenate([host_mass, p_mass], axis=0)
    all_soft = np.concatenate([host_soft, p_soft], axis=0)
    all_flag = np.concatenate([host_flag, p_flag], axis=0)

    _, _, captures_B = nbody.run_integration(
        all_pos, all_vel, all_mass, all_soft, all_flag,
        dt_int=dt_int, n_steps=nsteps_B, G=G, method=method, theta=theta,
        parallel=parallel, step0=nsteps_A, t0=cfg["t_relax"],
        snapshot_steps=snap_B, capture_steps=cap_B, energy_steps=ener_B, writer=writer)

    return dict(
        captures=captures_A + captures_B,
        captures_A=captures_A,
        captures_B=captures_B,
        placement=placement,
        snapshot_path=snapshot_path,
        n_snapshots=writer.n_snapshots,
        t_relax=cfg["t_relax"],
        nsteps_A=nsteps_A,
        config=cfg,
    )
