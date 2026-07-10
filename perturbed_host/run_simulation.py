"""CLI entry point for the perturbed-host pipeline.

Usage:
    python run_simulation.py --config config.yaml
    python run_simulation.py --config config_smoke.yaml --no-gif

Runs Phase A (relax) then Phase B (perturb), writes HDF5 snapshots, produces the
diagnostics PDFs, and renders the two-panel GIF.
"""

from __future__ import annotations

import argparse
import os
import yaml

import simulation
import diagnostics
import visualize


def load_config(path):
    with open(path, "r") as f:
        text = f.read()
    cfg = yaml.safe_load(text)
    cfg["_yaml_text"] = text
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Perturbed self-gravitating host N-body pipeline")
    ap.add_argument("--config", required=True, help="path to YAML config")
    ap.add_argument("--no-gif", action="store_true", help="skip GIF rendering")
    ap.add_argument("--no-diagnostics", action="store_true", help="skip diagnostics")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[run] {cfg['run_name']}: relaxing host (N={cfg['N_host']}) then "
          f"{cfg['perturber_type']} flyby ...")
    results = simulation.run_pipeline(cfg)

    p = results["placement"]
    print(f"[placement] r_start(apo)={p['r_start']:.3f} r_peri={p['r_peri']:.3f} "
          f"-> a={p['a']:.3f} e={p['ecc']:.3f} v_start={p['v_start']:.3f} "
          f"period={p['period']:.3f} t_peri_est={p['t_peri_estimate']:.3f}")
    print(f"[snapshots] wrote {results['n_snapshots']} snapshots -> {results['snapshot_path']}")

    if not args.no_diagnostics:
        summary = diagnostics.make_all(results, out_dir)
        e = summary["energy"]
        o = summary["orbit"]
        for phase, drift in e.items():
            print(f"[energy] {phase}: max |dE/E| = {drift:.2e}")
        print(f"[orbit] realized r_peri={o['realized_r_peri']:.3f} "
              f"(target {o['target_r_peri']:.3f}), "
              f"realized t_peri={o['realized_t_peri']:.3f} "
              f"(two-body est. {o['target_t_peri']:.3f})")

    if not args.no_gif:
        gif_path = os.path.join(out_dir, f"{cfg['run_name']}.gif")
        print(f"[gif] rendering {len(results['captures'])} frames ...")
        visualize.make_gif(results["captures"], gif_path, cfg)
        print(f"[gif] wrote {gif_path}")

    print("[done]")


if __name__ == "__main__":
    main()
