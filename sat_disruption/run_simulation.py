"""CLI entry point: read a YAML config, apply overrides, run the simulation.

Examples
--------
    python run_simulation.py --config config.yaml
    python run_simulation.py --config config.yaml --dt 100 --t-end 500
    python run_simulation.py --config config.yaml --dt 100 --name dt100_run

CLI flags override the corresponding YAML values, so the snapshot cadence
``dt`` (and everything else) can be swept without editing the config file.
"""

import argparse

import yaml

from nbody_runner import run


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, args):
    """Overlay non-None CLI args onto the config dict (CLI wins)."""
    if args.dt is not None:
        cfg["dt"] = args.dt
    if args.dt_int is not None:
        cfg["dt_int"] = args.dt_int
    if args.t_end is not None:
        cfg["t_end"] = args.t_end
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.out is not None:
        cfg["out_dir"] = args.out
    if args.name is not None:
        cfg["run_name"] = args.name
    if args.N is not None:
        cfg["satellite"]["N"] = args.N
    return cfg


def build_parser():
    p = argparse.ArgumentParser(description="Satellite-disruption N-body pipeline.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--dt", type=float, default=None, help="Snapshot cadence (Myr).")
    p.add_argument("--dt-int", dest="dt_int", type=float, default=None,
                   help="Internal integrator step (Myr).")
    p.add_argument("--t-end", dest="t_end", type=float, default=None,
                   help="Total integration time (Myr).")
    p.add_argument("--N", type=int, default=None, help="Number of particles.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    p.add_argument("--out", default=None, help="Output directory.")
    p.add_argument("--name", default=None, help="Run name / tag.")
    p.add_argument("--diagnostics", action="store_true",
                   help="Also generate diagnostic plots after the run.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)

    print(f"[run] name={cfg['run_name']} dt={cfg['dt']} Myr "
          f"dt_int={cfg.get('dt_int', 0.01)} Myr t_end={cfg['t_end']} Myr "
          f"N={cfg['satellite']['N']}")
    paths = run(cfg)
    print(f"[run] wrote {len(paths)} snapshots to "
          f"{cfg['out_dir']}/{cfg['run_name']}")

    if args.diagnostics:
        from diagnostics import make_all
        make_all(cfg["out_dir"], cfg["run_name"], cfg)
        print("[run] diagnostics written.")


if __name__ == "__main__":
    main()
