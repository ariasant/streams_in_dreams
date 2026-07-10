"""Render a two-panel GIF from a saved HDF5 snapshot file (after a run has finished).

Streams one snapshot at a time, so it is memory-safe even for very large N_host. Reads the
visualization parameters (gif_fps, gif_max_particles, seed) from the config embedded in the
snapshot file.

Usage:
    python make_gif.py runs/point_mass/point_mass_run_snapshots.h5
    python make_gif.py <snapshots.h5> --out my.gif
"""

from __future__ import annotations

import argparse
import os

import visualize


def main():
    ap = argparse.ArgumentParser(description="GIF from saved perturbed-host snapshots")
    ap.add_argument("snapshots", help="path to <run>_snapshots.h5")
    ap.add_argument("--out", help="output GIF path (default: alongside the snapshots)")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.snapshots)[0].replace("_snapshots", "") + ".gif"
    print(f"[gif] rendering from {args.snapshots} ...")
    visualize.make_gif_from_snapshots(args.snapshots, out)
    print(f"[gif] wrote {out}")


if __name__ == "__main__":
    main()
