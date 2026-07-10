"""Two-panel (x-y and x-z) simulation GIF via datashader.

The dense host population is rasterized with datashader (which handles overplotting and
renders fast for many-hundred-frame movies); the perturber is drawn as a distinct bright
overlay so it stays visible even as a single point mass. The frame cadence is deliberately
*decoupled* from the science snapshot cadence ``dt`` -- frames are the fine ``run_integration``
captures spanning ``t_relax + t_end`` -- so a coarse-``dt`` run still yields a smooth movie.
The axis range is fixed once from the whole trajectory so the view never rescales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from PIL import Image, ImageDraw
import h5py
import yaml

HOST_CMAP = ["#0b0b2a", "#3f7fd0", "#bfe0ff"]
PERT_COLOR = (255, 120, 40)
BG = (0, 0, 0)


def _fixed_extent(captures, host_pct=97.0, pad=1.15):
    """Symmetric half-width for the fixed view.

    Sized from the host *core* (a robust percentile of host-particle coordinates) plus the
    perturber's excursion, so a handful of particles flung out during a strong encounter do
    not zoom the whole view out and shrink the host to a dot.
    """
    host_vals = []
    pert_max = 0.0
    for c in captures:
        flag = c["is_perturber"]
        host_vals.append(np.abs(c["pos"][~flag]).reshape(-1))
        if flag.any():
            pert_max = max(pert_max, float(np.abs(c["pos"][flag]).max()))
    host_half = np.percentile(np.concatenate(host_vals), host_pct)
    return float(max(host_half, pert_max) * pad)


def _host_panel(df_host, xcol, ycol, extent, size):
    """Rasterize the host points for one projection into an RGB PIL image."""
    canvas = ds.Canvas(plot_width=size, plot_height=size,
                       x_range=(-extent, extent), y_range=(-extent, extent))
    if len(df_host):
        agg = canvas.points(df_host, xcol, ycol)
        img = tf.shade(agg, cmap=HOST_CMAP, how="eq_hist")
        img = tf.spread(img, px=1)   # fatten points for visibility, esp. at low N
        img = tf.set_background(img, "black")
        pil = img.to_pil().convert("RGB")
    else:
        pil = Image.new("RGB", (size, size), BG)
    return pil


def _overlay_perturber(pil, px, py, extent, size, radius=3):
    """Draw perturber particles as bright markers in the given projection."""
    draw = ImageDraw.Draw(pil)
    for x, y in zip(px, py):
        col = (x + extent) / (2 * extent) * size
        row = (extent - y) / (2 * extent) * size   # y up, matches datashader to_pil
        draw.ellipse([col - radius, row - radius, col + radius, row + radius],
                     fill=PERT_COLOR)
    return pil


def _compose(frame_capture, extent, size, gap=8):
    """Build one composite frame (x-y | x-z) from a single capture."""
    pos = frame_capture["pos"]
    flag = frame_capture["is_perturber"]
    host = pos[~flag]
    pert = pos[flag]
    df_host = pd.DataFrame(dict(x=host[:, 0], y=host[:, 1], z=host[:, 2]))

    xy = _host_panel(df_host, "x", "y", extent, size)
    xz = _host_panel(df_host, "x", "z", extent, size)
    if len(pert):
        _overlay_perturber(xy, pert[:, 0], pert[:, 1], extent, size)
        _overlay_perturber(xz, pert[:, 0], pert[:, 2], extent, size)

    canvas = Image.new("RGB", (2 * size + gap, size + 24), BG)
    canvas.paste(xy, (0, 24))
    canvas.paste(xz, (size + gap, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), "x-y", fill=(200, 200, 200))
    draw.text((size + gap + 4, 6), "x-z", fill=(200, 200, 200))
    draw.text((2 * size + gap - 90, 6), f"t = {frame_capture['t']:.1f}", fill=(200, 200, 200))
    return canvas


def make_gif(captures, out_path, cfg, size=500):
    """Render all captures into a GIF at ``gif_fps`` and write ``out_path``."""
    max_particles = cfg.get("gif_max_particles")
    fps = cfg["gif_fps"]

    # Fixed host subsample (indices are stable across frames; host particles keep their order).
    rng = np.random.default_rng(cfg["seed"])
    n_host = cfg["N_host"]
    if max_particles and n_host > max_particles:
        keep = np.sort(rng.choice(n_host, size=max_particles, replace=False))
    else:
        keep = None

    extent = _fixed_extent(captures)

    frames = []
    for c in captures:
        if keep is not None:
            flag = c["is_perturber"]
            host_idx = np.flatnonzero(~flag)[keep]
            pert_idx = np.flatnonzero(flag)
            sel = np.concatenate([host_idx, pert_idx])
            c = dict(t=c["t"], pos=c["pos"][sel], is_perturber=c["is_perturber"][sel])
        frames.append(_compose(c, extent, size))

    # PIL GIF assembly: duration is unambiguously milliseconds per frame; loop forever.
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(round(1000.0 / fps)), loop=0, disposal=2)
    return out_path


# --------------------------------------------------------------------------------------
# GIF from saved HDF5 snapshots (memory-safe; streams one snapshot at a time)
# --------------------------------------------------------------------------------------
def _extent_from_snapshots(path, host_pct=97.0, pad=1.15):
    """Fixed view half-width, computed by streaming snapshots one at a time (no full load)."""
    host_half = 0.0
    pert_max = 0.0
    with h5py.File(path, "r") as f:
        for name in sorted(k for k in f if k.startswith("snap_")):
            g = f[name]
            flag = g["is_perturber"][:]
            pos = g["pos"][:]
            h = np.abs(pos[~flag]).reshape(-1)
            if h.size:
                host_half = max(host_half, float(np.percentile(h, host_pct)))
            if flag.any():
                pert_max = max(pert_max, float(np.abs(pos[flag]).max()))
    return float(max(host_half, pert_max) * pad)


def make_gif_from_snapshots(snapshot_path, out_path, cfg=None, size=500):
    """Render a GIF from a saved HDF5 snapshot file, streaming one snapshot at a time.

    Unlike :func:`make_gif` (which uses the fine in-memory captures from a live run), this
    reads the dt-cadence science snapshots off disk, so it works after the fact and stays
    memory-safe at large ``N_host`` (only one snapshot, subsampled to ``gif_max_particles``,
    is resident at a time). Visualization parameters are read from the config embedded in the
    file unless ``cfg`` is supplied.
    """
    with h5py.File(snapshot_path, "r") as f:
        names = sorted(k for k in f if k.startswith("snap_"))
        if cfg is None:
            cfg = yaml.safe_load(f.attrs.get("config_yaml", "")) or {}
    if not names:
        raise ValueError(f"no snapshots found in {snapshot_path!r} (nothing to render)")

    max_particles = cfg.get("gif_max_particles")
    fps = cfg.get("gif_fps", 20)
    seed = cfg.get("seed", 0)
    extent = _extent_from_snapshots(snapshot_path)

    frames = []
    with h5py.File(snapshot_path, "r") as f:
        for name in names:
            g = f[name]
            flag = g["is_perturber"][:]
            pos = g["pos"][:]
            t = float(g.attrs["time"])
            host = pos[~flag]
            pert = pos[flag]
            if max_particles and len(host) > max_particles:
                # Stable subsample: host particle order is fixed across snapshots.
                idx = np.random.default_rng(seed).choice(len(host), size=max_particles,
                                                         replace=False)
                host = host[idx]
            allpos = np.concatenate([host, pert]) if len(pert) else host
            allflag = np.concatenate([np.zeros(len(host), bool), np.ones(len(pert), bool)])
            frames.append(_compose(dict(t=t, pos=allpos, is_perturber=allflag), extent, size))

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(round(1000.0 / fps)), loop=0, disposal=2)
    return out_path
