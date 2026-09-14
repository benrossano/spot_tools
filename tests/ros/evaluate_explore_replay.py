#!/usr/bin/env python3
"""Compare an exploration replay (occupancy_mapper + keyframe_recorder) with Hydra's output.

  evaluate_explore_replay.py <run_dir> --prior <hydra_run_dir> [--png out.png]

<run_dir>/occupancy_live.npz  vs  <prior>/occupancy_static.npz   (grid agreement, growth, flips)
<run_dir>/live_agents/        vs  <prior>/agents/                (calibration equality, pose error
                                                                  at matching timestamps)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def load_grid(path: pathlib.Path):
    with np.load(str(path)) as d:
        return np.asarray(d["grid"], dtype=np.int8), np.asarray(d["origin"], float).reshape(4, 4), float(d["resolution"])


def cells_of(grid, origin, res):
    """(x, y) world coordinates of every cell centre as flattened arrays plus values."""
    rows, cols = grid.shape
    iy, ix = np.mgrid[0:rows, 0:cols]
    x = origin[0, 3] + (ix + 0.5) * res
    y = origin[1, 3] + (iy + 0.5) * res
    return x.ravel(), y.ravel(), grid.ravel()


def lookup(grid, origin, res, x, y):
    ix = np.floor((x - origin[0, 3]) / res).astype(int)
    iy = np.floor((y - origin[1, 3]) / res).astype(int)
    rows, cols = grid.shape
    ok = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    out = np.full(x.shape, -2, dtype=np.int8)  # -2 = outside
    out[ok] = grid[iy[ok], ix[ok]]
    return out


def load_agents(directory: pathlib.Path, prefix: str):
    frames = []
    for meta in sorted(directory.glob(f"{prefix}_*_meta.json")):
        m = json.loads(meta.read_text())
        frames.append((int(m["timestamp_ns"]), np.asarray(m["world_T_body"], float).reshape(4, 4)))
    calib = json.loads((directory / "camera_calib.json").read_text()) if (directory / "camera_calib.json").exists() else None
    return frames, calib


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--prior", type=pathlib.Path, required=True)
    ap.add_argument("--png", type=pathlib.Path)
    a = ap.parse_args()
    ok = True

    # ---------------------------------------------------------------- grids
    live_g, live_o, res = load_grid(a.run_dir / "occupancy_live.npz")
    prior_g, prior_o, pres = load_grid(a.prior / "occupancy_static.npz")
    print(f"grid  live {live_g.shape[1]}x{live_g.shape[0]} origin ({live_o[0,3]:.1f},{live_o[1,3]:.1f})  "
          f"prior {prior_g.shape[1]}x{prior_g.shape[0]} origin ({prior_o[0,3]:.1f},{prior_o[1,3]:.1f})  res {res:.2f}/{pres:.2f}")
    lx, ly, lv = cells_of(live_g, live_o, res)
    pv = lookup(prior_g, prior_o, pres, lx, ly)
    known_live = lv >= 0
    known_prior = pv >= 0
    both = known_live & known_prior
    agree = (lv[both] == pv[both]).mean() if both.any() else float("nan")
    new_known = known_live & ~known_prior
    flips_to_occ = both & (pv == 0) & (lv == 100)
    flips_to_free = both & (pv == 100) & (lv == 0)
    lost = known_prior & ~known_live
    print(f"      prior known {int(known_prior.sum())} cells; live known {int(known_live.sum())} "
          f"(+{int(new_known.sum())} newly observed, {int(lost.sum())} prior cells now unknown)")
    print(f"      agreement on cells known to both: {100*agree:.1f}%  "
          f"(free->occupied {int(flips_to_occ.sum())}, occupied->free {int(flips_to_free.sum())})")
    if not (agree > 0.85):
        print("      FAIL agreement below 85%"); ok = False
    if lost.sum() > 0.01 * known_prior.sum():  # a few cells pass through unknown while flipping
        print("      FAIL more than 1% of prior cells became unknown"); ok = False

    # ------------------------------------------------------------ keyframes
    live_kf, live_calib = load_agents(a.run_dir / "live_agents", "agent")
    hydra_kf, hydra_calib = load_agents(a.prior / "agents", "agent")
    print(f"keyframes  live {len(live_kf)}   hydra {len(hydra_kf)}")
    if not live_kf or live_calib is None:
        print("      FAIL no live keyframes / camera_calib.json"); return 1
    for key in ("fx", "fy", "cx", "cy", "width", "height", "depth_scale", "depth_encoding"):
        if hydra_calib and live_calib.get(key) != hydra_calib.get(key) and not (
            isinstance(live_calib.get(key), float) and abs(live_calib[key] - hydra_calib[key]) < 1e-3
        ):
            print(f"      FAIL calib {key}: live {live_calib.get(key)} vs hydra {hydra_calib.get(key)}"); ok = False
    if hydra_calib:
        d = np.abs(np.asarray(live_calib["body_T_sensor"]) - np.asarray(hydra_calib["body_T_sensor"])).max()
        print(f"      body_T_sensor max |diff| vs hydra: {d:.4f}" + ("" if d < 0.01 else "   FAIL"))
        ok &= d < 0.01
    # pose error at matching timestamps (same TF source, so this should be tiny)
    ht = np.array([t for t, _ in hydra_kf]); errs = []
    for t, T in live_kf:
        if not len(ht):
            break
        j = int(np.argmin(np.abs(ht - t)))
        if abs(ht[j] - t) < 60e6:  # 60 ms
            errs.append(np.linalg.norm(T[:3, 3] - hydra_kf[j][1][:3, 3]))
    if errs:
        print(f"      {len(errs)} live keyframes within 60 ms of a hydra keyframe: pose |dt| median {np.median(errs)*100:.1f} cm, max {max(errs)*100:.1f} cm"
              + ("" if max(errs) < 0.10 else "   FAIL"))
        ok &= max(errs) < 0.10
    # all live keyframe positions must be on cells the live grid knows about
    kx = np.array([T[0, 3] for _, T in live_kf]); ky = np.array([T[1, 3] for _, T in live_kf])
    at = lookup(live_g, live_o, res, kx, ky)
    print(f"      keyframe positions on live grid: {int((at==0).sum())} free, {int((at==100).sum())} occupied, {int((at==-1).sum())} unknown, {int((at==-2).sum())} outside")

    # ------------------------------------------------------------------ png
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rgb = np.full(live_g.shape + (3,), 0.55)
        rgb[live_g == 0] = (0.92, 0.92, 0.92)
        rgb[live_g == 100] = (0.1, 0.1, 0.1)
        nk = new_known.reshape(live_g.shape)
        rgb[nk & (live_g == 0)] = (0.75, 0.95, 0.75)
        rgb[nk & (live_g == 100)] = (0.85, 0.2, 0.2)
        rgb[flips_to_occ.reshape(live_g.shape)] = (1.0, 0.55, 0.0)
        rgb[flips_to_free.reshape(live_g.shape)] = (0.2, 0.6, 1.0)
        ext = (live_o[0, 3], live_o[0, 3] + live_g.shape[1] * res, live_o[1, 3], live_o[1, 3] + live_g.shape[0] * res)
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.imshow(rgb, origin="lower", extent=ext, interpolation="nearest")
        hx = [T[0, 3] for _, T in hydra_kf]; hy = [T[1, 3] for _, T in hydra_kf]
        ax.plot(hx, hy, "kx", ms=4, label=f"hydra agents ({len(hydra_kf)})")
        ax.plot(kx, ky, "b.", ms=5, label=f"live keyframes ({len(live_kf)})")
        ax.set_title("exploration replay: light green/red = newly observed free/occupied, orange = prior free now occupied, blue = prior occupied now free")
        ax.set_aspect("equal"); ax.legend(loc="upper right"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.tight_layout(); fig.savefig(a.png, dpi=130)
        print(f"png   {a.png}")
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
