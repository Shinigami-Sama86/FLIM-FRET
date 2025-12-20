#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

from common_flim import (
    read_flim_cube_be_i32_f64,
    read_time_edges_be_i32,
    time_centres_from_edges,
    background_subtract_median,
    roi_mask_from_gated_intensity,
    tbar_map_ps,
)

def outline(mask: np.ndarray) -> np.ndarray:
    er = binary_erosion(mask)
    return mask & (~er)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--time-edges", required=True)
    ap.add_argument("--gate", nargs=2, type=int, default=[148, 228])
    ap.add_argument("--bg-bins", nargs=2, type=int, default=[450, 499])
    ap.add_argument("--roi-outline", action="store_true")
    ap.add_argument("--out", default="out_tbar_maps")
    ap.add_argument("--unit", choices=["ps","ns"], default="ns")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    edges_ps = read_time_edges_be_i32(args.time_edges)
    t_ps = time_centres_from_edges(edges_ps)

    gate0, gate1 = args.gate

    for p in args.inp:
        cube = read_flim_cube_be_i32_f64(p).counts
        cube = background_subtract_median(cube, args.bg_bins[0], args.bg_bins[1])
        tbar_ps = tbar_map_ps(cube, t_ps, gate0, gate1)
        show = tbar_ps if args.unit=="ps" else (tbar_ps/1000.0)

        np.save(outdir / f"{Path(p).stem}_tbar_{args.unit}.npy", show)

        plt.figure(figsize=(6.4, 5.3))
        im = plt.imshow(show, origin="lower", aspect="equal", vmin=args.vmin, vmax=args.vmax)
        plt.colorbar(im, label=f"⟨t⟩ ({args.unit})")
        plt.title(f"Amplitude-weighted mean time ⟨t⟩ (gate {gate0}–{gate1})\n{Path(p).name}")

        if args.roi_outline:
            roi, _ = roi_mask_from_gated_intensity(cube, gate0, gate1, min_pixels=100)
            ol = outline(roi)
            yy, xx = np.where(ol)
            plt.scatter(xx, yy, s=1)

        plt.tight_layout()
        plt.savefig(outdir / f"{Path(p).stem}_tbar_map.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
