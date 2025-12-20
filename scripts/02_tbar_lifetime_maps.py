#!/usr/bin/env python3
"""
02_tbar_lifetime_maps_paper.py

Produces publication-friendly ⟨t⟩ (mean photon arrival time) maps using the same
lifetime-index definition as the paper:

  ⟨t⟩ = Σ I_i t_i / Σ I_i   over bins 148–228 (active window)

This is the same per-pixel metric used for FRET-positive pixel classification.

Outputs:
  - outputs/tbar_map_green_only.png
  - outputs/tbar_map_green_orange_rep1.png
  - outputs/tbar_map_green_orange_rep2.png
  - outputs/tbar_map_delta_rep1_minus_green.png
  - outputs/tbar_map_delta_rep2_minus_green.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common_flim import (
    load_flim_histogram,
    background_subtract_median,
    tbar_seconds,
)


def save_map(path: Path, arr: np.ndarray, title: str, cbar_label: str) -> None:
    plt.rcParams.update({"figure.dpi": 220, "font.size": 11})
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    im = ax.imshow(arr, origin="lower")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-bin", default="/mnt/data/green test flim 450nm flim bins 1")
    ap.add_argument("--donor-time", default="/mnt/data/green test flim 450nm Flim times 2")
    ap.add_argument("--go1-bin", default="/mnt/data/mito green orange FLIM 450nm bin 1")
    ap.add_argument("--go1-time", default="/mnt/data/mito green orange FLIM 450nm time 1")
    ap.add_argument("--go2-bin", default="/mnt/data/mito green orange FLIM 450nm bin 2")
    ap.add_argument("--go2-time", default="/mnt/data/mito green orange FLIM 450nm time 2")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--gate0", type=int, default=148)
    ap.add_argument("--gate1", type=int, default=228)
    ap.add_argument("--bg0", type=int, default=450)
    ap.add_argument("--bg1", type=int, default=500)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gate = (args.gate0, args.gate1)
    bg = (args.bg0, args.bg1)

    donor = load_flim_histogram(args.donor_bin, args.donor_time, gate_bins_for_scale=gate)
    go1 = load_flim_histogram(args.go1_bin, args.go1_time, gate_bins_for_scale=gate)
    go2 = load_flim_histogram(args.go2_bin, args.go2_time, gate_bins_for_scale=gate)

    d_h = background_subtract_median(donor.hist, bg_bins=bg)
    g1_h = background_subtract_median(go1.hist, bg_bins=bg)
    g2_h = background_subtract_median(go2.hist, bg_bins=bg)

    t_d = tbar_seconds(d_h, donor.centres_s, gate_bins=gate) * 1e9
    t_1 = tbar_seconds(g1_h, go1.centres_s, gate_bins=gate) * 1e9
    t_2 = tbar_seconds(g2_h, go2.centres_s, gate_bins=gate) * 1e9

    # Convert to image arrays: original data is (n_pix,) with pixels ordered row-major (x then y)
    def to_img(v, nx, ny):
        return v.reshape(nx, ny).T  # (ny,nx)

    img_d = to_img(t_d, donor.nx, donor.ny)
    img_1 = to_img(t_1, go1.nx, go1.ny)
    img_2 = to_img(t_2, go2.nx, go2.ny)

    save_map(outdir / "tbar_map_green_only.png", img_d, "Green only (⟨t⟩ map)", "⟨t⟩ (ns)")
    save_map(outdir / "tbar_map_green_orange_rep1.png", img_1, "Green+Orange rep1 (⟨t⟩ map)", "⟨t⟩ (ns)")
    save_map(outdir / "tbar_map_green_orange_rep2.png", img_2, "Green+Orange rep2 (⟨t⟩ map)", "⟨t⟩ (ns)")

    # Δ⟨t⟩ maps only make sense if grids match (some exports differ in nx,ny).
    if img_d.shape == img_1.shape:
        save_map(outdir / "tbar_map_delta_rep1_minus_green.png", img_1 - img_d,
                 "Δ⟨t⟩: rep1 - green", "Δ⟨t⟩ (ns)")
    else:
        print("Skipping Δ⟨t⟩ rep1: grid mismatch", img_1.shape, "vs", img_d.shape)

    if img_d.shape == img_2.shape:
        save_map(outdir / "tbar_map_delta_rep2_minus_green.png", img_2 - img_d,
                 "Δ⟨t⟩: rep2 - green", "Δ⟨t⟩ (ns)")
    else:
        print("Skipping Δ⟨t⟩ rep2: grid mismatch", img_2.shape, "vs", img_d.shape)

    print("Saved maps to:", outdir)


if __name__ == "__main__":
    main()

