#!/usr/bin/env python3
"""
01_phasor_deltaR.py

Reproduces the paper's phasor centroid shift (Δr) analysis for FLIM-FRET.

- Loads LabVIEW-exported FLIM binary ("bin") + time-edge files.
- Histograms per-pixel photon microtimes into TCSPC decays.
- Background-subtracts (median of late bins).
- Computes per-pixel phasor (g,s) at f = 20 MHz and reports centroid shifts.

Outputs:
  - phasor_scatter.png / .pdf
  - phasor_centroids.csv (centroids and Δr)

Run (defaults are set to the project sandbox paths):
  python 01_phasor_deltaR.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common_flim import (
    load_flim_histogram,
    background_subtract_median,
    phasor_gs,
    universal_semicircle,
)


def weighted_centroid(g: np.ndarray, s: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    w = np.asarray(w, float)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return float(np.nanmean(g)), float(np.nanmean(s))
    return float(np.sum(w * g) / np.sum(w)), float(np.sum(w * s) / np.sum(w))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-bin", default="/mnt/data/green test flim 450nm flim bins 1")
    ap.add_argument("--donor-time", default="/mnt/data/green test flim 450nm Flim times 2")
    ap.add_argument("--go1-bin", default="/mnt/data/mito green orange FLIM 450nm bin 1")
    ap.add_argument("--go1-time", default="/mnt/data/mito green orange FLIM 450nm time 1")
    ap.add_argument("--go2-bin", default="/mnt/data/mito green orange FLIM 450nm bin 2")
    ap.add_argument("--go2-time", default="/mnt/data/mito green orange FLIM 450nm time 2")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--gate0", type=int, default=148, help="gate start bin (inclusive)")
    ap.add_argument("--gate1", type=int, default=228, help="gate end bin (exclusive)")
    ap.add_argument("--bg0", type=int, default=450, help="background start bin (inclusive)")
    ap.add_argument("--bg1", type=int, default=500, help="background end bin (exclusive)")
    ap.add_argument("--fmod-mhz", type=float, default=20.0, help="phasor modulation frequency (MHz)")
    ap.add_argument("--gmin", type=float, default=0.5)
    ap.add_argument("--gmax", type=float, default=1.0)
    ap.add_argument("--smin", type=float, default=0.0)
    ap.add_argument("--smax", type=float, default=0.6)
    ap.add_argument("--max-points", type=int, default=20000, help="max points per condition for plotting")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gate = (args.gate0, args.gate1)
    bg = (args.bg0, args.bg1)
    omega = 2 * np.pi * (args.fmod_mhz * 1e6)  # rad/s

    # Load
    donor = load_flim_histogram(args.donor_bin, args.donor_time, gate_bins_for_scale=gate)
    go1 = load_flim_histogram(args.go1_bin, args.go1_time, gate_bins_for_scale=gate)
    go2 = load_flim_histogram(args.go2_bin, args.go2_time, gate_bins_for_scale=gate)

    # Background-subtract
    d_h = background_subtract_median(donor.hist, bg_bins=bg)
    g1_h = background_subtract_median(go1.hist, bg_bins=bg)
    g2_h = background_subtract_median(go2.hist, bg_bins=bg)    # Phase-correct time axis (per-condition):
    # Use each condition's own global decay peak as an excitation/IRF reference so the
    # universal semicircle is oriented with s >= 0 in all panels.
    def shift_centres_by_peak(hist: np.ndarray, centres_s: np.ndarray) -> np.ndarray:
        peak_bin = int(np.argmax(hist.sum(axis=0)))
        return centres_s - float(centres_s[peak_bin])

    d_g, d_s = phasor_gs(d_h, shift_centres_by_peak(d_h, donor.centres_s), omega, gate_bins=gate)
    g1_g, g1_s = phasor_gs(g1_h, shift_centres_by_peak(g1_h, donor.centres_s), omega, gate_bins=gate)
    g2_g, g2_s = phasor_gs(g2_h, shift_centres_by_peak(g2_h, donor.centres_s), omega, gate_bins=gate)

    # Weights: gate photon counts (post background)
    d_w = d_h[:, gate[0]:gate[1]].sum(axis=1)
    g1_w = g1_h[:, gate[0]:gate[1]].sum(axis=1)
    g2_w = g2_h[:, gate[0]:gate[1]].sum(axis=1)

    d_c = weighted_centroid(d_g, d_s, d_w)
    g1_c = weighted_centroid(g1_g, g1_s, g1_w)
    g2_c = weighted_centroid(g2_g, g2_s, g2_w)

    def delta_r(c):
        return float(np.hypot(c[0] - d_c[0], c[1] - d_c[1]))

    dr1 = delta_r(g1_c)
    dr2 = delta_r(g2_c)

    # Save centroids
    import csv
    with open(outdir / "phasor_centroids.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "g_centroid", "s_centroid", "delta_r_from_donor"])
        w.writerow(["donor", d_c[0], d_c[1], 0.0])
        w.writerow(["green+orange_1", g1_c[0], g1_c[1], dr1])
        w.writerow(["green+orange_2", g2_c[0], g2_c[1], dr2])

    # Subsample for plotting (keeps visuals consistent & light)
    rng = np.random.default_rng(0)

    def subsample(g, s, max_n):
        n = g.size
        if n <= max_n:
            return g, s
        idx = rng.choice(n, size=max_n, replace=False)
        return g[idx], s[idx]

    d_gp, d_sp = subsample(d_g, d_s, args.max_points)
    g1_gp, g1_sp = subsample(g1_g, g1_s, args.max_points)
    g2_gp, g2_sp = subsample(g2_g, g2_s, args.max_points)

    # Plot
    plt.rcParams.update({"figure.dpi": 200, "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.3, 4.8))

    ug, us = universal_semicircle()
    ax.plot(ug, us, linewidth=1.2, label="universal semicircle")

    ax.scatter(d_gp, d_sp, s=8, marker=".", alpha=0.35, label="Green only")
    ax.scatter(g1_gp, g1_sp, s=8, marker=".", alpha=0.35, label="Green+Orange (rep 1)")
    ax.scatter(g2_gp, g2_sp, s=8, marker=".", alpha=0.35, label="Green+Orange (rep 2)")

    ax.scatter([d_c[0]], [d_c[1]], s=60, marker="x", linewidths=2.0, label="centroid (Green only)")
    ax.scatter([g1_c[0]], [g1_c[1]], s=60, marker="x", linewidths=2.0, label=f"centroid (GO1) Δr={dr1:.4f}")
    ax.scatter([g2_c[0]], [g2_c[1]], s=60, marker="x", linewidths=2.0, label=f"centroid (GO2) Δr={dr2:.4f}")

    ax.set_xlabel("g")
    ax.set_ylabel("s")
    ax.set_xlim(args.gmin, args.gmax)
    ax.set_ylim(args.smin, args.smax)
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(outdir / "phasor_scatter.png")
    fig.savefig(outdir / "phasor_scatter.pdf")
    plt.close(fig)

    print("Saved:")
    print(" -", outdir / "phasor_scatter.png")
    print(" -", outdir / "phasor_scatter.pdf")
    print(" -", outdir / "phasor_centroids.csv")
    print()
    print("Centroids:")
    print(" Donor:", d_c)
    print(" GO1  :", g1_c, "Δr=", dr1)
    print(" GO2  :", g2_c, "Δr=", dr2)


if __name__ == "__main__":
    main()

