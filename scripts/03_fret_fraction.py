#!/usr/bin/env python3
"""
03_fret_fraction_paper.py

Reproduces the paper's pixel-wise FRET-positive fraction analysis and the "global"
efficiency estimates from background-subtracted MTG decays.

Key steps (per paper text):
- Histogram per-pixel photon microtimes into 500-bin TCSPC decays.
- Background subtract (per-pixel constant, median of late bins).
- Compute amplitude-weighted mean photon arrival time ⟨t⟩ over the active window
  (bins 148–228) as a lifetime index.
- Use donor-only (Green only) ⟨t⟩ distribution (μD, σD) to classify FRET+ pixels:
    ⟨t⟩ < μD - k σD   (k = 1.5 default; also report k = 2.0).
- Global efficiency E from background-subtracted MTG decays:
    1) Sum decays over all pixels (global decay).
    2) Find the peak bin; compute mean arrival time in a tail window after the peak,
       using relative time (t - t_peak).
    3) E = 1 - (t̄_tail,DA / t̄_tail,D)

Outputs:
  - outputs/fret_fraction_summary.csv
  - outputs/tbar_distributions.png / .pdf
  - outputs/tbar_maps.png (optional small overview)

Run:
  python 03_fret_fraction_paper.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from common_flim import (
    load_flim_histogram,
    background_subtract_median,
    tbar_seconds,
)


def global_tail_efficiency(
    donor_hist: np.ndarray,
    acceptor_hist: np.ndarray,
    centres_s: np.ndarray,
    tail_start_offset_bins: int = 10,
    tail_length_bins: int = 100,
) -> float:
    """
    Compute global efficiency using a mean-arrival-time index on the post-peak tail.
    """
    d = donor_hist.sum(axis=0).astype(np.float64)
    a = acceptor_hist.sum(axis=0).astype(np.float64)

    peak_d = int(np.argmax(d))
    peak_a = int(np.argmax(a))

    # tail window is defined relative to each peak to reduce timing-offset sensitivity
    def tail_tbar(decay, peak):
        start = peak + tail_start_offset_bins
        end = min(decay.size, start + tail_length_bins)
        if end <= start + 5:
            return np.nan
        t_rel = (centres_s[start:end] - centres_s[peak])  # seconds
        y = np.clip(decay[start:end], 0.0, None)
        if y.sum() <= 0:
            return np.nan
        return float((y * t_rel).sum() / y.sum())

    td = tail_tbar(d, peak_d)
    ta = tail_tbar(a, peak_a)
    if not (np.isfinite(td) and np.isfinite(ta) and td > 0):
        return np.nan
    return float(1.0 - (ta / td))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-bin1", default="/mnt/data/green test flim 450nm flim bins 1")
    ap.add_argument("--donor-bin2", default="/mnt/data/green test flim 450nm flim bin 2")
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
    ap.add_argument("--k", type=float, nargs="*", default=[1.5, 2.0], help="sigma thresholds to report")
    ap.add_argument("--tail-start-offset-bins", type=int, default=59, help="bins after peak for global tail index")
    ap.add_argument("--tail-length-bins", type=int, default=90, help="tail window length (bins)")
    ap.add_argument("--save-maps", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gate = (args.gate0, args.gate1)
    bg = (args.bg0, args.bg1)

    # Load datasets
    d1 = load_flim_histogram(args.donor_bin1, args.donor_time, gate_bins_for_scale=gate)
    d2 = load_flim_histogram(args.donor_bin2, args.donor_time, gate_bins_for_scale=gate)
    go1 = load_flim_histogram(args.go1_bin, args.go1_time, gate_bins_for_scale=gate)
    go2 = load_flim_histogram(args.go2_bin, args.go2_time, gate_bins_for_scale=gate)

    # Background subtract (per pixel)
    d1_h = background_subtract_median(d1.hist, bg_bins=bg)
    d2_h = background_subtract_median(d2.hist, bg_bins=bg)
    go1_h = background_subtract_median(go1.hist, bg_bins=bg)
    go2_h = background_subtract_median(go2.hist, bg_bins=bg)

    # Per-pixel lifetime index ⟨t⟩ in the active window
    tbar_d = np.concatenate([
        tbar_seconds(d1_h, d1.centres_s, gate_bins=gate),
        tbar_seconds(d2_h, d2.centres_s, gate_bins=gate),
    ])
    tbar_go1 = tbar_seconds(go1_h, go1.centres_s, gate_bins=gate)
    tbar_go2 = tbar_seconds(go2_h, go2.centres_s, gate_bins=gate)

    mu_d = float(np.mean(tbar_d))
    sigma_d = float(np.std(tbar_d, ddof=1))

    # Global efficiencies from post-peak tail mean-arrival-time
    E_global_go1 = global_tail_efficiency(d1_h, go1_h, d1.centres_s,
                                          tail_start_offset_bins=args.tail_start_offset_bins,
                                          tail_length_bins=args.tail_length_bins)
    E_global_go2 = global_tail_efficiency(d1_h, go2_h, d1.centres_s,
                                          tail_start_offset_bins=args.tail_start_offset_bins,
                                          tail_length_bins=args.tail_length_bins)

    rows = []

    def add_rows(label, tbar_go):
        # Also report "subset efficiency" based on mean ⟨t⟩ in classified FRET+ pixels
        for k in args.k:
            thr = mu_d - float(k) * sigma_d
            mask = tbar_go < thr
            frac = float(mask.mean())
            mean_pos = float(np.mean(tbar_go[mask])) if mask.any() else float("nan")
            E_pos = float(1.0 - (mean_pos / mu_d)) if mask.any() else float("nan")
            rows.append({
                "condition": label,
                "k_sigma": float(k),
                "mu_D_ns": mu_d * 1e9,
                "sigma_D_ns": sigma_d * 1e9,
                "threshold_ns": thr * 1e9,
                "fret_pos_fraction": frac,
                "mean_tbar_fretpos_ns": mean_pos * 1e9 if np.isfinite(mean_pos) else np.nan,
                "E_from_fretpos_tbar": E_pos,
            })

    add_rows("green+orange_1", tbar_go1)
    add_rows("green+orange_2", tbar_go2)

    # Add global E rows
    rows.append({"condition": "green+orange_1", "k_sigma": "global_tail",
                 "mu_D_ns": mu_d * 1e9, "sigma_D_ns": sigma_d * 1e9,
                 "threshold_ns": "", "fret_pos_fraction": "",
                 "mean_tbar_fretpos_ns": "", "E_from_fretpos_tbar": E_global_go1})
    rows.append({"condition": "green+orange_2", "k_sigma": "global_tail",
                 "mu_D_ns": mu_d * 1e9, "sigma_D_ns": sigma_d * 1e9,
                 "threshold_ns": "", "fret_pos_fraction": "",
                 "mean_tbar_fretpos_ns": "", "E_from_fretpos_tbar": E_global_go2})

    # Save CSV summary
    out_csv = outdir / "fret_fraction_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["condition", "k_sigma", "mu_D_ns", "sigma_D_ns", "threshold_ns",
                      "fret_pos_fraction", "mean_tbar_fretpos_ns", "E_from_fretpos_tbar"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot distributions
    plt.rcParams.update({"figure.dpi": 200, "font.size": 11})
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Use ns for readability
    d_ns = tbar_d * 1e9
    go1_ns = tbar_go1 * 1e9
    go2_ns = tbar_go2 * 1e9

    bins = np.linspace(min(d_ns.min(), go1_ns.min(), go2_ns.min()),
                       max(d_ns.max(), go1_ns.max(), go2_ns.max()), 60)

    ax.hist(d_ns, bins=bins, density=True, alpha=0.35, label="Green only (donor ref)")
    ax.hist(go1_ns, bins=bins, density=True, alpha=0.35, label="Green+Orange (rep 1)")
    ax.hist(go2_ns, bins=bins, density=True, alpha=0.35, label="Green+Orange (rep 2)")

    for k in args.k:
        thr = (mu_d - float(k) * sigma_d) * 1e9
        ax.axvline(thr, linewidth=1.2, linestyle="--", label=f"threshold: μD - {k}σD")

    ax.set_xlabel("⟨t⟩ over bins 148–228 (ns)")
    ax.set_ylabel("probability density")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "tbar_distributions.png")
    fig.savefig(outdir / "tbar_distributions.pdf")
    plt.close(fig)

    # Optional: quick-look maps
    if args.save_maps:
        def to_map(tbar, nx, ny):
            return tbar.reshape(nx, ny).T  # transpose for visual (y,x)

        fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=200)
        for ax_, arr, title in [
            (axs[0], to_map(tbar_seconds(d1_h, d1.centres_s, gate_bins=gate), d1.nx, d1.ny) * 1e9, "Green only"),
            (axs[1], to_map(tbar_go1, go1.nx, go1.ny) * 1e9, "Green+Orange rep1"),
            (axs[2], to_map(tbar_go2, go2.nx, go2.ny) * 1e9, "Green+Orange rep2"),
        ]:
            im = ax_.imshow(arr, origin="lower")
            ax_.set_title(title)
            ax_.set_xticks([])
            ax_.set_yticks([])
            plt.colorbar(im, ax=ax_, fraction=0.046, pad=0.02, label="⟨t⟩ (ns)")
        fig.tight_layout()
        fig.savefig(outdir / "tbar_maps.png")
        plt.close(fig)

    # Console summary (paper-facing)
    print("Donor reference (combined Green-only):")
    print(f"  μD = {mu_d*1e9:.4f} ns, σD = {sigma_d*1e9:.4f} ns")
    print()
    print("Global efficiencies from background-subtracted MTG decays (post-peak tail index):")
    print(f"  Green+Orange rep1: E = {E_global_go1*100:.2f} %")
    print(f"  Green+Orange rep2: E = {E_global_go2*100:.2f} %")
    print()
    for k in args.k:
        thr = mu_d - float(k) * sigma_d
        mask1 = tbar_go1 < thr
        mask2 = tbar_go2 < thr
        Epos1 = 1 - (tbar_go1[mask1].mean() / mu_d) if mask1.any() else np.nan
        Epos2 = 1 - (tbar_go2[mask2].mean() / mu_d) if mask2.any() else np.nan
        print(f"Pixel-wise FRET+ fraction (k={k}σ):")
        print(f"  rep1: fraction={mask1.mean()*100:.2f}% ; E(FRET+)={Epos1*100:.2f}%")
        print(f"  rep2: fraction={mask2.mean()*100:.2f}% ; E(FRET+)={Epos2*100:.2f}%")

    print()
    print("Saved:")
    print(" -", out_csv)
    print(" -", outdir / "tbar_distributions.png")
    print(" -", outdir / "tbar_distributions.pdf")
    if args.save_maps:
        print(" -", outdir / "tbar_maps.png")


if __name__ == "__main__":
    main()

