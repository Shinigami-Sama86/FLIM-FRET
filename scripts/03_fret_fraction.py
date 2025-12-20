#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_flim import (
    read_flim_cube_be_i32_f64,
    read_time_edges_be_i32,
    time_centres_from_edges,
    background_subtract_median,
    roi_mask_from_gated_intensity,
    tbar_map_ps,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", nargs="+", required=True)
    ap.add_argument("--acceptor", nargs="+", required=True)
    ap.add_argument("--time-edges", required=True)
    ap.add_argument("--gate", nargs=2, type=int, default=[148, 228])
    ap.add_argument("--bg-bins", nargs=2, type=int, default=[450, 499])
    ap.add_argument("--sigma", nargs="+", type=float, default=[1.5, 2.0])
    ap.add_argument("--out", default="out_fret_fraction")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    edges_ps = read_time_edges_be_i32(args.time_edges)
    t_ps = time_centres_from_edges(edges_ps)

    gate0, gate1 = args.gate

    def collect(paths):
        vals_list = []
        meta = []
        for p in paths:
            cube = read_flim_cube_be_i32_f64(p).counts
            cube = background_subtract_median(cube, args.bg_bins[0], args.bg_bins[1])
            roi, thr = roi_mask_from_gated_intensity(cube, gate0, gate1, min_pixels=100)
            tbar = tbar_map_ps(cube, t_ps, gate0, gate1)
            vals = tbar[roi].ravel()
            vals = vals[np.isfinite(vals)]
            vals_list.append(vals)
            meta.append({
                "file": Path(p).name,
                "roi_pixels": int(roi.sum()),
                "roi_thr": float(thr),
                "n_vals": int(vals.size),
            })
        return vals_list, pd.DataFrame(meta)

    donor_vals, donor_meta = collect(args.donor)
    acc_vals, acc_meta = collect(args.acceptor)

    D = np.concatenate(donor_vals) if donor_vals else np.array([], float)
    muD = float(np.mean(D)) if D.size else float("nan")
    sigD = float(np.std(D, ddof=1)) if D.size > 1 else float("nan")

    rows = []
    for k in args.sigma:
        thr = muD - float(k)*sigD
        for p, vals in zip(args.acceptor, acc_vals):
            frac = float(np.mean(vals < thr)) if vals.size else float("nan")
            rows.append({
                "file": Path(p).name,
                "k_sigma": float(k),
                "muD_ps": muD,
                "sigD_ps": sigD,
                "threshold_ps": float(thr),
                "fret_positive_fraction": frac,
                "n_roi_pixels": int(vals.size),
            })

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "fret_positive_fraction.csv", index=False)
    donor_meta.to_csv(outdir / "donor_roi_summary.csv", index=False)
    acc_meta.to_csv(outdir / "acceptor_roi_summary.csv", index=False)

    if args.plot and D.size:
        plt.figure(figsize=(6.8, 4.8))
        plt.hist(D/1000.0, bins=60, density=True, alpha=0.4, label="Donor-only (ROI)")
        for p, vals in zip(args.acceptor, acc_vals):
            plt.hist(vals/1000.0, bins=60, density=True, alpha=0.25, label=Path(p).name)
        plt.xlabel("⟨t⟩ (ns) over gated window")
        plt.ylabel("Density")
        plt.title(f"Lifetime-index distributions (gate {gate0}–{gate1})")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / "tbar_distributions.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
