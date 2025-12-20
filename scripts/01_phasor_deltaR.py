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
    windowed_relative_phasor,
    centroid,
    delta_r,
    monoexp_relative_locus,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", nargs="+", required=True)
    ap.add_argument("--acceptor", nargs="+", required=True)
    ap.add_argument("--time-edges", required=True, help="Project time file (big-endian i32: n then n edges in ps)")
    ap.add_argument("--rep-rate", type=float, default=20e6)
    ap.add_argument("--gate", nargs=2, type=int, default=[148, 228])
    ap.add_argument("--bg-bins", nargs=2, type=int, default=[450, 499])
    ap.add_argument("--roi-min-pixels", type=int, default=100)
    ap.add_argument("--max-points", type=int, default=60000)
    ap.add_argument("--out", default="out_phasor")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    edges_ps = read_time_edges_be_i32(args.time_edges)
    t_ps = time_centres_from_edges(edges_ps)

    gate0, gate1 = args.gate

    def process(paths, label):
        rows, pts = [], []
        for p in paths:
            cube = read_flim_cube_be_i32_f64(p).counts
            cube = background_subtract_median(cube, args.bg_bins[0], args.bg_bins[1])
            roi, thr = roi_mask_from_gated_intensity(cube, gate0, gate1, min_pixels=args.roi_min_pixels)
            g, s = windowed_relative_phasor(cube, t_ps, rep_rate_hz=args.rep_rate, gate0=gate0, gate1=gate1)
            cg, cs = centroid(g, s, roi)
            rows.append({
                "file": Path(p).name, "label": label,
                "g": cg, "s": cs,
                "roi_pixels": int(roi.sum()), "roi_thr": float(thr)
            })
            gv, sv = g[roi].ravel(), s[roi].ravel()
            m = np.isfinite(gv) & np.isfinite(sv)
            gv, sv = gv[m], sv[m]
            if gv.size > args.max_points:
                idx = np.random.default_rng(0).choice(gv.size, size=args.max_points, replace=False)
                gv, sv = gv[idx], sv[idx]
            pts.append((gv, sv, Path(p).name))
        return pd.DataFrame(rows), pts

    d_df, d_pts = process(args.donor, "donor")
    a_df, a_pts = process(args.acceptor, "donor+acceptor")

    donor_mean = (float(d_df["g"].mean()), float(d_df["s"].mean()))
    d_df["delta_r_vs_donor_mean"] = d_df.apply(lambda r: delta_r((r["g"], r["s"]), donor_mean), axis=1)
    a_df["delta_r_vs_donor_mean"] = a_df.apply(lambda r: delta_r((r["g"], r["s"]), donor_mean), axis=1)

    cent = pd.concat([d_df, a_df], ignore_index=True)
    cent.to_csv(outdir / "phasor_centroids.csv", index=False)

    taus = np.logspace(-1, 1, 200)  # 0.1–10 ns
    g_l, s_l = monoexp_relative_locus(t_ps, rep_rate_hz=args.rep_rate, gate0=gate0, gate1=gate1, taus_ns=taus)

    plt.figure(figsize=(6.6, 5.4))
    plt.plot(g_l, s_l, linewidth=1.6, label="Relative mono-exp locus (0.1–10 ns)")
    for gv, sv, name in d_pts:
        plt.scatter(gv, sv, s=2, alpha=0.25, label=f"Donor pixels: {name}")
    for gv, sv, name in a_pts:
        plt.scatter(gv, sv, s=2, alpha=0.25, label=f"D+A pixels: {name}")
    plt.scatter([donor_mean[0]], [donor_mean[1]], s=90, marker="x", label="Donor mean centroid")
    for _, r in a_df.iterrows():
        plt.scatter([r["g"]], [r["s"]], s=70, marker="o",
                    label=f"{r['file']} centroid (Δr={r['delta_r_vs_donor_mean']:.4f})")
    plt.xlabel("g"); plt.ylabel("s")
    plt.title(f"Windowed-relative phasor (gate {gate0}–{gate1})")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1); plt.ylim(0, 0.8)
    plt.legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "phasor_plot.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()


