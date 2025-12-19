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
    pooled_roi_decay,
    load_irf_csv,
    fold_irf_to_period,
    coarse_align_irf,
    scan_fit_start_offsets,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--irf", required=True, help="FLIM_IRF.csv")
    ap.add_argument("--time-edges", required=True, help="Project time file (n then n edges in ps)")
    ap.add_argument("--donor", nargs="+", required=True)
    ap.add_argument("--acceptor", nargs="+", required=True)

    ap.add_argument("--rep-rate", type=float, default=20e6)
    ap.add_argument("--gate", nargs=2, type=int, default=[148, 228])
    ap.add_argument("--bg-bins", nargs=2, type=int, default=[450, 499])
    ap.add_argument("--roi-min-pixels", type=int, default=100)

    ap.add_argument("--offsets-ns", nargs="+", type=float, default=[0.5, 1, 2, 3, 5, 7.5, 10, 15])
    ap.add_argument("--span-ns", type=float, default=35.0)
    ap.add_argument("--tau-min", type=float, default=0.2)
    ap.add_argument("--tau-max", type=float, default=6.0)
    ap.add_argument("--tau-steps", type=int, default=160)

    ap.add_argument("--free-dt", action="store_true", help="Grid-search sub-bin IRF shift Δt as well")
    ap.add_argument("--dt-grid-ps", nargs=3, type=float, default=[-500, 500, 50], help="start stop step (ps)")
    ap.add_argument("--out", default="out_irf_E_stability")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    edges_ps = read_time_edges_be_i32(args.time_edges)
    t_centres_ps = time_centres_from_edges(edges_ps)
    dt_ps = float(np.median(np.diff(t_centres_ps)))

    irf_t_ps, irf_y = load_irf_csv(args.irf)
    irf_fold = fold_irf_to_period(irf_t_ps, irf_y, time_edges_ps=edges_ps, rep_rate_hz=args.rep_rate)

    delta_grid = None
    if args.free_dt:
        a, b, step = args.dt_grid_ps
        delta_grid = np.arange(a, b + 0.5*step, step)

    tau_grid = np.linspace(args.tau_min, args.tau_max, args.tau_steps)
    gate0, gate1 = args.gate

    def process(paths, label):
        rows = []
        for p in paths:
            cube = read_flim_cube_be_i32_f64(p).counts
            cube = background_subtract_median(cube, args.bg_bins[0], args.bg_bins[1])
            decay, roi = pooled_roi_decay(cube, gate0, gate1, min_pixels=args.roi_min_pixels)

            sh, score = coarse_align_irf(irf_fold, decay)
            irf_al = np.roll(irf_fold, sh)

            peak_bin = int(np.argmax(decay))
            scan = scan_fit_start_offsets(
                y=decay, irf=irf_al, dt_ps=dt_ps,
                peak_bin=peak_bin, offsets_ns=args.offsets_ns, span_ns=args.span_ns,
                tau_grid_ns=tau_grid, delta_grid_ps=delta_grid
            )

            for r in scan:
                r.update({
                    "file": Path(p).name,
                    "label": label,
                    "roi_pixels": int(roi.sum()),
                    "irf_shift_bins": int(sh),
                    "irf_align_score": float(score),
                    "peak_bin": peak_bin,
                    "total_counts": float(decay.sum()),
                })
                rows.append(r)
        return pd.DataFrame(rows)

    dfD = process(args.donor, "donor")
    dfA = process(args.acceptor, "donor+acceptor")
    df = pd.concat([dfD, dfA], ignore_index=True)
    df.to_csv(outdir / "reconv_scan_results.csv", index=False)

    don = df[df["label"]=="donor"].pivot_table(index="offset_ns", columns="file", values="tau_ns", aggfunc="mean")
    don_mean = don.mean(axis=1)

    eff_rows = []
    for f, sub in df[df["label"]=="donor+acceptor"].groupby("file"):
        for off, grp in sub.groupby("offset_ns"):
            tauDA = float(grp["tau_ns"].mean())
            tauD = float(don_mean.loc[off]) if off in don_mean.index else np.nan
            E = (1.0 - tauDA/tauD) if (np.isfinite(tauD) and np.isfinite(tauDA) and tauD > 0) else np.nan
            eff_rows.append({"file": f, "offset_ns": float(off), "tauD_ns": tauD, "tauDA_ns": tauDA, "E": E})
    eff = pd.DataFrame(eff_rows)
    eff.to_csv(outdir / "E_vs_offset.csv", index=False)

    plt.figure(figsize=(6.8, 4.8))
    for f, sub in eff.groupby("file"):
        sub = sub.sort_values("offset_ns")
        plt.plot(sub["offset_ns"], sub["E"], marker="o", label=f)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fit start offset from peak (ns)")
    plt.ylabel("E = 1 - τ_DA/τ_D")
    plt.title("E stability vs fit-start offset" + (" (Δt free)" if args.free_dt else " (Δt fixed)"))
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "E_vs_offset.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6.8, 3.9))
    plt.plot(t_centres_ps, irf_fold, linewidth=1.6)
    plt.xlabel("Time (ps)"); plt.ylabel("Folded IRF (normalised)")
    plt.title("Folded IRF on FLIM time grid")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "irf_folded.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
