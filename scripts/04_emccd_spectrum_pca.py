#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_flim import load_spectrum_csv, pca_svd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--wl-col", type=int, default=0)
    ap.add_argument("--y-col", type=int, default=1)
    ap.add_argument("--sep", default=None)
    ap.add_argument("--wl-min", type=float, default=None)
    ap.add_argument("--wl-max", type=float, default=None)
    ap.add_argument("--wl-step", type=float, default=1.0)
    ap.add_argument("--baseline-quantile", type=float, default=0.05)
    ap.add_argument("--norm", choices=["area","max","none"], default="area")
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--out", default="out_emccd_pca")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {indir}")

    wls, ys = [], []
    for f in files:
        wl, y = load_spectrum_csv(f, wl_col=args.wl_col, y_col=args.y_col, sep=args.sep)
        wls.append(wl); ys.append(y)

    wl_all = np.concatenate(wls)
    wl_min = float(np.nanmin(wl_all)) if args.wl_min is None else float(args.wl_min)
    wl_max = float(np.nanmax(wl_all)) if args.wl_max is None else float(args.wl_max)
    wl_grid = np.arange(wl_min, wl_max + args.wl_step, args.wl_step)

    X = []
    for wl, y in zip(wls, ys):
        yy = np.interp(wl_grid, wl, y, left=np.nan, right=np.nan)
        finite = yy[np.isfinite(yy)]
        base = float(np.nanquantile(finite, args.baseline_quantile)) if finite.size else 0.0
        yy = yy - base
        yy[yy < 0] = 0
        if args.norm == "max":
            yy = yy / (np.nanmax(yy) + 1e-12)
        elif args.norm == "area":
            yy = yy / (np.nansum(yy) + 1e-12)
        X.append(yy)

    X = np.vstack(X)
    pca = pca_svd(X, n_components=args.n_components)

    pd.DataFrame({"wavelength_nm": wl_grid}).to_csv(outdir / "wavelength_grid.csv", index=False)
    pd.DataFrame(X, index=[f.name for f in files]).to_csv(outdir / "spectra_matrix.csv")
    pd.DataFrame(pca["components"], columns=[f"{w:.2f}" for w in wl_grid]).to_csv(outdir / "pca_components.csv", index=False)
    pd.DataFrame(pca["scores"], index=[f.name for f in files],
                 columns=[f"PC{i+1}" for i in range(pca["scores"].shape[1])]).to_csv(outdir / "pca_scores.csv")
    pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(pca["explained_variance_ratio"].size)],
        "explained_variance_ratio": pca["explained_variance_ratio"],
    }).to_csv(outdir / "pca_explained_variance.csv", index=False)

    plt.figure(figsize=(6.8, 4.8))
    for i, f in enumerate(files):
        plt.plot(wl_grid, X[i], alpha=0.6, label=f.name)
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Preprocessed intensity (normalised)")
    plt.title("EMCCD spectra (preprocessed)")
    plt.legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "spectra_preprocessed.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6.8, 4.8))
    for k in range(min(3, pca["components"].shape[0])):
        plt.plot(wl_grid, pca["components"][k], label=f"PC{k+1} ({pca['explained_variance_ratio'][k]*100:.1f}%)")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Loading (a.u.)")
    plt.title("PCA loadings")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / "pca_loadings.png", dpi=300)
    plt.close()

    if pca["scores"].shape[1] >= 2:
        plt.figure(figsize=(5.8, 5.0))
        plt.scatter(pca["scores"][:, 0], pca["scores"][:, 1], s=70)
        for i, f in enumerate(files):
            plt.text(pca["scores"][i, 0], pca["scores"][i, 1], f.stem, fontsize=8)
        plt.xlabel("PC1 score"); plt.ylabel("PC2 score")
        plt.title("PCA scores (PC1 vs PC2)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "pca_scores_PC1_PC2.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
