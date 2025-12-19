#!/usr/bin/env python3
"""
Load EMCCD spectra from a folder (each file: 2 columns wavelength,intensity),
baseline subtract, normalize area, PCA via SVD.

Outputs:
- outputs/emccd_pca_loadings.png
- outputs/emccd_pca_scores_PC1_PC2.png
- outputs/*.npy arrays (wl, processed spectra, scores, loadings, explained)

Dependencies: numpy, matplotlib
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "outputs"
SPECTRA_DIR = os.environ.get("SPECTRA_DIR", "data/emccd_spectra")  # EDIT or set env var
EXTS = (".csv", ".txt", ".asc")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_spectrum_two_col(path: str):
    data = np.genfromtxt(path, delimiter=None)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected 2+ columns in {path}")
    wl = data[:, 0].astype(float)
    I = data[:, 1].astype(float)
    return wl, I

def load_spectra_folder(folder: str):
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(EXTS)]
    if not files:
        raise ValueError(f"No spectra found in {folder}")
    wl0, I0 = load_spectrum_two_col(files[0])
    X = [I0]
    for fp in files[1:]:
        wl, I = load_spectrum_two_col(fp)
        if wl.shape != wl0.shape or np.max(np.abs(wl - wl0)) > 1e-6:
            I = np.interp(wl0, wl, I, left=np.nan, right=np.nan)
        X.append(I)
    return wl0, np.vstack(X), [os.path.basename(f) for f in files]

def baseline_subtract_poly(wl: np.ndarray, X: np.ndarray, deg: int = 2, fit_frac: float = 0.2):
    n = wl.size
    k = max(10, int(round(fit_frac * n)))
    Xc = X.copy().astype(float)
    for i in range(X.shape[0]):
        y = X[i].astype(float)
        idx = np.argsort(y)[:k]  # lowest intensities
        coef = np.polyfit(wl[idx], y[idx], deg=deg)
        base = np.polyval(coef, wl)
        Xc[i] = y - base
    return Xc

def normalize_area(X: np.ndarray):
    Xn = X.copy().astype(float)
    for i in range(Xn.shape[0]):
        y = np.where(np.isfinite(Xn[i]), Xn[i], 0.0)
        y = np.maximum(y, 0.0)
        area = float(np.sum(y))
        Xn[i] = (y / area) if area > 0 else y
    return Xn

def pca_svd(X: np.ndarray, n_components: int = 3, center: bool = True):
    X = np.array(X, dtype=float)
    mean = np.mean(X, axis=0) if center else np.zeros(X.shape[1], dtype=float)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    loadings = Vt[:n_components]
    scores = U[:, :n_components] * S[:n_components]
    var = (S**2) / (X.shape[0] - 1)
    explained = var[:n_components] / np.sum(var)
    return scores, loadings, explained, mean

def main():
    ensure_dir(OUTDIR)

    wl, Xraw, names = load_spectra_folder(SPECTRA_DIR)
    Xc = baseline_subtract_poly(wl, Xraw, deg=2, fit_frac=0.2)
    Xn = normalize_area(Xc)

    scores, loadings, explained, mean = pca_svd(Xn, n_components=3, center=True)

    np.save(os.path.join(OUTDIR, "emccd_wavelength_nm.npy"), wl)
    np.save(os.path.join(OUTDIR, "emccd_spectra_raw.npy"), Xraw)
    np.save(os.path.join(OUTDIR, "emccd_spectra_processed.npy"), Xn)
    np.save(os.path.join(OUTDIR, "pca_scores.npy"), scores)
    np.save(os.path.join(OUTDIR, "pca_loadings.npy"), loadings)
    np.save(os.path.join(OUTDIR, "pca_explained.npy"), explained)

    plt.figure(figsize=(7, 4.5))
    plt.plot(wl, np.mean(Xn, axis=0), label="Mean (processed)")
    for i in range(3):
        plt.plot(wl, loadings[i], label=f"PC{i+1} loading ({explained[i]*100:.1f}%)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Arb. units (processed)")
    plt.title("EMCCD spectra PCA loadings (SVD)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "emccd_pca_loadings.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(scores[:, 0], scores[:, 1], s=40)
    for i, n in enumerate(names):
        plt.text(scores[i, 0], scores[i, 1], n, fontsize=7)
    plt.xlabel("PC1 score")
    plt.ylabel("PC2 score")
    plt.title("EMCCD PCA scores")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "emccd_pca_scores_PC1_PC2.png"), dpi=220)
    plt.close()

    print("Saved PCA outputs to outputs/")

if __name__ == "__main__":
    main()
