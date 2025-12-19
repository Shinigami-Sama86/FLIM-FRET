#!/usr/bin/env python3
"""
IRF folding + alignment + reconvolution-based tau fit stability vs fit-start.

Model:
  y(t) ≈ a*IRF(t-Δt) + b*(IRF(t-Δt) * exp(-t/τ)) + c
Solved by grid search over τ (and optionally Δt) with linear least squares for a,b,c.

Outputs:
- outputs/irf_alignment_report.txt
- outputs/E_vs_offset_fixedDt.png
- outputs/E_vs_offset_freeDt.png

Dependencies: numpy, matplotlib
"""

from __future__ import annotations
import os, struct
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG (EDIT THESE)
# =========================
DATA = {
    "donor_bin_1": "data/green test flim 450nm flim bins 1",
    "donor_bin_2": "data/green test flim 450nm flim bin 2",
    "go_bin_1":    "data/mito green orange FLIM 450nm bin 1",
    "go_bin_2":    "data/mito green orange FLIM 450nm bin 2",
    "time_edges":  "data/mito green orange FLIM 450nm time 1",
    "irf_csv":     "data/FLIM_IRF.csv",
}
OUTDIR = "outputs"
PARAMS = dict(
    rep_rate_hz=20e6,
    gate0=148, gate1=228,
    bg0=450, bg1=499,
    roi_percentile=90.0, roi_min_pixels=500,
)

# =========================
# Helpers
# =========================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def read_lv_time_edges_int32_be(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack(">i", f.read(4))[0]
        arr = np.frombuffer(f.read(), dtype=">i4", count=n)
    return arr.astype(float)

def read_lv_cube_be_f64(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        nx, ny, nt = struct.unpack(">3i", f.read(12))
        payload = np.frombuffer(f.read(), dtype=">f8", count=nx*ny*nt)
    if payload.size != nx*ny*nt:
        raise ValueError(f"File size mismatch for LV cube: {path}")
    return payload.reshape((nx, ny, nt)).astype(float)

def median_bg_subtract(cube: np.ndarray, bg0: int, bg1: int) -> np.ndarray:
    bg = np.median(cube[..., bg0:bg1+1], axis=-1)
    out = cube - bg[..., None]
    out[out < 0] = 0.0
    return out

def gated_intensity(cube: np.ndarray, gate0: int, gate1: int) -> np.ndarray:
    return np.sum(cube[..., gate0:gate1+1], axis=-1)

def roi_mask_from_intensity(inten: np.ndarray, percentile: float, min_pixels: int) -> np.ndarray:
    vals = inten.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(inten, dtype=bool)
    p = float(percentile)
    thr = np.nanpercentile(vals, p)
    mask = inten > thr
    while mask.sum() < int(min_pixels) and p > 10:
        p -= 5
        thr = np.nanpercentile(vals, p)
        mask = inten > thr
    return mask

def read_irf_csv(path: str):
    # expects headers containing "BinCenter" and "Histo01" (or similar)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    names = data.dtype.names

    def pick(candidates):
        for c in candidates:
            for n in names:
                if n.strip() == c:
                    return n
        for c in candidates:
            c2 = c.lower().replace(" ", "")
            for n in names:
                if c2 in n.lower().replace(" ", ""):
                    return n
        raise ValueError(f"Could not find {candidates} in columns {names}")

    tcol = pick(["BinCenter (ps)", "BinCenter"])
    ycol = pick(["Histo01", "Counts", "Histo"])
    return np.array(data[tcol], float), np.array(data[ycol], float)

def fold_irf_to_period(t_ps: np.ndarray, y: np.ndarray, time_edges_ps: np.ndarray, rep_rate_hz: float):
    T_ps = 1e12 / rep_rate_hz
    t0 = t_ps[np.argmax(y)]
    t_fold = (t_ps - t0) % T_ps
    irf, _ = np.histogram(t_fold, bins=time_edges_ps, weights=y)
    return np.maximum(irf.astype(float), 0.0)

def coarse_align_by_xcorr(irf: np.ndarray, decay: np.ndarray, search_bins: int = 180, use_bins: int = 350):
    irf0 = irf - np.mean(irf[:50])
    d0 = decay - np.mean(decay[:50])
    irf0 = irf0 / (np.linalg.norm(irf0) + 1e-12)
    d0 = d0 / (np.linalg.norm(d0) + 1e-12)
    best_shift, best_score = 0, -1e9
    for sh in range(-search_bins, search_bins+1):
        sc = float(np.dot(np.roll(irf0, sh)[:use_bins], d0[:use_bins]))
        if sc > best_score:
            best_score, best_shift = sc, sh
    return best_shift, best_score

def _lsq_3col(x1, x2, y):
    ones = np.ones_like(y)
    A = np.vstack([x1, x2, ones]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, c = coef
    rss = float(np.sum((y - A @ coef) ** 2))
    return float(a), float(b), float(c), rss

def reconv_fit_tau_dt_grid(y: np.ndarray, irf: np.ndarray, dt_ps: float,
                           start_bin: int, end_bin: int,
                           tau_ns_grid: np.ndarray,
                           delta_ps_grid: np.ndarray):
    """Grid search over (tau, delta) with FFT conv and LS for (a,b,c)."""
    n = len(y)
    L = 1
    while L < 2*n:
        L *= 2

    y_w = y[start_bin:end_bin+1].astype(float)
    t_ps = np.arange(n) * dt_ps

    exp_ffts = []
    for tau_ns in tau_ns_grid:
        expd = np.exp(-t_ps / (tau_ns * 1000.0))  # tau_ns -> ps
        pad = np.zeros(L); pad[:n] = expd
        exp_ffts.append(np.fft.rfft(pad))
    exp_ffts = np.array(exp_ffts)

    best = None
    x = np.arange(n) * dt_ps

    for delta_ps in delta_ps_grid:
        x_src = x - delta_ps
        irf_shift = np.interp(x_src, x, irf, left=0.0, right=0.0)

        irf_pad = np.zeros(L); irf_pad[:n] = irf_shift
        irf_fft = np.fft.rfft(irf_pad)

        conv = np.fft.irfft(irf_fft[None, :] * exp_ffts, n=L)[:, :n]
        irf_w = irf_shift[start_bin:end_bin+1]

        for i, tau_ns in enumerate(tau_ns_grid):
            conv_w = conv[i, start_bin:end_bin+1]
            a, b, c, rss = _lsq_3col(irf_w, conv_w, y_w)
            if a < 0 or b < 0:
                continue
            if best is None or rss < best["rss"]:
                best = dict(tau_ns=float(tau_ns), delta_ps=float(delta_ps), a=a, b=b, c=c, rss=rss)

    return best

# =========================
# Main
# =========================
def main():
    ensure_dir(OUTDIR)

    gate0, gate1 = int(PARAMS["gate0"]), int(PARAMS["gate1"])
    bg0, bg1 = int(PARAMS["bg0"]), int(PARAMS["bg1"])
    rep_rate_hz = float(PARAMS["rep_rate_hz"])

    t_edges = read_lv_time_edges_int32_be(DATA["time_edges"])
    dt_ps = float(t_edges[1] - t_edges[0])

    # IRF fold
    t_irf_ps, y_irf = read_irf_csv(DATA["irf_csv"])
    irf_fold = fold_irf_to_period(t_irf_ps, y_irf, t_edges, rep_rate_hz)

    # Load + bg subtract cubes
    cubes = {
        "D1": median_bg_subtract(read_lv_cube_be_f64(DATA["donor_bin_1"]), bg0, bg1),
        "D2": median_bg_subtract(read_lv_cube_be_f64(DATA["donor_bin_2"]), bg0, bg1),
        "GO1": median_bg_subtract(read_lv_cube_be_f64(DATA["go_bin_1"]), bg0, bg1),
        "GO2": median_bg_subtract(read_lv_cube_be_f64(DATA["go_bin_2"]), bg0, bg1),
    }

    # ROI pooled decays
    pooled = {}
    roi_pix = {}
    for k, cube in cubes.items():
        inten = gated_intensity(cube, gate0, gate1)
        roi = roi_mask_from_intensity(inten, PARAMS["roi_percentile"], PARAMS["roi_min_pixels"])
        roi_pix[k] = int(roi.sum())
        pooled[k] = np.sum(cube[roi], axis=0)

    # align IRF to each pooled decay
    irf_al = {}
    shift_info = {}
    peaks = {k: int(np.argmax(pooled[k])) for k in pooled}
    for k in pooled:
        sh, sc = coarse_align_by_xcorr(irf_fold, pooled[k], search_bins=180, use_bins=350)
        irf_al[k] = np.roll(irf_fold, sh)
        shift_info[k] = (sh, sc)

    # Fit-start stability scan
    offsets_ns = [0.5, 1, 2, 3, 5, 7.5, 10, 15]
    offsets_bins = [int(round(ns*1000.0/dt_ps)) for ns in offsets_ns]
    end_offset_bins = int(round(20_000.0/dt_ps))  # end = peak + 20 ns

    tau_grid = np.linspace(0.2, 6.0, 160)
    delta_fixed = np.array([0.0])
    delta_free  = np.arange(-500, 501, 50, dtype=float)

    def scan_one(key: str, allow_dt: bool):
        y = pooled[key]
        irf = irf_al[key]
        p = peaks[key]
        end = min(len(y)-1, p + end_offset_bins)
        out = []
        for ns, ob in zip(offsets_ns, offsets_bins):
            start = p + ob
            if start >= end - 10:
                out.append((ns, np.nan, np.nan))
                continue
            best = reconv_fit_tau_dt_grid(
                y=y, irf=irf, dt_ps=dt_ps,
                start_bin=start, end_bin=end,
                tau_ns_grid=tau_grid,
                delta_ps_grid=(delta_free if allow_dt else delta_fixed),
            )
            if best is None:
                out.append((ns, np.nan, np.nan))
            else:
                out.append((ns, best["tau_ns"], best["delta_ps"]))
        return np.array(out, float)  # cols: offset_ns, tau_ns, delta_ps

    def run_mode(allow_dt: bool):
        mode = "freeDt" if allow_dt else "fixedDt"
        sD1 = scan_one("D1", allow_dt)
        sD2 = scan_one("D2", allow_dt)
        tauD = 0.5 * (sD1[:, 1] + sD2[:, 1])

        plt.figure(figsize=(7.2, 4.6))
        for go in ("GO1", "GO2"):
            sGO = scan_one(go, allow_dt)
            E = 1.0 - (sGO[:, 1] / tauD)
            plt.plot(sGO[:, 0], E, marker="o", label=f"{go} ({mode})")
        plt.axhline(0, linestyle="--")
        plt.xlabel("Fit start offset from peak (ns)")
        plt.ylabel("E = 1 - τ_DA/τ_D")
        plt.title(f"Reconvolution efficiency stability ({mode})")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"E_vs_offset_{mode}.png"), dpi=220)
        plt.close()

    run_mode(False)
    run_mode(True)

    # Write alignment report
    rep = []
    rep.append("IRF folding/alignment report\n")
    rep.append(f"rep_rate={rep_rate_hz/1e6:.1f} MHz, dt={dt_ps:.1f} ps\n\n")
    for k in ("D1", "D2", "GO1", "GO2"):
        sh, sc = shift_info[k]
        rep.append(f"{k}: ROI pixels={roi_pix[k]}, peak_bin={peaks[k]}, irf_shift_bins={sh}, score={sc:.4f}\n")
    with open(os.path.join(OUTDIR, "irf_alignment_report.txt"), "w", encoding="utf-8") as f:
        f.writelines(rep)

    print("Wrote outputs/irf_alignment_report.txt and E-vs-offset plots")

if __name__ == "__main__":
    main()
