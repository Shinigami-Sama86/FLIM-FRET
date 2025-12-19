#!/usr/bin/env python3
"""
Windowed phasor (relative) + centroid Δr + windowed mono-exp locus projection
-> tau_D, tau_DA and E = 1 - tau_DA/tau_D
+ distance conversion r(E) using R0.

Dependencies: numpy, matplotlib
"""

from __future__ import annotations
import os, struct, math
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
}
OUTDIR = "outputs"

PARAMS = dict(
    rep_rate_hz=20e6,     # 20 MHz rep-rate
    f_mhz=20.0,           # phasor modulation frequency (MHz)
    gate0=148, gate1=228, # gated window bins
    bg0=450, bg1=499,     # background bins (median subtraction)
    roi_percentile=90.0,  # ROI = top percentile of gated intensity (relaxes if too small)
    roi_min_pixels=500,
    R0_nm=5.0,            # EDIT: Förster radius for distance conversion
)

# =========================
# IO helpers
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

def read_counts_uint32(path: str, nbins: int = 1000, header_ints: int = 3) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint32)
    n_pix = (raw.size - header_ints) // nbins
    if n_pix <= 0:
        raise ValueError(f"Not enough data for uint32 counts format: {path}")
    counts = raw[header_ints:header_ints + n_pix*nbins].reshape((n_pix, nbins))
    return counts.astype(float)

def load_flim_cube(path: str) -> np.ndarray:
    # Robust loader: tries LV cube, else flat uint32 counts
    try:
        return read_lv_cube_be_f64(path)
    except Exception:
        return read_counts_uint32(path)

# =========================
# FLIM helpers
# =========================
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

# =========================
# Phasor helpers (windowed relative)
# =========================
def phasor_window_relative(cube: np.ndarray, t_centres_ps: np.ndarray, f_mhz: float, gate0: int, gate1: int):
    w = 2 * math.pi * (f_mhz * 1e6)
    sl = slice(gate0, gate1+1)
    tt = (t_centres_ps[sl] - t_centres_ps[gate0]) * 1e-12  # seconds, relative to gate start
    I = cube[..., sl]
    denom = np.sum(I, axis=-1)
    denom = np.where(denom == 0, np.nan, denom)
    C = np.cos(w * tt)
    S = np.sin(w * tt)
    g = np.sum(I * C[None, None, :], axis=-1) / denom
    s = np.sum(I * S[None, None, :], axis=-1) / denom
    return g, s

def centroid(g: np.ndarray, s: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    return float(np.nanmean(g[mask])), float(np.nanmean(s[mask]))

def delta_r(c1: tuple[float,float], c2: tuple[float,float]) -> float:
    return float(math.hypot(c1[0]-c2[0], c1[1]-c2[1]))

def windowed_monoexp_locus(t_centres_ps: np.ndarray, f_mhz: float, gate0: int, gate1: int, tau_ns_grid: np.ndarray):
    w = 2 * math.pi * (f_mhz * 1e6)
    tt_ps = t_centres_ps[gate0:gate1+1] - t_centres_ps[gate0]
    tt = tt_ps * 1e-12
    C = np.cos(w * tt)
    S = np.sin(w * tt)
    g_curve = np.zeros_like(tau_ns_grid, dtype=float)
    s_curve = np.zeros_like(tau_ns_grid, dtype=float)
    for i, tau_ns in enumerate(tau_ns_grid):
        tau_s = tau_ns * 1e-9
        I = np.exp(-tt / tau_s)
        denom = np.sum(I)
        g_curve[i] = np.sum(I * C) / denom
        s_curve[i] = np.sum(I * S) / denom
    return g_curve, s_curve

def project_to_locus(g0: float, s0: float, g_curve: np.ndarray, s_curve: np.ndarray, tau_ns_grid: np.ndarray) -> float:
    d2 = (g_curve - g0)**2 + (s_curve - s0)**2
    idx = int(np.nanargmin(d2))
    return float(tau_ns_grid[idx])

def fret_distance_nm(E: float, R0_nm: float) -> float:
    # r = R0 * ((1/E)-1)^(1/6)
    if not np.isfinite(E) or E <= 0:
        return float("nan")
    return float(R0_nm * ((1.0/E) - 1.0) ** (1.0/6.0))

# =========================
# Main
# =========================
def main():
    ensure_dir(OUTDIR)

    gate0, gate1 = int(PARAMS["gate0"]), int(PARAMS["gate1"])
    bg0, bg1 = int(PARAMS["bg0"]), int(PARAMS["bg1"])
    f_mhz = float(PARAMS["f_mhz"])
    R0_nm = float(PARAMS["R0_nm"])

    t_edges = read_lv_time_edges_int32_be(DATA["time_edges"])
    t_centres_ps = (t_edges[:-1] + t_edges[1:]) / 2.0

    cubes = {
        "D1": load_flim_cube(DATA["donor_bin_1"]),
        "D2": load_flim_cube(DATA["donor_bin_2"]),
        "GO1": load_flim_cube(DATA["go_bin_1"]),
        "GO2": load_flim_cube(DATA["go_bin_2"]),
    }
    cubes = {k: median_bg_subtract(v, bg0, bg1) for k,v in cubes.items()}

    roi = {}
    for k, cube in cubes.items():
        inten = gated_intensity(cube, gate0, gate1)
        roi[k] = roi_mask_from_intensity(inten, PARAMS["roi_percentile"], PARAMS["roi_min_pixels"])

    ph = {}
    cent = {}
    for k, cube in cubes.items():
        g, s = phasor_window_relative(cube, t_centres_ps, f_mhz, gate0, gate1)
        ph[k] = (g, s)
        cent[k] = centroid(g, s, roi[k])

    # donor centroid = mean of donor runs
    cD = ((cent["D1"][0] + cent["D2"][0]) / 2.0, (cent["D1"][1] + cent["D2"][1]) / 2.0)
    dr_go1 = delta_r(cent["GO1"], cD)
    dr_go2 = delta_r(cent["GO2"], cD)

    # Windowed mono-exp locus projection for "effective tau" (no IRF)
    tau_grid = np.linspace(0.1, 6.0, 250)  # ns (edit if you expect longer)
    g_curve, s_curve = windowed_monoexp_locus(t_centres_ps, f_mhz, gate0, gate1, tau_grid)

    tauD  = project_to_locus(cD[0], cD[1], g_curve, s_curve, tau_grid)
    tau1  = project_to_locus(cent["GO1"][0], cent["GO1"][1], g_curve, s_curve, tau_grid)
    tau2  = project_to_locus(cent["GO2"][0], cent["GO2"][1], g_curve, s_curve, tau_grid)

    E1 = 1.0 - (tau1 / tauD)
    E2 = 1.0 - (tau2 / tauD)

    r1 = fret_distance_nm(E1, R0_nm)
    r2 = fret_distance_nm(E2, R0_nm)

    # Write summary
    out_txt = os.path.join(OUTDIR, "phasor_deltaR_efficiency_distance.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Windowed phasor (relative gate) summary\n")
        f.write(f"gate bins: {gate0}..{gate1}, f={f_mhz} MHz\n\n")
        for k in ["D1","D2","GO1","GO2"]:
            f.write(f"{k} centroid: g={cent[k][0]:.6f}, s={cent[k][1]:.6f}, ROI pixels={roi[k].sum()}\n")
        f.write(f"\nDonor centroid (mean of D1,D2): g={cD[0]:.6f}, s={cD[1]:.6f}\n")
        f.write(f"Δr(GO1, donor)={dr_go1:.8f}\n")
        f.write(f"Δr(GO2, donor)={dr_go2:.8f}\n\n")
        f.write("Projected effective lifetimes from windowed mono-exp locus (no IRF):\n")
        f.write(f"tau_D   ≈ {tauD:.3f} ns\n")
        f.write(f"tau_GO1 ≈ {tau1:.3f} ns  -> E≈{E1:.3f}\n")
        f.write(f"tau_GO2 ≈ {tau2:.3f} ns  -> E≈{E2:.3f}\n\n")
        f.write(f"Distance conversion using R0={R0_nm:.2f} nm:\n")
        f.write(f"r_GO1 ≈ {r1:.3f} nm\n")
        f.write(f"r_GO2 ≈ {r2:.3f} nm\n")

    # Plot phasor scatter + locus
    plt.figure(figsize=(6.2, 5.4))
    for k in ["D1","D2","GO1","GO2"]:
        g, s = ph[k]
        m = roi[k]
        plt.scatter(g[m], s[m], s=4, alpha=0.35, label=k)
    plt.plot(g_curve, s_curve, linewidth=2, label="Windowed mono-exp locus")
    plt.scatter([cD[0]], [cD[1]], s=90, marker="x", label="Donor centroid")
    plt.xlabel("g")
    plt.ylabel("s")
    plt.title("Windowed (gated) phasor plot (relative)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(OUTDIR, "phasor_windowed_scatter.png")
    plt.savefig(out_png, dpi=220)
    plt.close()

    print("Wrote:", out_txt)
    print("Wrote:", out_png)

if __name__ == "__main__":
    main()
