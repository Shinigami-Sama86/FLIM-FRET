#!/usr/bin/env python3
"""
FRET-positive fraction from amplitude-weighted mean arrival time over bins gate0..gate1.

Workflow:
- Build donor-only reference distribution from D1 + D2 ROI pixels:
    mu_D, sigma_D
- Classify GO pixels as FRET+ if:
    ⟨t⟩ < mu_D - k*sigma_D   (k = 1.5 and 2.0)

Outputs:
- outputs/fret_positive_fraction.txt

Dependencies: numpy
"""

from __future__ import annotations
import os, struct
import numpy as np

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
PARAMS = dict(gate0=148, gate1=228, bg0=450, bg1=499, roi_percentile=90.0, roi_min_pixels=500)

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

def mean_arrival_time_ps(cube: np.ndarray, t_centres_ps: np.ndarray, gate0: int, gate1: int) -> np.ndarray:
    I = cube[..., gate0:gate1+1]
    t = t_centres_ps[gate0:gate1+1]
    denom = np.sum(I, axis=-1)
    denom = np.where(denom == 0, np.nan, denom)
    return np.sum(I * t[None, None, :], axis=-1) / denom

def main():
    ensure_dir(OUTDIR)

    gate0, gate1 = int(PARAMS["gate0"]), int(PARAMS["gate1"])
    bg0, bg1 = int(PARAMS["bg0"]), int(PARAMS["bg1"])

    t_edges = read_lv_time_edges_int32_be(DATA["time_edges"])
    t_centres_ps = (t_edges[:-1] + t_edges[1:]) / 2.0

    cubes = {
        "D1": median_bg_subtract(read_lv_cube_be_f64(DATA["donor_bin_1"]), bg0, bg1),
        "D2": median_bg_subtract(read_lv_cube_be_f64(DATA["donor_bin_2"]), bg0, bg1),
        "GO1": median_bg_subtract(read_lv_cube_be_f64(DATA["go_bin_1"]), bg0, bg1),
        "GO2": median_bg_subtract(read_lv_cube_be_f64(DATA["go_bin_2"]), bg0, bg1),
    }

    roi = {}
    tbar = {}
    for k, cube in cubes.items():
        inten = gated_intensity(cube, gate0, gate1)
        roi[k] = roi_mask_from_intensity(inten, PARAMS["roi_percentile"], PARAMS["roi_min_pixels"])
        tbar[k] = mean_arrival_time_ps(cube, t_centres_ps, gate0, gate1)  # ps

    donor_vals = np.concatenate([tbar["D1"][roi["D1"]].ravel(), tbar["D2"][roi["D2"]].ravel()])
    muD = float(np.nanmean(donor_vals))
    sigD = float(np.nanstd(donor_vals, ddof=1))

    lines = []
    lines.append("FRET-positive fraction by lifetime-index thresholding\n")
    lines.append(f"gate bins: {gate0}..{gate1}\n")
    lines.append(f"Donor-only ⟨t⟩: mu={muD/1000.0:.4f} ns, sigma={sigD/1000.0:.4f} ns\n\n")

    for ksig in (1.5, 2.0):
        thr = muD - ksig * sigD
        lines.append(f"Threshold: ⟨t⟩ < mu - {ksig:.1f}σ  =>  {thr/1000.0:.4f} ns\n")
        for cond in ("GO1", "GO2"):
            m = roi[cond] & (tbar[cond] < thr)
            frac = float(m.sum() / max(1, roi[cond].sum()))
            lines.append(f"  {cond}: FRET+ pixels={m.sum()} / ROI={roi[cond].sum()} => frac={frac:.6f}\n")
        lines.append("\n")

    out_txt = os.path.join(OUTDIR, "fret_positive_fraction.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("Wrote:", out_txt)

if __name__ == "__main__":
    main()
