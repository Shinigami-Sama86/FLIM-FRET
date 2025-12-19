#!/usr/bin/env python3
"""
Amplitude-weighted mean arrival time maps over bins gate0..gate1:
  ⟨t⟩ = Σ I_i t_i / Σ I_i

Outputs:
- outputs/awlt_<dataset>.png
- outputs/awlt_<dataset>_ROI.png
- outputs/awlt_<dataset>.npy (numerical map in ns)

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

def save_map(path: str, img: np.ndarray, title: str):
    plt.figure(figsize=(6.2, 5.2))
    plt.imshow(img, origin="lower")
    plt.colorbar(label="⟨t⟩ (ns)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def main():
    ensure_dir(OUTDIR)

    gate0, gate1 = int(PARAMS["gate0"]), int(PARAMS["gate1"])
    bg0, bg1 = int(PARAMS["bg0"]), int(PARAMS["bg1"])

    t_edges = read_lv_time_edges_int32_be(DATA["time_edges"])
    t_centres_ps = (t_edges[:-1] + t_edges[1:]) / 2.0

    cubes = {
        "D1": read_lv_cube_be_f64(DATA["donor_bin_1"]),
        "D2": read_lv_cube_be_f64(DATA["donor_bin_2"]),
        "GO1": read_lv_cube_be_f64(DATA["go_bin_1"]),
        "GO2": read_lv_cube_be_f64(DATA["go_bin_2"]),
    }
    cubes = {k: median_bg_subtract(v, bg0, bg1) for k,v in cubes.items()}

    for k, cube in cubes.items():
        inten = gated_intensity(cube, gate0, gate1)
        roi = roi_mask_from_intensity(inten, PARAMS["roi_percentile"], PARAMS["roi_min_pixels"])

        tbar_ns = mean_arrival_time_ps(cube, t_centres_ps, gate0, gate1) / 1000.0
        save_map(os.path.join(OUTDIR, f"awlt_{k}.png"), tbar_ns,
                 f"{k}: ⟨t⟩ map (bins {gate0}-{gate1})")

        tbar_roi = tbar_ns.copy()
        tbar_roi[~roi] = np.nan
        save_map(os.path.join(OUTDIR, f"awlt_{k}_ROI.png"), tbar_roi,
                 f"{k}: ⟨t⟩ map (ROI only)")

        np.save(os.path.join(OUTDIR, f"awlt_{k}.npy"), tbar_ns)

    print("Saved AWLT maps to outputs/")

if __name__ == "__main__":
    main()

