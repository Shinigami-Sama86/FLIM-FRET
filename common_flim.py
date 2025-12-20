from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# ============================
# IO: confirmed project format
# ============================

@dataclass
class FLIMCube:
    nx: int
    ny: int
    nbins: int
    counts: np.ndarray  # (ny, nx, nbins), float64


def read_flim_cube_be_i32_f64(path: str | Path) -> FLIMCube:
    '''
    Project FLIM cube format (confirmed by inspection):
      header: 3x big-endian int32: nx, ny, nbins
      payload: big-endian float64 counts, length nx*ny*nbins
      reshape -> (ny, nx, nbins)
    '''
    path = Path(path)
    with path.open("rb") as f:
        hdr = f.read(12)
        if len(hdr) != 12:
            raise ValueError(f"{path} too short for 12-byte header")
        nx, ny, nbins = struct.unpack(">3i", hdr)
        payload = np.frombuffer(f.read(), dtype=">f8")
    expected = nx * ny * nbins
    if payload.size != expected:
        raise ValueError(
            f"{path}: payload size {payload.size} != expected {expected} (nx,ny,nbins={nx,ny,nbins})"
        )
    counts = payload.reshape((ny, nx, nbins)).astype(np.float64, copy=False)
    return FLIMCube(nx=nx, ny=ny, nbins=nbins, counts=counts)


def read_time_edges_be_i32(path: str | Path) -> np.ndarray:
    '''
    Project time file format (confirmed by inspection):
      first: big-endian int32 n
      then: n x big-endian int32 values (ps), typically edges (length nbins+1)
    Returns float64 array in ps of length n.
    '''
    path = Path(path)
    raw = np.fromfile(path, dtype=">i4")
    if raw.size < 2:
        raise ValueError(f"{path}: too short to be time file")
    n = int(raw[0])
    arr = raw[1:].astype(np.float64, copy=False)
    if arr.size != n:
        raise ValueError(f"{path}: expected {n} values after leading n, got {arr.size}")
    return arr


def time_centres_from_edges(edges_ps: np.ndarray) -> np.ndarray:
    edges_ps = np.asarray(edges_ps, dtype=np.float64)
    if edges_ps.ndim != 1 or edges_ps.size < 2:
        raise ValueError("edges_ps must be 1D with >=2 elements")
    return (edges_ps[:-1] + edges_ps[1:]) / 2.0


# ============================
# Preprocess / ROI
# ============================

def background_subtract_median(cube: np.ndarray, b0: int, b1: int) -> np.ndarray:
    '''
    Subtract per-pixel median of bins [b0:b1] (inclusive), clip at 0.
    '''
    bg = np.median(cube[..., b0:b1+1], axis=-1)
    out = cube - bg[..., None]
    out[out < 0] = 0
    return out


def otsu_threshold(values: np.ndarray, bins: int = 128) -> float:
    '''
    Otsu threshold on log1p(values). Returns threshold in original units.
    '''
    x = values[np.isfinite(values)]
    if x.size == 0:
        return 0.0
    z = np.log1p(np.maximum(x, 0))
    zmin, zmax = float(z.min()), float(z.max())
    if zmax <= zmin:
        return float(np.expm1(zmin))
    hist, edges = np.histogram(z, bins=bins, range=(zmin, zmax))
    p = hist.astype(float) / (hist.sum() + 1e-12)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2.0)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    thr_z = float((edges[idx] + edges[idx+1]) / 2.0)
    return float(np.expm1(thr_z))


def roi_mask_from_gated_intensity(
    cube: np.ndarray,
    gate0: int,
    gate1: int,
    min_pixels: int = 100,
) -> Tuple[np.ndarray, float]:
    '''
    ROI from integrated intensity over [gate0:gate1] using Otsu threshold.
    Falls back to 90th percentile if too few pixels.
    '''
    inten = cube[..., gate0:gate1+1].sum(axis=-1)
    thr = otsu_threshold(inten.ravel())
    mask = inten > thr
    if int(mask.sum()) < min_pixels:
        q = float(np.nanquantile(inten.ravel(), 0.90))
        mask = inten > q
    return mask, float(thr)


# ============================
# Lifetime index: amplitude-weighted mean time
# ============================

def tbar_map_ps(cube: np.ndarray, t_centres_ps: np.ndarray, gate0: int, gate1: int) -> np.ndarray:
    sl = slice(gate0, gate1+1)
    I = cube[..., sl]
    t = t_centres_ps[sl]
    denom = I.sum(axis=-1)
    denom = np.where(denom == 0, np.nan, denom)
    return (I * t[None, None, :]).sum(axis=-1) / denom


# ============================
# Windowed-relative phasor
# ============================

def windowed_relative_phasor(
    cube: np.ndarray,
    t_centres_ps: np.ndarray,
    rep_rate_hz: float,
    gate0: int,
    gate1: int
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Phasor computed on a windowed decay where t=0 is set to the first gate bin centre.
    '''
    w = 2.0 * math.pi * float(rep_rate_hz)
    sl = slice(gate0, gate1+1)
    t0 = float(t_centres_ps[gate0])
    t_s = (t_centres_ps[sl] - t0) * 1e-12
    I = cube[..., sl]
    C = np.cos(w * t_s)
    S = np.sin(w * t_s)
    denom = I.sum(axis=-1)
    denom = np.where(denom == 0, np.nan, denom)
    g = (I * C[None, None, :]).sum(axis=-1) / denom
    s = (I * S[None, None, :]).sum(axis=-1) / denom
    return g, s


def centroid(g: np.ndarray, s: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    gv, sv = g[mask], s[mask]
    return float(np.nanmean(gv)), float(np.nanmean(sv))


def delta_r(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    return float(math.hypot(c1[0] - c2[0], c1[1] - c2[1]))


def monoexp_relative_locus(
    t_centres_ps: np.ndarray,
    rep_rate_hz: float,
    gate0: int,
    gate1: int,
    taus_ns: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Relative mono-exponential locus for the chosen gate window, using the same
    "t=0 at first gate bin" convention as windowed_relative_phasor().
    '''
    w = 2.0 * math.pi * float(rep_rate_hz)
    t0 = float(t_centres_ps[gate0])
    t = (t_centres_ps[gate0:gate1+1] - t0) * 1e-12
    C = np.cos(w * t)
    S = np.sin(w * t)
    gs, ss = [], []
    for tau_ns in np.asarray(taus_ns, float):
        tau_ps = tau_ns * 1000.0
        I = np.exp(-(t_centres_ps[gate0:gate1+1] - t0) / tau_ps)
        denom = I.sum()
        gs.append(float((I * C).sum() / denom))
        ss.append(float((I * S).sum() / denom))
    return np.asarray(gs), np.asarray(ss)


# ============================
# FRET E <-> distance
# ============================

def E_from_taus(tauD: float, tauDA: float) -> float:
    return float(1.0 - tauDA / tauD)


def E_from_distance(r_nm: float, R0_nm: float) -> float:
    return float(1.0 / (1.0 + (float(r_nm) / float(R0_nm))**6))


def distance_from_E(E: float, R0_nm: float) -> float:
    E = float(E)
    if E <= 0:
        return float("nan")
    if E >= 1:
        return 0.0
    return float(R0_nm * ((1.0 / E) - 1.0)**(1.0 / 6.0))


# ============================
# Spectra + PCA (SVD)
# ============================

def load_spectrum_csv(path: str | Path, wl_col=0, y_col=1, sep: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep=sep)
    wl = df.iloc[:, wl_col].to_numpy(float) if isinstance(wl_col, int) else df[wl_col].to_numpy(float)
    y = df.iloc[:, y_col].to_numpy(float) if isinstance(y_col, int) else df[y_col].to_numpy(float)
    return wl, y


def pca_svd(X: np.ndarray, n_components: int = 5) -> dict:
    X = np.asarray(X, float)
    mean = np.nanmean(X, axis=0)
    Xc = np.nan_to_num(X - mean[None, :], nan=0.0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    scores = U[:, :n_components] * S[:n_components][None, :]
    n = Xc.shape[0]
    var = (S**2) / max(n - 1, 1)
    evr = var[:n_components] / (var.sum() + 1e-12)
    return {"mean": mean, "components": comps, "scores": scores, "explained_variance_ratio": evr}


# ============================
# IRF folding/alignment + reconvolution grid fit
# ============================

def load_irf_csv(path: str | Path, time_col: str = "BinCenter (ps)", hist_col: str = "Histo01") -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df[time_col].to_numpy(float), df[hist_col].to_numpy(float)


def fold_irf_to_period(
    irf_time_ps: np.ndarray,
    irf_counts: np.ndarray,
    time_edges_ps: np.ndarray,
    rep_rate_hz: float
) -> np.ndarray:
    '''
    Fold IRF timestamps to the repetition period and bin onto the FLIM bin edges.
    Returns a length-nbins array on the same grid as the FLIM counts.
    '''
    T_ps = 1e12 / float(rep_rate_hz)
    t0 = float(irf_time_ps[np.argmax(irf_counts)])
    t_fold = (irf_time_ps - t0) % T_ps
    irf_binned, _ = np.histogram(t_fold, bins=time_edges_ps, weights=irf_counts)
    irf_binned = np.maximum(irf_binned.astype(np.float64), 0.0)
    return irf_binned / (irf_binned.max() + 1e-12)


def coarse_align_irf(irf: np.ndarray, decay: np.ndarray, search_bins: int = 160, use_bins: int = 350) -> Tuple[int, float]:
    '''
    Coarse integer-bin alignment by maximising dot-product between early parts.
    '''
    irf0 = irf - np.mean(irf[:50])
    d0 = decay - np.mean(decay[:50])
    irf0 /= (np.linalg.norm(irf0) + 1e-12)
    d0 /= (np.linalg.norm(d0) + 1e-12)
    best_shift, best_score = 0, -1e9
    for sh in range(-search_bins, search_bins + 1):
        sc = float(np.dot(np.roll(irf0, sh)[:use_bins], d0[:use_bins]))
        if sc > best_score:
            best_score, best_shift = sc, sh
    return best_shift, best_score


def shift_irf_subbin(irf: np.ndarray, delta_ps: float, dt_ps: float) -> np.ndarray:
    n = irf.size
    x = np.arange(n, dtype=float) * float(dt_ps)
    x_src = x - float(delta_ps)
    return np.interp(x_src, x, irf, left=0.0, right=0.0)


def pooled_roi_decay(cube: np.ndarray, gate0: int, gate1: int, min_pixels: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    roi, _ = roi_mask_from_gated_intensity(cube, gate0, gate1, min_pixels=min_pixels)
    decay = cube[roi].sum(axis=0).astype(np.float64, copy=False)
    return decay, roi


def _lsq_3term(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    ones = np.ones_like(y)
    A = np.vstack([x1, x2, ones]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rss = float(np.sum((y - A @ coef)**2))
    a, b, c = coef
    return float(a), float(b), float(c), rss


def reconv_singleexp_fit_grid(
    y: np.ndarray,
    irf: np.ndarray,
    dt_ps: float,
    start_bin: int,
    end_bin: int,
    tau_grid_ns: np.ndarray,
    delta_grid_ps: Optional[np.ndarray] = None,
    enforce_nonneg: bool = True,
) -> Optional[dict]:
    '''
    Model on [start_bin:end_bin]:
      y(t) ≈ a*IRF(t) + b*(IRF ⊗ exp(-t/τ)) + c
    Grid-search over τ (and optionally Δt sub-bin IRF shift). Returns best RSS fit.
    '''
    y = np.asarray(y, float)
    irf = np.asarray(irf, float)
    n = y.size

    L = 1
    while L < 2*n:
        L *= 2

    t_ps = np.arange(n, dtype=float) * float(dt_ps)
    tau_grid_ns = np.asarray(tau_grid_ns, float)

    exp_ffts = []
    for tau_ns in tau_grid_ns:
        tau_ps = tau_ns * 1000.0
        expd = np.exp(-t_ps / tau_ps)
        exp_pad = np.zeros(L, float); exp_pad[:n] = expd
        exp_ffts.append(np.fft.rfft(exp_pad))
    exp_ffts = np.asarray(exp_ffts)

    if delta_grid_ps is None:
        delta_grid_ps = np.array([0.0], float)
    else:
        delta_grid_ps = np.asarray(delta_grid_ps, float)

    y_w = y[start_bin:end_bin+1]
    best = None

    for delta_ps in delta_grid_ps:
        irf_shift = shift_irf_subbin(irf, float(delta_ps), float(dt_ps))
        irf_pad = np.zeros(L, float); irf_pad[:n] = irf_shift
        irf_fft = np.fft.rfft(irf_pad)

        conv = np.fft.irfft(irf_fft[None, :] * exp_ffts, n=L)[:, :n]

        irf_w = irf_shift[start_bin:end_bin+1]
        for i, tau_ns in enumerate(tau_grid_ns):
            conv_w = conv[i, start_bin:end_bin+1]
            a, b, c, rss = _lsq_3term(irf_w, conv_w, y_w)
            if enforce_nonneg and (a < 0 or b < 0):
                continue
            cand = {
                "tau_ns": float(tau_ns),
                "delta_ps": float(delta_ps),
                "a": a, "b": b, "c": c,
                "rss": rss,
                "start_bin": int(start_bin),
                "end_bin": int(end_bin),
            }
            if best is None or cand["rss"] < best["rss"]:
                best = cand

    return best


def scan_fit_start_offsets(
    y: np.ndarray,
    irf: np.ndarray,
    dt_ps: float,
    peak_bin: int,
    offsets_ns: Iterable[float],
    span_ns: float,
    tau_grid_ns: np.ndarray,
    delta_grid_ps: Optional[np.ndarray] = None,
) -> list[dict]:
    out = []
    end_bin = int(min(len(y) - 1, peak_bin + round(span_ns * 1000.0 / dt_ps)))
    for off_ns in offsets_ns:
        start_bin = int(peak_bin + round(float(off_ns) * 1000.0 / dt_ps))
        if start_bin >= end_bin - 10:
            continue
        best = reconv_singleexp_fit_grid(
            y=y, irf=irf, dt_ps=dt_ps,
            start_bin=start_bin, end_bin=end_bin,
            tau_grid_ns=tau_grid_ns,
            delta_grid_ps=delta_grid_ps
        )
        if best is None:
            out.append({"offset_ns": float(off_ns), "tau_ns": np.nan, "delta_ps": np.nan, "rss": np.nan})
        else:
            out.append({"offset_ns": float(off_ns), **best})
    return out

