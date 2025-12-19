#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_flim import distance_from_E, E_from_distance

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R0", type=float, required=True, help="Förster radius (nm)")
    ap.add_argument("--E", nargs="*", type=float, default=[], help="Efficiencies to annotate")
    ap.add_argument("--labels", nargs="*", default=[], help="Labels for each E")
    ap.add_argument("--shade", nargs=2, type=float, default=[0.10, 0.22], help="Shade E-range")
    ap.add_argument("--out", default="out_fret_curve")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    r = np.linspace(0.5, 12.0, 800)
    E = np.array([E_from_distance(ri, args.R0) for ri in r])

    plt.figure(figsize=(6.8, 4.8))
    plt.plot(r, E, linewidth=2.0, label=f"Theory (R0={args.R0:g} nm)")

    Emin, Emax = args.shade
    r_a = distance_from_E(Emax, args.R0)
    r_b = distance_from_E(Emin, args.R0)
    if np.isfinite(r_a) and np.isfinite(r_b):
        rlo, rhi = sorted([r_a, r_b])
        rr = r[(r >= rlo) & (r <= rhi)]
        if rr.size > 5:
            plt.fill_between(rr, [E_from_distance(x, args.R0) for x in rr], Emin, alpha=0.2,
                             label=f"Shaded E={Emin:g}–{Emax:g}")

    rows = []
    for i, Ei in enumerate(args.E):
        ri = distance_from_E(Ei, args.R0)
        lab = args.labels[i] if i < len(args.labels) else f"E{i+1}"
        rows.append({"label": lab, "E": float(Ei), "r_nm": float(ri)})
        plt.scatter([ri], [Ei], s=70, label=f"{lab}: E={Ei:.3f}, r={ri:.2f} nm")
        plt.plot([ri, ri], [0, Ei], linestyle="--", linewidth=1.0)
        plt.plot([0.5, ri], [Ei, Ei], linestyle="--", linewidth=1.0)

    plt.xlabel("Distance r (nm)"); plt.ylabel("FRET efficiency E")
    plt.xlim(0.5, 12.0); plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "E_vs_distance.png", dpi=300)
    plt.close()

    if rows:
        pd.DataFrame(rows).to_csv(outdir / "annotated_points.csv", index=False)

if __name__ == "__main__":
    main()
