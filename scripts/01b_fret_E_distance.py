#!/usr/bin/env python3
"""
01b_fret_E_distance_paper.py

Convert FRET efficiency E into donor–acceptor separation distance r using the
Förster relation:
    E = 1 / (1 + (r/R0)^6)  =>  r = R0 * ((1/E) - 1)^(1/6)

The paper uses an R0 range of 2.8–4.9 nm (aqueous vs membrane-like environments),
and reports distances for the observed efficiency range.

Inputs:
- Either provide E values directly (--E) or an E range (--E-min/--E-max),
- Or point to outputs/fret_fraction_summary.csv and use the global_tail entries.

Outputs:
  - outputs/fret_distance_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import math

import numpy as np


def r_from_E(E: float, R0_nm: float) -> float:
    if not (0 < E < 1):
        return float("nan")
    return float(R0_nm * ((1.0 / E) - 1.0) ** (1.0 / 6.0))


def load_global_E_from_summary(path: Path) -> list[float]:
    Es = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if str(row.get("k_sigma", "")).strip() == "global_tail":
                try:
                    Es.append(float(row["E_from_fretpos_tbar"]))
                except Exception:
                    pass
    return [e for e in Es if np.isfinite(e)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--summary-csv", default="outputs/fret_fraction_summary.csv",
                    help="optional: use global_tail efficiencies from this file if it exists")
    ap.add_argument("--E", type=float, nargs="*", default=[],
                    help="one or more efficiencies (e.g. 0.046 0.043)")
    ap.add_argument("--E-min", type=float, default=None)
    ap.add_argument("--E-max", type=float, default=None)
    ap.add_argument("--R0-min-nm", type=float, default=4.9)
    ap.add_argument("--R0-max-nm", type=float, default=4.9)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    E_vals = list(args.E)

    # If no explicit E supplied, try to load from summary csv
    summary_path = Path(args.summary_csv)
    if not E_vals and summary_path.exists():
        E_vals = load_global_E_from_summary(summary_path)

    # Else use E range
    if not E_vals and (args.E_min is not None) and (args.E_max is not None):
        E_vals = [float(args.E_min), float(args.E_max)]

    if not E_vals:
        # Fall back to the paper's headline range
        E_vals = [0.043, 0.046]

    E_min = float(np.min(E_vals))
    E_max = float(np.max(E_vals))

    R0_min = float(args.R0_min_nm)
    R0_max = float(args.R0_max_nm)

    # Compute extreme distances across the rectangle of (E,R0)
    combos = [
        ("E_min,R0_min", E_min, R0_min),
        ("E_min,R0_max", E_min, R0_max),
        ("E_max,R0_min", E_max, R0_min),
        ("E_max,R0_max", E_max, R0_max),
    ]
    rows = []
    for label, E, R0 in combos:
        rows.append({
            "label": label,
            "E": E,
            "R0_nm": R0,
            "r_nm": r_from_E(E, R0),
        })

    r_vals = [r["r_nm"] for r in rows if np.isfinite(r["r_nm"])]
    r_min = float(np.min(r_vals))
    r_max = float(np.max(r_vals))

    out_csv = outdir / "fret_distance_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label", "E", "R0_nm", "r_nm"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("E range used:", E_min, "to", E_max)
    print("R0 range used (nm):", R0_min, "to", R0_max)
    print(f"Distance range (nm): {r_min:.2f} to {r_max:.2f}")
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()

