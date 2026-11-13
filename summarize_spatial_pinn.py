# -*- coding: utf-8 -*-
"""Combine spatial-PINN holdout metrics across windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_spatial_pinn")
    ap.add_argument("--out", default="")
    ap.add_argument("--pattern", default="W*_2017_2023/spatial_pinn_metrics.csv")
    args = ap.parse_args()
    root = Path(args.root)
    frames = []
    for path in sorted(root.glob(args.pattern)):
        df = pd.read_csv(path)
        df = df[df["model"].isin(["PINN", "persistence", "climatology", "PINN_pattern"])].copy()
        df["window"] = path.parent.name.split("_", 1)[0]
        frames.append(df)
    if not frames:
        raise SystemExit(f"no W*_2017_2023/spatial_pinn_metrics.csv under {root}")
    all_rows = pd.concat(frames, ignore_index=True)
    out = Path(args.out) if args.out else root / "multiwindow_holdout_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out, index=False, encoding="utf-8-sig")
    cols = ["year", "variable", "model", "rmse", "mae", "r2"]
    print(all_rows[cols].sort_values(["year", "variable", "model"]).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
