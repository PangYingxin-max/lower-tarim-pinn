# -*- coding: utf-8 -*-
"""QC for the fixed Sentinel-2 windows exported from Google Earth Engine."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def read_meta(path: Path):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        tags = page.tags
        scale = tags["ModelPixelScaleTag"].value if "ModelPixelScaleTag" in tags else None
        tie = tags["ModelTiepointTag"].value if "ModelTiepointTag" in tags else None
        arr = tf.asarray().astype(float)
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3 and arr.shape[0] > arr.shape[2]:
        arr = np.moveaxis(arr, -1, 0)
    return arr, scale, tie


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", default="results_spatial_pinn/raster_qc.csv")
    args = ap.parse_args()
    folder = Path(args.folder)
    rows = []
    for path in sorted(folder.glob("W*.tif")):
        # Browser downloads may append a duplicate suffix, e.g. W1_2015(1).tif.
        m = re.match(r"^(W\d+)_(\d{4})(?:\(\d+\))?$", path.stem)
        if not m:
            continue
        arr, scale, tie = read_meta(path)
        dx = float(scale[0]) if scale else np.nan
        dy = float(scale[1]) if scale else np.nan
        row = {
            "file": path.name,
            "window": m.group(1),
            "year": int(m.group(2)),
            "bands": int(arr.shape[0]),
            "height": int(arr.shape[1]),
            "width": int(arr.shape[2]),
            "dx_m": dx,
            "dy_m": dy,
            "x0": float(tie[3]) if tie else np.nan,
            "y0": float(tie[4]) if tie else np.nan,
            "mask_present": (folder / f"{m.group(1)}_MASK.tif").exists(),
        }
        for i, name in enumerate(("NDVI", "LST_C", "NDWI")):
            a = arr[i] if i < arr.shape[0] else np.full(arr.shape[1:], np.nan)
            row[f"{name}_min"] = float(np.nanmin(a)) if np.isfinite(a).any() else np.nan
            row[f"{name}_max"] = float(np.nanmax(a)) if np.isfinite(a).any() else np.nan
            row[f"{name}_mean"] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
            row[f"{name}_valid_frac"] = float(np.isfinite(a).mean())
        if arr.shape[0] >= 4:
            valid_band = arr[3]
            row["VALID_frac"] = float(np.mean(valid_band > 0.5))
        else:
            row["VALID_frac"] = np.nan
        row["valid"] = bool(
            5 <= dx <= 20 and 5 <= dy <= 20
            and row["mask_present"]
            and row["NDVI_max"] > row["NDVI_min"]
            and row["LST_C_max"] > row["LST_C_min"]
            and row["NDWI_max"] > row["NDWI_min"]
        )
        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\nwrote {out}")
    if len(df) == 0 or not bool(df["valid"].all()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
