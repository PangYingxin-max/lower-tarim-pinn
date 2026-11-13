"""Convert the raw Google Earth Engine exports into the inputs used by v1.py.

Two basins are supported.

Tarim (lower reaches, 1 km corridor around the mapped channel, 2000-2023):
  - tarim_annual_ndvi_lst_2000_2023_series(1).csv   annual corridor means
  - tarim_points_corridor_2000_2023(1).csv          3000 corridor points x 24 yr
  - tarim_points_region_2000_2023(1).csv            3000 fitting-box points x 24 yr
  - data/tarim_water_official_2000_2023.csv         ecological water conveyance
    (2000-2020 from Wang et al. 2023, J. Desert Res. 43(4):231-240, Table 2,
     the event-by-event Daxihaizi Reservoir release; 2021-2022 from the Tarim
     River Basin Administration annual chronicles; 2023 derived from the
     published cumulative total and flagged as such)

Ejina (lower Heihe, vegetation-map ROI, 2000-2023):
  - Ejina_NDVI_2000_2023_Success.csv                annual NDVI
  - Ejina_LST_2000_2023.csv                         annual LST (kelvin)
  - optional point export for the NDVI-LST feature space (needed for TVDI)

TVDI is not a satellite product: it is derived here from the NDVI-LST feature
space after Sandholt et al. (2002).  For every year the dry edge (maximum LST
per NDVI bin) and the wet edge (minimum LST per NDVI bin) are fitted by least
squares on the fitting set, and

    TVDI = (LST - LST_wet(NDVI)) / (LST_dry(NDVI) - LST_wet(NDVI))

is evaluated on the study-area points and averaged per year.  The edge
coefficients are written out so the procedure can be audited year by year.

Ejina TVDI needs two point exports: a region table for fitting the dry and wet
edges and an oasis table for evaluating TVDI.  Paste and run
``basins/ejina_gee_points_v1.js`` in GEE, then supply the two CSV downloads with
--ejina-region-points and --ejina-oasis-points.  The older single-file option
(--ejina-points) remains supported when it contains a ``zone`` column.

Heihe water input is read from an audited CSV or workbook.  Zhengyixia release
is preferred as the model forcing; East Juyan inflow is retained as a second
diagnostic series.  Missing water-years remain missing and are never filled by
interpolation.

Usage
-----
  python gee_to_inputs.py --basin tarim --outdir gee_inputs/tarim
  python gee_to_inputs.py --basin ejina --outdir gee_inputs/ejina
  python gee_to_inputs.py --basin ejina --outdir gee_inputs/ejina \
    --ejina-region-points region.csv --ejina-oasis-points oasis.csv \
    --ejina-water data/heihe_water_official_2000_2023.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DESKTOP = Path(r"C:\Users\ACER\Desktop")
PINN_DATA_DIR = DESKTOP / "PINN" / "pinn数据"
TARIM_DIR = PINN_DATA_DIR if (PINN_DATA_DIR / "tarim_annual_ndvi_lst_2000_2023_series(1).csv").exists() else DESKTOP / "vensim"
EJINA_DIR = PINN_DATA_DIR
WATER_CSV = Path(__file__).resolve().parent / "data" / "tarim_water_official_2000_2023.csv"
HEIHE_WATER_CSV = Path(__file__).resolve().parent / "data" / "heihe_water_official_2000_2023.csv"
ROOT = Path(__file__).resolve().parent

NDVI_BIN_EDGES = np.arange(-0.10, 0.62, 0.02)
MIN_POINTS_PER_BIN = 50
EDGE_QUANTILE = 2.0  # per-bin LST percentile used for the dry (98) and wet (2) edge


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least-squares slope/intercept with one 2-sigma refit pass."""
    a, b = np.polyfit(x, y, 1)
    for _ in range(2):
        resid = y - (a * x + b)
        sd = float(np.std(resid))
        if sd <= 0:
            break
        keep = np.abs(resid) <= 2.0 * sd
        if keep.sum() < 3:
            break
        a, b = np.polyfit(x[keep], y[keep], 1)
    return float(a), float(b)


def bin_extremes(ndvi: np.ndarray, lst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centres plus robust per-bin high and low LST quantiles.

    Raw per-bin maxima and minima are driven by single pixels, and with only a
    few thousand points they produce unstable, often unphysical edge slopes.
    The 98th and 2nd percentiles of each bin are used instead.
    """
    idx = np.digitize(ndvi, NDVI_BIN_EDGES) - 1
    centres, highs, lows = [], [], []
    for b in range(len(NDVI_BIN_EDGES) - 1):
        sel = idx == b
        if sel.sum() < MIN_POINTS_PER_BIN:
            continue
        centres.append(0.5 * (NDVI_BIN_EDGES[b] + NDVI_BIN_EDGES[b + 1]))
        highs.append(float(np.percentile(lst[sel], 100.0 - EDGE_QUANTILE)))
        lows.append(float(np.percentile(lst[sel], EDGE_QUANTILE)))
    return np.asarray(centres), np.asarray(highs), np.asarray(lows)


def r_squared(x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def tvdi_for_year(fit: pd.DataFrame, target: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """Fit dry/wet edges on the fitting set, evaluate mean TVDI on the target set."""
    fit = fit.rename(columns={"ndvi": "NDVI", "lst": "LST"})
    target = target.rename(columns={"ndvi": "NDVI", "lst": "LST"})
    centres, highs, lows = bin_extremes(fit["NDVI"].to_numpy(float), fit["LST"].to_numpy(float))
    if len(centres) < 4:
        raise ValueError("too few usable NDVI bins to fit the feature space")
    a_dry, b_dry = fit_line(centres, highs)
    a_wet, b_wet = fit_line(centres, lows)

    nd = target["NDVI"].to_numpy(float)
    ls = target["LST"].to_numpy(float)
    dry = a_dry * nd + b_dry
    wet = a_wet * nd + b_wet
    span = dry - wet
    ok = span > 0.5
    if ok.sum() < 10:
        raise ValueError("degenerate feature space: dry edge does not exceed wet edge")
    tvdi = np.clip((ls[ok] - wet[ok]) / span[ok], 0.0, 1.0)
    info = {
        "n_bins": int(len(centres)),
        "ndvi_min": float(np.min(centres)),
        "ndvi_max": float(np.max(centres)),
        "dry_slope": a_dry,
        "dry_intercept": b_dry,
        "dry_r2": r_squared(centres, highs, a_dry, b_dry),
        "wet_slope": a_wet,
        "wet_intercept": b_wet,
        "wet_r2": r_squared(centres, lows, a_wet, b_wet),
        "n_points_used": int(ok.sum()),
        "mean_dry_minus_wet": float(np.mean(span[ok])),
    }
    return float(np.mean(tvdi)), info


def read_annual(path: Path, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    year_col = "year" if "year" in df.columns else "Year"
    out = pd.DataFrame({"Year": df[year_col].astype(int), "value": df[value_col].astype(float)})
    return out.sort_values("Year").reset_index(drop=True)


def tarim_inputs(outdir: Path) -> pd.DataFrame:
    ann = pd.read_csv(TARIM_DIR / "tarim_annual_ndvi_lst_2000_2023_series(1).csv")
    ann = ann.rename(columns={"year": "Year"}).sort_values("Year").reset_index(drop=True)
    ann["Year"] = ann["Year"].astype(int)

    corridor = pd.read_csv(TARIM_DIR / "tarim_points_corridor_2000_2023(1).csv")
    region = pd.read_csv(TARIM_DIR / "tarim_points_region_2000_2023(1).csv")

    rows: List[Dict[str, float]] = []
    for year in sorted(corridor["year"].unique()):
        sub_reg = region[region["year"] == year]
        sub_cor = corridor[corridor["year"] == year]
        # Fit the feature space on the pooled sample so that the fitted NDVI
        # range covers the corridor instead of being extrapolated into it.
        fit = pd.concat([sub_reg[["NDVI", "LST"]], sub_cor[["NDVI", "LST"]]], ignore_index=True)
        tvdi, info = tvdi_for_year(fit, sub_cor)
        tvdi_reg, _ = tvdi_for_year(fit, sub_reg)
        rows.append(
            {
                "Year": int(year),
                "TVDI": tvdi,
                "TVDI_region": tvdi_reg,
                "LST_anomaly_C": float(sub_cor["LST"].mean() - sub_reg["LST"].mean()),
                **info,
            }
        )
    tvdi_df = pd.DataFrame(rows)

    point_year = corridor.groupby("year")[["NDVI", "LST"]].agg(["mean", "std", "count"]).reset_index()
    point_year.columns = ["Year", "NDVI_point_mean", "NDVI_point_sd", "n", "LST_point_mean", "LST_point_sd", "n2"]
    point_year = point_year[["Year", "NDVI_point_mean", "NDVI_point_sd", "LST_point_mean", "LST_point_sd", "n"]]

    series = pd.DataFrame(
        {
            "Year": ann["Year"],
            "NDVI": ann["NDVI_mean"].to_numpy(float),
            "NDVI_p50": ann["NDVI_p50"].to_numpy(float),
            "NDVI_p90": ann["NDVI_p90"].to_numpy(float),
            "LST_C": ann["LST_C"].to_numpy(float),
        }
    ).merge(tvdi_df[["Year", "TVDI", "TVDI_region", "LST_anomaly_C"]], on="Year").merge(point_year, on="Year")

    # Water input: the event-level Daxihaizi Reservoir release compiled from
    # Wang et al. (2023, J. Desert Res. 43(4):231-240, Table 2) for 2000-2020 and
    # from the basin-authority annual chronicles for 2021-2022.  The previous
    # workbook (Lower Tarim River_R_2000-2023.xlsx) is no longer used: its
    # allocation column is not the downstream release and its 2021-2023 rows
    # were trend-extrapolated rather than observed.
    water = pd.read_csv(WATER_CSV)
    water = water.rename(columns={"W_downstream_1e8m3": "R_raw"})
    water = water[["Year", "R_raw", "Source", "Reliability", "Note"]].copy()
    water["Year"] = water["Year"].astype(int)
    water = water.sort_values("Year").reset_index(drop=True)
    water["Q_total"] = water["R_raw"].cumsum()

    # Emit the workbook with the column names v1.load_real_data() looks for so
    # that v1.py can be pointed straight at gee_inputs/tarim/water.xlsx.
    water_out = pd.DataFrame(
        {
            "Year": water["Year"],
            "Ecological water allocation (10^8 m3)": water["R_raw"],
            "Total water discharge (10^8 m3)": water["Q_total"],
            "Source of data": water["Source"],
            "reliability": water["Reliability"],
            "Notes": water["Note"],
        }
    )

    series = series.merge(water[["Year", "R_raw", "Q_total"]], on="Year", how="left")
    series.to_csv(outdir / "series.csv", index=False, encoding="utf-8-sig")
    tvdi_df.to_csv(outdir / "tvdi_edges.csv", index=False, encoding="utf-8-sig")
    point_year.to_csv(outdir / "points_annual.csv", index=False, encoding="utf-8-sig")

    series[["Year", "NDVI"]].to_excel(outdir / "NDVI.xlsx", index=False)
    # Primary stress variable: regional TVDI.  The corridor TVDI saturates
    # (0.90-0.99) because MOD11A2 LST is 1 km while the corridor is 1 km wide,
    # so it is kept only as a diagnostic column.
    series[["Year", "TVDI_region"]].rename(columns={"TVDI_region": "TVDI"}).to_excel(
        outdir / "TVDI.xlsx", index=False
    )
    series[["Year", "TVDI"]].rename(columns={"TVDI": "TVDI_corridor"}).to_excel(
        outdir / "TVDI_corridor.xlsx", index=False
    )
    water_out.to_excel(outdir / "water.xlsx", index=False)

    corridor_out = corridor.rename(columns={"year": "Year"}).groupby(
        ["Year", "lon", "lat"], as_index=False
    )[["NDVI", "LST"]].mean()
    corridor_out.to_csv(outdir / "corridor_points_annual.csv", index=False)

    # The cumulative total-discharge column is an exact running sum of the
    # allocation column, so it carries no independent information.
    q = water["Q_total"].to_numpy(float)
    r = water["R_raw"].to_numpy(float)
    check = bool(np.allclose(np.diff(q), r[1:], atol=1e-6))
    print(f"  water: cumulative column equals running sum of allocation: {check}")
    return series


def read_points_file(path: Path) -> pd.DataFrame:
    pts = pd.read_csv(path)
    pts.columns = [c.lower() for c in pts.columns]
    if "lst" not in pts.columns:
        if "lst_k" not in pts.columns:
            raise ValueError(f"{path} must contain LST or LST_K")
        pts["lst"] = pts["lst_k"] - 273.15
    if "ndvi" not in pts.columns or "year" not in pts.columns:
        raise ValueError(f"{path} must contain year and NDVI")
    pts["year"] = pts["year"].astype(int)
    pts["ndvi"] = pd.to_numeric(pts["ndvi"], errors="coerce")
    pts["lst"] = pd.to_numeric(pts["lst"], errors="coerce")
    return pts.replace([np.inf, -np.inf], np.nan).dropna(subset=["ndvi", "lst"])


def read_points(path: Path) -> pd.DataFrame:
    """Read one CSV or a directory of yearly Ejina point exports."""
    if path.is_dir():
        patterns = ("Ejina_reg_*.csv", "Ejina_oas_*.csv", "Ejina_points_*.csv")
        files = sorted({f for pattern in patterns for f in path.glob(pattern)})
        if not files:
            raise ValueError(f"{path} contains no Ejina point CSV files")
        frames = [read_points_file(f).assign(source_file=f.name) for f in files]
        pts = pd.concat(frames, ignore_index=True)
        print(f"  point export: {len(files)} files in {path}")
        return pts
    return read_points_file(path)


def point_set_audit(fit: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Flag point exports that are duplicates rather than independent samples."""
    rows = []
    for year in sorted(set(fit["year"]).intersection(target["year"])):
        year_fit = fit[fit["year"] == year].reset_index(drop=True)
        year_target = target[target["year"] == year].reset_index(drop=True)
        same_geo = False
        if ".geo" in year_fit and ".geo" in year_target and len(year_fit) == len(year_target):
            same_geo = year_fit[".geo"].astype(str).equals(year_target[".geo"].astype(str))
        if "lon" in year_fit and "lat" in year_fit and "lon" in year_target and "lat" in year_target and len(year_fit) == len(year_target):
            same_geo = bool(
                np.allclose(year_fit["lon"].to_numpy(float), year_target["lon"].to_numpy(float), equal_nan=True)
                and np.allclose(year_fit["lat"].to_numpy(float), year_target["lat"].to_numpy(float), equal_nan=True)
            )
        same_values = False
        if len(year_fit) == len(year_target):
            same_values = bool(
                np.allclose(year_fit["ndvi"].to_numpy(float), year_target["ndvi"].to_numpy(float), equal_nan=True)
                and np.allclose(year_fit["lst"].to_numpy(float), year_target["lst"].to_numpy(float), equal_nan=True)
            )
        overlap = 0
        if "lon" in year_fit and "lat" in year_fit and "lon" in year_target and "lat" in year_target:
            overlap = int(
                len(
                    year_fit[["lon", "lat"]]
                    .drop_duplicates()
                    .merge(year_target[["lon", "lat"]].drop_duplicates(), on=["lon", "lat"])
                )
            )
        rows.append(
            {
                "Year": int(year),
                "n_fit": int(len(year_fit)),
                "n_target": int(len(year_target)),
                "same_geometry": bool(same_geo),
                "same_ndvi_lst": bool(same_values),
                "overlap_coordinates": overlap,
            }
        )
    return pd.DataFrame(rows)


def plot_ejina_tvdi(
    fit_pts: pd.DataFrame,
    target_pts: pd.DataFrame,
    tvdi_df: pd.DataFrame,
    outdir: Path,
) -> None:
    """Plot the annual TVDI result, edge fit quality, feature space, and samples."""
    years = tvdi_df["Year"].to_numpy(int)
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8))

    ax = axes[0, 0]
    ax.plot(years, tvdi_df["TVDI"], "o-", color="#b2182b", lw=1.8, ms=4)
    ax.set(xlabel="Year", ylabel="Mean TVDI", title="Ejina annual TVDI")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.plot(years, tvdi_df["dry_r2"], "o-", color="#d95f02", lw=1.5, ms=3.5, label="Dry edge $R^2$")
    ax.plot(years, tvdi_df["wet_r2"], "s-", color="#1b9e77", lw=1.5, ms=3.5, label="Wet edge $R^2$")
    ax.set(xlabel="Year", ylabel="Edge-fit $R^2$", title="NDVI-LST edge quality", ylim=(-0.05, 1.02))
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.2)

    ax = axes[0, 2]
    ax.plot(years, tvdi_df["n_bins"], "o-", color="#4d4d4d", lw=1.5, ms=4)
    ax.set(xlabel="Year", ylabel="Usable NDVI bins", title="Feature-space support")
    ax.grid(alpha=0.2)

    selected_years = [int(years[0]), int(years[-1])]
    for ax, year in zip(axes[1, :2], selected_years):
        fit = fit_pts[fit_pts["year"] == year]
        target = target_pts[target_pts["year"] == year]
        row = tvdi_df[tvdi_df["Year"] == year].iloc[0]
        ax.scatter(fit["ndvi"], fit["lst"], s=4, alpha=0.18, color="#999999", label="Edge-fit sample")
        ax.scatter(target["ndvi"], target["lst"], s=4, alpha=0.22, color="#2c7fb8", label="Oasis sample")
        nd = np.linspace(float(row["ndvi_min"]), float(row["ndvi_max"]), 200)
        ax.plot(nd, row["dry_slope"] * nd + row["dry_intercept"], color="#d95f02", lw=1.8, label="Dry edge")
        ax.plot(nd, row["wet_slope"] * nd + row["wet_intercept"], color="#1b9e77", lw=1.8, label="Wet edge")
        ax.set(xlabel="NDVI", ylabel="LST (deg C)", title=f"NDVI-LST feature space, {year}")
        ax.legend(fontsize=7, loc="upper right", markerscale=2.5, framealpha=0.9)
        ax.grid(alpha=0.15)

    ax = axes[1, 2]
    year = selected_years[-1]
    fit = fit_pts[fit_pts["year"] == year]
    target = target_pts[target_pts["year"] == year]
    ax.scatter(fit["lon"], fit["lat"], s=3, alpha=0.18, color="#d95f02", label="Region sample")
    ax.scatter(target["lon"], target["lat"], s=3, alpha=0.28, color="#1b9e77", label="Oasis sample")
    ax.set(xlabel="Longitude", ylabel="Latitude", title=f"Sample locations, {year}")
    ax.legend(fontsize=8, loc="best", markerscale=3.0)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(outdir / "fig_ejina_tvdi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_water_table(path: Path) -> pd.DataFrame:
    """Read a water table without inventing missing annual values."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        water = pd.read_excel(path, sheet_name=0)
    else:
        water = pd.read_csv(path, encoding="utf-8-sig")
    water.columns = [str(c).lstrip("\ufeff") for c in water.columns]
    year_candidates = [c for c in water.columns if c.lower() == "year"]
    if not year_candidates:
        raise ValueError(f"{path} has no Year column")
    water = water.rename(columns={year_candidates[0]: "Year"})
    water["Year"] = pd.to_numeric(water["Year"], errors="coerce")
    water = water.dropna(subset=["Year"]).copy()
    water["Year"] = water["Year"].astype(int)
    return water.sort_values("Year").reset_index(drop=True)


def heihe_water_series(path: Path) -> Tuple[pd.DataFrame, str]:
    """Return model forcing plus provenance, preferring Zhengyixia release."""
    water = read_water_table(path)
    primary_candidates = [
        "Zhengyixia_1e8m3",
        "R_raw",
        "Ecological water allocation (10^8 m3)",
        "Ecological water allocation",
        "EastJuyan_1e8m3",
    ]
    primary = next((c for c in primary_candidates if c in water.columns), None)
    if primary is None:
        raise ValueError(f"{path} has no supported water-volume column")

    out = water.copy()
    out["R_raw"] = pd.to_numeric(out[primary], errors="coerce")
    if "Source" in out.columns:
        out["R_source"] = out["Source"]
    elif "Source_R" in out.columns:
        out["R_source"] = out["Source_R"]
    else:
        out["R_source"] = f"{path.name}:{primary}"
    if "Reliability" in out.columns:
        out["R_reliability"] = out["Reliability"]
    elif "Reliability_R" in out.columns:
        out["R_reliability"] = out["Reliability_R"]
    else:
        out["R_reliability"] = "not stated"
    if "Note" in out.columns:
        out["R_note"] = out["Note"]
    elif "Note_R" in out.columns:
        out["R_note"] = out["Note_R"]
    else:
        out["R_note"] = ""

    # The cumulative field is produced only for pipeline compatibility.  It is
    # never used as an independent observation or to fill a missing water-year.
    out["Q_total"] = out["R_raw"].cumsum(skipna=True)
    return out, primary


def ejina_inputs(
    outdir: Path,
    points_path: Optional[Path],
    water_path: Optional[Path],
    region_path: Optional[Path] = None,
    oasis_path: Optional[Path] = None,
) -> pd.DataFrame:
    ndvi = read_annual(EJINA_DIR / "Ejina_NDVI_2000_2023_Success.csv", "NDVI").rename(columns={"value": "NDVI"})
    lst = read_annual(EJINA_DIR / "Ejina_LST_2000_2023.csv", "LST_K").rename(columns={"value": "LST_K"})
    series = ndvi.merge(lst, on="Year")
    series["LST_C"] = series["LST_K"] - 273.15

    fit_pts: Optional[pd.DataFrame] = None
    target_pts: Optional[pd.DataFrame] = None
    label = ""
    if region_path is not None and oasis_path is not None:
        fit_pts, target_pts = read_points(region_path), read_points(oasis_path)
        label = f"{region_path.name} (edges) + {oasis_path.name} (TVDI)"
    elif points_path is not None:
        pts = read_points(points_path)
        if "zone" in pts.columns:
            zones = pts["zone"].astype(str).str.lower()
            fit_pts = pts[zones == "region"] if (zones == "region").any() else pts
            target_pts = pts[zones != "region"] if (zones != "region").any() else pts
        else:
            fit_pts = target_pts = pts
        label = points_path.name

    if fit_pts is not None and target_pts is not None:
        audit = point_set_audit(fit_pts, target_pts)
        audit.to_csv(outdir / "point_audit.csv", index=False, encoding="utf-8-sig")
        n_duplicate = int((audit["same_geometry"] | audit["same_ndvi_lst"]).sum())
        if n_duplicate:
            print(
                "  WARNING: region and oasis points are identical in "
                f"{n_duplicate}/{len(audit)} years; TVDI is exploratory, not an independent validation."
            )
        rows = []
        for year in sorted(target_pts["year"].unique()):
            fit = fit_pts[fit_pts["year"] == year]
            tgt = target_pts[target_pts["year"] == year]
            tvdi, info = tvdi_for_year(fit, tgt)
            rows.append({"Year": int(year), "TVDI": tvdi, **info})
        tvdi_df = pd.DataFrame(rows)
        tvdi_df.to_csv(outdir / "tvdi_edges.csv", index=False, encoding="utf-8-sig")
        point_year = (
            target_pts.groupby("year", as_index=False)
            .agg(
                NDVI_point_mean=("ndvi", "mean"),
                LST_point_mean_C=("lst", "mean"),
                n_points=("ndvi", "size"),
            )
            .rename(columns={"year": "Year"})
        )
        point_year["Year"] = point_year["Year"].astype(int)
        point_year.to_csv(outdir / "point_means.csv", index=False, encoding="utf-8-sig")
        series = series.merge(tvdi_df[["Year", "TVDI"]], on="Year")
        series = series.merge(point_year, on="Year", how="left")
        series[["Year", "TVDI"]].to_excel(outdir / "TVDI.xlsx", index=False)
        plot_ejina_tvdi(fit_pts, target_pts, tvdi_df, outdir)
        print(f"  TVDI derived from {label}")
    else:
        print("  TVDI skipped: no point export for the NDVI-LST feature space")

    if water_path is not None:
        water, primary = heihe_water_series(water_path)
        merge_cols = [
            c
            for c in [
                "Year",
                "WaterYear",
                "R_raw",
                "Q_total",
                "Zhengyixia_1e8m3",
                "EastJuyan_1e8m3",
                "R_source",
                "R_reliability",
                "R_note",
            ]
            if c in water.columns
        ]
        series = series.merge(water[merge_cols], on="Year", how="left")
        water_out = water.copy()
        water_out.insert(0, "Ecological water allocation (10^8 m3)", water_out["R_raw"])
        water_out.insert(1, "Total water discharge (10^8 m3)", water_out["Q_total"])
        water_out.to_excel(outdir / "water.xlsx", index=False)
        water_out.to_csv(outdir / "water.csv", index=False, encoding="utf-8-sig")
        n_valid = int(water["R_raw"].notna().sum())
        print(f"  water forcing: {primary}; {n_valid}/{len(water)} water-years observed")
    else:
        print("  water input skipped: no Heihe conveyance file supplied")

    series.to_csv(outdir / "series.csv", index=False, encoding="utf-8-sig")
    if "NDVI_point_mean" in series:
        point_series = series.copy()
        point_series["NDVI"] = point_series["NDVI_point_mean"]
        point_series["LST_C"] = point_series["LST_point_mean_C"]
        point_series["LST_K"] = point_series["LST_C"] + 273.15
        point_series.to_csv(outdir / "series_point_sample.csv", index=False, encoding="utf-8-sig")
    series[["Year", "NDVI"]].to_excel(outdir / "NDVI.xlsx", index=False)
    return series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin", choices=["tarim", "ejina"])
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--ejina-points", type=str, default="", help="one point CSV or a directory of yearly Ejina point exports")
    parser.add_argument("--ejina-region-points", type=str, default="")
    parser.add_argument("--ejina-oasis-points", type=str, default="")
    parser.add_argument("--ejina-water", type=str, default="")
    parser.add_argument("--print-ejina-gee", action="store_true")
    args = parser.parse_args()

    if args.print_ejina_gee:
        print((ROOT / "basins" / "ejina_gee_points_v1.js").read_text(encoding="utf-8"))
        return
    if not args.basin or not args.outdir:
        parser.error("--basin and --outdir are required unless --print-ejina-gee is used")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.basin == "tarim":
        series = tarim_inputs(outdir)
        provenance = {
            "basin": "lower Tarim River",
            "ndvi": "MODIS/061/MOD13Q1 growing-season mean over the 1 km channel corridor (GEE script 数据2.txt, v6)",
            "lst": "MODIS/061/MOD11A2 LST_Day_1km growing-season mean over the same corridor",
            "tvdi": "derived from the NDVI-LST feature space (Sandholt et al. 2002), edges fitted on the pooled fitting-box and corridor points; TVDI.xlsx holds the regional mean (TVDI_region)",
            "tvdi_corridor": "corridor mean TVDI; diagnostic only, saturates near 0.95 because MOD11A2 LST (1 km) is coarser than the 1 km corridor",
            "water": "Lower Tarim River_R_2000-2023.xlsx (basin authority and published assessments)",
            "season": "April-October",
            "corridor": "1 km buffer on both sides of the mapped channel, Daxihaizi Reservoir to Taitema Lake",
        }
    else:
        ejina_water = Path(args.ejina_water) if args.ejina_water else HEIHE_WATER_CSV
        series = ejina_inputs(
            outdir,
            Path(args.ejina_points) if args.ejina_points else None,
            ejina_water if ejina_water.exists() else None,
            Path(args.ejina_region_points) if args.ejina_region_points else None,
            Path(args.ejina_oasis_points) if args.ejina_oasis_points else None,
        )
        provenance = {
            "basin": "lower Heihe / Ejina Oasis",
            "roi": "Heihe River Basin 1:100000 vegetation map (2015) v3.0, TPDC, doi:10.11888/Ecolo.tpdc.270869",
            "ndvi": "MODIS/061/MOD13Q1 growing-season mean (May-September), GEE export Ejina_NDVI_2000_2023_Success",
            "lst": "MODIS/061/MOD11A2 LST_Day_1km growing-season mean (May-September), GEE export Ejina_LST_2000_2023",
            "season": "May-September (differs from the Tarim window of April-October)",
            "tvdi": "derived from region-point dry/wet edges and oasis-point evaluation (Sandholt et al. 2002)",
            "water": "Zhengyixia hydrological-station release where available; East Juyan inflow retained as a secondary column; missing water-years are not interpolated",
            "water_year_convention": "Year is the start year of the 11 November-10 November water-scheduling year (for example, 2000 means 2000-2001)",
        }
    (outdir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{args.basin}] -> {outdir}")
    with pd.option_context("display.width", 220):
        print(series.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
