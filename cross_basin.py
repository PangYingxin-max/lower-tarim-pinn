"""Cross-basin comparison of the lower Tarim River and the lower Heihe (Ejina).

Both series come from the same MODIS products through the GEE exports, so the
NDVI and LST trajectories are directly comparable once the two caveats are
stated: the growing-season window differs (April-October for the Tarim corridor,
May-September for Ejina) and the spatial units differ (a 1 km channel corridor
versus the vegetation-map ROI of the Ejina Oasis).

Outputs: cross_basin_trends.png, cross_basin_metrics.csv, cross_basin_notes.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ols_trend(years: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Slope per year, intercept and R2 of a linear trend."""
    a, b = np.polyfit(years, values, 1)
    pred = a * years + b
    ss_res = float(np.sum((values - pred) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return float(a), float(b), r2


def summarise(name: str, df: pd.DataFrame) -> dict:
    years = df["Year"].to_numpy(float)
    ndvi = df["NDVI"].to_numpy(float)
    slope, _, r2 = ols_trend(years, ndvi)
    out = {
        "basin": name,
        "n_years": int(len(df)),
        "ndvi_mean": float(ndvi.mean()),
        "ndvi_first": float(ndvi[0]),
        "ndvi_last": float(ndvi[-1]),
        "ndvi_change": float(ndvi[-1] - ndvi[0]),
        "ndvi_change_pct": float(100.0 * (ndvi[-1] - ndvi[0]) / ndvi[0]),
        "ndvi_slope_per_year": slope,
        "ndvi_slope_per_decade": slope * 10.0,
        "ndvi_trend_r2": r2,
        "ndvi_cv": float(ndvi.std(ddof=1) / ndvi.mean()),
    }
    if "LST_C" in df:
        lst = df["LST_C"].to_numpy(float)
        s_lst, _, r2_lst = ols_trend(years, lst)
        out.update(
            {
                "lst_mean_C": float(lst.mean()),
                "lst_slope_per_decade": s_lst * 10.0,
                "lst_trend_r2": r2_lst,
                "corr_ndvi_lst": float(np.corrcoef(ndvi, lst)[0, 1]),
            }
        )
    if "R_raw" in df:
        r = df["R_raw"].to_numpy(float)
        ok = np.isfinite(r) & np.isfinite(ndvi)
        out["n_water"] = int(ok.sum())
        if ok.sum() >= 3:
            out["corr_ndvi_water"] = float(np.corrcoef(ndvi[ok], r[ok])[0, 1])
    tvdi_col = "TVDI_region" if "TVDI_region" in df else ("TVDI" if "TVDI" in df else None)
    if tvdi_col is not None:
        t = df[tvdi_col].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(ndvi)
        out["n_tvdi"] = int(ok.sum())
        if ok.any():
            out["tvdi_mean"] = float(t[ok].mean())
        if ok.sum() >= 3:
            out["corr_ndvi_tvdi"] = float(np.corrcoef(ndvi[ok], t[ok])[0, 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarim", default="gee_inputs/tarim/series.csv")
    ap.add_argument("--ejina", default="gee_inputs/ejina/series.csv")
    ap.add_argument("--outdir", default="results_cross_basin")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tarim = pd.read_csv(args.tarim)
    ejina = pd.read_csv(args.ejina)
    metrics = pd.DataFrame([summarise("lower Tarim (1 km corridor)", tarim), summarise("lower Heihe (Ejina Oasis)", ejina)])
    metrics.to_csv(outdir / "cross_basin_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.6), layout="constrained")
    ax = axes[0, 0]
    ax.plot(tarim["Year"], tarim["NDVI"], "o-", color="#1f77b4", label="lower Tarim (Apr-Oct)")
    ax.plot(ejina["Year"], ejina["NDVI"], "s-", color="#d62728", label="lower Heihe / Ejina (May-Sep)")
    ax.set_xlabel("year")
    ax.set_ylabel("growing-season NDVI")
    ax.set_title("Vegetation condition")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.plot(tarim["Year"], tarim["LST_C"], "o-", color="#1f77b4", label="lower Tarim")
    ax.plot(ejina["Year"], ejina["LST_C"], "s-", color="#d62728", label="lower Heihe / Ejina")
    ax.set_xlabel("year")
    ax.set_ylabel("growing-season LST (deg C)")
    ax.set_title("Land-surface temperature")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[0, 2]
    tarim_tvdi = "TVDI_region" if "TVDI_region" in tarim else ("TVDI" if "TVDI" in tarim else None)
    ejina_tvdi = "TVDI" if "TVDI" in ejina else ("TVDI_region" if "TVDI_region" in ejina else None)
    if tarim_tvdi:
        ax.plot(tarim["Year"], tarim[tarim_tvdi], "o-", color="#1f77b4", label="lower Tarim")
    if ejina_tvdi:
        ax.plot(ejina["Year"], ejina[ejina_tvdi], "s-", color="#d62728", label="lower Heihe / Ejina")
    ax.set(xlabel="year", ylabel="growing-season TVDI", title="Dryness index")
    if tarim_tvdi or ejina_tvdi:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    for df, colour, label in ((tarim, "#1f77b4", "lower Tarim"), (ejina, "#d62728", "lower Heihe / Ejina")):
        ref = df["NDVI"].to_numpy(float)[:11].mean()
        ax.plot(df["Year"], df["NDVI"].to_numpy(float) / ref - 1.0, "o-", color=colour, label=label)
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("year")
    ax.set_ylabel("NDVI relative to the 2000-2010 mean")
    ax.set_title("Relative recovery signal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 2]
    for df, colour, label, marker in ((tarim, "#1f77b4", "lower Tarim", "o"), (ejina, "#d62728", "lower Heihe / Ejina", "s")):
        ok = df["R_raw"].notna() if "R_raw" in df else pd.Series(False, index=df.index)
        if ok.any():
            ax.scatter(df.loc[ok, "R_raw"], df.loc[ok, "NDVI"], s=30, c=colour, marker=marker, label=label)
    ax.set(xlabel=r"annual water input ($10^8$ m$^3$)", ylabel="growing-season NDVI", title="Vegetation vs water")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.scatter(tarim["LST_C"], tarim["NDVI"], s=32, color="#1f77b4", label="lower Tarim")
    ax.scatter(ejina["LST_C"], ejina["NDVI"], s=32, color="#d62728", marker="s", label="lower Heihe / Ejina")
    ax.set_xlabel("growing-season LST (deg C)")
    ax.set_ylabel("growing-season NDVI")
    ax.set_title("Vegetation vs temperature")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.savefig(outdir / "cross_basin_trends.png", dpi=300)
    fig.savefig(Path(__file__).resolve().parent / "figure" / "fig_cross_basin_trends.png", dpi=300)
    plt.close(fig)

    ejina_tvdi_col = "TVDI" if "TVDI" in ejina.columns else ("TVDI_region" if "TVDI_region" in ejina.columns else None)
    ejina_water_n = int(ejina["R_raw"].notna().sum()) if "R_raw" in ejina.columns else 0
    tvdi_quality_lines = []
    tvdi_edge_file = Path(args.ejina).with_name("tvdi_edges.csv")
    if tvdi_edge_file.exists():
        edges = pd.read_csv(tvdi_edge_file)
        tvdi_quality_lines = [
            "  - Ejina TVDI uses independent region and oasis point samples; "
            f"annual mean TVDI is {edges['TVDI'].mean():.3f} "
            f"(range {edges['TVDI'].min():.3f}-{edges['TVDI'].max():.3f}).",
            "    Edge-fit medians: "
            f"dry-edge R2={edges['dry_r2'].median():.3f}, "
            f"wet-edge R2={edges['wet_r2'].median():.3f}; "
            f"usable NDVI bins={int(edges['n_bins'].min())}-{int(edges['n_bins'].max())}.",
            "    Low dry-edge R2 (<0.10) occurs in "
            + ", ".join(str(int(y)) for y in edges.loc[edges["dry_r2"] < 0.10, "Year"])
            + "; report this annual uncertainty rather than treating TVDI as an in-situ measurement.",
        ]
    notes = [
        "Cross-basin comparison notes",
        "",
        "Common protocol: MODIS/061/MOD13Q1 NDVI and MODIS/061/MOD11A2 LST_Day_1km,",
        "annual growing-season means, quality-filtered, exported through Google Earth Engine.",
        "",
        "Differences that must be stated in the manuscript:",
        "  - season: lower Tarim April-October, Ejina May-September;",
        "  - spatial unit: lower Tarim = 1 km buffer on both sides of the mapped channel,",
        "    Ejina = Heihe River Basin 1:100000 vegetation map (2015) v3.0 clipped to the oasis;",
        "  - Ejina TVDI " + ("is available from the point-based NDVI-LST feature space." if ejina_tvdi_col else "is not yet available; run the two GEE point exports and gee_to_inputs.py."),
        *tvdi_quality_lines,
        "  - Ejina water forcing uses Zhengyixia release when observed. "
        + f"{ejina_water_n}/{len(ejina)} water-years are present; missing years stay blank.",
        "  - the Heihe water-year label is the start year (for example, 2000 = 2000-2001).",
        "",
        "Data citation for the Ejina ROI:",
        "  Zheng, Y., Zhou, J. (2019). 1:100000 vegetation map of Heihe River Basin (2015)",
        "  (version 3.0). National Tibetan Plateau / Third Pole Environment Data Center.",
        "  https://doi.org/10.11888/Ecolo.tpdc.270869 (CSTR 18406.11.Ecolo.tpdc.270869)",
        "  Zhang, X., Zhou, J., Cai, W., Gao, N., Zheng, Y. (2018). Vegetation mapping of the",
        "  Heihe River Basin at 1:100000 based on 3S technology. Journal of Northwest Normal",
        "  University (Natural Science), (2), 95-101.",
    ]
    (outdir / "cross_basin_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    with pd.option_context("display.width", 200):
        print(metrics.round(4).to_string(index=False))
    print(f"written to {outdir}")


if __name__ == "__main__":
    main()
