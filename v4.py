"""
Spatial patch maps and Turing-pattern verification for the Lower Tarim River PINN model.

This script builds on the cached annual PINN output and the digitized spatial
figures extracted from the literature:
  - results_real/tarim_coverage_spatial.csv
  - results_real/tarim_trend_spatial.csv

It adds:
  1. Tarim River spatial patch mosaic
  2. Turing-mode verification for a PINN-guided reduced vegetation-water model
  3. power-spectrum check of the simulated pattern

Run:
  python v4.py --outdir results_v4
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.interpolate import griddata

import v3


@dataclasses.dataclass
class V4Config:
    ndvi: Path
    water: Path
    tvdi: Path
    outdir: Path
    artifacts: Path
    retrain: bool
    train_end_year: int
    epochs: int
    lbfgs: int
    collocation: int
    restarts: int
    forecast_years: int
    seed: int
    coverage_csv: Optional[Path] = None
    trend_csv: Optional[Path] = None
    spatial_years: str = "2000,2005,2010,2015,2020"
    patch_grid: int = 160
    sim_grid: int = 160
    sim_steps: int = 800
    sim_dt: float = 0.02


@dataclasses.dataclass
class TuringModel:
    r: float
    q: float
    c: float
    a: float
    m: float
    s: float
    du: float
    dv: float
    u_eq: float
    v_eq: float
    growth_peak: float = 0.0
    k_peak: float = 0.0
    classical_turing: bool = False

    def jacobian(self) -> np.ndarray:
        fu = self.r * self.u_eq * self.v_eq * (1.0 + self.a - 2.0 * self.u_eq)
        fv = self.r * self.u_eq * (self.u_eq - self.a) * (1.0 - self.u_eq)
        gu = -self.c * self.v_eq
        gv = -(self.q + self.c * self.u_eq)
        return np.array([[fu, fv], [gu, gv]], dtype=float)


GEE_INPUT_NAMES = {
    "Lower Tarim River_NDVI_2000-2023.xlsx": "NDVI.xlsx",
    "Lower Tarim River_R_2000-2023.xlsx": "water.xlsx",
    "Lower Tarim River_TVDI_2000-2023.xlsx": "TVDI.xlsx",
}


def resolve_desktop_file(filename: str, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    # Inputs are produced by gee_to_inputs.py from the GEE exports; the legacy
    # hand-compiled Excel workbooks are kept only as a fallback.
    gee = Path(__file__).resolve().parent / "gee_inputs" / "tarim" / GEE_INPUT_NAMES.get(filename, filename)
    if gee.exists():
        return gee
    desktop = Path.home() / "Desktop"
    matches = list(desktop.rglob(filename))
    if matches:
        return matches[0]
    return desktop / filename


def select_years(available_years: Sequence[int], requested_years: Sequence[int]) -> List[int]:
    available = np.asarray(sorted(set(int(x) for x in available_years)), dtype=int)
    if len(available) == 0:
        return [int(x) for x in requested_years]
    selected = []
    for yr in requested_years:
        idx = int(np.argmin(np.abs(available - int(yr))))
        selected.append(int(available[idx]))
    return selected


def value_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in {"Year", "x", "y", "Source", "source"}]


def spatial_grid_from_points(
    df: pd.DataFrame,
    value_col: str,
    ngrid: int = 160,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df[value_col].to_numpy(dtype=float)
    xi = np.linspace(np.nanmin(x), np.nanmax(x), ngrid)
    yi = np.linspace(np.nanmin(y), np.nanmax(y), ngrid)
    XI, YI = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (XI, YI), method="linear")
    mask = np.isfinite(Z)
    if np.isnan(Z).any():
        Z2 = griddata((x, y), z, (XI, YI), method="nearest")
        Z = np.where(mask, Z, Z2)
    return XI, YI, Z, mask


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    return (arr - lo) / (hi - lo + 1e-12)


def load_bundle(cfg: V4Config) -> v3.v1.DataBundle:
    return v3.load_bundle(cfg)


def load_or_train(cfg: V4Config, bundle: v3.v1.DataBundle) -> Dict[str, object]:
    return v3.load_or_train(cfg, bundle)


def plot_spatial_patch_mosaic(
    coverage_df: pd.DataFrame,
    trend_df: Optional[pd.DataFrame],
    outdir: Path,
    years: Sequence[int],
    grid_size: int = 160,
) -> None:
    if coverage_df is None or len(coverage_df) == 0:
        return

    val_col = value_columns(coverage_df)[0]
    sel_years = select_years(coverage_df["Year"].unique(), years)

    cmap_cov = ListedColormap(["#b2182b", "#ef8a62", "#fddbc7", "#d9ef8b", "#1a9850"])
    cmap_trend = ListedColormap(["#b2182b", "#ef8a62", "#fee08b", "#d9ef8b", "#1a9850"])
    boundaries = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
    norm = BoundaryNorm(boundaries, cmap_cov.N, clip=True)

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.6), squeeze=False)
    axes = axes.ravel()

    all_panels = []
    for yr in sel_years:
        sub = coverage_df[coverage_df["Year"] == yr]
        XI, YI, Z, mask = spatial_grid_from_points(sub, val_col, ngrid=grid_size)
        Zs = normalize01(Z)
        all_panels.append((yr, XI, YI, Zs, mask))

    vmin = 0.0
    vmax = 1.0
    last_im = None
    for ax, (yr, XI, YI, Zs, mask) in zip(axes[:5], all_panels):
        last_im = ax.imshow(
            np.ma.array(Zs, mask=~mask),
            origin="lower",
            extent=(XI.min(), XI.max(), YI.min(), YI.max()),
            cmap=cmap_cov,
            norm=norm,
            aspect="auto",
        )
        ax.contour(XI, YI, Zs, levels=boundaries[1:-1], colors="white", linewidths=0.5, alpha=0.6)
        ax.set_title(f"Coverage {yr}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if trend_df is not None and len(trend_df) > 0:
        tcol = value_columns(trend_df)[0]
        sub = trend_df[trend_df["Year"] == int(trend_df["Year"].iloc[0])]
        XI, YI, Z, mask = spatial_grid_from_points(sub, tcol, ngrid=grid_size)
        Zs = normalize01(Z)
        ax = axes[5]
        last_im = ax.imshow(
            np.ma.array(Zs, mask=~mask),
            origin="lower",
            extent=(XI.min(), XI.max(), YI.min(), YI.max()),
            cmap=cmap_trend,
            norm=norm,
            aspect="auto",
        )
        ax.contour(XI, YI, Zs, levels=boundaries[1:-1], colors="white", linewidths=0.5, alpha=0.6)
        ax.set_title("Trend 2000-2020")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        axes[5].axis("off")

    if last_im is not None:
        cb = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.82, pad=0.02)
        cb.set_label("Normalized class")
        cb.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9])
        cb.set_ticklabels(["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
    fig.suptitle("Lower Tarim River spatial patch mosaic")
    v3.save_fig(fig, outdir / "fig16_spatial_patch_mosaic.png")


def build_turing_model(bundle: v3.v1.DataBundle, params: Dict[str, float]) -> TuringModel:
    u_eq = float(bundle.y_all[:, 0].mean())
    v_eq = float(1.0 - bundle.y_all[:, 1].mean())
    r = float(params["a1"])
    q = float(params["b3"])
    c = float(params["b1"])

    k_vals = np.linspace(0.0, 80.0, 1200)
    du_grid = np.logspace(-6, -3, 6)
    dv_grid = np.logspace(0.0, 7.0, 15)
    a_hi = min(u_eq - 0.02, 0.68)
    a_lo = max(0.03, min(0.18, 0.25 * a_hi))
    a_grid = np.linspace(a_lo, a_hi, 72) if a_hi > a_lo else np.array([max(0.05, min(0.3, u_eq - 0.03))], dtype=float)

    fallback: Optional[TuringModel] = None
    best: Optional[TuringModel] = None

    for a in a_grid:
        if a >= u_eq:
            continue
        m = r * (u_eq - a) * (1.0 - u_eq) * v_eq
        s = q * v_eq + c * u_eq * v_eq
        candidate = TuringModel(r=r, q=q, c=c, a=float(a), m=float(m), s=float(s), du=du_grid[0], dv=dv_grid[0], u_eq=u_eq, v_eq=v_eq)
        J = candidate.jacobian()
        trace = float(np.trace(J))
        det = float(np.linalg.det(J))
        if not (trace < 0.0 and det > 0.0):
            continue

        for du in du_grid:
            for dv in dv_grid:
                growth, k_peak = dispersion_relation(J, float(du), float(dv), k_vals)
                classical = classical_turing_conditions(J, float(du), float(dv))
                if growth > 1e-4 and k_peak > 0.0:
                    return TuringModel(
                        r=r, q=q, c=c, a=float(a), m=float(m), s=float(s),
                        du=float(du), dv=float(dv), u_eq=u_eq, v_eq=v_eq,
                        growth_peak=float(growth), k_peak=float(k_peak), classical_turing=bool(classical),
                    )
                if fallback is None or growth > fallback.growth_peak:
                    fallback = TuringModel(
                        r=r, q=q, c=c, a=float(a), m=float(m), s=float(s),
                        du=float(du), dv=float(dv), u_eq=u_eq, v_eq=v_eq,
                        growth_peak=float(growth), k_peak=float(k_peak), classical_turing=bool(classical),
                    )

    if fallback is not None and fallback.growth_peak > 0.0:
        return fallback

    a = max(0.05, min(0.5 * u_eq, u_eq - 0.03))
    m = r * (u_eq - a) * (1.0 - u_eq) * v_eq
    s = q * v_eq + c * u_eq * v_eq
    du = 1e-6
    dv = 1e7
    growth, k_peak = dispersion_relation(
        TuringModel(r, q, c, a, m, s, du, dv, u_eq, v_eq).jacobian(),
        du,
        dv,
        k_vals,
    )
    return TuringModel(
        r=r,
        q=q,
        c=c,
        a=a,
        m=m,
        s=s,
        du=du,
        dv=dv,
        u_eq=u_eq,
        v_eq=v_eq,
        growth_peak=float(growth),
        k_peak=float(k_peak),
        classical_turing=bool(classical_turing_conditions(TuringModel(r, q, c, a, m, s, du, dv, u_eq, v_eq).jacobian(), du, dv)),
    )


def dispersion_relation(J: np.ndarray, du: float, dv: float, k_vals: np.ndarray) -> Tuple[float, float]:
    growth = []
    for k in k_vals:
        M = J - np.diag([du * k * k, dv * k * k])
        growth.append(float(np.max(np.linalg.eigvals(M).real)))
    growth = np.asarray(growth, dtype=float)
    idx = int(np.argmax(growth))
    return float(growth[idx]), float(k_vals[idx])


def classical_turing_conditions(J: np.ndarray, du: float, dv: float) -> bool:
    fu, fv = float(J[0, 0]), float(J[0, 1])
    gu, gv = float(J[1, 0]), float(J[1, 1])
    trace = fu + gv
    det = fu * gv - fv * gu
    cond1 = trace < 0.0
    cond2 = det > 0.0
    cond3 = dv * fu + du * gv > 0.0
    cond4 = (dv * fu + du * gv) ** 2 - 4.0 * du * dv * det > 0.0
    return bool(cond1 and cond2 and cond3 and cond4)


def simulate_turing(model: TuringModel, init_u: np.ndarray, init_v: np.ndarray, steps: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    u = init_u.astype(float).copy()
    v = init_v.astype(float).copy()
    n = u.shape[0]

    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    diff_u = np.exp(-model.du * k2 * dt)
    diff_v = np.exp(-model.dv * k2 * dt)

    for _ in range(steps):
        ru = model.r * u * (u - model.a) * (1.0 - u) * v - model.m * u
        rv = model.s - model.q * v - model.c * u * v
        u = u + dt * ru
        v = v + dt * rv
        u = np.real(np.fft.ifftn(np.fft.fftn(u) * diff_u))
        v = np.real(np.fft.ifftn(np.fft.fftn(v) * diff_v))
        u = np.clip(u, 0.0, 1.5)
        v = np.clip(v, 0.0, 1.5)
    return u, v


def radial_spectrum(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = field.shape[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    F = np.abs(np.fft.fftshift(np.fft.fftn(field - field.mean()))) ** 2
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    KR = np.sqrt(KX ** 2 + KY ** 2)
    bins = np.linspace(0.0, float(KR.max()), 90)
    ids = np.digitize(KR.ravel(), bins) - 1
    spec = np.array([np.nanmean(F.ravel()[ids == i]) if np.any(ids == i) else np.nan for i in range(len(bins))], dtype=float)
    return bins, spec


def plot_turing_verification(
    bundle: v3.v1.DataBundle,
    fit_df: pd.DataFrame,
    params: Dict[str, float],
    coverage_df: Optional[pd.DataFrame],
    outdir: Path,
    sim_grid: int = 160,
    sim_steps: int = 800,
    sim_dt: float = 0.02,
) -> None:
    model = build_turing_model(bundle, params)
    J = model.jacobian()
    k_vals = np.linspace(0.0, 40.0, 800)
    growth = []
    for k in k_vals:
        M = J - np.diag([model.du * k * k, model.dv * k * k])
        growth.append(float(np.max(np.linalg.eigvals(M).real)))
    growth = np.asarray(growth, dtype=float)
    k_peak = float(k_vals[int(np.argmax(growth))])
    g_peak = float(np.max(growth))

    if coverage_df is not None and len(coverage_df) > 0:
        val_col = value_columns(coverage_df)[0]
        year0 = int(sorted(coverage_df["Year"].unique())[0])
        sub = coverage_df[coverage_df["Year"] == year0]
        XI, YI, Z, mask = spatial_grid_from_points(sub, val_col, ngrid=sim_grid)
        seed = normalize01(Z)
        seed = np.where(mask, seed, np.nanmean(seed))
    else:
        years = fit_df["Year"].to_numpy(dtype=float)
        z = normalize01(np.interp(np.linspace(years.min(), years.max(), sim_grid), years, fit_df["NDVI"]))
        seed = np.tile(z[:, None], (1, sim_grid))
        mask = np.ones_like(seed, dtype=bool)

    rng = np.random.default_rng(7)
    seed_noise = 0.015 * rng.normal(size=seed.shape)
    u0 = model.u_eq + 0.08 * (seed - np.nanmean(seed)) + seed_noise
    v0 = model.v_eq - 0.05 * (seed - np.nanmean(seed)) - seed_noise
    u0 = np.clip(np.where(mask, u0, model.u_eq), 0.0, 1.5)
    v0 = np.clip(np.where(mask, v0, model.v_eq), 0.0, 1.5)

    u_final, v_final = simulate_turing(model, u0, v0, steps=sim_steps, dt=sim_dt)
    bins, spec = radial_spectrum(u_final)
    peak_idx = int(np.nanargmax(spec[1:]) + 1) if len(spec) > 1 else 0
    peak_k = float(bins[peak_idx]) if peak_idx < len(bins) else 0.0
    scale_pool = np.concatenate([u0.ravel(), u_final.ravel()])
    vmin = float(np.nanpercentile(scale_pool, 2.0))
    vmax = float(np.nanpercentile(scale_pool, 98.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(u_final))
        vmax = float(np.nanmax(u_final))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.2))
    ax = axes[0, 0]
    ax.plot(k_vals, growth, color="#1f77b4", lw=1.8)
    ax.axhline(0.0, color="gray", lw=1, ls=":")
    ax.axvline(k_peak, color="crimson", lw=1.2, ls="--")
    ax.scatter([k_peak], [g_peak], color="crimson", s=40, zorder=5)
    ax.set_xlabel("Wave number k")
    ax.set_ylabel("Max growth rate")
    ax.set_title("Dispersion relation")
    summary = (
        f"u*={model.u_eq:.3f}, v*={model.v_eq:.3f}\n"
        f"a={model.a:.3f}, Du={model.du:.1e}, Dv={model.dv:.1e}\n"
        f"peak={g_peak:.3e} at k={k_peak:.2f}\n"
        f"classical Turing = {model.classical_turing}"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    ax.grid(alpha=0.2)

    cmap = "YlGn"
    ax = axes[0, 1]
    im1 = ax.imshow(u0, cmap=cmap, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title("Initial vegetation seed")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im1, ax=ax, shrink=0.86, label="U")

    ax = axes[1, 0]
    im2 = ax.imshow(u_final, cmap=cmap, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title("Simulated Turing pattern")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im2, ax=ax, shrink=0.86, label="U")

    ax = axes[1, 1]
    ax.semilogy(bins[1:], spec[1:], color="#2ca02c", lw=1.7)
    ax.axvline(peak_k, color="crimson", lw=1.2, ls="--")
    ax.set_xlabel("Wave number k")
    ax.set_ylabel("Radial power")
    ax.set_title("Spectral peak away from k=0")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    v3.save_fig(fig, outdir / "fig17_turing_verification.png")

    summary_df = pd.DataFrame(
        [
            {
                "u_eq": model.u_eq,
                "v_eq": model.v_eq,
                "a": model.a,
                "m": model.m,
                "s": model.s,
                "r": model.r,
                "q": model.q,
                "c": model.c,
                "du": model.du,
                "dv": model.dv,
                "growth_peak": g_peak,
                "k_peak": k_peak,
                "radial_peak": peak_k,
                "classical_turing": bool(model.classical_turing),
            }
        ]
    )
    summary_df.to_csv(outdir / "turing_summary.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        outdir / "turing_fields.npz",
        u0=u0,
        v0=v0,
        u_final=u_final,
        v_final=v_final,
        growth=growth,
        k_vals=k_vals,
        bins=bins,
        spectrum=spec,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndvi", type=str, default="")
    parser.add_argument("--water", type=str, default="")
    parser.add_argument("--tvdi", type=str, default="")
    parser.add_argument("--outdir", type=str, default="results_v4")
    parser.add_argument("--artifacts", type=str, default="results_real/predictions_and_params.npz")
    parser.add_argument("--train-end-year", type=int, default=2018)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lbfgs", type=int, default=200)
    parser.add_argument("--collocation", type=int, default=3000)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--forecast-years", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--coverage-csv", type=str, default="results_real/tarim_coverage_spatial.csv")
    parser.add_argument("--trend-csv", type=str, default="results_real/tarim_trend_spatial.csv")
    parser.add_argument("--spatial-years", type=str, default="2000,2005,2010,2015,2020")
    parser.add_argument("--patch-grid", type=int, default=160)
    parser.add_argument("--sim-grid", type=int, default=160)
    parser.add_argument("--sim-steps", type=int, default=800)
    parser.add_argument("--sim-dt", type=float, default=0.02)
    args = parser.parse_args()

    ndvi = resolve_desktop_file("Lower Tarim River_NDVI_2000-2023.xlsx", args.ndvi)
    water = resolve_desktop_file("Lower Tarim River_R_2000-2023.xlsx", args.water)
    tvdi = resolve_desktop_file("Lower Tarim River_TVDI_2000-2023.xlsx", args.tvdi)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = V4Config(
        ndvi=ndvi,
        water=water,
        tvdi=tvdi,
        outdir=outdir,
        artifacts=Path(args.artifacts),
        retrain=args.retrain,
        train_end_year=args.train_end_year,
        epochs=args.epochs,
        lbfgs=args.lbfgs,
        collocation=args.collocation,
        restarts=args.restarts,
        forecast_years=args.forecast_years,
        seed=args.seed,
        coverage_csv=Path(args.coverage_csv) if args.coverage_csv else None,
        trend_csv=Path(args.trend_csv) if args.trend_csv else None,
        spatial_years=args.spatial_years,
        patch_grid=args.patch_grid,
        sim_grid=args.sim_grid,
        sim_steps=args.sim_steps,
        sim_dt=args.sim_dt,
    )

    v3.v1.set_seed(cfg.seed)
    bundle = load_bundle(cfg)
    pack = load_or_train(cfg, bundle)

    print(f"Source: {pack['source']}")
    print("Parameters:")
    for k, v in pack["params"].items():
        print(f"  {k:>4s}: {v:.6f}")

    coverage_df = v3.load_spatial_data(cfg.coverage_csv, None) if cfg.coverage_csv else None
    trend_df = v3.load_spatial_data(cfg.trend_csv, None) if cfg.trend_csv else None

    years = [int(x) for x in cfg.spatial_years.split(",") if x.strip()]
    if coverage_df is not None:
        plot_spatial_patch_mosaic(coverage_df, trend_df, outdir, years=years, grid_size=cfg.patch_grid)
    else:
        print("No spatial coverage data found; patch mosaic skipped.")

    fit_df = pack["fit_df"]
    plot_turing_verification(
        bundle=bundle,
        fit_df=fit_df,
        params=pack["params"],
        coverage_df=coverage_df,
        outdir=outdir,
        sim_grid=cfg.sim_grid,
        sim_steps=cfg.sim_steps,
        sim_dt=cfg.sim_dt,
    )

    print("Done.")


if __name__ == "__main__":
    main()
