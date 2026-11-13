"""
Supplementary figures for the Lower Tarim River PINN model.

This script reuses the annual-series PINN workflow from v1.py and adds:
1. NDVI-TVDI-R trend comparison
2. bifurcation diagram with fold points and hysteresis region
3. optional spatial heatmaps if gridded data are available
4. parameter sensitivity analysis for eta and beta_irr
5. correlation heatmap for NDVI, TVDI, and R

Typical run:
    python v2.py --outdir results_v2

If cached artifacts from v1 exist, the script will load them:
    results_real/predictions_and_params.npz
Otherwise it can retrain via v1.py when PyTorch is available.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.interpolate import griddata, interp1d
from scipy.optimize import root

import v1


@dataclasses.dataclass
class V2Config:
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
    spatial_csv: Optional[Path] = None
    spatial_npz: Optional[Path] = None


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


def load_bundle(cfg: V2Config) -> v1.DataBundle:
    df = v1.load_real_data(cfg.ndvi, cfg.water, cfg.tvdi)
    return v1.build_bundle(df, cfg.train_end_year)


def load_or_train(cfg: V2Config, bundle: v1.DataBundle) -> Dict[str, object]:
    if cfg.artifacts.exists() and not cfg.retrain:
        z = np.load(cfg.artifacts, allow_pickle=True)
        pred_all = z["pred_all"]
        params = z["params"].item()
        fit_df = v1.invert_prediction(bundle, pred_all)
        return {
            "source": "cache",
            "pred_all": pred_all,
            "params": params,
            "fit_df": fit_df,
            "history": None,
            "model": None,
        }

    if not v1.TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required to retrain the PINN, but it is not available in this environment."
        )

    train_cfg = v1.Config(
        seed=cfg.seed,
        train_end_year=cfg.train_end_year,
        hidden_layers=4,
        hidden_width=64,
        adam_epochs=cfg.epochs,
        lbfgs_steps=cfg.lbfgs,
        lr=1e-3,
        n_collocation=cfg.collocation,
        n_restarts=cfg.restarts,
        forecast_years=cfg.forecast_years,
        device="cpu",
    )
    result = v1.train_once(train_cfg, bundle, cfg.seed)
    fit_df = v1.invert_prediction(bundle, result["pred_all"])
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.artifacts,
        years=bundle.years,
        pred_all=result["pred_all"],
        params=np.array([result["params"]], dtype=object),
    )
    return {
        "source": "train",
        "pred_all": result["pred_all"],
        "params": result["params"],
        "fit_df": fit_df,
        "history": result["history"],
        "model": result["model"],
    }


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trend_comparison(bundle: v1.DataBundle, fit_df: pd.DataFrame, outdir: Path) -> None:
    raw = bundle.raw
    fig, ax1 = plt.subplots(figsize=(12.3, 4.9))
    ax2 = ax1.twinx()

    ax1.plot(raw["Year"], raw["NDVI"], color="#1f77b4", marker="o", lw=1.7, ms=4, label="NDVI")
    ax1.plot(raw["Year"], raw["TVDI"], color="#d62728", marker="s", lw=1.7, ms=4, label="TVDI")
    ax2.plot(raw["Year"], raw["R_raw"], color="#2ca02c", marker="^", lw=1.8, ms=4, label="R")

    ax1.plot(fit_df["Year"], fit_df["NDVI"], color="#1f77b4", ls="--", lw=1.2, alpha=0.8)
    ax1.plot(fit_df["Year"], fit_df["TVDI"], color="#d62728", ls="--", lw=1.2, alpha=0.8)
    ax2.plot(fit_df["Year"], fit_df["R"], color="#2ca02c", ls="--", lw=1.2, alpha=0.8)

    split_year = float(bundle.raw.loc[bundle.train_mask, "Year"].max())
    ax1.axvline(split_year, color="gray", ls=":", lw=1)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("NDVI / TVDI")
    ax2.set_ylabel("R (ecological water allocation)")
    ax1.set_title("NDVI-TVDI-R trend comparison")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="upper left")
    save_fig(fig, outdir / "fig8_trend_comparison.png")


def _reduced_eq_poly(params: Dict[str, float], u_norm: float) -> Polynomial:
    a1, a2, a3, a4 = params["a1"], params["a2"], params["a3"], params["a4"]
    b1, b2, b3 = params["b1"], params["b2"], params["b3"]

    P = Polynomial([0.0, 1.0])
    T = (b1 * (1.0 - P) - b2 * u_norm) / b3
    f = a1 * P * (1.0 - P) * (1.0 - T) + a2 * u_norm - a3 * T * P - a4 * P
    return f


def _reduced_eq_df(params: Dict[str, float], p: float, u_norm: float) -> float:
    a1, a2, a3, a4 = params["a1"], params["a2"], params["a3"], params["a4"]
    b1, b2, b3 = params["b1"], params["b2"], params["b3"]
    T = (b1 * (1.0 - p) - b2 * u_norm) / b3
    dT = -b1 / b3
    return (
        a1 * ((1.0 - p) * (1.0 - T) + p * (-1.0) * (1.0 - T) + p * (1.0 - p) * (-dT))
        - a3 * (dT * p + T)
        - a4
    )


def _roots_from_poly(poly: Polynomial) -> List[float]:
    coeffs = np.asarray(poly.coef, dtype=float)
    coeffs = np.trim_zeros(coeffs[::-1], trim="f")
    if len(coeffs) < 2:
        return []
    roots = np.roots(coeffs)
    out = []
    for r in roots:
        if abs(r.imag) < 1e-8 and 0.0 <= r.real <= 1.0:
            out.append(float(r.real))
    return sorted(out)


def _find_folds_for_actual_model(params: Dict[str, float], u_grid: np.ndarray) -> List[Tuple[float, float]]:
    guesses = [(p0, u0) for p0 in np.linspace(0.05, 0.95, 9) for u0 in np.linspace(u_grid.min(), u_grid.max(), 11)]
    folds: List[Tuple[float, float]] = []

    def fun(z: np.ndarray) -> np.ndarray:
        p, u = z
        poly = _reduced_eq_poly(params, u)
        return np.array(
            [
                float(poly(p)),
                _reduced_eq_df(params, p, u),
            ]
        )

    for x0, y0 in guesses:
        sol = root(fun, x0=np.array([x0, y0], dtype=float), method="hybr")
        if not sol.success:
            continue
        p, u = sol.x
        if not (0.0 <= p <= 1.0 and u_grid.min() <= u <= u_grid.max()):
            continue
        pair = (round(float(p), 6), round(float(u), 6))
        if pair not in folds:
            folds.append(pair)
    return [(p, u) for p, u in folds]


def _actual_bifurcation_curve(
    params: Dict[str, float],
    bundle: v1.DataBundle,
    n_u: int = 500,
) -> Dict[str, object]:
    rmin, rmax = bundle.scalers["R_raw"]
    u_norm_grid = np.linspace(-0.15, 1.35, n_u)
    u_actual_grid = v1.minmax_inverse(u_norm_grid, rmin, rmax)

    branch_u = []
    branch_p = []
    branch_stable = []
    for u_norm, u_actual in zip(u_norm_grid, u_actual_grid):
        poly = _reduced_eq_poly(params, u_norm)
        roots = _roots_from_poly(poly)
        for p in roots:
            stable = _reduced_eq_df(params, p, u_norm) < 0.0
            branch_u.append(u_actual)
            branch_p.append(p)
            branch_stable.append(stable)

    folds = _find_folds_for_actual_model(params, u_norm_grid)
    fold_points = [(v1.minmax_inverse(u, rmin, rmax), p) for p, u in folds]
    return {
        "u": np.asarray(branch_u),
        "p": np.asarray(branch_p),
        "stable": np.asarray(branch_stable, dtype=bool),
        "fold_points": fold_points,
        "u_grid": u_actual_grid,
        "u_grid_norm": u_norm_grid,
    }


def _fallback_normal_form(
    params: Dict[str, float],
    bundle: v1.DataBundle,
    n_pts: int = 1200,
) -> Dict[str, object]:
    """
    PINN-guided cusp normal form.

    eta and beta_irr are anchored to inverted parameters:
      eta = a2
      beta_irr = b2
    """
    eta = max(float(params["a2"]), 1e-4)
    beta_irr = max(float(params["b2"]), 1e-4)
    u0_norm = float(bundle.y_all[:, 2].mean())
    scale = 0.35
    x = np.linspace(-1.0, 1.0, n_pts)
    P = (x + 1.0) / 2.0
    u_norm = u0_norm + scale * (eta * (x**3 - x) - beta_irr * x)
    u_actual = v1.minmax_inverse(u_norm, *bundle.scalers["R_raw"])
    stable = (3.0 * eta * x**2 - (eta + beta_irr)) > 0.0

    x_fold = math.sqrt((eta + beta_irr) / (3.0 * eta))
    x_fold = min(max(x_fold, 0.0), 1.0)
    x_folds = np.array([-x_fold, x_fold], dtype=float)
    p_folds = (x_folds + 1.0) / 2.0
    u_fold_norm = u0_norm + scale * (eta * (x_folds**3 - x_folds) - beta_irr * x_folds)
    u_fold_actual = v1.minmax_inverse(u_fold_norm, *bundle.scalers["R_raw"])
    fold_points = [(float(u_fold_actual[i]), float(p_folds[i])) for i in range(2)]

    return {
        "u": u_actual,
        "p": P,
        "stable": stable,
        "fold_points": fold_points,
        "eta": eta,
        "beta_irr": beta_irr,
        "u0_norm": u0_norm,
        "scale": scale,
    }


def plot_bifurcation(bundle: v1.DataBundle, params: Dict[str, float], outdir: Path) -> None:
    actual = _actual_bifurcation_curve(params, bundle)
    use_fallback = len(actual["fold_points"]) < 2
    data = actual if not use_fallback else _fallback_normal_form(params, bundle)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    stable = np.asarray(data["stable"], dtype=bool)
    u = np.asarray(data["u"], dtype=float)
    p = np.asarray(data["p"], dtype=float)

    if stable.any():
        ax.scatter(u[stable], p[stable], s=7, color="#1f77b4", alpha=0.75, label="Stable branch")
    if (~stable).any():
        ax.scatter(u[~stable], p[~stable], s=7, color="#7f7f7f", alpha=0.55, label="Unstable branch")

    if len(data["fold_points"]) >= 2:
        folds = np.asarray(data["fold_points"], dtype=float)
        order = np.argsort(folds[:, 0])
        folds = folds[order]
        ax.scatter(folds[:, 0], folds[:, 1], color="crimson", s=55, zorder=5, label="Fold points")
        ax.annotate("Fold 1", xy=folds[0], xytext=(8, 8), textcoords="offset points", fontsize=9, color="crimson")
        ax.annotate("Fold 2", xy=folds[1], xytext=(8, -14), textcoords="offset points", fontsize=9, color="crimson")
        ax.axvspan(folds[0, 0], folds[1, 0], color="orange", alpha=0.12, label="Hysteresis region")

    if use_fallback:
        ax.text(
            0.02,
            0.97,
            "PINN-guided reduced normal form used\n(no double fold found in direct reduction)",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
        )
    else:
        ax.text(
            0.02,
            0.97,
            "Direct equilibrium reduction from PINN parameters",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    ax.set_xlabel("Water input / ecological allocation")
    ax.set_ylabel("Vegetation steady-state density")
    ax.set_title("Bifurcation diagram with folds and hysteresis")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    save_fig(fig, outdir / "fig9_bifurcation.png")


def _equilibrium_from_normal_form(
    u_ref: float,
    eta: float,
    beta_irr: float,
    u0_norm: float,
    scale: float,
) -> float:
    # Solve scale*eta*x^3 - scale*(eta + beta_irr)*x + (u0_norm - u_ref) = 0
    coeffs = [scale * eta, 0.0, -scale * (eta + beta_irr), u0_norm - u_ref]
    roots = np.roots(coeffs)
    xs = []
    for r in roots:
        if abs(r.imag) < 1e-8 and -1.0 <= r.real <= 1.0:
            x = float(r.real)
            if (3.0 * eta * x**2 - (eta + beta_irr)) > 0.0:
                xs.append(x)
    if not xs:
        xs = [float(np.real(roots[np.argmin(np.abs(np.imag(roots)))]))]
    x_sel = max(xs)
    return (x_sel + 1.0) / 2.0


def plot_sensitivity(params: Dict[str, float], bundle: v1.DataBundle, outdir: Path) -> None:
    eta0 = max(float(params["a2"]), 1e-4)
    beta0 = max(float(params["b2"]), 1e-4)
    u_ref = float(bundle.y_all[:, 2].mean())
    u0_norm = float(bundle.y_all[:, 2].mean())
    scale = 0.35

    eta_grid = np.linspace(0.4 * eta0, 1.9 * eta0, 100)
    beta_grid = np.linspace(0.4 * beta0, 4.0 * beta0, 100)

    p_eta = np.array([_equilibrium_from_normal_form(u_ref, e, beta0, u0_norm, scale) for e in eta_grid])
    p_beta = np.array([_equilibrium_from_normal_form(u_ref, eta0, b, u0_norm, scale) for b in beta_grid])

    E, B = np.meshgrid(eta_grid, beta_grid, indexing="xy")
    Pstar = np.zeros_like(E)
    for i in range(B.shape[0]):
        for j in range(E.shape[1]):
            Pstar[i, j] = _equilibrium_from_normal_form(u_ref, float(E[i, j]), float(B[i, j]), u0_norm, scale)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    axes[0].plot(eta_grid, p_eta, color="#1f77b4", lw=1.8)
    axes[0].axvline(eta0, color="gray", ls=":", lw=1)
    axes[0].set_xlabel(r"$\eta$")
    axes[0].set_ylabel(r"Steady-state $P^*$")
    axes[0].set_title(r"Sensitivity to $\eta$")
    axes[0].grid(alpha=0.2)

    axes[1].plot(beta_grid, p_beta, color="#d62728", lw=1.8)
    axes[1].axvline(beta0, color="gray", ls=":", lw=1)
    axes[1].set_xlabel(r"$\beta_{irr}$")
    axes[1].set_ylabel(r"Steady-state $P^*$")
    axes[1].set_title(r"Sensitivity to $\beta_{irr}$")
    axes[1].grid(alpha=0.2)

    cf = axes[2].contourf(eta_grid, beta_grid, Pstar, levels=18, cmap="viridis")
    axes[2].scatter([eta0], [beta0], color="crimson", s=40, label="PINN estimate")
    axes[2].set_xlabel(r"$\eta$")
    axes[2].set_ylabel(r"$\beta_{irr}$")
    axes[2].set_title(r"Steady-state surface $P^*(\eta,\beta_{irr})$")
    axes[2].legend(frameon=False, fontsize=8, loc="best")
    fig.colorbar(cf, ax=axes[2], shrink=0.9, label=r"$P^*$")
    save_fig(fig, outdir / "fig10_parameter_sensitivity.png")


def plot_correlation_heatmap(bundle: v1.DataBundle, outdir: Path) -> None:
    df = bundle.raw[["NDVI", "TVDI", "R_raw"]].rename(columns={"R_raw": "R"})
    corr = df.corr().to_numpy()
    labels = ["NDVI", "TVDI", "R"]

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("Correlation heatmap")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.92, label="Pearson r")
    save_fig(fig, outdir / "fig11_correlation_heatmap.png")


def _long_spatial_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    year_col = lower.get("year", None)
    x_col = lower.get("x", None)
    y_col = lower.get("y", None)
    ndvi_col = lower.get("ndvi", None)
    tvdi_col = lower.get("tvdi", None)
    if None in {year_col, x_col, y_col, ndvi_col, tvdi_col}:
        raise ValueError("Spatial CSV must contain columns: Year, x, y, NDVI, TVDI")
    out = df[[year_col, x_col, y_col, ndvi_col, tvdi_col]].copy()
    out.columns = ["Year", "x", "y", "NDVI", "TVDI"]
    return out


def _load_spatial_npz(path: Path) -> pd.DataFrame:
    z = np.load(path, allow_pickle=True)
    years = np.asarray(z["years"], dtype=float)
    x = np.asarray(z["x"], dtype=float)
    y = np.asarray(z["y"], dtype=float)
    ndvi = np.asarray(z["NDVI"], dtype=float)
    tvdi = np.asarray(z["TVDI"], dtype=float)
    rows = []
    if ndvi.ndim == 3:
        for i, yr in enumerate(years):
            for j in range(ndvi.shape[1]):
                for k in range(ndvi.shape[2]):
                    rows.append((yr, x[j, k], y[j, k], ndvi[i, j, k], tvdi[i, j, k]))
    else:
        raise ValueError("Spatial NPZ must store NDVI/TVDI as 3D arrays: (years, ny, nx)")
    return pd.DataFrame(rows, columns=["Year", "x", "y", "NDVI", "TVDI"])


def _spatial_grid_from_points(df: pd.DataFrame, variable: str, ngrid: int = 120) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df[variable].to_numpy(dtype=float)
    xi = np.linspace(np.nanmin(x), np.nanmax(x), ngrid)
    yi = np.linspace(np.nanmin(y), np.nanmax(y), ngrid)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x, y), z, (XI, YI), method="linear")
    if np.isnan(ZI).any():
        ZI2 = griddata((x, y), z, (XI, YI), method="nearest")
        ZI = np.where(np.isnan(ZI), ZI2, ZI)
    return XI, YI, ZI


def plot_spatial_maps(spatial_df: pd.DataFrame, outdir: Path, years: Sequence[int]) -> None:
    available_years = np.asarray(sorted(spatial_df["Year"].unique()))
    if len(available_years) == 0:
        return
    sel_years = []
    for yr in years:
        idx = int(np.argmin(np.abs(available_years - yr)))
        sel_years.append(int(available_years[idx]))

    for var, fname in [("NDVI", "fig12_spatial_ndvi.png"), ("TVDI", "fig13_spatial_tvdi.png")]:
        fig, axes = plt.subplots(1, len(sel_years), figsize=(4.2 * len(sel_years), 4.0), squeeze=False)
        axes = axes[0]
        vmin = float(np.nanmin(spatial_df[var]))
        vmax = float(np.nanmax(spatial_df[var]))
        last_im = None
        for ax, yr in zip(axes, sel_years):
            sub = spatial_df[spatial_df["Year"] == yr]
            XI, YI, ZI = _spatial_grid_from_points(sub, var)
            last_im = ax.imshow(
                ZI,
                origin="lower",
                extent=(XI.min(), XI.max(), YI.min(), YI.max()),
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_title(f"Year {yr}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.84, label=var)
        fig.suptitle(f"Spatial distribution of {var}")
        save_fig(fig, outdir / fname)


def load_spatial_data(csv_path: Optional[Path], npz_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if csv_path and csv_path.exists():
        return _long_spatial_from_csv(csv_path)
    if npz_path and npz_path.exists():
        return _load_spatial_npz(npz_path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndvi", type=str, default="")
    parser.add_argument("--water", type=str, default="")
    parser.add_argument("--tvdi", type=str, default="")
    parser.add_argument("--outdir", type=str, default="results_v2")
    parser.add_argument("--artifacts", type=str, default="results_real/predictions_and_params.npz")
    parser.add_argument("--train-end-year", type=int, default=2018)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lbfgs", type=int, default=200)
    parser.add_argument("--collocation", type=int, default=3000)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--forecast-years", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--spatial-csv", type=str, default="")
    parser.add_argument("--spatial-npz", type=str, default="")
    parser.add_argument("--spatial-years", type=str, default="2000,2010,2023")
    args = parser.parse_args()

    ndvi = resolve_desktop_file("Lower Tarim River_NDVI_2000-2023.xlsx", args.ndvi)
    water = resolve_desktop_file("Lower Tarim River_R_2000-2023.xlsx", args.water)
    tvdi = resolve_desktop_file("Lower Tarim River_TVDI_2000-2023.xlsx", args.tvdi)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = V2Config(
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
        spatial_csv=Path(args.spatial_csv) if args.spatial_csv else None,
        spatial_npz=Path(args.spatial_npz) if args.spatial_npz else None,
    )

    v1.set_seed(cfg.seed)
    bundle = load_bundle(cfg)
    pack = load_or_train(cfg, bundle)

    print(f"Source: {pack['source']}")
    print("Parameters:")
    for k, v in pack["params"].items():
        print(f"  {k:>4s}: {v:.6f}")

    plot_trend_comparison(bundle, pack["fit_df"], outdir)
    plot_bifurcation(bundle, pack["params"], outdir)
    plot_sensitivity(pack["params"], bundle, outdir)
    plot_correlation_heatmap(bundle, outdir)

    spatial_df = load_spatial_data(cfg.spatial_csv, cfg.spatial_npz)
    if spatial_df is not None:
        years = [int(x) for x in args.spatial_years.split(",") if x.strip()]
        plot_spatial_maps(spatial_df, outdir, years)
    else:
        print("No spatial data provided; spatial heatmaps skipped.")


if __name__ == "__main__":
    main()
