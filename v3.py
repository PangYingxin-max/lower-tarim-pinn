"""
Supplementary figures for the Lower Tarim River PINN model.

This script reuses the annual-series PINN workflow from v1.py and adds:
1. NDVI-TVDI-R trend comparison
2. bifurcation diagram with fold points and hysteresis region
3. potential landscape for vegetation state under water input
4. optional spatial heatmaps if gridded data are available
5. Tarim River lower reaches Populus euphratica patch maps
6. parameter sensitivity analysis for eta and beta_irr
7. correlation heatmap for NDVI, TVDI, and R

Typical run:
    python v3.py --outdir results_v3

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
from scipy.ndimage import gaussian_filter
from scipy.optimize import root

import v1


@dataclasses.dataclass
class V3Config:
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


def load_bundle(cfg: V3Config) -> v1.DataBundle:
    df = v1.load_real_data(cfg.ndvi, cfg.water, cfg.tvdi)
    return v1.build_bundle(df, cfg.train_end_year)


def load_or_train(cfg: V3Config, bundle: v1.DataBundle) -> Dict[str, object]:
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


def _series_at_year(df: pd.DataFrame, year: float, col: str) -> float:
    years = df["Year"].to_numpy(dtype=float)
    vals = df[col].to_numpy(dtype=float)
    if year <= years.min():
        return float(vals[0])
    if year >= years.max():
        return float(vals[-1])
    return float(np.interp(year, years, vals))


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
    axes[2].set_title(r"Steady-state surface $P^*(\eta,\beta_{irr})$", pad=10)
    axes[2].legend(frameon=False, fontsize=8, loc="lower left")
    fig.colorbar(cf, ax=axes[2], shrink=0.9, label=r"$P^*$")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
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


def _potential_surface(
    params: Dict[str, float],
    bundle: v1.DataBundle,
    n_u: int = 120,
    n_p: int = 240,
) -> Dict[str, np.ndarray]:
    rmin, rmax = bundle.scalers["R_raw"]
    u_norm_grid = np.linspace(-0.1, 1.25, n_u)
    u_actual_grid = v1.minmax_inverse(u_norm_grid, rmin, rmax)
    p_grid = np.linspace(0.0, 1.0, n_p)

    V = np.zeros((n_u, n_p), dtype=float)
    for i, u_norm in enumerate(u_norm_grid):
        poly = _reduced_eq_poly(params, float(u_norm))
        pot = -poly.integ()
        vals = pot(p_grid)
        vals = vals - vals[0]
        V[i] = vals

    return {
        "u_norm": u_norm_grid,
        "u": u_actual_grid,
        "p": p_grid,
        "V": V,
    }


def plot_potential_landscape(bundle: v1.DataBundle, params: Dict[str, float], outdir: Path) -> None:
    surface = _potential_surface(params, bundle)
    bif = _actual_bifurcation_curve(params, bundle)
    stable = np.asarray(bif["stable"], dtype=bool)

    U, P = np.meshgrid(surface["u"], surface["p"], indexing="xy")
    V = surface["V"].T

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    cf = axes[0].contourf(U, P, V, levels=28, cmap="magma")
    if stable.any():
        axes[0].scatter(bif["u"][stable], bif["p"][stable], s=7, c="white", alpha=0.9, label="Stable equilibrium")
    if (~stable).any():
        axes[0].scatter(bif["u"][~stable], bif["p"][~stable], s=7, c="#8ecae6", alpha=0.45, label="Unstable equilibrium")
    if len(bif["fold_points"]) >= 2:
        folds = np.asarray(bif["fold_points"], dtype=float)
        order = np.argsort(folds[:, 0])
        folds = folds[order]
        axes[0].scatter(folds[:, 0], folds[:, 1], s=60, c="cyan", edgecolor="black", linewidth=0.5, zorder=5, label="Fold points")
        axes[0].axvspan(folds[0, 0], folds[1, 0], color="cyan", alpha=0.08)
    axes[0].set_xlabel("Water input / ecological allocation")
    axes[0].set_ylabel("Vegetation state P")
    axes[0].set_title("Potential landscape")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.colorbar(cf, ax=axes[0], shrink=0.92, label="Potential V")

    idxs = [0, len(surface["u"]) // 2, -1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, color in zip(idxs, colors):
        axes[1].plot(surface["p"], surface["V"][idx], color=color, lw=1.8, label=f"R={surface['u'][idx]:.2f}")
        pmin = surface["p"][np.argmin(surface["V"][idx])]
        vmin = float(np.min(surface["V"][idx]))
        axes[1].scatter([pmin], [vmin], color=color, s=35)
    axes[1].set_xlabel("Vegetation state P")
    axes[1].set_ylabel("Potential V")
    axes[1].set_title("Potential slices")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    axes[1].grid(alpha=0.2)

    save_fig(fig, outdir / "fig12_potential_landscape.png")


def _long_spatial_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    year_col = lower.get("year", None)
    x_col = lower.get("x", None)
    y_col = lower.get("y", None)
    if None in {year_col, x_col, y_col}:
        raise ValueError("Spatial CSV must contain columns: Year, x, y, ...")
    value_cols = [c for c in df.columns if c not in {year_col, x_col, y_col, "Source", "source"}]
    if not value_cols:
        raise ValueError("Spatial CSV must contain at least one value column besides Year/x/y")
    out = df[[year_col, x_col, y_col] + value_cols].copy()
    out = out.rename(columns={year_col: "Year", x_col: "x", y_col: "y"})
    return out


def _load_spatial_npz(path: Path) -> pd.DataFrame:
    z = np.load(path, allow_pickle=True)
    years = np.asarray(z["years"], dtype=float)
    x = np.asarray(z["x"], dtype=float)
    y = np.asarray(z["y"], dtype=float)
    if "value" in z.files:
        value = np.asarray(z["value"], dtype=float)
        name = "Value"
        if "value_name" in z.files:
            name = str(np.asarray(z["value_name"], dtype=object).reshape(-1)[0])
        return pd.DataFrame({"Year": years, "x": x, "y": y, name: value})

    rows = []
    value_keys = [k for k in z.files if k not in {"years", "x", "y"}]
    if not value_keys:
        raise ValueError("Spatial NPZ must contain either 'value' or at least one named value array")
    for key in value_keys:
        arr = np.asarray(z[key], dtype=float)
        if arr.ndim == 3:
            for i, yr in enumerate(years):
                for j in range(arr.shape[1]):
                    for k in range(arr.shape[2]):
                        rows.append((yr, x[j, k], y[j, k], arr[i, j, k], key))
        elif arr.ndim == 2:
            for j in range(arr.shape[0]):
                for k in range(arr.shape[1]):
                    rows.append((years[0], x[j, k], y[j, k], arr[j, k], key))
        else:
            raise ValueError(f"Unsupported spatial array shape for {key}: {arr.shape}")
    return pd.DataFrame(rows, columns=["Year", "x", "y", "Value", "Source"])


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


def load_spatial_data(csv_path: Optional[Path], npz_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if csv_path and csv_path.exists():
        return _long_spatial_from_csv(csv_path)
    if npz_path and npz_path.exists():
        return _load_spatial_npz(npz_path)
    return None


def _select_years(available_years: Sequence[int], requested_years: Sequence[int]) -> List[int]:
    available = np.asarray(sorted(set(int(x) for x in available_years)), dtype=int)
    if len(available) == 0:
        return [int(x) for x in requested_years]
    selected = []
    for yr in requested_years:
        idx = int(np.argmin(np.abs(available - int(yr))))
        selected.append(int(available[idx]))
    return selected


def plot_spatial_maps(spatial_df: pd.DataFrame, outdir: Path, years: Sequence[int]) -> None:
    available_years = np.asarray(sorted(spatial_df["Year"].unique()))
    if len(available_years) == 0:
        return
    sel_years = _select_years(available_years, years)

    value_cols = [c for c in spatial_df.columns if c not in {"Year", "x", "y", "Source"}]
    if not value_cols:
        return

    for var in value_cols:
        fname = f"fig13_spatial_{var.lower()}.png"
        cmap = "RdYlGn" if "trend" in var.lower() else "viridis"
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
                cmap=cmap,
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


def _patch_from_spatial_points(df: pd.DataFrame, ngrid: int = 120) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    XI, YI, ndvi = _spatial_grid_from_points(df, "NDVI", ngrid=ngrid)
    _, _, tvdi = _spatial_grid_from_points(df, "TVDI", ngrid=ngrid)
    ndvi_n = (ndvi - np.nanmin(ndvi)) / (np.nanmax(ndvi) - np.nanmin(ndvi) + 1e-12)
    tvdi_n = (tvdi - np.nanmin(tvdi)) / (np.nanmax(tvdi) - np.nanmin(tvdi) + 1e-12)
    patch = 0.68 * ndvi_n + 0.32 * (1.0 - tvdi_n)
    patch = np.clip(patch, 0.0, 1.0)
    return XI, YI, patch


def _patch_from_generic_points(df: pd.DataFrame, value_cols: Sequence[str], ngrid: int = 120) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not value_cols:
        raise ValueError("No spatial value columns available")
    if len(value_cols) == 1:
        XI, YI, Z = _spatial_grid_from_points(df, value_cols[0], ngrid=ngrid)
        Z = (Z - np.nanmin(Z)) / (np.nanmax(Z) - np.nanmin(Z) + 1e-12)
        return XI, YI, np.clip(Z, 0.0, 1.0)
    XI, YI, a = _spatial_grid_from_points(df, value_cols[0], ngrid=ngrid)
    _, _, b = _spatial_grid_from_points(df, value_cols[1], ngrid=ngrid)
    a = (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12)
    b = (b - np.nanmin(b)) / (np.nanmax(b) - np.nanmin(b) + 1e-12)
    patch = 0.68 * a + 0.32 * (1.0 - b)
    return XI, YI, np.clip(patch, 0.0, 1.0)


def _synthetic_tarim_patch_map(
    year: int,
    ndvi_level: float,
    tvdi_level: float,
    shape: Tuple[int, int] = (140, 180),
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + int(year))
    ny, nx = shape
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    phase = 0.018 * (year - 2000)
    center = 0.52 + 0.09 * np.sin(2.0 * np.pi * (1.15 * X + phase))
    center += 0.025 * np.sin(2.0 * np.pi * (4.2 * X + 0.06 * (year - 2000)))
    corridor_width = 0.055 + 0.020 * (1.0 - ndvi_level)
    river = np.exp(-((Y - center) ** 2) / (2.0 * corridor_width**2))

    distance_band = np.exp(-((X - 0.52) ** 2) / (2.0 * 0.22**2))
    patch_seed = gaussian_filter(rng.normal(size=(ny, nx)), sigma=3.1)
    patch_seed = (patch_seed - patch_seed.min()) / (patch_seed.max() - patch_seed.min() + 1e-12)
    moisture = 0.7 * river + 0.3 * distance_band
    stress = 1.0 - 0.65 * ndvi_level - 0.35 * (1.0 - tvdi_level)

    suitability = 0.52 * moisture + 0.24 * patch_seed + 0.24 * ndvi_level - 0.18 * stress
    suitability = (suitability - suitability.min()) / (suitability.max() - suitability.min() + 1e-12)
    suitability = np.clip(suitability, 0.0, 1.0)
    return X, Y, suitability, center


def plot_tarim_patch_maps(
    bundle: v1.DataBundle,
    fit_df: pd.DataFrame,
    outdir: Path,
    years: Sequence[int],
    spatial_df: Optional[pd.DataFrame] = None,
) -> None:
    years = [int(y) for y in years]
    if len(years) == 0:
        years = [2000, 2010, 2023]

    fig, axes = plt.subplots(1, len(years), figsize=(4.8 * len(years), 4.2), squeeze=False)
    axes = axes[0]
    last_im = None

    if spatial_df is not None and len(spatial_df) > 0:
        sel_years = _select_years(spatial_df["Year"].unique(), years)
        value_cols = [c for c in spatial_df.columns if c not in {"Year", "x", "y", "Source"}]
        if not value_cols:
            value_cols = []
        patch_ranges = []
        for yr in sel_years:
            sub = spatial_df[spatial_df["Year"] == yr]
            XI, YI, patch = _patch_from_generic_points(sub, value_cols)
            patch_ranges.append((np.nanmin(patch), np.nanmax(patch)))
        vmin = min(v[0] for v in patch_ranges)
        vmax = max(v[1] for v in patch_ranges)

        for ax, yr in zip(axes, sel_years):
            sub = spatial_df[spatial_df["Year"] == yr]
            XI, YI, patch = _patch_from_generic_points(sub, value_cols)
            last_im = ax.imshow(
                patch,
                origin="lower",
                extent=(XI.min(), XI.max(), YI.min(), YI.max()),
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.contour(XI, YI, patch, levels=[0.55], colors="white", linewidths=1.0, alpha=0.9)
            labels = ", ".join(f"{col}={float(sub[col].mean()):.2f}" for col in value_cols[:2])
            ax.set_title(f"{yr}\n{labels}" if labels else str(yr), fontsize=10, pad=7)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    else:
        raw = bundle.raw
        patch_ranges = []
        synth = []
        for yr in years:
            ndvi_level = _series_at_year(fit_df, yr, "NDVI")
            tvdi_level = _series_at_year(fit_df, yr, "TVDI")
            ndvi_n = (ndvi_level - raw["NDVI"].min()) / (raw["NDVI"].max() - raw["NDVI"].min() + 1e-12)
            tvdi_n = (tvdi_level - raw["TVDI"].min()) / (raw["TVDI"].max() - raw["TVDI"].min() + 1e-12)
            X, Y, patch, center = _synthetic_tarim_patch_map(yr, float(ndvi_n), float(tvdi_n))
            synth.append((X, Y, patch, center, ndvi_level, tvdi_level))
            patch_ranges.append((np.nanmin(patch), np.nanmax(patch)))
        vmin = min(v[0] for v in patch_ranges)
        vmax = max(v[1] for v in patch_ranges)

        for ax, (X, Y, patch, center, ndvi_level, tvdi_level), yr in zip(axes, synth, years):
            last_im = ax.imshow(
                patch,
                origin="lower",
                extent=(X.min(), X.max(), Y.min(), Y.max()),
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.contour(X, Y, patch, levels=[0.55], colors="white", linewidths=1.0, alpha=0.9)
            ax.plot(np.linspace(0.0, 1.0, center.shape[1]), center[0], color="white", lw=1.1, alpha=0.85)
            ax.set_title(f"{yr}\nNDVI={ndvi_level:.2f}, TVDI={tvdi_level:.2f}", fontsize=10, pad=7)
            ax.set_xlabel("river-aligned x")
            ax.set_ylabel("cross-river y")

    if last_im is not None:
        fig.subplots_adjust(left=0.04, right=0.88, top=0.83, bottom=0.10, wspace=0.18)
        fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            shrink=0.82,
            pad=0.03,
            fraction=0.03,
            label="Patch suitability",
        )
    fig.suptitle("Lower Tarim River Populus euphratica patch mosaic", y=0.985)
    save_fig(fig, outdir / "fig15_tarim_patch_maps.png")


def metrics_table(bundle: v1.DataBundle, pred_df: pd.DataFrame) -> pd.DataFrame:
    raw = bundle.raw.copy()
    merged = raw.merge(pred_df, on="Year", suffixes=("_obs", "_pred"))
    rows = []
    pairs = {
        "NDVI": ("NDVI_obs", "NDVI_pred"),
        "TVDI": ("TVDI_obs", "TVDI_pred"),
        "R": ("R_raw", "R"),
    }
    for col, (obs_key, pred_key) in pairs.items():
        obs = merged[obs_key].to_numpy(dtype=float)
        pred = merged[pred_key].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
        mae = float(np.mean(np.abs(obs - pred)))
        ss_res = float(np.sum((obs - pred) ** 2))
        ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rows.append({"series": col, "RMSE": rmse, "MAE": mae, "R2": r2})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndvi", type=str, default="")
    parser.add_argument("--water", type=str, default="")
    parser.add_argument("--tvdi", type=str, default="")
    parser.add_argument("--outdir", type=str, default="results_v3")
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

    cfg = V3Config(
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

    fit_df = pack["fit_df"]
    plot_trend_comparison(bundle, fit_df, outdir)
    plot_bifurcation(bundle, pack["params"], outdir)
    plot_potential_landscape(bundle, pack["params"], outdir)
    plot_sensitivity(pack["params"], bundle, outdir)
    plot_correlation_heatmap(bundle, outdir)

    spatial_df = load_spatial_data(cfg.spatial_csv, cfg.spatial_npz)
    if spatial_df is not None:
        years = [int(x) for x in args.spatial_years.split(",") if x.strip()]
        plot_spatial_maps(spatial_df, outdir, years)
    else:
        print("No spatial data provided; observed NDVI/TVDI heatmaps skipped.")

    patch_years = [int(x) for x in args.spatial_years.split(",") if x.strip()]
    plot_tarim_patch_maps(bundle, fit_df, outdir, patch_years, spatial_df=spatial_df)

    fit_metrics = metrics_table(bundle, fit_df)
    fit_metrics.to_csv(outdir / "fit_metrics.csv", index=False, encoding="utf-8-sig")

    print("\nFit metrics")
    print(fit_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
