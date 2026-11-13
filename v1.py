"""
Real-data PINN for the Lower Tarim River annual series.

Inputs:
  - NDVI (2000-2023)
  - Ecological water allocation / total discharge (2000-2023)
  - TVDI (2000-2023)

Workflow:
  1. load and normalize the Excel data
  2. fit a time-dependent PINN
  3. infer mechanistic coefficients
  4. evaluate train/test fit
  5. run baseline / no-water / enhanced-water scenarios
  6. plot paper-style figures

Run:
  python v1.py --outdir results

Requires:
  torch, pandas, openpyxl, scipy, matplotlib, numpy
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = exc

    class _CudaStub:
        @staticmethod
        def manual_seed_all(seed: int) -> None:
            return None

    class _TorchStub:
        cuda = _CudaStub()

        @staticmethod
        def manual_seed(seed: int) -> None:
            return None

        @staticmethod
        def tensor(*args, **kwargs):
            return None

        @staticmethod
        def as_tensor(*args, **kwargs):
            return None

        @staticmethod
        def device(name: str):
            return name

    class _NNStub:
        class Module:
            def __init__(self, *args, **kwargs):
                pass

        class Linear:
            def __init__(self, *args, **kwargs):
                pass

        class Sequential:
            def __init__(self, *args, **kwargs):
                pass

        class Parameter:
            def __init__(self, value):
                self.value = value

    class _FStub:
        @staticmethod
        def softplus(x):
            return x

    torch = _TorchStub()
    nn = _NNStub()
    F = _FStub()


def _resolve_default(filename: str) -> Path:
    """Locate a default data file, whatever Desktop folder it currently sits in."""
    desktop = Path.home() / "Desktop"
    candidates = [
        desktop / "PINN" / filename,
        desktop / "参考文献" / filename,
    ]
    if desktop.exists():
        candidates.extend(sorted(desktop.rglob(filename)))
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


# Inputs are assembled by gee_to_inputs.py from the GEE exports; the legacy
# hand-compiled Excel workbooks are only a fallback.
_GEE_INPUT_DIR = Path(__file__).resolve().parent / "gee_inputs" / "tarim"


def _gee_input(name: str, legacy: str) -> Path:
    path = _GEE_INPUT_DIR / name
    return path if path.exists() else _resolve_default(legacy)


DEFAULT_NDVI_FILE = _gee_input("NDVI.xlsx", "Lower Tarim River_NDVI_2000-2023.xlsx")
DEFAULT_WATER_FILE = _gee_input("water.xlsx", "Lower Tarim River_R_2000-2023.xlsx")
DEFAULT_TVDI_FILE = _gee_input("TVDI.xlsx", "Lower Tarim River_TVDI_2000-2023.xlsx")


@dataclasses.dataclass
class Config:
    seed: int = 7
    train_end_year: int = 2018
    hidden_layers: int = 4
    hidden_width: int = 64
    adam_epochs: int = 2500
    lbfgs_steps: int = 200
    lr: float = 1e-3
    n_collocation: int = 3000
    n_restarts: int = 4
    forecast_years: int = 5
    device: str = "cpu"


@dataclasses.dataclass
class DataBundle:
    raw: pd.DataFrame
    years: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_all: np.ndarray
    y_all: np.ndarray
    scalers: Dict[str, Tuple[float, float]]
    forcing_fn: Callable[[np.ndarray], np.ndarray]
    year_min: float
    year_max: float


TRUE_LABELS = ("NDVI", "TVDI", "R")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_torch() -> None:
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for training this PINN. "
            f"Import error: {TORCH_IMPORT_ERROR}"
        )


def pick_column(columns: Iterable[str], keywords: Iterable[str]) -> str:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for kw in keywords:
        for c in cols:
            if kw in c.lower():
                return c
    raise KeyError(f"Cannot find a column matching keywords {list(keywords)} in {cols}")


def load_excel_series(path: Path, value_keywords: Iterable[str], sheet_name: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name)
    year_col = pick_column(df.columns, ["year"])
    value_col = pick_column(df.columns, value_keywords)
    out = df[[year_col, value_col]].copy()
    out.columns = ["Year", value_col]
    return out


def load_real_data(ndvi_file: Path, water_file: Path, tvdi_file: Path) -> pd.DataFrame:
    ndvi = load_excel_series(ndvi_file, ["ndvi"])
    water = pd.read_excel(water_file, sheet_name=0)
    tvdi = load_excel_series(tvdi_file, ["tvdi"])

    year_col = pick_column(water.columns, ["year"])
    eco_col = pick_column(water.columns, ["ecological water allocation"])
    total_col = pick_column(water.columns, ["total water discharge"])
    water = water[[year_col, eco_col, total_col]].copy()
    water.columns = ["Year", "R_raw", "Q_total"]

    df = ndvi.merge(water, on="Year", how="inner").merge(tvdi, on="Year", how="inner")
    tvdi_val_col = pick_column(df.columns, ["tvdi"])
    ndvi_val_col = pick_column(df.columns, ["ndvi"])
    df = df[["Year", ndvi_val_col, tvdi_val_col, "R_raw", "Q_total"]].copy()
    df.columns = ["Year", "NDVI", "TVDI", "R_raw", "Q_total"]
    df = df.sort_values("Year").reset_index(drop=True)
    return df


def minmax_fit(values: np.ndarray) -> Tuple[float, float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def minmax_transform(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return (values - vmin) / (vmax - vmin)


def minmax_inverse(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return values * (vmax - vmin) + vmin


def build_bundle(df: pd.DataFrame, train_end_year: int) -> DataBundle:
    years = df["Year"].to_numpy(dtype=float)
    train_mask = years <= train_end_year
    test_mask = ~train_mask

    scalers: Dict[str, Tuple[float, float]] = {}
    x_train = years[train_mask].reshape(-1, 1)
    x_all = years.reshape(-1, 1)
    y_all = np.zeros((len(df), 3), dtype=np.float32)
    y_train = np.zeros((train_mask.sum(), 3), dtype=np.float32)

    for i, col in enumerate(["NDVI", "TVDI", "R_raw"]):
        vmin, vmax = minmax_fit(df.loc[train_mask, col].to_numpy(dtype=float))
        scalers[col] = (vmin, vmax)
        y_all[:, i] = minmax_transform(df[col].to_numpy(dtype=float), vmin, vmax)
        y_train[:, i] = minmax_transform(df.loc[train_mask, col].to_numpy(dtype=float), vmin, vmax)

    forcing_years = years.copy()
    forcing_values = y_all[:, 2].copy()

    def forcing_fn(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.interp(t, forcing_years, forcing_values, left=forcing_values[0], right=forcing_values[-1])

    return DataBundle(
        raw=df,
        years=years,
        train_mask=train_mask,
        test_mask=test_mask,
        x_train=x_train,
        y_train=y_train,
        x_all=x_all,
        y_all=y_all,
        scalers=scalers,
        forcing_fn=forcing_fn,
        year_min=float(years.min()),
        year_max=float(years.max()),
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: int, hidden_width: int):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden_width] * hidden_layers + [out_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TarimAnnualPINN(nn.Module):
    def __init__(self, hidden_layers: int, hidden_width: int, year_min: float, year_max: float):
        super().__init__()
        self.backbone = MLP(1, 3, hidden_layers, hidden_width)
        self.year_min = float(year_min)
        self.year_scale = max(float(year_max - year_min), 1.0)

        self.raw_a1 = nn.Parameter(torch.tensor(math.log(1.0), dtype=torch.float32))
        self.raw_a2 = nn.Parameter(torch.tensor(math.log(0.5), dtype=torch.float32))
        self.raw_a3 = nn.Parameter(torch.tensor(math.log(0.4), dtype=torch.float32))
        self.raw_a4 = nn.Parameter(torch.tensor(math.log(0.2), dtype=torch.float32))
        self.raw_b1 = nn.Parameter(torch.tensor(math.log(0.5), dtype=torch.float32))
        self.raw_b2 = nn.Parameter(torch.tensor(math.log(0.4), dtype=torch.float32))
        self.raw_b3 = nn.Parameter(torch.tensor(math.log(0.2), dtype=torch.float32))
        self.raw_c1 = nn.Parameter(torch.tensor(math.log(1.0), dtype=torch.float32))

    @staticmethod
    def pos(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-6

    def phys_params(self) -> Dict[str, torch.Tensor]:
        return {
            "a1": self.pos(self.raw_a1),
            "a2": self.pos(self.raw_a2),
            "a3": self.pos(self.raw_a3),
            "a4": self.pos(self.raw_a4),
            "b1": self.pos(self.raw_b1),
            "b2": self.pos(self.raw_b2),
            "b3": self.pos(self.raw_b3),
            "c1": self.pos(self.raw_c1),
        }

    def _scale_year(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.year_min) / self.year_scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        z = self._scale_year(t)
        return torch.sigmoid(self.backbone(z))

    def predict(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(t)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def residuals(self, t: torch.Tensor, forcing_fn: Callable[[np.ndarray], np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = t.clone().detach().requires_grad_(True)
        P, T, R = self.predict(t)

        dP = torch.autograd.grad(P, t, grad_outputs=torch.ones_like(P), create_graph=True, retain_graph=True)[0]
        dT = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
        dR = torch.autograd.grad(R, t, grad_outputs=torch.ones_like(R), create_graph=True, retain_graph=True)[0]

        forcing_np = forcing_fn(t.detach().cpu().numpy())
        forcing = torch.as_tensor(forcing_np, dtype=t.dtype, device=t.device).view(-1, 1)

        p = self.phys_params()
        rhs_p = p["a1"] * P * (1.0 - P) * (1.0 - T) + p["a2"] * R - p["a3"] * T * P - p["a4"] * P
        rhs_t = p["b1"] * (1.0 - P) - p["b2"] * R - p["b3"] * T
        rhs_r = p["c1"] * (forcing - R)

        return dP - rhs_p, dT - rhs_t, dR - rhs_r


def to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def train_once(cfg: Config, bundle: DataBundle, seed: int) -> Dict[str, object]:
    ensure_torch()
    set_seed(seed)
    device = torch.device(cfg.device)
    model = TarimAnnualPINN(cfg.hidden_layers, cfg.hidden_width, bundle.year_min, bundle.year_max).to(device)

    t_train = to_tensor(bundle.x_train, device)
    y_train = to_tensor(bundle.y_train, device)
    t_all = to_tensor(bundle.x_all, device)
    y_all = to_tensor(bundle.y_all, device)
    forcing_fn = bundle.forcing_fn

    coll_t = np.random.uniform(bundle.year_min, bundle.year_max, size=(cfg.n_collocation, 1)).astype(np.float32)
    t_coll = to_tensor(coll_t, device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = {"total": [], "data": [], "pde": [], "ic": []}

    x0 = t_train[:1]
    y0 = y_train[:1]

    for epoch in range(cfg.adam_epochs):
        opt.zero_grad()
        pred_train = model(t_train)
        loss_data = mse(pred_train, y_train)

        rP, rT, rR = model.residuals(t_coll, forcing_fn)
        loss_pde = mse(rP, torch.zeros_like(rP)) + mse(rT, torch.zeros_like(rT)) + mse(rR, torch.zeros_like(rR))

        pred0 = model(x0)
        loss_ic = mse(pred0, y0)

        loss = 2.0 * loss_data + 10.0 * loss_pde + 5.0 * loss_ic
        loss.backward()
        opt.step()

        history["total"].append(float(loss.detach().cpu()))
        history["data"].append(float(loss_data.detach().cpu()))
        history["pde"].append(float(loss_pde.detach().cpu()))
        history["ic"].append(float(loss_ic.detach().cpu()))

        if (epoch + 1) % 500 == 0:
            print(
                f"epoch {epoch+1:5d} "
                f"loss={history['total'][-1]:.4e} "
                f"data={history['data'][-1]:.4e} "
                f"pde={history['pde'][-1]:.4e} "
                f"ic={history['ic'][-1]:.4e}"
            )

    if cfg.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=cfg.lbfgs_steps,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            pred_train = model(t_train)
            loss_data = mse(pred_train, y_train)
            rP, rT, rR = model.residuals(t_coll, forcing_fn)
            loss_pde = mse(rP, torch.zeros_like(rP)) + mse(rT, torch.zeros_like(rT)) + mse(rR, torch.zeros_like(rR))
            pred0 = model(x0)
            loss_ic = mse(pred0, y0)
            loss = 2.0 * loss_data + 10.0 * loss_pde + 5.0 * loss_ic
            loss.backward()
            return loss

        lbfgs.step(closure)

    with torch.no_grad():
        pred_all = model(t_all).cpu().numpy()
        params = {k: float(v.detach().cpu()) for k, v in model.phys_params().items()}

    return {
        "model": model,
        "pred_all": pred_all,
        "history": history,
        "params": params,
    }


def simulate_ode(
    model: TarimAnnualPINN,
    bundle: DataBundle,
    params: Dict[str, float],
    start_year: float,
    end_year: float,
    forcing_scale: float = 1.0,
    forcing_shift: float = 0.0,
    zero_water: bool = False,
) -> pd.DataFrame:
    forcing_raw = bundle.raw["R_raw"].to_numpy(dtype=float)
    years = bundle.years.astype(float)
    forcing_fn = interp1d(years, minmax_transform(forcing_raw, *bundle.scalers["R_raw"]), kind="linear", fill_value="extrapolate")
    zero_level = float(minmax_transform(np.array([0.0]), *bundle.scalers["R_raw"])[0])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        P, T, R = y
        u = zero_level if zero_water else float(forcing_fn(t + forcing_shift)) * forcing_scale
        a1, a2, a3, a4 = params["a1"], params["a2"], params["a3"], params["a4"]
        b1, b2, b3 = params["b1"], params["b2"], params["b3"]
        c1 = params["c1"]
        dP = a1 * P * (1.0 - P) * (1.0 - T) + a2 * R - a3 * T * P - a4 * P
        dT = b1 * (1.0 - P) - b2 * R - b3 * T
        dR = c1 * (u - R)
        return np.array([dP, dT, dR], dtype=float)

    y0 = bundle.y_all[0].astype(float)
    t_eval = np.arange(start_year, end_year + 0.01, 0.1)
    sol = solve_ivp(rhs, (start_year, end_year), y0, t_eval=t_eval, rtol=1e-6, atol=1e-8)
    out = pd.DataFrame({"Year": sol.t, "NDVI": sol.y[0], "TVDI": sol.y[1], "R": sol.y[2]})
    out["NDVI"] = minmax_inverse(out["NDVI"].to_numpy(dtype=float), *bundle.scalers["NDVI"])
    out["TVDI"] = minmax_inverse(out["TVDI"].to_numpy(dtype=float), *bundle.scalers["TVDI"])
    out["R"] = minmax_inverse(out["R"].to_numpy(dtype=float), *bundle.scalers["R_raw"])
    return out


def compute_indices(df: pd.DataFrame, scalers: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
    if scalers is None:
        p = df["NDVI"].to_numpy(dtype=float)
        t = df["TVDI"].to_numpy(dtype=float)
        r_col = "R" if "R" in df.columns else "R_raw"
        r = df[r_col].to_numpy(dtype=float)
    else:
        p = minmax_transform(df["NDVI"].to_numpy(dtype=float), *scalers["NDVI"])
        t = minmax_transform(df["TVDI"].to_numpy(dtype=float), *scalers["TVDI"])
        r_col = "R" if "R" in df.columns else "R_raw"
        r = minmax_transform(df[r_col].to_numpy(dtype=float), *scalers["R_raw"])
    # Use vegetation, moisture complement, and water input as the three components.
    q1 = np.clip(p, 1e-8, None)
    q2 = np.clip(1.0 - t, 1e-8, None)
    q3 = np.clip(r, 1e-8, None)
    s = q1 + q2 + q3
    q1, q2, q3 = q1 / s, q2 / s, q3 / s
    H = -(q1 * np.log(q1) + q2 * np.log(q2) + q3 * np.log(q3))
    E = H / np.log(3.0)
    out = df.copy()
    out["H"] = H
    out["E"] = E
    return out


def invert_prediction(bundle: DataBundle, pred_norm: np.ndarray) -> pd.DataFrame:
    ndvi_min, ndvi_max = bundle.scalers["NDVI"]
    tvdi_min, tvdi_max = bundle.scalers["TVDI"]
    r_min, r_max = bundle.scalers["R_raw"]
    return pd.DataFrame(
        {
            "Year": bundle.years,
            "NDVI": minmax_inverse(pred_norm[:, 0], ndvi_min, ndvi_max),
            "TVDI": minmax_inverse(pred_norm[:, 1], tvdi_min, tvdi_max),
            "R": minmax_inverse(pred_norm[:, 2], r_min, r_max),
        }
    )


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_and_data(bundle: DataBundle, history: Dict[str, List[float]], outdir: Path) -> None:
    raw = bundle.raw
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    train = raw[bundle.train_mask]
    test = raw[bundle.test_mask]
    axes[0].scatter(train["Year"], train["NDVI"], s=28, c="#1f77b4", label="NDVI train")
    axes[0].scatter(test["Year"], test["NDVI"], s=28, c="#1f77b4", marker="x", label="NDVI test")
    axes[0].scatter(train["Year"], train["TVDI"], s=28, c="#d62728", label="TVDI train")
    axes[0].scatter(test["Year"], test["TVDI"], s=28, c="#d62728", marker="x", label="TVDI test")
    axes[0].scatter(train["Year"], train["R_raw"], s=28, c="#2ca02c", label="Water train")
    axes[0].scatter(test["Year"], test["R_raw"], s=28, c="#2ca02c", marker="x", label="Water test")
    axes[0].axvline(bundle.raw.loc[bundle.train_mask, "Year"].max(), color="gray", ls="--", lw=1)
    axes[0].set_title("Observed annual series")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Value")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    axes[1].semilogy(history["total"], lw=1.6, label="Total")
    axes[1].semilogy(history["data"], lw=1.2, label="Data")
    axes[1].semilogy(history["pde"], lw=1.2, label="PDE")
    axes[1].semilogy(history["ic"], lw=1.2, label="IC")
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=True, fontsize=8)
    save_fig(fig, outdir / "fig2_data_and_loss.png")


def plot_fit(bundle: DataBundle, fit_df: pd.DataFrame, outdir: Path, name: str = "fig3_fit.png") -> None:
    raw = bundle.raw
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.5), sharex=True)
    specs = [("NDVI", "#1f77b4"), ("TVDI", "#d62728"), ("R_raw", "#2ca02c")]
    for ax, (col, color) in zip(axes, specs):
        ax.scatter(raw["Year"], raw[col], s=22, color=color, alpha=0.65, label="Observed")
        ax.plot(fit_df["Year"], fit_df["NDVI" if col == "NDVI" else "TVDI" if col == "TVDI" else "R"], color=color, lw=1.8, label="PINN fit")
        ax.axvline(bundle.raw.loc[bundle.train_mask, "Year"].max(), color="gray", ls="--", lw=1)
        ax.set_ylabel(col)
        ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("Year")
    save_fig(fig, outdir / name)


def plot_parameter_boxplots(runs: List[Dict[str, float]], outdir: Path) -> None:
    groups = [
        ("Vegetation", ["a1", "a2", "a3", "a4"]),
        ("Drought", ["b1", "b2", "b3"]),
        ("Water", ["c1"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    for ax, (title, keys) in zip(axes, groups):
        data = [[run[k] for run in runs] for k in keys]
        ax.boxplot(data, patch_artist=True, medianprops={"color": "black"})
        ax.set_xticks(range(1, len(keys) + 1))
        ax.set_xticklabels(keys)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    save_fig(fig, outdir / "fig4_parameter_boxplots.png")


def plot_scenarios(scenarios: Dict[str, pd.DataFrame], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True)
    colors = {
        "baseline": "#1f77b4",
        "no_water": "#d62728",
        "plus20": "#2ca02c",
        "delay1y": "#9467bd",
    }
    for name, df in scenarios.items():
        axes[0].plot(df["Year"], df["NDVI"], color=colors.get(name, None), lw=1.8, label=name)
        axes[1].plot(df["Year"], df["TVDI"], color=colors.get(name, None), lw=1.8, label=name)
    axes[0].set_ylabel("NDVI")
    axes[1].set_ylabel("TVDI")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    save_fig(fig, outdir / "fig5_scenarios.png")


def plot_indices(df_obs: pd.DataFrame, df_fit: pd.DataFrame, scalers: Dict[str, Tuple[float, float]], outdir: Path) -> None:
    obs_idx = compute_indices(df_obs, scalers=scalers)
    fit_idx = compute_indices(df_fit, scalers=scalers)
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=True)
    axes[0].plot(obs_idx["Year"], obs_idx["H"], color="red", lw=1.6, label="Observed H")
    axes[0].plot(fit_idx["Year"], fit_idx["H"], color="black", lw=1.6, label="PINN H")
    axes[1].plot(obs_idx["Year"], obs_idx["E"], color="blue", lw=1.6, label="Observed E")
    axes[1].plot(fit_idx["Year"], fit_idx["E"], color="black", lw=1.6, label="PINN E")
    axes[0].set_ylabel("H")
    axes[1].set_ylabel("E")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    save_fig(fig, outdir / "fig7_indices.png")


def metrics_table(bundle: DataBundle, pred_df: pd.DataFrame) -> pd.DataFrame:
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


def run_experiment(cfg: Config, bundle: DataBundle, outdir: Path) -> None:
    data = train_once(cfg, bundle, cfg.seed)
    fit_df = invert_prediction(bundle, data["pred_all"])

    plot_loss_and_data(bundle, data["history"], outdir)
    plot_fit(bundle, fit_df, outdir)

    # Additional restart runs for parameter spread.
    runs = [data["params"]]
    for i in range(1, cfg.n_restarts):
        local_cfg = dataclasses.replace(
            cfg,
            seed=cfg.seed + 17 * i,
            adam_epochs=max(900, cfg.adam_epochs // 2),
            lbfgs_steps=max(80, cfg.lbfgs_steps // 2),
            n_collocation=max(1200, cfg.n_collocation // 2),
        )
        runs.append(train_once(local_cfg, bundle, local_cfg.seed)["params"])
    plot_parameter_boxplots(runs, outdir)

    # Scenarios.
    base = simulate_ode(
        data["model"], bundle, data["params"],
        start_year=float(bundle.years.min()),
        end_year=float(bundle.years.max()) + cfg.forecast_years,
    )
    no_water = simulate_ode(
        data["model"], bundle, data["params"],
        start_year=float(bundle.years.min()),
        end_year=float(bundle.years.max()) + cfg.forecast_years,
        zero_water=True,
    )
    plus20 = simulate_ode(
        data["model"], bundle, data["params"],
        start_year=float(bundle.years.min()),
        end_year=float(bundle.years.max()) + cfg.forecast_years,
        forcing_scale=1.2,
    )
    delay1y = simulate_ode(
        data["model"], bundle, data["params"],
        start_year=float(bundle.years.min()),
        end_year=float(bundle.years.max()) + cfg.forecast_years,
        forcing_shift=1.0,
    )
    plot_scenarios(
        {
            "baseline": base,
            "no_water": no_water,
            "plus20": plus20,
            "delay1y": delay1y,
        },
        outdir,
    )
    plot_indices(bundle.raw, fit_df, bundle.scalers, outdir)

    fit_metrics = metrics_table(bundle, fit_df)
    fit_metrics.to_csv(outdir / "fit_metrics.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        outdir / "predictions_and_params.npz",
        years=bundle.years,
        pred_all=data["pred_all"],
        params=np.array([data["params"]], dtype=object),
    )

    print("\nInferred parameters")
    for k, v in data["params"].items():
        print(f"  {k:>4s}: {v:.6f}")

    print("\nFit metrics")
    print(fit_metrics.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndvi", type=str, default=str(DEFAULT_NDVI_FILE))
    parser.add_argument("--water", type=str, default=str(DEFAULT_WATER_FILE))
    parser.add_argument("--tvdi", type=str, default=str(DEFAULT_TVDI_FILE))
    parser.add_argument("--outdir", type=str, default="results_real")
    parser.add_argument("--train-end-year", type=int, default=2018)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lbfgs", type=int, default=200)
    parser.add_argument("--collocation", type=int, default=3000)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--forecast-years", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = Config(
        seed=args.seed,
        train_end_year=args.train_end_year,
        adam_epochs=args.epochs,
        lbfgs_steps=args.lbfgs,
        n_collocation=args.collocation,
        n_restarts=args.restarts,
        forecast_years=args.forecast_years,
    )
    set_seed(cfg.seed)

    df = load_real_data(Path(args.ndvi), Path(args.water), Path(args.tvdi))
    bundle = build_bundle(df, cfg.train_end_year)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE:  # pragma: no cover
        print("Torch is not installed, so training cannot run here.")
        print("Install torch and rerun on your machine.")
        return

    run_experiment(cfg, bundle, outdir)


if __name__ == "__main__":
    main()
