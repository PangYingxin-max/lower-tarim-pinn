"""Recompute windowed evaluation metrics for the PINN calibration.

The earlier manuscript reported a single 2019--2023 score.  This script
refits the PINN on a named series and reports three separate windows:

  * calibration reconstruction (2000--2018)
  * withheld observations (2019--2020)
  * trend-projection window (2021--2023, not an independent validation)

Run:
  python split_eval.py --series legacy --outdir results_split_legacy
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import v1

ROOT = Path(__file__).resolve().parent


def load_series(kind: str) -> pd.DataFrame:
    if kind == "legacy":
        df = pd.read_csv(ROOT / "results_real" / "series_old.csv")
        return df.rename(columns={"TVDI_region": "TVDI"})[["Year", "NDVI", "TVDI", "R_raw"]]
    df = pd.read_csv(ROOT / "gee_inputs" / "tarim" / "series.csv")
    return df[["Year", "NDVI", "TVDI_region", "R_raw"]].rename(columns={"TVDI_region": "TVDI"})


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((obs - pred) ** 2)) / ss_tot


def window_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    return {
        "n": int(obs.size),
        "R2": r2(obs, pred),
        "RMSE": float(np.sqrt(np.mean((obs - pred) ** 2))),
        "MAE": float(np.mean(np.abs(obs - pred))),
    }


def lag1_freerun(y_train: np.ndarray, n_steps: int) -> np.ndarray:
    lag, nxt = y_train[:-1], y_train[1:]
    design = np.column_stack([np.ones_like(lag), lag])
    coef, *_ = np.linalg.lstsq(design, nxt, rcond=None)
    out, last = [], float(y_train[-1])
    for _ in range(n_steps):
        last = float(coef[0] + coef[1] * last)
        out.append(last)
    return np.asarray(out)


def ode_forward(bundle: v1.DataBundle, params: Dict[str, float]) -> np.ndarray:
    """Integrate the calibrated ODE forward from the first observation."""
    years = bundle.years

    def rhs(t: float, y: np.ndarray) -> List[float]:
        P, T, R = y
        forcing = float(bundle.forcing_fn(np.array([t]))[0])
        return [
            params["a1"] * P * (1.0 - P) * (1.0 - T) + params["a2"] * R
            - params["a3"] * T * P - params["a4"] * P,
            params["b1"] * (1.0 - P) - params["b2"] * R - params["b3"] * T,
            params["c1"] * (forcing - R),
        ]

    sol = solve_ivp(rhs, (years[0], years[-1]), bundle.y_all[0].astype(float),
                    t_eval=years, rtol=1e-9, atol=1e-11)
    return sol.y.T if sol.success else np.full((len(years), 3), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", choices=["legacy", "gee"], default="legacy")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--lbfgs", type=int, default=200)
    ap.add_argument("--collocation", type=int, default=3000)
    ap.add_argument("--train-end-year", type=int, default=2018)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_series(args.series)
    bundle = v1.build_bundle(raw, args.train_end_year)
    years = bundle.years
    tr = bundle.train_mask
    withheld = (years >= 2019) & (years <= 2020)
    projection = years >= 2021

    v1.set_seed(7)
    base_cfg = v1.Config(
        seed=7,
        train_end_year=args.train_end_year,
        adam_epochs=args.epochs,
        lbfgs_steps=args.lbfgs,
        n_collocation=args.collocation,
        n_restarts=args.restarts,
        forecast_years=5,
    )

    runs = []
    for i in range(args.restarts):
        seed = 7 + 17 * i
        local = base_cfg if i == 0 else dataclasses.replace(
            base_cfg, seed=seed,
            adam_epochs=max(1200, args.epochs // 2),
            lbfgs_steps=max(80, args.lbfgs // 2),
            n_collocation=max(1500, args.collocation // 2),
        )
        run = v1.train_once(local, bundle, seed)
        train_mse = float(np.mean((run["pred_all"][tr] - bundle.y_all[tr]) ** 2))
        runs.append({"seed": seed, "run": run, "train_mse": train_mse})
        print("restart seed=%d train_mse=%.6f" % (seed, train_mse), flush=True)

    best = runs[0]  # seed 7 is the primary run; the others give the restart spread
    pred = best["run"]["pred_all"].astype(float)
    params = best["run"]["params"]
    history = best["run"]["history"]

    ode_pred = ode_forward(bundle, params)
    persistence = np.concatenate([bundle.y_all[:1], bundle.y_all[:-1]]).astype(float)
    ar1 = np.vstack([
        bundle.y_all[tr],
        np.column_stack([lag1_freerun(bundle.y_all[tr, k], int((~tr).sum())) for k in range(3)]),
    ]).astype(float)

    names = ["NDVI", "TVDI", "Water input"]
    windows = [
        ("Calibration reconstruction (2000-2018)", tr),
        ("Withheld observations (2019-2020)", withheld),
        ("Trend-projection window (2021-2023)", projection),
        ("Combined evaluation window (2019-2023)", ~tr),
    ]
    models = [("PINN", pred), ("Nonlinear ODE (forward)", ode_pred),
              ("Persistence", persistence), ("AR(1) free run", ar1)]

    rows = []
    for model, series in models:
        for k, name in enumerate(names):
            for label, mask in windows:
                rows.append({"model": model, "variable": name, "window": label,
                             **window_metrics(bundle.y_all[mask, k], series[mask, k])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "split_metrics.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([params]).to_csv(outdir / "params.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        outdir / "predictions.npz",
        years=years,
        pred_all=pred,
        y_all=bundle.y_all,
        train_mask=tr,
        history_total=np.asarray(history["total"], float),
        history_data=np.asarray(history["data"], float),
        history_pde=np.asarray(history["pde"], float),
        history_ic=np.asarray(history["ic"], float),
    )

    scalers = {"NDVI": bundle.scalers["NDVI"], "TVDI": bundle.scalers["TVDI"],
               "Water input": bundle.scalers["R_raw"]}
    raw_rows = []
    for k, name in enumerate(names):
        lo, hi = scalers[name]
        obs_raw = v1.minmax_inverse(bundle.y_all[:, k], lo, hi)
        pred_raw = v1.minmax_inverse(pred[:, k], lo, hi)
        for label, mask in windows:
            raw_rows.append({"variable": name, "window": label,
                             **window_metrics(obs_raw[mask], pred_raw[mask])})
    pd.DataFrame(raw_rows).to_csv(outdir / "split_metrics_physical.csv",
                                  index=False, encoding="utf-8-sig")

    print("\nPINN windowed metrics (normalized scale)")
    print(metrics[metrics["model"] == "PINN"].to_string(index=False))

    (outdir / "summary.json").write_text(json.dumps(
        {"series": args.series, "seed": best["seed"], "params": params,
         "restart_params": [{"seed": r["seed"], **r["run"]["params"]} for r in runs],
         "restart_train_mse": [r["train_mse"] for r in runs]},
        indent=2), encoding="utf-8")
    print("\nWrote", outdir)


if __name__ == "__main__":
    main()
