"""Same-budget ablation of the PINN loss terms on the GEE Tarim series."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import v1


def train(bundle, mode: str, seed: int, epochs: int, lbfgs_steps: int):
    import torch

    v1.set_seed(seed)
    model = v1.TarimAnnualPINN(4, 64, bundle.year_min, bundle.year_max)
    t_train = torch.tensor(bundle.x_train, dtype=torch.float32)
    y_train = torch.tensor(bundle.y_train, dtype=torch.float32)
    t_all = torch.tensor(bundle.x_all, dtype=torch.float32)
    t_coll = torch.tensor(np.random.uniform(bundle.year_min, bundle.year_max,
                                            (3000, 1)), dtype=torch.float32)
    x0, y0 = t_train[:1], y_train[:1]
    fn = bundle.forcing_fn

    def loss_parts():
        ld = v1.mse(model(t_train), y_train)
        rp, rt, rr = model.residuals(t_coll, fn)
        lp = v1.mse(rp, torch.zeros_like(rp)) + v1.mse(rt, torch.zeros_like(rt)) + v1.mse(rr, torch.zeros_like(rr))
        li = v1.mse(model(x0), y0)
        return ld, lp, li

    def total():
        ld, lp, li = loss_parts()
        if mode == "data_only":
            return 2.0 * ld
        if mode == "data_ode":
            return 2.0 * ld + 10.0 * lp
        return 2.0 * ld + 10.0 * lp + 5.0 * li

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = total()
        loss.backward()
        opt.step()
    if lbfgs_steps > 0 and mode != "data_only":
        opt2 = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_steps,
                                 tolerance_grad=1e-9, tolerance_change=1e-11)

        def closure():
            opt2.zero_grad()
            loss = total()
            loss.backward()
            return loss

        opt2.step(closure)
    with torch.no_grad():
        pred = model(t_all).numpy()
    return pred


def metrics(bundle, pred):
    out = {}
    for i, name in enumerate(["NDVI", "TVDI", "R"]):
        y = bundle.y_all[bundle.test_mask, i]
        p = pred[bundle.test_mask, i]
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        ss = float(np.sum((y - y.mean()) ** 2))
        out[f"{name}_RMSE"] = rmse
        out[f"{name}_R2"] = 1.0 - float(np.sum((y - p) ** 2)) / ss
    out["avg_RMSE"] = float(np.mean([out["NDVI_RMSE"], out["TVDI_RMSE"], out["R_RMSE"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="gee_inputs/tarim/series.csv")
    ap.add_argument("--outdir", default="results_ablation")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lbfgs", type=int, default=60)
    ap.add_argument("--restarts", type=int, default=2)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bundle = v1.build_bundle(pd.read_csv(args.series), 2018)
    rows = []
    for mode in ["data_only", "data_ode", "full"]:
        for k in range(args.restarts):
            seed = 7 + 17 * k
            print("mode", mode, "restart", k + 1, flush=True)
            pred = train(bundle, mode, seed, args.epochs, args.lbfgs)
            row = {"mode": mode, "seed": seed, **metrics(bundle, pred)}
            rows.append(row)
            pd.DataFrame(rows).to_csv(outdir / "ablation_metrics.csv", index=False)
    df = pd.DataFrame(rows)
    summary = df.groupby("mode").mean(numeric_only=True).reset_index()
    summary.to_csv(outdir / "ablation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
