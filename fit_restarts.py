"""Multi-restart PINN ensemble for uncertainty propagation.

Trains the physics-informed network of :mod:`v1` from several independent
initialisations and stores every inferred parameter set.  The ensemble is the
input of the restart-robustness check in ``bistability_pinn.py``: a bifurcation
statement is only reported as robust if the fold/threshold locations agree
across restarts.

Usage
-----
    python fit_restarts.py --ndvi gee_inputs/tarim/NDVI.xlsx \
        --water gee_inputs/tarim/water.xlsx --tvdi gee_inputs/tarim/TVDI.xlsx \
        --out results_gee_tarim/restart_params.csv --n 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import v1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndvi", default=r"gee_inputs/tarim/NDVI.xlsx")
    ap.add_argument("--water", default=r"gee_inputs/tarim/water.xlsx")
    ap.add_argument("--tvdi", default=r"gee_inputs/tarim/TVDI.xlsx")
    ap.add_argument("--out", default=r"results_gee_tarim/restart_params.csv")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2500)
    ap.add_argument("--lbfgs", type=int, default=200)
    ap.add_argument("--collocation", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-end-year", type=int, default=2018)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = v1.load_real_data(Path(args.ndvi), Path(args.water), Path(args.tvdi))
    bundle = v1.build_bundle(df, args.train_end_year)

    rows = []
    t_start = time.time()
    for i in range(args.n):
        seed = args.seed + 17 * i
        cfg = v1.Config(
            seed=seed,
            train_end_year=args.train_end_year,
            adam_epochs=args.epochs,
            lbfgs_steps=args.lbfgs,
            n_collocation=args.collocation,
            n_restarts=1,
        )
        print(f"[restart {i + 1}/{args.n}] seed={seed}", flush=True)
        run = v1.train_once(cfg, bundle, seed)
        row = {"restart": i, "seed": seed, **run["params"]}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(
            "    " + "  ".join(f"{k}={v:.4g}" for k, v in run["params"].items()),
            flush=True,
        )
        print(f"    elapsed {time.time() - t_start:.0f}s", flush=True)

    print(f"wrote {out} ({len(rows)} restarts)")


if __name__ == "__main__":
    main()
