"""Parameter-perturbation robustness scan for the full three-dimensional ODE.

`formal_ode_verification.py` continues the equilibria of the calibrated model
and applies the standard nondegeneracy (fold) and Routh-Hurwitz/transversality
(Hopf) screens to that single parameter set. This script asks the follow-up
question: how stable is that negative result when the PINN parameters are
perturbed?

Only one calibrated parameter dictionary was archived, so restart-level
uncertainty cannot be measured directly. Instead the parameters are perturbed
multiplicatively by +/-10%, +/-20% and +/-30% (200 draws each) and the analytic
equilibrium branches are screened for fold and Hopf crossings over the same
normalized water-input interval that the observations span. The result is a
statement about parameter sensitivity, not about restart statistics.

Run:  python bifurcation_robustness.py --outdir results_robustness
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import full_ode_bifurcation as fb
import v1

PARAM_NAMES = ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "c1"]


def equilibria_analytic(mu: np.ndarray, p: Dict[str, float]) -> List[np.ndarray]:
    """Analytic equilibrium branches of the three-state ODE at constant forcing mu.

    With R* = mu the model reduces to a quadratic in P, so branches can be
    evaluated on a dense grid without repeated root finding.
    """
    a1, a2, a3, a4 = p["a1"], p["a2"], p["a3"], p["a4"]
    b1, b2, b3 = p["b1"], p["b2"], p["b3"]
    t0, t1 = b1 / b3, b2 / b3
    A = 1.0 - t0 + t1 * mu
    q2 = -a1 * t0 * np.ones_like(mu)
    q1 = a1 * (t0 - A) + a3 * t0
    q0 = a1 * A + a2 * mu - a3 * t0 + a3 * t1 * mu - a4
    disc = q1 * q1 - 4.0 * q2 * q0
    ok = disc >= 0.0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    branches = [np.zeros_like(mu)]
    for sign in (1.0, -1.0):
        branches.append(np.where(ok, (-q1 + sign * sq) / (2.0 * q2), np.nan))
    out = []
    for P in branches:
        T = t0 * (1.0 - P) - t1 * mu
        keep = np.isfinite(P) & (P >= -1e-9) & (P <= 1.5) & (T >= -1e-9) & (T <= 1.5)
        out.append(np.where(keep, P, np.nan))
    return out


def jacobian_stack(P: np.ndarray, T: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    J = np.zeros(P.shape + (3, 3), dtype=float)
    J[..., 0, 0] = p["a1"] * (1.0 - 2.0 * P) * (1.0 - T) - p["a3"] * T - p["a4"]
    J[..., 0, 1] = -p["a1"] * P * (1.0 - P) - p["a3"] * P
    J[..., 0, 2] = p["a2"]
    J[..., 1, 0] = -p["b1"]
    J[..., 1, 1] = -p["b3"]
    J[..., 1, 2] = -p["b2"]
    J[..., 2, 2] = -p["c1"]
    return J


def diagnostics(P: np.ndarray, T: np.ndarray, p: Dict[str, float]) -> Dict[str, np.ndarray]:
    J = jacobian_stack(P, T, p)
    det = np.linalg.det(J)
    A1 = -(J[..., 0, 0] + J[..., 1, 1] + J[..., 2, 2])
    A2 = (J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
          + J[..., 0, 0] * J[..., 2, 2] - J[..., 0, 2] * J[..., 2, 0]
          + J[..., 1, 1] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 1])
    A3 = det
    return {"det": det, "A1": A1, "A2": A2, "A3": A3, "RH": A1 * A2 - A3}


def crossings(mu: np.ndarray, values: np.ndarray) -> List[float]:
    """Interpolated mu positions where a continuous branch changes sign."""
    out: List[float] = []
    finite = np.isfinite(values)
    for i in range(len(mu) - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        v0, v1 = values[i], values[i + 1]
        if v0 == 0.0:
            out.append(float(mu[i]))
        elif v0 * v1 < 0.0:
            w = v0 / (v0 - v1)
            out.append(float(mu[i] + w * (mu[i + 1] - mu[i])))
    return out


def screen(params: Dict[str, float], mu: np.ndarray) -> Dict[str, object]:
    folds: List[float] = []
    hopfs: List[float] = []
    min_abs_det = np.inf
    t0, t1 = params["b1"] / params["b3"], params["b2"] / params["b3"]
    for P in equilibria_analytic(mu, params):
        if not np.any(np.isfinite(P)):
            continue
        T = t0 * (1.0 - P) - t1 * mu
        d = diagnostics(P, T, params)
        finite_det = np.abs(d["det"][np.isfinite(d["det"])])
        if finite_det.size:
            min_abs_det = min(min_abs_det, float(finite_det.min()))
        folds.extend(crossings(mu, d["det"]))
        hopf_values = np.where((d["A1"] > 0) & (d["A3"] > 0), d["RH"], np.nan)
        hopfs.extend(crossings(mu, hopf_values))
    return {"fold_mu": sorted(set(round(f, 4) for f in folds)),
            "hopf_mu": sorted(set(round(h, 4) for h in hopfs)),
            "min_abs_det": min_abs_det}


def observed_mu_range(ndvi: Path, water: Path, tvdi: Path, train_end_year: int):
    df = v1.load_real_data(ndvi, water, tvdi)
    years = df["Year"].to_numpy(dtype=float)
    r_raw = df["R_raw"].to_numpy(dtype=float)
    vmin, vmax = v1.minmax_fit(r_raw[years <= train_end_year])
    return years, (r_raw - vmin) / (vmax - vmin)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndvi", default=str(v1.DEFAULT_NDVI_FILE))
    ap.add_argument("--water", default=str(v1.DEFAULT_WATER_FILE))
    ap.add_argument("--tvdi", default=str(v1.DEFAULT_TVDI_FILE))
    ap.add_argument("--params", default="results_real/predictions_and_params.npz")
    ap.add_argument("--outdir", default="results_robustness")
    ap.add_argument("--train-end-year", type=int, default=2018)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    params = fb.load_params(Path(args.params))
    years, mu_obs = observed_mu_range(Path(args.ndvi), Path(args.water), Path(args.tvdi), args.train_end_year)
    lo, hi = float(mu_obs.min()), float(mu_obs.max())
    mu = np.linspace(min(0.0, lo) - 0.05, hi + 0.05, 801)

    base = screen(params, mu)
    rng = np.random.default_rng(args.seed)
    rows = []
    for scale in (0.10, 0.20, 0.30):
        for draw in range(args.draws):
            eps = rng.uniform(-scale, scale, size=len(PARAM_NAMES))
            pert = {k: float(params[k] * (1.0 + e)) for k, e in zip(PARAM_NAMES, eps)}
            res = screen(pert, mu)
            rows.append({
                "scale": scale,
                "draw": draw,
                "n_fold": len(res["fold_mu"]),
                "n_hopf": len(res["hopf_mu"]),
                "fold_mu": ";".join(format(f, ".4f") for f in res["fold_mu"]),
                "hopf_mu": ";".join(format(h, ".4f") for h in res["hopf_mu"]),
                "min_abs_det": res["min_abs_det"],
            })
    df = pd.DataFrame(rows)
    in_range = lambda s: any(lo <= float(v) <= hi for v in s.split(";") if v)
    df["fold_in_observed"] = df["fold_mu"].apply(in_range)
    df["hopf_in_observed"] = df["hopf_mu"].apply(in_range)
    df.to_csv(outdir / "robustness_draws.csv", index=False, encoding="utf-8-sig")

    summary = df.groupby("scale").agg(
        draws=("draw", "size"),
        fold_rate=("fold_in_observed", "mean"),
        hopf_rate=("hopf_in_observed", "mean"),
        any_fold=("n_fold", lambda s: float((s > 0).mean())),
        any_hopf=("n_hopf", lambda s: float((s > 0).mean())),
        min_abs_det=("min_abs_det", "min"),
    ).reset_index()
    summary.to_csv(outdir / "robustness_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "Full-ODE critical-point robustness under parameter perturbation",
        "Parameter source: " + str(args.params),
        "Observed normalized water-input range: [" + format(lo, ".3f") + ", " + format(hi, ".3f") + "]",
        "Scanned interval: [" + format(float(mu.min()), ".3f") + ", " + format(float(mu.max()), ".3f") + "] with " + str(len(mu)) + " points",
        "Nominal set: folds=" + str(base["fold_mu"] or "none") + ", hopf=" + str(base["hopf_mu"] or "none")
        + ", min|det J|=" + format(base["min_abs_det"], ".3e"),
        "",
        summary.to_string(index=False),
        "",
        "Interpretation: a rate of 0 means no perturbed parameter set produced a sign change of det J (fold screen) or of the Routh-Hurwitz indicator with A1>0, A3>0 (Hopf screen) inside the observed water-input interval.",
    ]
    (outdir / "robustness_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, col, title in ((axes[0], "fold_mu", "Fold-type det J crossings"),
                           (axes[1], "hopf_mu", "Hopf-type RH crossings")):
        vals = []
        for s in df[col]:
            vals.extend(float(v) for v in s.split(";") if v)
        if vals:
            ax.hist(vals, bins=40, color="#1769aa", alpha=0.85)
        else:
            ax.text(0.5, 0.5, "no crossings in any draw", ha="center", va="center", transform=ax.transAxes)
        ax.axvspan(lo, hi, color="#c62828", alpha=0.12, label="observed range")
        ax.set_xlabel("Normalized water input")
        ax.set_ylabel("Draws")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
