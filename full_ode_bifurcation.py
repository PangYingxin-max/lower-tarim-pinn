"""Full three-dimensional ODE equilibrium and bifurcation screening.

The time-dependent PINN uses the ecological-water series as forcing.  For
critical-point analysis we replace that forcing by a constant dimensionless
level mu and analyse the original three-dimensional ODE directly:

    P' = a1 P(1-P)(1-T) + a2 R - a3 T P - a4 P
    T' = b1(1-P) - b2 R - b3 T
    R' = c1(mu-R)

This is a numerical screening/continuation aid.  A point is reported as a
candidate fold or Hopf point only after residual and eigenvalue checks.  It
does not by itself establish an ecological threshold.

Example:
  python full_ode_bifurcation.py --outdir results_full_bifurcation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import root

import v1


def load_params(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Parameter cache not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "params" not in data:
        raise KeyError(f"No 'params' array in {path}")
    obj = data["params"].item() if data["params"].ndim == 0 else data["params"].ravel()[0]
    if not isinstance(obj, dict):
        raise TypeError("Cached params must contain a dictionary")
    required = ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "c1"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise KeyError(f"Missing parameters: {missing}")
    return {k: float(obj[k]) for k in required}


def rhs(x: np.ndarray, mu: float, p: Dict[str, float]) -> np.ndarray:
    P, T, R = x
    return np.array([
        p["a1"] * P * (1 - P) * (1 - T) + p["a2"] * R - p["a3"] * T * P - p["a4"] * P,
        p["b1"] * (1 - P) - p["b2"] * R - p["b3"] * T,
        p["c1"] * (mu - R),
    ], dtype=float)


def jacobian(x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    P, T, _ = x
    return np.array([
        [p["a1"] * (1 - 2 * P) * (1 - T) - p["a3"] * T - p["a4"],
         -p["a1"] * P * (1 - P) - p["a3"] * P, p["a2"]],
        [-p["b1"], -p["b3"], -p["b2"]],
        [0.0, 0.0, -p["c1"]],
    ], dtype=float)


def char_coefficients(J: np.ndarray) -> Tuple[float, float, float]:
    # det(lambda I - J) = lambda^3 + A1 lambda^2 + A2 lambda + A3
    coeff = np.poly(J).real
    return float(coeff[1]), float(coeff[2]), float(coeff[3])


def unique_rows(rows: List[np.ndarray], tol: float = 1e-5) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for row in rows:
        if not any(np.linalg.norm(row - old) < tol for old in out):
            out.append(row)
    return out


def equilibria_at_mu(mu: float, p: Dict[str, float], guesses: np.ndarray) -> List[np.ndarray]:
    sols: List[np.ndarray] = []
    for guess in guesses:
        sol = root(lambda x: rhs(x, mu, p), guess, method="hybr")
        if not sol.success or np.max(np.abs(rhs(sol.x, mu, p))) > 1e-7:
            continue
        x = sol.x.astype(float)
        if np.all(np.isfinite(x)) and np.all(x >= -1e-6) and np.all(x <= 1.5):
            sols.append(x)
    return unique_rows(sols)


def scan_branch(p: Dict[str, float], mu_grid: np.ndarray) -> pd.DataFrame:
    guesses = np.array([
        [P, T, R] for P in np.linspace(0.02, 0.98, 9)
        for T in np.linspace(0.02, 0.98, 7)
        for R in np.linspace(0.0, 1.0, 4)
    ], dtype=float)
    rows = []
    for mu in mu_grid:
        eqs = equilibria_at_mu(float(mu), p, guesses)
        # The previous grid solutions are useful continuation guesses.
        if eqs:
            guesses = np.vstack([guesses, np.asarray(eqs)])
        for branch_id, x in enumerate(eqs):
            J = jacobian(x, p)
            eig = np.linalg.eigvals(J)
            A1, A2, A3 = char_coefficients(J)
            order = np.argsort(eig.real)[::-1]
            eig = eig[order]
            rows.append({
                "mu": float(mu), "P": x[0], "T": x[1], "R": x[2],
                "residual": float(np.max(np.abs(rhs(x, mu, p)))),
                "detJ": float(np.linalg.det(J)), "traceJ": float(np.trace(J)),
                "max_real": float(np.max(eig.real)),
                "eig1_real": float(eig[0].real), "eig1_imag": float(eig[0].imag),
                "eig2_real": float(eig[1].real), "eig2_imag": float(eig[1].imag),
                "eig3_real": float(eig[2].real), "eig3_imag": float(eig[2].imag),
                "A1": A1, "A2": A2, "A3": A3, "hopf_indicator": A1 * A2 - A3,
                "branch_id": branch_id,
            })
    return pd.DataFrame(rows)


def fold_equations(z: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    x = z[:3]
    mu = float(z[3])
    return np.r_[rhs(x, mu, p), np.linalg.det(jacobian(x, p))]


def hopf_equations(z: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    x = z[:3]
    mu = float(z[3])
    J = jacobian(x, p)
    A1, A2, A3 = char_coefficients(J)
    return np.r_[rhs(x, mu, p), A1 * A2 - A3]


def solve_candidates(df: pd.DataFrame, p: Dict[str, float], mu_min: float, mu_max: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    seeds = df.iloc[::max(1, len(df) // 100)].copy()
    folds, hopfs = [], []
    for _, row in seeds.iterrows():
        # Use explicit column indexing; attribute access can collide with
        # pandas Series attributes (notably for short column names).
        z0 = np.array([row["P"], row["T"], row["R"], row["mu"]], dtype=float)
        for kind, fun, target in [("fold", fold_equations, folds), ("hopf", hopf_equations, hopfs)]:
            sol = root(fun, z0, args=(p,), method="hybr")
            if not sol.success or np.max(np.abs(fun(sol.x, p))) > 1e-5:
                continue
            z = sol.x
            if not (mu_min - 1e-6 <= z[3] <= mu_max + 1e-6 and np.all(z[:3] >= -1e-5) and np.all(z[:3] <= 1.5)):
                continue
            J = jacobian(z[:3], p)
            eig = np.linalg.eigvals(J)
            if kind == "fold":
                valid = np.min(np.abs(eig)) < 1e-4 and np.sum(np.abs(eig) < 1e-4) == 1
            else:
                pair = np.sort(np.abs(eig.real))[:2]
                valid = pair[1] < 1e-4 and np.max(np.abs(eig.imag)) > 1e-5 and np.max(eig.real[np.abs(eig.imag) < 1e-5]) < -1e-5
            if valid:
                record = {"mu": float(z[3]), "P": float(z[0]), "T": float(z[1]), "R": float(z[2]),
                          "residual": float(np.max(np.abs(rhs(z[:3], z[3], p)))),
                          "detJ": float(np.linalg.det(J)), "eigvals": ";".join(f"{e.real:.8g}{e.imag:+.8g}i" for e in eig)}
                if not any(abs(record["mu"] - old["mu"]) < 1e-4 for old in target):
                    target.append(record)
    return pd.DataFrame(folds), pd.DataFrame(hopfs)


def make_plots(df: pd.DataFrame, folds: pd.DataFrame, hopfs: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    if df.empty:
        raise RuntimeError("No equilibria were found; inspect parameter cache and mu range.")
    stable = df["max_real"] < 0
    axes[0].scatter(df.loc[stable, "mu"], df.loc[stable, "P"], s=5, c="#1769aa", label="Stable")
    axes[0].scatter(df.loc[~stable, "mu"], df.loc[~stable, "P"], s=5, c="#888888", label="Unstable")
    if not folds.empty:
        axes[0].scatter(folds["mu"], folds["P"], c="#c62828", marker="X", s=65, label="Fold candidate")
    if not hopfs.empty:
        axes[0].scatter(hopfs["mu"], hopfs["P"], c="#ef6c00", marker="D", s=48, label="Hopf candidate")
    axes[0].set(xlabel=r"Constant normalized water input $\mu$", ylabel=r"Equilibrium vegetation $P^*$", title="Full ODE equilibrium branches")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter(df.mu, df.max_real, s=5, c=np.where(stable, "#1769aa", "#888888"))
    axes[1].axhline(0, c="black", lw=0.8)
    for table, color, label in [(folds, "#c62828", "Fold"), (hopfs, "#ef6c00", "Hopf")]:
        if not table.empty:
            axes[1].scatter(table["mu"], np.zeros(len(table)), c=color, marker="X", s=55, label=label)
    axes[1].set(xlabel=r"$\mu$", ylabel=r"$\max Re(\lambda)$", title="Linear stability")
    if not folds.empty or not hopfs.empty:
        axes[1].legend(frameon=False, fontsize=8)

    axes[2].scatter(df["mu"], df["T"], s=5, c="#7b3294", label=r"$T^*$")
    axes[2].scatter(df["mu"], df["R"], s=5, c="#2ca25f", label=r"$R^*$")
    axes[2].set(xlabel=r"$\mu$", ylabel="Equilibrium state", title="Other equilibrium components")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "fig_full_ode_bifurcation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndvi", default=str(v1.DEFAULT_NDVI_FILE))
    ap.add_argument("--water", default=str(v1.DEFAULT_WATER_FILE))
    ap.add_argument("--tvdi", default=str(v1.DEFAULT_TVDI_FILE))
    ap.add_argument("--params", default="results_real/predictions_and_params.npz")
    ap.add_argument("--outdir", default="results_full_bifurcation")
    ap.add_argument("--mu-min", type=float, default=-0.15)
    ap.add_argument("--mu-max", type=float, default=1.35)
    ap.add_argument("--n-mu", type=int, default=501)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df_raw = v1.load_real_data(Path(args.ndvi), Path(args.water), Path(args.tvdi))
    bundle = v1.build_bundle(df_raw, train_end_year=2018)
    params = load_params(Path(args.params))
    mu_grid = np.linspace(args.mu_min, args.mu_max, args.n_mu)
    df = scan_branch(params, mu_grid)
    folds, hopfs = solve_candidates(df, params, args.mu_min, args.mu_max)
    df.to_csv(outdir / "full_ode_equilibria.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(outdir / "fold_candidates.csv", index=False, encoding="utf-8-sig")
    hopfs.to_csv(outdir / "hopf_candidates.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(outdir / "full_ode_bifurcation.npz", mu=mu_grid, params=np.array([params], dtype=object))
    make_plots(df, folds, hopfs, outdir)
    print(f"Equilibria: {len(df)}")
    print(f"Fold candidates: {len(folds)}")
    print(f"Hopf candidates: {len(hopfs)}")
    if folds.empty:
        print("No validated fold candidate found in the scanned mu range.")
    if hopfs.empty:
        print("No validated Hopf candidate found in the scanned mu range.")


if __name__ == "__main__":
    main()
