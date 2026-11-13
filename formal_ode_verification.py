"""Pseudo-arclength continuation and bifurcation validation for the full ODE.

This script analyses the original three-state model used by v1.py.  It is a
lightweight, reproducible continuation implementation for this one-parameter
equilibrium problem; it does not replace AUTO/MATCONT/COCO for periodic-orbit
continuation.  It reports nondegeneracy and transversality diagnostics when
candidate points exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import root

import full_ode_bifurcation as fb
import v1


def f_mu(x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    out = np.zeros(3, dtype=float)
    out[2] = p["c1"]
    return out


def null_tangent(x: np.ndarray, mu: float, p: Dict[str, float], previous: np.ndarray | None = None) -> np.ndarray:
    A = np.column_stack([fb.jacobian(x, p), f_mu(x, p)])
    _, _, vh = np.linalg.svd(A)
    t = vh[-1]
    t = t / np.linalg.norm(t)
    if previous is not None and np.dot(t, previous) < 0:
        t = -t
    return t


def continue_branch(x0: np.ndarray, mu0: float, p: Dict[str, float], ds: float, steps: int, mu_end: float) -> pd.DataFrame:
    z = np.r_[x0, mu0].astype(float)
    tangent = null_tangent(x0, mu0, p)
    rows: List[dict] = []
    for step in range(steps + 1):
        x, mu = z[:3], float(z[3])
        J = fb.jacobian(x, p)
        eig = np.linalg.eigvals(J)
        A1, A2, A3 = fb.char_coefficients(J)
        rows.append({
            "step": step, "P": x[0], "T": x[1], "R": x[2], "mu": mu,
            "residual": np.max(np.abs(fb.rhs(x, mu, p))),
            "detJ": np.linalg.det(J), "max_real": np.max(eig.real),
            "max_imag": np.max(np.abs(eig.imag)), "RH": A1 * A2 - A3,
        })
        z_pred = z + ds * tangent

        def correct(w: np.ndarray) -> np.ndarray:
            zz = w[:4]
            return np.r_[fb.rhs(zz[:3], zz[3], p), np.dot(zz - z_pred, tangent)]

        sol = root(correct, z_pred, method="hybr", options={"xtol": 1e-10})
        if not sol.success or np.max(np.abs(correct(sol.x))) > 1e-7:
            break
        z_new = sol.x.astype(float)
        if not np.all(np.isfinite(z_new)) or np.any(z_new[:3] < -1e-5) or np.any(z_new[:3] > 1.5):
            break
        if z_new[3] > mu_end + 0.25:
            break
        tangent_new = null_tangent(z_new[:3], z_new[3], p, tangent)
        z, tangent = z_new, tangent_new
    return pd.DataFrame(rows)


def second_directional_derivative(x: np.ndarray, mu: float, p: Dict[str, float], v: np.ndarray, h: float = 1e-5) -> np.ndarray:
    # F_xx[v,v], with the parameter held fixed.
    return (fb.rhs(x + h * v, mu, p) - 2 * fb.rhs(x, mu, p) + fb.rhs(x - h * v, mu, p)) / h**2


def fold_diagnostics(row: pd.Series, p: Dict[str, float]) -> dict:
    x = row[["P", "T", "R"]].to_numpy(float)
    mu = float(row["mu"])
    J = fb.jacobian(x, p)
    u, s, vh = np.linalg.svd(J)
    v = vh[-1]
    w = u[:, -1]
    a = float(np.dot(w, second_directional_derivative(x, mu, p, v)))
    b = float(np.dot(w, f_mu(x, p)))
    eig = np.linalg.eigvals(J)
    return {"mu": mu, "P": x[0], "T": x[1], "R": x[2], "detJ": np.linalg.det(J),
            "smallest_singular": s[-1], "fold_a_wFxxvv": a, "fold_b_wFmu": b,
            "fold_nondegenerate": bool(abs(a) > 1e-5 and abs(b) > 1e-5),
            "eigvals": ";".join(f"{e.real:.8g}{e.imag:+.8g}i" for e in eig)}


def hopf_diagnostics(branch: pd.DataFrame, p: Dict[str, float]) -> pd.DataFrame:
    # Locate sign changes of the Routh-Hurwitz Hopf indicator A1*A2-A3.
    out = []
    if len(branch) < 3:
        return pd.DataFrame()
    for i in range(1, len(branch) - 1):
        left, cur, right = branch.iloc[i - 1], branch.iloc[i], branch.iloc[i + 1]
        if left["RH"] * right["RH"] > 0:
            continue
        x = cur[["P", "T", "R"]].to_numpy(float)
        mu = float(cur["mu"])
        J = fb.jacobian(x, p)
        eig = np.linalg.eigvals(J)
        complex_pair = np.sum(np.abs(eig.imag) > 1e-6) >= 2
        omega = float(np.max(np.abs(eig.imag)))
        # Finite-difference crossing rate along the computed equilibrium branch.
        trans = float((right["max_real"] - left["max_real"]) / (right["mu"] - left["mu"])) if abs(right["mu"] - left["mu"]) > 1e-12 else np.nan
        other_real = [e.real for e in eig if abs(e.imag) < 1e-6]
        out.append({"mu": mu, "P": x[0], "T": x[1], "R": x[2], "RH": cur["RH"],
                    "omega": omega, "transversality_dReLambda_dmu": trans,
                    "complex_pair": bool(complex_pair),
                    "hopf_nondegenerate": bool(complex_pair and omega > 1e-6 and abs(trans) > 1e-5),
                    "other_real_eigenvalue": other_real[0] if other_real else np.nan,
                    "eigvals": ";".join(f"{e.real:.8g}{e.imag:+.8g}i" for e in eig)})
    return pd.DataFrame(out)


def plot(branch: pd.DataFrame, fold: pd.DataFrame, hopf: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    stable = branch.max_real < 0
    ax[0].plot(branch.loc[stable, "mu"], branch.loc[stable, "P"], ".", ms=3, label="Stable")
    ax[0].plot(branch.loc[~stable, "mu"], branch.loc[~stable, "P"], ".", ms=3, color="0.55", label="Unstable")
    if not fold.empty:
        ax[0].scatter(fold.mu, fold.P, marker="X", s=70, color="crimson", label="Validated fold")
    if not hopf.empty:
        ax[0].scatter(hopf.mu, hopf.P, marker="D", s=45, color="#e67e22", label="Hopf candidate")
    ax[0].set(xlabel=r"Constant normalized water input $\mu$", ylabel=r"$P^*$", title="Pseudo-arclength equilibrium continuation")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(branch.mu, branch.max_real, color="#1769aa", lw=1)
    ax[1].axhline(0, color="black", lw=.8)
    ax[1].set(xlabel=r"$\mu$", ylabel=r"$\max Re(\lambda)$", title="Stability along continued branch")
    for a in ax:
        a.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(outdir / "fig_formal_continuation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndvi", default=str(v1.DEFAULT_NDVI_FILE))
    ap.add_argument("--water", default=str(v1.DEFAULT_WATER_FILE))
    ap.add_argument("--tvdi", default=str(v1.DEFAULT_TVDI_FILE))
    ap.add_argument("--params", default="results_real/predictions_and_params.npz")
    ap.add_argument("--outdir", default="results_formal_verification")
    ap.add_argument("--mu-start", type=float, default=-0.15)
    ap.add_argument("--mu-end", type=float, default=1.35)
    ap.add_argument("--ds", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=1800)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = v1.load_real_data(Path(args.ndvi), Path(args.water), Path(args.tvdi))
    bundle = v1.build_bundle(data, 2018)
    p = fb.load_params(Path(args.params))
    # Find feasible seed equilibria automatically.  Some extrapolated mu
    # values have no equilibrium in the admissible state box.
    guesses = np.array([[P, T, R] for P in np.linspace(.02, .98, 9) for T in np.linspace(.02, .98, 7) for R in np.linspace(0, 1, 4)])
    seed_mu = None
    eqs = []
    for candidate_mu in np.linspace(args.mu_start, args.mu_end, 31):
        eqs = fb.equilibria_at_mu(float(candidate_mu), p, guesses)
        if eqs:
            seed_mu = float(candidate_mu)
            break
    if seed_mu is None:
        raise RuntimeError("No initial equilibrium found in the requested mu interval. Expand state bounds or inspect parameters.")
    branches = []
    for x0 in eqs:
        # Continue in both tangent directions to capture folds and return
        # the branch to a common coordinate orientation.
        for direction, signed_ds in (("forward", args.ds), ("backward", -args.ds)):
            br = continue_branch(x0, seed_mu, p, signed_ds, args.steps, args.mu_end)
            if len(br):
                br["direction"] = direction
                branches.append(br)
    if not branches:
        raise RuntimeError("No initial equilibrium found. Expand the state bounds or inspect parameters.")
    all_branch = pd.concat(branches, ignore_index=True)
    all_branch.to_csv(outdir / "continued_equilibria.csv", index=False, encoding="utf-8-sig")
    fold_rows = []
    # Fold candidates are local minima of |det J| on each continued branch.
    for br in branches:
        for i in range(1, len(br) - 1):
            if abs(br.iloc[i].detJ) <= abs(br.iloc[i - 1].detJ) and abs(br.iloc[i].detJ) <= abs(br.iloc[i + 1].detJ) and abs(br.iloc[i].detJ) < 1e-4:
                fold_rows.append(fold_diagnostics(br.iloc[i], p))
    folds = pd.DataFrame(fold_rows).drop_duplicates(subset=["mu"], keep="first") if fold_rows else pd.DataFrame()
    hopf_rows = [hopf_diagnostics(br, p) for br in branches]
    hopfs = pd.concat([x for x in hopf_rows if not x.empty], ignore_index=True) if any(not x.empty for x in hopf_rows) else pd.DataFrame()
    folds.to_csv(outdir / "fold_validation.csv", index=False, encoding="utf-8-sig")
    hopfs.to_csv(outdir / "hopf_validation.csv", index=False, encoding="utf-8-sig")
    report = [
        "Full ODE continuation verification",
        f"Parameter source: {args.params}",
        f"Branches continued: {len(branches)}",
        f"Continuation step: {args.ds}",
        f"Requested mu range: [{args.mu_start}, {args.mu_end}]",
        f"Continuation seed mu: {seed_mu}",
        f"Fold candidates passing nondegeneracy screen: {len(folds)}",
        f"Hopf candidates passing RH/crossing screen: {len(hopfs)}",
        "PINN restart uncertainty: not evaluated; the supplied cache contains one parameter dictionary.",
        "Interpretation: absence of candidates is evidence only for this parameter set and scanned range.",
    ]
    (outdir / "verification_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    plot(all_branch, folds, hopfs, outdir)
    print("\n".join(report))


if __name__ == "__main__":
    main()
