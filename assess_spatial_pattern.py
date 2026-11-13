# -*- coding: utf-8 -*-
"""Apply transparent pass/fail rules to spatial-PINN holdout metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    path = Path("results_spatial_pinn/residual_multiwindow_metrics.csv")
    df = pd.read_csv(path)
    pattern = df[df["model"] == "PINN_pattern"].groupby(
        ["year", "variable"], as_index=False
    ).agg(
        spectral_cosine=("spectral_cosine", "mean"),
        pixel_correlation=("pixel_correlation", "mean"),
        null_pixel_correlation=("null_pixel_correlation", "mean"),
        observed_wavelength_m=("observed_peak_wavelength_m", "mean"),
        predicted_wavelength_m=("peak_wavelength_m", "mean"),
    )
    level = df[(df["model"] == "PINN")].groupby(
        ["year", "variable"], as_index=False
    ).agg(rmse=("rmse", "mean"), mae=("mae", "mean"), r2=("r2", "mean"))
    out = pattern.merge(level, on=["year", "variable"], how="left")
    out["pattern_pass"] = (
        (out["spectral_cosine"] >= 0.95)
        & (out["pixel_correlation"] >= 0.5)
        & (out["pixel_correlation"] >= out["null_pixel_correlation"] + 0.2)
    )
    out["level_pass"] = out["r2"] >= 0.5
    out["primary_claim_pass"] = out["pattern_pass"] & out["level_pass"]
    out.to_csv("results_spatial_pinn/residual_holdout_assessment.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    per_window = df[df["model"] == "PINN"][["window", "year", "variable", "rmse", "mae", "r2"]].merge(
        df[df["model"] == "PINN_pattern"][[
            "window", "year", "variable", "spectral_cosine",
            "pixel_correlation", "null_pixel_correlation",
            "peak_wavelength_m", "observed_peak_wavelength_m",
        ]],
        on=["window", "year", "variable"], how="left",
    )
    per_window["pass"] = (
        (per_window["spectral_cosine"] >= 0.95)
        & (per_window["pixel_correlation"] >= 0.5)
        & (per_window["pixel_correlation"] >= per_window["null_pixel_correlation"] + 0.2)
        & (per_window["r2"] >= 0.5)
    )
    per_window.to_csv("results_spatial_pinn/residual_per_window_assessment.csv", index=False, encoding="utf-8-sig")
    print("\nPrimary P/R claims require both pattern_pass and level_pass.")
    print("T is retained as a secondary dryness diagnostic, not a headline R2 claim.")


if __name__ == "__main__":
    main()
