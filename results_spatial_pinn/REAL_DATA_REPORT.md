# Real 10 m window QC and spatial-PINN check

Date: 2026-10-31

## Raster QC

Input folder: `C:\Users\ACER\Desktop\PINN\pinn数据`

- W1: 128 rows x 129 columns, 10 m x 10 m pixels.
- W2: 129 rows x 129 columns, 10 m x 10 m pixels.
- W3: 128 rows x 129 columns, 10 m x 10 m pixels.
- All three `MASK.tif` files are present and are entirely 1.
- 2016--2023 have finite NDVI, LST and NDWI values and `VALID=1` in the
  exported window.
- 2015 has Landsat LST but no valid Sentinel-2 NDVI or NDWI. It is an empty
  Sentinel-2 year and is excluded automatically.
- 2016 uses Sentinel-2 TOA; treat it as low-reliability supporting data. The
  primary analysis should use 2017--2023 surface reflectance.

QC table: `results_spatial_pinn/downloaded_raster_qc.csv`

## Model run

Protocol: train 2017--2021, hold out 2022 and 2023, 1500 epochs, SIREN PINN,
three windows. The model input is normalized `(x,y,t)` and the output is
`(P,T,R)` with reaction--diffusion residuals, zero-flux boundaries and a raster
observation loss.

The model fits the calibration years, but it does not generalize to 2022--2023.
Mean holdout metrics across W1--W3:

| Year | Variable | Model | RMSE | MAE | R2 |
|---|---|---|---:|---:|---:|
| 2022 | P | PINN | 0.286 | 0.218 | -0.188 |
| 2022 | P | persistence | 0.075 | 0.054 | 0.901 |
| 2022 | P | climatology | 0.085 | 0.057 | 0.890 |
| 2022 | R | PINN | 0.292 | 0.226 | -0.292 |
| 2022 | R | persistence | 0.130 | 0.103 | 0.619 |
| 2022 | R | climatology | 0.126 | 0.099 | 0.704 |
| 2022 | T | PINN | 0.237 | 0.194 | -10.435 |
| 2022 | T | persistence | 0.249 | 0.236 | -16.114 |
| 2022 | T | climatology | 0.141 | 0.127 | -5.167 |
| 2023 | P | PINN | 0.291 | 0.220 | -0.442 |
| 2023 | P | persistence | 0.068 | 0.047 | 0.917 |
| 2023 | P | climatology | 0.072 | 0.047 | 0.910 |
| 2023 | R | PINN | 0.282 | 0.216 | -0.438 |
| 2023 | R | persistence | 0.102 | 0.078 | 0.782 |
| 2023 | R | climatology | 0.087 | 0.064 | 0.854 |
| 2023 | T | PINN | 0.234 | 0.199 | -18.743 |
| 2023 | T | persistence | 0.306 | 0.297 | -19.625 |
| 2023 | T | climatology | 0.198 | 0.191 | -11.021 |

The standard PINN is therefore not yet a validated spatial predictor. The
2022--2023 outputs are out-of-range temporal extrapolations. They must not be
described as future pattern forecasts or as evidence that the field model
reproduces observed patch change.

## Persistence-anchored residual PINN

The residual variant uses the 2021 field as an exact spatial anchor and adds a
bounded residual proportional to time from 2021. At the anchor year the
prediction is exactly the observed field, so extrapolation cannot drift to an
arbitrary time-dependent surface. The residual amplitude is 0.2 in normalized
state units.

Mean absolute holdout metrics across W1--W3:

| Year | Variable | Model | RMSE | MAE | R2 |
|---|---|---|---:|---:|---:|
| 2022 | P | residual PINN | 0.075 | 0.055 | 0.903 |
| 2022 | P | persistence | 0.075 | 0.054 | 0.901 |
| 2022 | R | residual PINN | 0.132 | 0.107 | 0.601 |
| 2022 | R | persistence | 0.130 | 0.103 | 0.619 |
| 2023 | P | residual PINN | 0.085 | 0.065 | 0.875 |
| 2023 | P | persistence | 0.068 | 0.047 | 0.917 |
| 2023 | R | residual PINN | 0.138 | 0.116 | 0.629 |
| 2023 | R | persistence | 0.102 | 0.078 | 0.782 |

The pattern-level comparison is the stronger test for this application. Mean
across the three windows:

| Year | Variable | spectral cosine | pixel correlation | random-phase null | observed/predicted wavelength |
|---|---|---:|---:|---:|---:|
| 2022 | P | 0.992 | 0.942 | 0.019 | 592 / 592 m |
| 2022 | R | 0.987 | 0.903 | -0.023 | 592 / 592 m |
| 2022 | T | 0.978 | 0.556 | 0.021 | 1123 / 748 m |
| 2023 | P | 0.986 | 0.944 | 0.018 | 592 / 592 m |
| 2023 | R | 0.989 | 0.920 | -0.024 | 592 / 592 m |
| 2023 | T | 0.936 | 0.465 | 0.011 | 562 / 935 m |

The residual PINN therefore passes a vegetation/water pattern holdout test: the
spectral shape and the pixel-level pattern are far above the random-phase null,
and the dominant P/R wavelength is reproduced. The absolute level is still
slightly worse than persistence in 2023, and W3 in 2022 has a large R mean shift
that the residual model does not recover. T is a secondary dryness diagnostic:
its annual absolute level is not predictable, so T R2 must not be used as a
headline validation metric.

The machine-readable pass table is `residual_holdout_assessment.csv`. With the
rules `spectral cosine >= 0.95`, `pixel correlation >= 0.5`, `pixel correlation
>= null + 0.2`, and `R2 >= 0.5`, P and R pass in both 2022 and 2023; T does not
pass. T should therefore be reported as a diagnostic state, not as a validated
absolute-level prediction.

The direct standard-versus-residual comparison is in
`standard_vs_residual_holdout.csv`:

| Year | Variable | Standard PINN R2 | Residual PINN R2 | Residual pass |
|---|---|---:|---:|---|
| 2022 | P | -0.188 | 0.903 | yes |
| 2022 | R | -0.292 | 0.601 | yes |
| 2023 | P | -0.442 | 0.875 | yes |
| 2023 | R | -0.438 | 0.629 | yes |

## Interpretation

- The GEE export is now geometrically usable: the grid is 10 m, the masks are
  present, and the annual files are not the earlier all-zero placeholders.
- 2015 cannot be used for the primary analysis.
- The persistence-anchored residual model is implemented and is the version to
  use for spatial pattern claims. The standard unanchored PINN remains a
  negative control and should not be presented as a validated predictor.
- The existing temporal PINN and exploratory Turing analysis should remain the
  manuscript's main quantitative claims until the spatial model passes a
  holdout comparison against persistence and a random-phase null.

## Files

- `results_spatial_pinn/W1_2017_2023/`
- `results_spatial_pinn/W2_2017_2023/`
- `results_spatial_pinn/W3_2017_2023/`
- `results_spatial_pinn/W1_residual_2017_2023/`
- `results_spatial_pinn/W2_residual_2017_2023/`
- `results_spatial_pinn/W3_residual_2017_2023/`
- `results_spatial_pinn/multiwindow_holdout_summary.csv`
- `results_spatial_pinn/multiwindow_holdout_means.csv`
- `results_spatial_pinn/residual_multiwindow_metrics.csv`
- `results_spatial_pinn/residual_pattern_means.csv`
- `results_spatial_pinn/residual_holdout_assessment.csv`
- `results_spatial_pinn/residual_per_window_assessment.csv`
- `results_spatial_pinn/standard_vs_residual_holdout.csv`
