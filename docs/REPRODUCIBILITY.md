# Reproducibility guide

## Environment

Tested with Python 3.10+ and a CPU PyTorch runtime. Install:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

MATLAB is needed only for the optional MATCONT/COCO templates in matcont/.

## 1. Temporal PINN

~~~powershell
python v1.py --outdir results_real
~~~

Inputs are gee_inputs/tarim/NDVI.xlsx, gee_inputs/tarim/TVDI.xlsx, and gee_inputs/tarim/water.xlsx. The calibration window ends in 2018. The archived outputs are in results_real/.

For the explicit split and baseline comparison:

~~~powershell
python split_eval.py --series gee --outdir results_split --restarts 4
python fit_restarts.py --n 8 --train-end-year 2018
python ablation_study.py --series gee_inputs/tarim/series.csv --outdir results_ablation
~~~

Important interpretation: 2019--2020 are withheld observations; 2021--2023 are trend-projected forcings. The archived PINN has negative holdout R2 and does not beat persistence, so the temporal model is a parameter-inversion tool, not a forecaster.

## 2. Full-system continuation and bifurcation audit

~~~powershell
python formal_ode_verification.py --params results_real/predictions_and_params.npz --outdir results_formal_verification
python full_ode_bifurcation.py --params results_real/predictions_and_params.npz --outdir results_full_bifurcation
python bifurcation_robustness.py --params results_real/predictions_and_params.npz --outdir results_robustness
~~~

The archived report is results_formal_verification/verification_report.txt. For the supplied PINN parameter set, no validated fold or Hopf point was found in the searched range. The reduced cusp normal form is exploratory and uses normalized scenario coordinates; it is not a calibrated threshold.

## 3. Reduced Turing compatibility test

~~~powershell
python turing_linear_check.py
~~~

This writes results_v4/turing_summary.csv, results_v4/turing_fields.npz, and results_v4/fig17_turing_verification.png. The reduced model is a compatibility check, not a spatial calibration. Its parameters were selected by search, not estimated from the PINN.

## 4. Three 10 m spatial windows

The three windows are approximately 1.28 km square and are exported in EPSG:32645 at 10 m grid spacing. The fixed lower-left coordinates stored in the archived grid metadata are:

| Window | x0 (m) | y0 (m) | Width x height |
|---|---:|---:|---|
| W1 | 555410 | 4487660 | 129 x 128 |
| W2 | 602590 | 4466170 | 129 x 129 |
| W3 | 621790 | 4395900 | 129 x 128 |

Export them by running basins/sentinel2_pattern_export_v2.js in the GEE Code Editor. Put W1_2015.tif ... W3_2023.tif and W1_MASK.tif ... W3_MASK.tif in data/raw_spatial_windows/.

~~~powershell
python check_spatial_windows.py --folder data/raw_spatial_windows
~~~

Primary calibration uses 2017--2021; 2022 and 2023 are held out. 2015 is excluded because the Sentinel-2 optical bands are empty. 2016 uses TOA and is retained only as low-reliability supporting data.

Run the persistence-anchored residual PINN for each window:

~~~powershell
python spatial_pinn.py --stack data/raw_spatial_windows --window W1 --train-years 2017-2021 --val-years 2022 --test-years 2023 --epochs 1500 --persistence-anchor --residual-scale 0.2 --outdir results_spatial_pinn/W1_repro
~~~

Repeat with --window W2 and --window W3. Then score the archived multi-window metrics:

~~~powershell
python assess_spatial_pattern.py
python summarize_spatial_pinn.py --root results_spatial_pinn
~~~

The archived pass/fail table is results_spatial_pinn/residual_holdout_assessment.csv. P passes in all six window-years; R passes in four of six and fails the absolute-level criterion in W3; T is diagnostic only. These are pattern-level claims, not basin-wide predictions.

## 5. Cross-basin comparison

~~~powershell
python cross_basin.py --tarim gee_inputs/tarim/series.csv --ejina gee_inputs/ejina/series.csv --outdir results_cross_basin
~~~

The lower Heihe water forcing is incomplete (11/24 water years). Do not interpolate missing values or replace them with East Juyan inflow for a continuous forcing.

## 6. Checks performed for the archived release

- LaTeX compilation of the manuscript and supplementary material.
- Citation resolution and bibliography audit.
- Source-link and DOI checks for satellite products and the Heihe vegetation map.
- Row-level water-record audit with reliability labels.
- Spatial raster QC: 10 m spacing, valid masks, finite annual fields.
- Holdout evaluation against persistence and climatology.
- Random-phase null for the spatial pattern test.
