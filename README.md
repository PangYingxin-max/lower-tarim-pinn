# Physics-informed ecohydrological modelling of the lower Tarim River

This repository accompanies the manuscript **"A physics-informed framework for ecohydrological parameter inversion and dynamical exploration in the lower Tarim River"** submitted to *Ecological Informatics*.

It contains the processed data, Google Earth Engine export scripts, PINN and dynamical-systems code, result tables, and manuscript figures needed to inspect and reproduce the analyses. Raw third-party satellite scenes and licensed source documents are not redistributed; their stable download locations and access conditions are listed in docs/DATA_SOURCES.md and data_sources.csv.

## Scientific scope

The manuscript makes four distinct claims, and they should not be conflated:

1. **Temporal PINN calibration.** The coupled three-state model is calibrated for 2000--2018. Calibration reconstruction reaches R2 = 0.993 for NDVI, 0.996 for TVDI and 0.792 for cumulative water transfer.
2. **Out-of-range transfer.** The 2019--2020 observations are a true holdout, while 2021--2023 contains trend-projected forcings. Holdout R2 is negative for all three states, and the PINN does not outperform one-year persistence. The paper therefore presents the temporal PINN as a parameter-inversion and dynamical-exploration tool, not as a forecasting system.
3. **Dynamical verification.** The full calibrated three-state ODE is continued and screened for fold and Hopf points. No validated fold or Hopf bifurcation was found in the supplied parameter range. The reported 0.35/0.65 hysteresis values belong to a reduced cusp normal form and are scenario values, not calibrated ecological thresholds.
4. **Spatial pattern evidence.** A persistence-anchored residual spatiotemporal PINN is evaluated on three georeferenced 1.28 km windows at 10 m grid spacing. Vegetation and water pattern structure passes the predefined holdout criteria, but the absolute TVDI level does not. This is pattern-level evidence, not a basin-wide spatial forecast.

The lower Heihe (Ejina Oasis) data are used only for descriptive cross-basin comparison. Zhengyixia release is available for 11 of 24 water years; missing years are retained as missing and are not interpolated.

## Repository layout

| Path | Contents |
|---|---|
| v1.py--v4.py | Temporal PINN calibration, holdout diagnostics, scenarios, and exploratory bifurcation/Turing figures |
| spatial_pinn.py | Spatiotemporal reaction--diffusion PINN, including the persistence-anchored residual variant |
| check_spatial_windows.py | QC of the exported 10 m GeoTIFF windows |
| assess_spatial_pattern.py | Pre-specified pattern pass/fail criteria |
| summarize_spatial_pinn.py | Aggregation of multi-window spatial results |
| full_ode_bifurcation.py, formal_ode_verification.py | Full-system continuation and fold/Hopf validation |
| bifurcation_robustness.py | Parameter-perturbation audit of the bifurcation screen |
| turing_linear_check.py | Reduced reaction--diffusion linear stability and mode check |
| cross_basin.py | Descriptive Tarim--Heihe comparison |
| split_eval.py, fit_restarts.py, ablation_study.py | Holdout split, restart spread, and ablation analyses |
| gee_to_inputs.py | Conversion of GEE point exports to annual series and TVDI edge tables |
| basins/ | Earth Engine JavaScript used to export the Tarim and Ejina inputs and the three 10 m windows |
| matcont/ | MATLAB templates for independent MATCONT/COCO checks |
| gee_inputs/ | Processed Tarim and Heihe annual series, provenance, and TVDI edge diagnostics |
| data/ | Audited water-conveyance tables and provenance notes |
| results_real/, results_v4/, results_cross_basin/ | Temporal, Turing, and cross-basin result tables and figures |
| results_formal_verification/, results_robustness/ | Full-system continuation and sensitivity outputs |
| results_spatial_pinn/ | Spatial-PINN QC, holdout metrics, pattern checks, and residual-model fields |
| figure/, graphical_abstract/ | Manuscript figures and graphical abstract |
| paper/ | Submitted manuscript and supplementary LaTeX/PDF copies for reference |

## Data sources

The complete machine-readable source list is data_sources.csv. Human-readable notes are in docs/DATA_SOURCES.md.

Primary products:

- MOD13Q1 v061 NDVI: https://doi.org/10.5067/MODIS/MOD13Q1.061
- MOD11A2 v061 LST: https://doi.org/10.5067/MODIS/MOD11A2.061
- Sentinel-2 surface reflectance: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- Landsat 8/9 Collection 2 surface temperature: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2 and https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2
- Heihe vegetation map v3.0: https://doi.org/10.11888/Ecolo.tpdc.270869

The Tarim and Heihe water records are compiled tables, not single downloadable observation files. Their row-level sources, reliability labels, and missing-value rules are in data/README_tarim_water.md, data/README_heihe_water.md, and data/water_records_audit.csv.

## Installation

Python 3.10 or newer is recommended.

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

For a CPU-only PyTorch installation, follow the selector at https://pytorch.org/get-started/locally/ if the default wheel is unsuitable for your platform.

## Reproduction entry points

Run commands from the repository root. The processed inputs are already present, so the temporal analyses can be rerun without downloading satellite scenes.

~~~powershell
# Temporal PINN calibration and diagnostics
python v1.py --outdir results_real

# Independent holdout split and baselines
python split_eval.py --series gee --outdir results_split

# Full-system continuation and fold/Hopf screening
python formal_ode_verification.py --params results_real/predictions_and_params.npz --outdir results_formal_verification

# Parameter-perturbation audit
python bifurcation_robustness.py --params results_real/predictions_and_params.npz --outdir results_robustness

# Reduced Turing compatibility check
python turing_linear_check.py

# Descriptive cross-basin comparison
python cross_basin.py --tarim gee_inputs/tarim/series.csv --ejina gee_inputs/ejina/series.csv --outdir results_cross_basin
~~~

The spatial PINN needs the 10 m GeoTIFF windows, which are not committed. Export them with basins/sentinel2_pattern_export_v2.js, place W1--W3 in data/raw_spatial_windows/, then run:

~~~powershell
python check_spatial_windows.py --folder data/raw_spatial_windows
python spatial_pinn.py --stack data/raw_spatial_windows --window W1 --train-years 2017-2021 --val-years 2022 --test-years 2023 --epochs 1500 --persistence-anchor --residual-scale 0.2 --outdir results_spatial_pinn/W1_repro
python assess_spatial_pattern.py
python summarize_spatial_pinn.py --root results_spatial_pinn
~~~

Detailed commands, expected outputs, and known limitations are in docs/REPRODUCIBILITY.md. Figure-to-file mappings are in docs/RESULTS_GUIDE.md.

## What is intentionally not included

- Raw Sentinel-2, Landsat, and MODIS scenes. They are large and governed by provider terms; the repository provides the GEE scripts, asset IDs, and stable catalog links.
- The original licensed PDFs, reports, and web pages used for the water records. The derived annual tables and source citations are included instead.
- Journal cover letters, editorial correspondence, and temporary build artifacts.
- Exploratory runs that are not part of the submitted manuscript, including the separate Win = 0.6997 three-state Hopf extension.

## Citation

Citation metadata are in CITATION.cff. The manuscript is under review; use the manuscript title and authors until a journal DOI is assigned. A versioned archive should be deposited in Zenodo before the final submission so that the Data Availability Statement can cite a persistent DOI rather than a mutable branch.

## Licences

Code is released under the MIT License. Author-derived tables, metadata, and result files are released under CC BY 4.0. Third-party products retain their original licences and terms; see DATA_LICENSE.md.

## Contact

Jianwei Shen, School of Mathematics and Statistics, North China University of Water Resources and Electric Power, Zhengzhou 450046, China. Email: shenjianwei@ncwu.edu.cn
