# 10 m spatiotemporal pattern workflow

This workflow replaces the qualitative Turing demonstration with a georeferenced
reaction--diffusion PINN. It does not require a river-vector asset.

## 1. Export fixed Sentinel-2 windows

Open `basins/sentinel2_pattern_export_v2.js` in the Earth Engine Code Editor and
run the three export groups. The script uses three fixed centres already present
in the lower-Tarim corridor point export and creates three approximately
1.28 km x 1.28 km windows at 10 m. Every year is exported from the same region
with `crs='EPSG:32645'` and `scale=10`; the actual grid transform is read back
from the GeoTIFF rather than assembled from Earth Engine number objects.

The output folder contains:

```text
W1_2015.tif ... W1_2023.tif    bands: NDVI, LST_C, NDWI, VALID
W1_MASK.tif
W1_GRID.csv
W2_...
W3_...
```

Before training, validate the downloaded files:

```powershell
D:\Anaconda\python.exe check_spatial_windows.py --folder C:\path\to\tarim_s2_pattern_10m
```

Every row must have `dx_m=10`, `dy_m=10`, `mask_present=True`, non-zero
NDVI/LST/NDWI ranges, and `valid=True`. A 2--3 KB GeoTIFF with all values equal to
zero is an empty export, not a usable raster.

The first real export (30 October 2026) passed the 10 m grid and mask checks for
2016--2023. `W1_2015`, `W2_2015`, and `W3_2015` contained valid Landsat LST but
no valid Sentinel-2 NDVI or NDWI, so 2015 is rejected automatically and excluded
from model calibration. The 2016 TOA year is retained only as low-reliability
supporting data; the primary calibration uses 2017--2021, with 2022 and 2023 held
out.

`VALID` is 1 only where both the Sentinel-2 and Landsat observations are usable.
`spatial_pinn.py` automatically reads this fourth band when present and masks
empty-year placeholders; a zero-valued placeholder must never enter the loss.

If the Code Editor reports `Image.clip: Required argument (geometry) missing`,
remove `.clip(region)` from inside `ImageCollection.map(...)`; apply it once to
the median image (`col.median().clip(region)`). The mapped callback should not
carry the custom window object through GEE serialization.

For 2015--2016 the `COPERNICUS/S2` collection in the Code Editor exposes
`MSK_CLASSI_OPAQUE`, `MSK_CLASSI_CIRRUS`, and `MSK_CLASSI_SNOW_ICE`, not `QA60`.
Use those masks; a `QA60 did not match any bands` task failure is not a window
geometry failure.

Sentinel-2 starts in 2015. Do not extend this 10 m stack back to 2000. The LST
band is Landsat 8/9 at 100 m native resolution, resampled to the 10 m grid. NDVI
and NDWI are Sentinel-2 10 m bands; the exported LST is not a 10 m thermal
observation.

The 2015--2016 exports use `COPERNICUS/S2` (TOA) with QA60 cloud masking because
the harmonised surface-reflectance collection starts in 2017. The 2017--2023
exports use `COPERNICUS/S2_SR_HARMONIZED`. Keep this product change explicit in
the provenance record.

The 2015--2016 TOA products expose inconsistent quality-band schemas across
scenes. If a scene has no `QA60` or `MSK_CLASSI_*` band, a `B1 > -9999`
availability mask lets the task complete, but it is not a cloud mask. Treat
those years as low-reliability and use 2017--2023 for the primary pattern
analysis.

The current script also accepts browser-style duplicate suffixes such as
`W1_2015(1).tif`. Do not mix files from different GEE runs in one stack; remove
or move the older unmatched copies before a final analysis.

## 2. Run the spatiotemporal PINN

Install the CPU runtime if needed:

```powershell
D:\Anaconda\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu --no-deps torch==2.2.2+cpu
D:\Anaconda\python.exe -m pip install tifffile
```

Then run one held-out temporal window at a time:

```powershell
D:\Anaconda\python.exe spatial_pinn.py `
  --stack C:\path\to\tarim_s2_pattern_10m `
  --window W1 `
  --train-years 2017-2021 `
  --val-years 2022 `
  --test-years 2023 `
  --epochs 1500 `
  --outdir results_spatial_pinn\W1
```

The model input is normalized `(x,y,t)`. It outputs `(P,T,R)`. The observation
loss uses the raster pixels, while the PDE loss uses autograd derivatives and the
three reaction--diffusion equations. Boundary samples impose zero-flux conditions
and the initial year supplies an initial-condition loss.

## 3. Outputs to inspect

```text
spatial_stack.npz             normalized P/T/R/F cube and fixed grid
spatial_observations.csv      long-form audited raster observations
grid_metadata.csv             CRS transform and window dimensions
loss_history.csv              full PINN loss decomposition
spatial_pinn.pt               checkpoint
spatial_pinn_metrics.csv      PINN vs persistence and climatology
spatial_pinn_fields.npz       predicted and observed test-year fields
fig_spatiotemporal_pinn.png   observed/predicted/residual maps and spectra
fig_pattern_validation.png    spectral and pixel-level pattern checks
inverted_parameters.csv       reaction rates, diffusivities, linear stability
```

The pattern comparison includes a random-phase null with the same power
spectrum. A high spectral cosine alone is not sufficient: the pixel-level
correlation and patch statistics must also beat the null.

The real-data run on the three windows shows that the current standard PINN does
not yet pass a predictive holdout test. Training fits the calibration years, but
the 2022--2023 persistence and climatology baselines have much lower RMSE and
higher R^2 than the PINN. The 2022--2023 PINN predictions must therefore be
reported as an out-of-range extrapolation and must not be presented as validated
pattern forecasts.

Use the persistence-anchored residual model for the spatial pattern analysis:

```powershell
D:\Anaconda\python.exe spatial_pinn.py `
  --stack C:\path\to\tarim_s2_pattern_10m `
  --window W1 --train-years 2017-2021 --val-years 2022 --test-years 2023 `
  --epochs 1500 --persistence-anchor --residual-scale 0.2 `
  --outdir results_spatial_pinn\W1_residual_2017_2023
```

This adds a bounded residual to the 2021 observed field. At 2021 the residual is
exactly zero, so the holdout prediction cannot drift to an arbitrary surface.
Across W1--W3, the mean P pattern test gives spectral cosine 0.99 and pixel
correlation 0.94, with a random-phase null near 0.02; R gives 0.99 and 0.90--0.92.
The P/R dominant wavelength is reproduced. The absolute 2023 level is slightly
worse than persistence, and W3's 2022 R mean shift is not recovered. T remains a
secondary dryness diagnostic; its annual mean level is not predictive, so T R2
is not a headline validation metric.

## 4. Synthetic implementation check

The following command was run successfully in this workspace. It creates a
synthetic stack and verifies that the autograd Laplacian, training loop,
predictions, metrics and figures execute end to end. It is a numerical smoke test,
not a satellite result.

```powershell
D:\Anaconda\python.exe spatial_pinn.py --synthetic --synthetic-size 64 `
  --epochs 200 --outdir results_spatial_pinn\smoke_siren
```

## 5. Interpretation guardrails

- The current manuscript result remains a temporal PINN plus an exploratory
  reduced-system Turing compatibility check until the downloaded rasters are
  trained and the held-out-window metrics are reported.
- Do not call `T` a measured temperature at 10 m; it is a dry-edge TVDI state on
  a 10 m analysis grid with 100 m thermal support.
- Do not call the spatial figures future forecasts. They are held-out-year and
  held-out-window reconstructions.
- A Turing-compatible spectrum is necessary but not sufficient. Report the
  random-phase null, the pixel correlation, the patch-size distribution and the
  failure of the homogeneous mode.
