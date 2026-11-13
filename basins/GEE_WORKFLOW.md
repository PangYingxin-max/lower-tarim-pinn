# Google Earth Engine workflow

All scripts are designed to be pasted into https://code.earthengine.google.com.

## Tarim annual series

tarim_rasters_v1.js computes the lower Tarim 1 km corridor NDVI and LST series. It uses:

- MODIS/061/MOD13Q1 for NDVI;
- MODIS/061/MOD11A2 for LST.

## Ejina point exports

ejina_gee_points_v1.js exports annual point samples within the lower Heihe/Ejina Oasis. The analysis extent follows the Heihe River Basin vegetation map v3.0:

https://doi.org/10.11888/Ecolo.tpdc.270869

The region-point and oasis-point exports are combined by gee_to_inputs.py. TVDI is derived from the annual NDVI--LST dry and wet edges. It is an empirical feature-space index, not an in-situ measurement.

## 10 m pattern windows

sentinel2_pattern_export_v2.js exports three fixed, approximately 1.28 km windows at 10 m spacing in EPSG:32645. The script uses Sentinel-2 SR harmonized data for 2017--2023, Sentinel-2 TOA for 2015--2016, and Landsat 8/9 thermal data resampled to the 10 m grid.

The export is a fixed-window pattern test. It is not a basin-wide spatial sample.
