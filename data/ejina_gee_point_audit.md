# Ejina GEE point-export audit

## Corrected export (current)

Checked files:

- `C:\Users\ACER\Desktop\PINN\pinn数据\Ejina_points_region_2000_2023.csv`
- `C:\Users\ACER\Desktop\PINN\pinn数据\Ejina_points_oasis_2000_2023.csv`

Both files contain 72,000 records (3,000 points for each year from 2000 to
2023) and use the fields `year, zone, NDVI, LST, LST_K, lon, lat`.

For every year, the region and oasis point coordinates are different. The
coordinate overlap is zero in all 24 years. The audit is written to
`gee_inputs/ejina/point_audit.csv` and can be regenerated with:

```powershell
python gee_to_inputs.py --basin ejina --outdir gee_inputs\ejina `
  --ejina-region-points Ejina_points_region_2000_2023.csv `
  --ejina-oasis-points Ejina_points_oasis_2000_2023.csv `
  --ejina-water data\heihe_water_official_2000_2023.csv
```

The region sample fits the annual dry/wet edges. The independent oasis sample
is used to evaluate annual mean TVDI. This is a valid point-based TVDI
workflow, but it is not an in-situ validation: the dry/wet edges are empirical
fits and their fit quality should be reported for every year.

The point sample also yields annual mean NDVI and LST. These are stored as
`NDVI_point_mean` and `LST_point_mean_C` in `series.csv`, and as the canonical
`NDVI`/`LST_C` columns in `series_point_sample.csv`. They differ substantially
from the older annual exports: the point-based Ejina LST is about 36-38.5 C,
whereas the old annual file gives about 22.9-28.4 C. Do not mix the two
definitions in one cross-basin figure or model run.

## Earlier export (superseded)

The earlier `Ejina_reg_YYYY.csv` and `Ejina_oas_YYYY.csv` files were identical
within each year: both called `randomPoints(ROI, 3000, 42)` on the same ROI.
They are retained only as a record of the superseded export and should not be
used for independent fitting and evaluation.
