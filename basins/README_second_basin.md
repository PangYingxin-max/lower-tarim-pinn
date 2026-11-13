# Adding a second basin to the lower Tarim River study

This folder holds the template for an independent validation basin.  The
template contains **column names and years only**.  Every value cell is
deliberately empty: the numbers must come from real observations, not from a
language model.

## Why the values cannot be generated for you

The manuscript's central claims (physics-informed parameter inversion, a
no-fold/no-Hopf result, an exploratory Turing compatibility test) already rest
on a 24-row annual record for a single river reach.  Adding a second basin with
invented numbers would not strengthen those claims; it would invalidate the
paper.  Editors and reviewers at *Ecological Informatics* routinely request the
underlying series, and a fabricated series is detectable because it will not
match the cited source tables.

So the deliverable here is a fill-in template plus the exact acquisition chain.
Once the four columns are filled from real sources, the existing pipeline in
`v1.py` runs on the new basin unchanged.

## Candidate basins

Prefer a basin that, like the lower Tarim, received managed ecological water
conveyance after 2000.  That keeps the model structure interpretable.

| Candidate | Why it works | Ecological water-conveyance series |
|---|---|---|
| Heihe River lower reaches / Ejina Oasis (Inner Mongolia) | Canonical analogue: unified water diversion since 2000, documented NDVI recovery, groundwater response | Heihe River Basin Water Resources Commission annual reports; papers in *Journal of Hydrology* / *Ecological Indicators* |
| Yarkant River / Kashgar oasis | Large Tarim tributary with comparable arid riparian vegetation | Xinjiang Tarim River Basin Authority yearbooks |
| Kaidu-Konqi River / Bosten Lake | Closed inland system with documented water-transfer records | Xinjiang water-resources statistical yearbooks |
| Shule River / Dunhuang Xihu | Very water-limited, strong restoration signal | Provincial water-resources bulletins |

## Data sources per column

1. **NDVI** - MODIS MOD13Q1 (Terra, 16-day, 250 m) or MOD13A3 (monthly, 1 km).
   Free download through NASA LP DAAC / Earthdata, or through Google Earth
   Engine (`MODIS/061/MOD13Q1`).  Aggregate to growing-season means exactly as
   for the Tarim record (May-September, quality flags retained).
2. **TVDI** - derived, not downloaded.  Compute from the NDVI / land-surface
   temperature feature space following Sandholt et al. (2002), using MOD11A2
   (8-day, 1 km) LST.  The dry and wet edges must be fitted per basin.
3. **R_eco_water_1e8m3** - annual ecological water conveyance volume in
   10^8 m^3.  Sources are basin-authority annual reports or published tables.
   Record the exact table or page in the `Source_water` column.
4. **Q_total_1e8m3** - total discharge at the controlling hydrological station,
   same source family as the conveyance volume.

## Required provenance fields

Fill `Source_NDVI`, `Source_TVDI`, and `Source_water` with a citable string
(author, year, journal, table or figure number).  Fill `Reliability` with one of
`high` / `medium` / `low`.  The lower Tarim record used those three levels, and
2019-2023 entries were partly extrapolated.  Use `low` honestly where a value is
an interpolation.

## Running the audit

```
python data_quality_report.py --csv basins/second_basin_filled.csv --outdir reports_quality_second_basin --label "Heihe lower reaches"
```

The audit reports: year continuity, duplicate years, missing values per
variable, plausible-range violations, calibration-period ranges, and how many
holdout observations fall outside the calibration range.

## Running the model on the new basin

Convert the filled CSV into the same three-file Excel layout used for the Tarim
record, then run:

```
python v1.py --ndvi <new_NDVI.xlsx> --water <new_R.xlsx> --tvdi <new_TVDI.xlsx> --outdir results_second_basin
```

Everything downstream (`v2.py`, `v3.py`, `full_ode_bifurcation.py`,
`formal_ode_verification.py`) then works on the new basin by pointing `--outdir`
and the input paths at the new files.
