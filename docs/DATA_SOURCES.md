# Data sources and provenance

This file documents every external data source used by the submitted analyses. The machine-readable version is data_sources.csv.

## Satellite products

| Variable | Product | Stable link | GEE asset ID | Resolution |
|---|---|---|---|---|
| NDVI | MOD13Q1 v061 | https://doi.org/10.5067/MODIS/MOD13Q1.061 | MODIS/061/MOD13Q1 | 250 m |
| LST | MOD11A2 v061 | https://doi.org/10.5067/MODIS/MOD11A2.061 | MODIS/061/MOD11A2 | 1 km |
| 10 m reflectance | Sentinel-2 SR harmonized | https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED | COPERNICUS/S2_SR_HARMONIZED | 10/20/60 m |
| 2015-2016 reflectance | Sentinel-2 TOA | https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2 | COPERNICUS/S2 | 10/20/60 m |
| Thermal | Landsat 8 C2 T1 L2 | https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2 | LANDSAT/LC08/C02/T1_L2 | 30 m optical; thermal resampled |
| Thermal | Landsat 9 C2 T1 L2 | https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2 | LANDSAT/LC09/C02/T1_L2 | 30 m optical; thermal resampled |

The exported 10 m grids are not native thermal observations. In the spatial model, P (NDVI) is the only native 10 m optical state; LST is derived from a coarser thermal product and resampled to the 10 m grid. The manuscript states this limitation.

## Lower Heihe study extent

The lower Heihe / Ejina Oasis extent follows the 1:100,000 Heihe River Basin vegetation map (2015), version 3.0:

- DOI: https://doi.org/10.11888/Ecolo.tpdc.270869
- Data page: https://data.tpdc.ac.cn/en/data/3897882a-1604-4f8e-a4bc-aeda739fe8b6/

The map defines the ROI; it is not a land-cover validation dataset for the Tarim model.

## Tarim ecological water conveyance

The water series is an audited compilation, not a single station download.

- 2000--2020: Wang et al. (2023), Journal of Desert Research, 43(4), 231--240. DOI: https://doi.org/10.7522/j.issn.1000-694X.2023.00009. Table 2 is aggregated to annual Daxihaizi release.
- 2021: Tarim River Basin Administration annual chronicle: https://www.tahe.gov.cn/zwgk/dsj/YN3QFb.htm. The 22nd lower-Tarim conveyance was 3.48 x 10^8 m3.
- 2022: Tarim River Basin Administration annual chronicle: https://www.tahe.gov.cn/zwgk/dsj/AjY7Bf.htm. Full-year ecological water delivered was 7.20 x 10^8 m3.
- 2023: cumulative checkpoint: https://www.tahe.gov.cn/zwgk/dsj/RjaYBz.htm. The annual value 3.15 x 10^8 m3 is derived from official cumulative checkpoints and is labelled derived, not directly observed.

The authoritative audit is data/water_records_audit.csv. The 2008 row is a documented no-conveyance gap. The 2021--2023 values are not to be described as direct continuous station measurements.

## Lower Heihe water release

The preferred forcing is Zhengyixia_1e8m3; East Juyan inflow is a secondary ecological response, not a substitute for Zhengyixia release. The available record covers 11 of 24 water years. Missing years remain blank.

Source pages:

- 2000--2002: http://www.waterpub.com.cn/info/detail_2719.html
- 2013--2014 public reproduction: https://www.docin.com/p-1167886375.html
- 2014--2015 public reproduction: https://m.sohu.com/a/50592676_119854
- 2016: http://yrcc.gov.cn/xwdt/jcdt/202312/t20231220_348735.html
- 2021--2022 public reproduction: https://m.sohu.com/a/622429901_121123815/
- 2023--2024: http://www.yrcc.gov.cn/zwzc/tzgg/202412/t20241225_438774.html

The row-level source strings and reliability labels are in data/heihe_water_official_2000_2023.csv.

## What is not redistributed

Raw satellite scenes, licensed PDFs, and web-page snapshots are not included. The repository supplies:

1. the exact GEE scripts;
2. the resulting annual tables and TVDI edge diagnostics;
3. the water-record audit with source URLs and reliability labels;
4. the result tables and figures derived from these inputs.

This keeps the release small and avoids redistributing third-party material outside its licence.
