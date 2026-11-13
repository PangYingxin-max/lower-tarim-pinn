# Heihe lower-reach annual water input, 2000-2023

Source table: `data/heihe_water_official_2000_2023.csv`.

## Definition

`Year` is the start year of the Heihe water-scheduling year (normally
11 November to 10 November).  For example, `2000` means `2000-2001`.

The preferred model forcing is `Zhengyixia_1e8m3`, the measured annual release
at the Zhengyixia hydrological station in `1e8 m3`.  `EastJuyan_1e8m3` is the
reported inflow to East Juyan Lake and is retained as a secondary ecological
response series.  It is not used to fill Zhengyixia and the two quantities must
not be mixed in one continuous series.

## Verified public sources

* 2000-2002: Yellow River Conservancy Commission, *Heihe water-allocation record
  and chronology*, http://www.waterpub.com.cn/info/detail_2719.html.  The page
  gives Zhengyixia releases of 6.76, 9.23 and 12.56 x1e8 m3 and East Juyan
  inflows of 0.2350+0.2574 and 0.4247 x1e8 m3.
* 2005: YRCC news, *Heihe 2006 annual water-scheduling result* (10 November
  2006), reports a Zhengyixia release of 11.45 x1e8 m3.  The legacy page now
  returns 404, so the row is marked `verified_indexed`; title and value were
  checked through the YRCC public search index.
* 2006: YRCC news, *Heihe annual release reaches an eight-year high* (2007),
  reports 11.96 x1e8 m3 at Zhengyixia.  This legacy page also returns 404 and
  is therefore marked `verified_indexed`.
* 2013-2014 and 2014-2015: public reproductions of the annual bulletins,
  https://www.docin.com/p-1167886375.html and
  https://m.sohu.com/a/50592676_119854.  These values are marked `reproduced`.
* 2016: YRCC news, *Heihe hydrological survey completes monitoring in the
  wettest year*, http://yrcc.gov.cn/xwdt/jcdt/202312/t20231220_348735.html,
  reports 15.92 x1e8 m3 at Zhengyixia and 0.6452 x1e8 m3 into East Juyan.
* 2021-2022: public reproduction of the annual bulletin,
  https://m.sohu.com/a/622429901_121123815/, reports 13.71 x1e8 m3 at
  Zhengyixia and 0.64 x1e8 m3 into East Juyan.
* 2022-2023: China Water Resources News/China Zhangye Web, 20 December 2023,
  reports 8.49 x1e8 m3 at Zhengyixia.
* 2023-2024: YRCC official bulletin,
  http://www.yrcc.gov.cn/zwzc/tzgg/202412/t20241225_438774.html, reports
  10.58 x1e8 m3 at Zhengyixia and 0.70 x1e8 m3 into East Juyan.

## Not yet complete

The present file is an audited first release, not a complete 24-year forcing
series.  Years without a directly checked annual value are blank and marked
`missing`; they are deliberately not interpolated, reconstructed from a trend,
or replaced by a partial-period news figure.  A complete series requires the
missing annual water-scheduling bulletins or the relevant Hydrological Yearbook
tables.  Until those are obtained, this table can support descriptive cross-basin
comparison and data-quality checks, but it must not be used as if it were a fully
observed PINN training forcing.

## How to regenerate the Ejina inputs

After downloading the two point exports from GEE:

```powershell
python gee_to_inputs.py --basin ejina --outdir gee_inputs\ejina `
  --ejina-region-points Ejina_points_region_2000_2023.csv `
  --ejina-oasis-points Ejina_points_oasis_2000_2023.csv `
  --ejina-water data\heihe_water_official_2000_2023.csv
```

The command writes `series.csv`, `TVDI.xlsx`, `water.xlsx`, `water.csv`, and
`provenance.json`.  Missing water-years stay blank in all of them.
