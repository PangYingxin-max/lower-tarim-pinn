// ============================================================
//  Ejina / lower Heihe - point tables for the NDVI-LST feature space (TVDI)
//  Companion to Ejina_NDVI_2000_2023_Success.csv and Ejina_LST_2000_2023.csv
//  Run at https://code.earthengine.google.com
//
//  Why this export exists
//  ----------------------
//  TVDI is not a satellite product. It is computed from the dry and wet edges
//  of the NDVI-LST scatter, which needs many pixels per year, not one annual
//  mean. This script produces exactly two tables, with the same schema as the
//  Tarim point exports, so one Python routine can build TVDI for both basins:
//
//      year, zone, NDVI, LST, LST_K, lon, lat
//
//  Export 1  Ejina_points_region_2000_2023   dry/wet edge fitting set
//            random points over oases + surrounding desert (the full feature
//            space, so both the dry and the wet edge are populated)
//  Export 2  Ejina_points_oasis_2000_2023    TVDI evaluation set
//            random points inside the vegetation-map ROI only
//
//  Run the two exports ONE AT A TIME (Export > Run, wait for Completed).
//  Queueing both at once fails on the free tier.
// ============================================================

// ---------- 0. settings ---------------------------------------------------
// Replace this with the asset you imported for the Heihe vegetation-map ROI.
// It must be a FeatureCollection (or table), not the annual NDVI/LST image.
var table = ee.FeatureCollection('PUT_YOUR_HEIHE_VEGMAP_ASSET_HERE');

var YEARS    = ee.List.sequence(2000, 2023);
var MON_FROM = 5, MON_TO = 9;      // matches the annual export; change to 4/10
                                   // only if you also re-run the annual one
var SEED     = 42;
var N_POINTS = 3000;               // same count as the Tarim point tables

var ROI = table.geometry().simplify(1000);

// Fitting box for the dry and wet edges: the oasis plus enough bare desert to
// give the dry edge real support.
var FITBOX = ROI.buffer(40000).bounds();

var NDVI_ID = 'MODIS/061/MOD13Q1';   // 250 m, 16 day
var LST_ID  = 'MODIS/061/MOD11A2';   // 1 km,   8 day

// ---------- 1. composites -------------------------------------------------
function ndviYear(y) {
  var start = ee.Date.fromYMD(y, MON_FROM, 1);
  var end   = ee.Date.fromYMD(y, MON_TO, 1).advance(1, 'month');
  return ee.ImageCollection(NDVI_ID)
    .filterDate(start, end)
    .map(function (img) {
      return img.select('NDVI').updateMask(img.select('SummaryQA').lte(1));
    })
    .mean()
    .multiply(0.0001)
    .rename('NDVI');
}

function lstYear(y) {
  var start = ee.Date.fromYMD(y, MON_FROM, 1);
  var end   = ee.Date.fromYMD(y, MON_TO, 1).advance(1, 'month');
  var k = ee.ImageCollection(LST_ID)
    .filterDate(start, end)
    .map(function (img) {
      // QC_Day bits 0-1: 00 = good, 01 = other quality, 10/11 = poor.
      var good = img.select('QC_Day').bitwiseAnd(3).lte(1);
      return img.select('LST_Day_1km').updateMask(good);
    })
    .mean().multiply(0.02).rename('LST_K');
  return k.addBands(k.subtract(273.15).rename('LST'));
}

function stackYear(y) {
  var nd = ndviYear(y);
  var ls = lstYear(y).resample('bilinear')
    .reproject({ crs: nd.select('NDVI').projection(), scale: 250 });
  return nd.addBands(ls);
}

// ---------- 2. points -----------------------------------------------------
var ptsRegion = ee.FeatureCollection.randomPoints(FITBOX, N_POINTS, SEED);
var ptsOasis  = ee.FeatureCollection.randomPoints(ROI, N_POINTS, SEED);

function sampleYear(y, pts, zone) {
  var stack = stackYear(ee.Number(y))
    .addBands(ee.Image.constant(y).toFloat().rename('year'));
  return stack.reduceRegions({
      collection: pts,
      reducer: ee.Reducer.mean(),
      scale: 250,
      tileScale: 4
    })
    .map(function (f) {
      var c = f.geometry().coordinates();
      return f.set({
        zone: zone,
        lon: ee.Number(c.get(0)),
        lat: ee.Number(c.get(1))
      });
    });
}

var regionTable = ee.FeatureCollection(
  YEARS.map(function (y) { return sampleYear(y, ptsRegion, 'region'); })).flatten();
var oasisTable = ee.FeatureCollection(
  YEARS.map(function (y) { return sampleYear(y, ptsOasis, 'oasis'); })).flatten();

// ---------- 3. self-check before exporting --------------------------------
print('ROI area [km2]', ROI.area().divide(1e6));
print('QA - region points, four years',
      ee.FeatureCollection(ee.List([2000, 2010, 2020, 2023])
        .map(function (y) { return sampleYear(y, ptsRegion, 'region'); })));
print('QA - oasis points, four years',
      ee.FeatureCollection(ee.List([2000, 2010, 2020, 2023])
        .map(function (y) { return sampleYear(y, ptsOasis, 'oasis'); })));

// ---------- 4. map --------------------------------------------------------
Map.centerObject(ROI, 8);
Map.addLayer(ee.ImageCollection(NDVI_ID).select('NDVI')
  .filterDate('2000-01-01', '2024-01-01')
  .filter(ee.Filter.calendarRange(MON_FROM, MON_TO, 'month'))
  .mean().multiply(0.0001),
  { min: 0.0, max: 0.35, palette: ['white', 'yellow', 'green', 'darkgreen'] },
  'mean NDVI 2000-2023');
Map.addLayer(ROI,    { color: 'red'    }, 'vegetation-map ROI', true);
Map.addLayer(FITBOX, { color: 'blue'   }, 'edge fitting box',   false);
Map.addLayer(ptsRegion, { color: 'orange' }, 'region points', false);
Map.addLayer(ptsOasis,  { color: 'cyan'   }, 'oasis points',  false);

// ---------- 5. exports ----------------------------------------------------
Export.table.toDrive({
  collection: regionTable,
  description: 'Ejina_points_region_2000_2023',
  folder: 'GEE_Ejina',
  fileFormat: 'CSV',
  selectors: ['year', 'zone', 'NDVI', 'LST', 'LST_K', 'lon', 'lat']
});

Export.table.toDrive({
  collection: oasisTable,
  description: 'Ejina_points_oasis_2000_2023',
  folder: 'GEE_Ejina',
  fileFormat: 'CSV',
  selectors: ['year', 'zone', 'NDVI', 'LST', 'LST_K', 'lon', 'lat']
});
