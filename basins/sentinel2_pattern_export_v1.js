// NOTE: For the current manuscript workflow use
// basins/sentinel2_pattern_export_v2.js.  That script exports fixed 128 x 128
// windows and avoids automatic large-blob ROI selection on an entire oasis.
// ============================================================
//  Lower Tarim - 10 m Sentinel-2 PATTERN EXPORT (no river vector needed)
//  Run at https://code.earthengine.google.com
//
//  Outputs, on ONE fixed 10 m grid (EPSG:32645):
//      S2NDVI_YYYY.tif   growing-season NDVI, Sentinel-2 SR harmonised
//      L8LST_YYYY.tif    growing-season LST (deg C), Landsat 8/9 C02 L2
//      MASK.tif          1 = analysed riparian pixel
//      ROIAUTO.tif       1 = auto-derived corridor (largest vegetated blob)
//      GRID.json         crsTransform / width / height for Python
//
//  Why no ROI asset: the corridor is derived from the imagery itself. A NDVI
//  threshold is applied to the 2015-2023 median, the largest connected blob is
//  kept, and the export grid is its bounding box. Draw a rectangle only if you
//  want to restrict the search (optional CFG.bboxOverride).
//
//  Sentinel-2 starts in 2015. For 2000-2014 use Landsat 5/7 at 30 m; do not
//  stretch Sentinel-2 backwards.
// ============================================================

var CFG = {
  epsg: 'EPSG:32645',
  scale: 10,                 // metres
  folder: 'tarim_s2_pattern',
  start: 2015,
  end: 2023,
  months: [6, 9],            // growing season, June-September
  ndviVeg: 0.15,             // threshold used only to derive the corridor
  bboxOverride: null,        // e.g. [xmin, ymin, xmax, ymax] in EPSG:32645
  minSidePx: 64              // refuse windows smaller than this
};

// ---------- 1. Sentinel-2 NDVI --------------------------------------------

var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');

function maskS2(img) {
  var scl = img.select('SCL');
  // 4 vegetation, 5 bare, 6 water, 11 snow; drop cloud/shadow/cirrus/saturated
  var bad = scl.eq(1).or(scl.eq(3)).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return ndvi.updateMask(bad.not()).updateMask(scl.eq(4).or(scl.eq(5)).or(scl.eq(6)));
}

function s2Year(y) {
  var start = ee.Date.fromYMD(y, CFG.months[0], 1);
  var end = ee.Date.fromYMD(y, CFG.months[1] + 1, 1);
  return S2.filterDate(start, end)
    .filterBounds(searchArea())
    .map(maskS2)
    .median()
    .rename('NDVI')
    .set('system:time_start', start.millis());
}

// ---------- 2. Landsat LST (Sentinel-2 has no thermal band) ----------------

var L89 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'));

function lstYear(y) {
  var start = ee.Date.fromYMD(y, CFG.months[0], 1);
  var end = ee.Date.fromYMD(y, CFG.months[1] + 1, 1);
  var col = L89.filterDate(start, end).filterBounds(searchArea()).map(function (img) {
    var qa = img.select('QA_PIXEL');
    var cloud = qa.bitwiseAnd(1 << 3).neq(0).or(qa.bitwiseAnd(1 << 4).neq(0));
    var lst = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15);
    return lst.updateMask(cloud.not()).rename('LST');
  });
  return col.median().rename('LST').set('system:time_start', start.millis());
}

// ---------- 3. auto corridor -------------------------------------------------

function searchArea() {
  if (CFG.bboxOverride) {
    return ee.Geometry.Rectangle(CFG.bboxOverride, CFG.epsg, false);
  }
  // coarse search box; refined below from the vegetation blob
  return ee.Geometry.Rectangle([84.0, 39.0, 90.0, 42.0], 'EPSG:4326', false);
}

var vegMedian = S2.filterDate('2015-06-01', '2023-10-01')
  .filterBounds(searchArea())
  .map(maskS2)
  .median()
  .gt(CFG.ndviVeg);

var blobs = vegMedian.selfMask().reduceToVectors({
  geometry: searchArea(),
  crs: CFG.epsg,
  scale: 100,
  maxPixels: 1e10,
  geometryType: 'polygon',
  eightConnected: true
});
var biggest = ee.Feature(blobs.sort('count', false).first());
var corridor = ee.FeatureCollection([biggest]).geometry();
print('corridor area (km2)', corridor.area().divide(1e6));

// ---------- 4. one shared 10 m grid -----------------------------------------

var proj = ee.Projection(CFG.epsg);
var bounds = corridor.bounds(1, proj).coordinates().get(0);
var xs = bounds.map(function (p) { return ee.List(p).get(0); });
var ys = bounds.map(function (p) { return ee.List(p).get(1); });
var xmin = ee.Number(xs.reduce(ee.Reducer.min()));
var xmax = ee.Number(xs.reduce(ee.Reducer.max()));
var ymin = ee.Number(ys.reduce(ee.Reducer.min()));
var ymax = ee.Number(ys.reduce(ee.Reducer.max()));

var S = CFG.scale;
var x0 = xmin.divide(S).floor().multiply(S);
var y0 = ymax.divide(S).ceil().multiply(S);
var W = xmax.subtract(x0).divide(S).ceil();
var H = y0.subtract(ymin).divide(S).ceil();
var crsTransform = [S, 0, x0, 0, -S, y0];
print('grid', ee.Dictionary({x0: x0, y0: y0, width: W, height: H, scale: S}));

// ---------- 5. exports -------------------------------------------------------

function exportImage(img, name, band) {
  Export.image.toDrive({
    image: img.select(band).toFloat(),
    description: name,
    folder: CFG.folder,
    fileNamePrefix: name,
    region: corridor.bounds(1, proj),
    crs: CFG.epsg,
    crsTransform: crsTransform,
    dimensions: ee.String(W.format('%d')).cat('x').cat(H.format('%d')),
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
}

for (var y = CFG.start; y <= CFG.end; y++) {
  exportImage(s2Year(y), 'S2NDVI_' + y, 'NDVI');
  exportImage(lstYear(y), 'L8LST_' + y, 'LST');
}

var vegFrac = ee.ImageCollection(
  ee.List.sequence(CFG.start, CFG.end).map(function (y) {
    return s2Year(ee.Number(y)).gt(0.10).toFloat();
  })
).mean();
exportImage(vegFrac.gte(0.5).rename('MASK'), 'MASK', 'MASK');
exportImage(ee.Image.constant(1).clip(corridor).rename('ROI'), 'ROIAUTO', 'ROI');

print('GRID.json', ee.Dictionary({
  epsg: CFG.epsg, scale: S, x0: x0, y0: y0, width: W, height: H,
  crsTransform: crsTransform, years: [CFG.start, CFG.end]
}));
