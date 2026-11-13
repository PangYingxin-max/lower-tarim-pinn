// ============================================================
//  Lower Tarim corridor - ANNUAL GEOREFERENCED RASTER EXPORT
//  Run at https://code.earthengine.google.com
//
//  Purpose
//  -------
//  The point exports used so far give one annual mean per point. A pattern
//  (Turing) analysis needs the actual spatial field, on a fixed grid, in a
//  projected CRS, with identical geometry in every year. This script produces
//
//      NDVI_YYYY.tif    growing-season NDVI       (250 m)
//      LST_YYYY.tif     growing-season LST day    (250 m, reprojected)
//      MASK.tif         1 = analysed riparian pixel
//      GRID.json        EPSG, crsTransform, width, height, scale
//
//  Every export uses ONE crsTransform, computed once from the ROI bounds and
//  snapped to the pixel grid, so year-to-year differencing is valid.
//
//  Run the exports one at a time (Tasks -> Run). 24 years x 2 variables is
//  48 tasks; the free tier will refuse to queue them all at once.
// ============================================================

// ---------- 0. settings ---------------------------------------------------

var CFG = {
  asset: 'projects/YOUR-PROJECT/assets/tarim_lower_corridor', // <- ROI FeatureCollection
  buffer_m: 5000,          // corridor half-width around the mapped channel
  epsg: 'EPSG:32645',      // UTM 45N covers 84-90 E
  scale: 250,              // metres
  folder: 'tarim_rasters',
  start: 2000,
  end: 2023,
  season: [4, 10]          // April-October
};

// ---------- 1. ROI and ONE shared pixel grid ------------------------------

var proj = ee.Projection(CFG.epsg);
var river = ee.FeatureCollection(CFG.asset);
var roi = river.geometry().buffer(CFG.buffer_m).bounds(1, proj);
var bounds = roi.bounds(1, proj).coordinates().get(0);

// bounds is [[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin],[xmin,ymin]]
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

// GEE crsTransform = [xScale, xShear, xTranslate, yShear, yScale, yTranslate]
var crsTransform = [S, 0, x0, 0, -S, y0];
print('grid', ee.Dictionary({x0: x0, y0: y0, width: W, height: H, scale: S}));

// ---------- 2. annual composites -----------------------------------------

var SR = ee.ImageCollection('MODIS/061/MOD13Q1');   // NDVI 250 m, 16-day
var LST = ee.ImageCollection('MODIS/061/MOD11A2');  // LST 1 km, 8-day

function seasonMask(img) {
  var m = ee.Date(img.get('system:time_start')).get('month');
  return img.set('month', m);
}

function annualNDVI(y) {
  var start = ee.Date.fromYMD(y, CFG.season[0], 1);
  var end = ee.Date.fromYMD(y, CFG.season[1] + 1, 1);
  var col = SR.filterDate(start, end)
    .map(function (img) {
      // MOD13Q1: SummaryQA 0 = good, 1 = marginal
      var qa = img.select('SummaryQA');
      var good = qa.lte(1);
      return img.select('NDVI').multiply(0.0001).updateMask(good);
    });
  return col.mean().rename('NDVI')
    .set('system:time_start', start.millis());
}

function annualLST(y) {
  var start = ee.Date.fromYMD(y, CFG.season[0], 1);
  var end = ee.Date.fromYMD(y, CFG.season[1] + 1, 1);
  var col = LST.filterDate(start, end)
    .map(function (img) {
      // MOD11A2: QC_Day bits 0-1 (00 good, 01 other quality)
      var qc = img.select('QC_Day');
      var good = qc.bitwiseAnd(3).lte(1);
      return img.select('LST_Day_1km').multiply(0.02).subtract(273.15)
        .updateMask(good).rename('LST');
    });
  return col.mean().rename('LST')
    .set('system:time_start', start.millis());
}

// ---------- 3. exports ----------------------------------------------------

function exportBand(img, name, band) {
  Export.image.toDrive({
    image: img.select(band).toFloat(),
    description: name,
    folder: CFG.folder,
    fileNamePrefix: name,
    region: roi,
    crs: CFG.epsg,
    crsTransform: crsTransform,
    dimensions: ee.String(W.format('%d')).cat('x').cat(H.format('%d')),
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
}

for (var y = CFG.start; y <= CFG.end; y++) {
  exportBand(annualNDVI(y), 'NDVI_' + y, 'NDVI');
  exportBand(annualLST(y), 'LST_' + y, 'LST');
}

// Static analysis mask: pixels that carry vegetation signal in most years.
var vegFrac = ee.ImageCollection(
  ee.List.sequence(CFG.start, CFG.end).map(function (y) {
    return annualNDVI(ee.Number(y)).gt(0.05).toFloat();
  })
).mean();
exportBand(vegFrac.gte(0.6).rename('MASK'), 'MASK', 'MASK');

// ---------- 4. grid manifest ---------------------------------------------
// Copy the printed values into grid.json next to the GeoTIFFs.
print('GRID.json', ee.Dictionary({
  epsg: CFG.epsg,
  scale: S,
  x0: x0,
  y0: y0,
  width: W,
  height: H,
  crsTransform: crsTransform
}));
