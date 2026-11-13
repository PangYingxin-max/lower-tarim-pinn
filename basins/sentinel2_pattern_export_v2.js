// Lower Tarim 10 m pattern windows (no river-vector asset required).
//
// Run in the Earth Engine Code Editor. Each window is approximately
// 1.28 km x 1.28 km at 10 m. Three windows are placed on
// coordinates already present in the corridor point export. No connected-vector
// ROI or hand-drawn river polygon is used.
//
// Exported per window and year (one three-band GeoTIFF):
//   NDVI   Sentinel-2 10 m native (SR 2017+, TOA 2015-2016)
//   LST_C  Landsat 8/9 Collection 2 surface temperature, resampled to 10 m
//   NDWI   Sentinel-2 green/NIR water-moisture proxy, 10 m native
// plus WINDOW_MASK.tif and GRID_WINDOW.csv.
//
// Sentinel-2 starts in 2015. Do not extend this stack back to 2000. Use a
// separate Landsat 30 m stack for historical context if needed.

var CFG = {
  epsg: 'EPSG:32645',
  scale: 10,
  sizePx: 128,
  folder: 'tarim_s2_pattern_10m',
  start: 2015,
  end: 2023,
  months: [6, 9],
  // Centers are deliberately specified in lon/lat so the script does not
  // require a river asset. They lie on the sampled lower-Tarim corridor.
  windows: [
    {id: 'W1', lon: 87.66186439, lat: 40.53198588},
    {id: 'W2', lon: 88.21532633, lat: 40.33385845},
    {id: 'W3', lon: 88.42803607, lat: 39.69838378}
  ]
};

// SR harmonised starts in 2017.  The 2015--2016 fallback is the original
// Sentinel-2 TOA collection with the QA60 cloud mask.
var S2SR = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var S2TOA = ee.ImageCollection('COPERNICUS/S2');
var L89 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'));

function windowGrid(w) {
  var proj = ee.Projection(CFG.epsg);
  // Keep the region as a client-side geometry.  Constructing a UTM rectangle
  // from ee.Number objects was the source of the empty 128 x 128 exports: GEE
  // fell back to a projection-wide extent when the transform was serialised.
  var half = CFG.scale * CFG.sizePx / 2;
  var region = ee.Geometry.Point([w.lon, w.lat])
    .buffer(half, CFG.scale)
    .bounds(CFG.scale, proj);
  return {
    id: w.id,
    lon: w.lon,
    lat: w.lat,
    region: region
  };
}

function s2Year(y, region) {
  var start = ee.Date.fromYMD(y, CFG.months[0], 1);
  var end = ee.Date.fromYMD(y, CFG.months[1] + 1, 1);
  var useSR = y >= 2017;
  var col = (useSR ? S2SR : S2TOA).filterDate(start, end)
    .filterBounds(region)
    .map(function (img) {
      var valid;
      if (useSR) {
        var scl = img.select('SCL');
        // Keep vegetation, bare soil and water; remove cloud, shadow, cirrus.
        valid = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7))
          .or(scl.eq(11));
      } else {
        // COPERNICUS/S2 has inconsistent quality-band schemas across scenes
        // and Code Editor builds.  Prefer QA60, then CLASSI, and finally use
        // a data-availability mask.  The last fallback is not a cloud mask.
        var bands = img.bandNames();
        var hasQA = bands.contains('QA60');
        var hasCLS = bands.contains('MSK_CLASSI_OPAQUE');
        var qaValid = img.select('QA60').bitwiseAnd(1 << 10).eq(0)
          .and(img.select('QA60').bitwiseAnd(1 << 11).eq(0));
        var clsValid = img.select('MSK_CLASSI_OPAQUE').eq(0)
          .and(img.select('MSK_CLASSI_CIRRUS').eq(0));
        var available = img.select('B1').gt(-9999);
        valid = ee.Image(ee.Algorithms.If(
          hasQA, qaValid,
          ee.Image(ee.Algorithms.If(hasCLS, clsValid, available))));
      }
      var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
      var ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI');
      var validBand = valid.toFloat().rename('VALID');
      return ndvi.addBands(ndwi).addBands(validBand).updateMask(valid);
    });
  // Keep the band schema even when a year has no usable scenes.  VALID=0 is
  // essential: the zero values must never be treated as observations.
  var safe = ee.Image.constant([0, 0, 0])
    .rename(['NDVI', 'NDWI', 'VALID']).toFloat().updateMask(0);
  var reduced = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), safe));
  return reduced.clip(region).set('system:time_start', start.millis());
}

function lstYear(y, region) {
  var start = ee.Date.fromYMD(y, CFG.months[0], 1);
  var end = ee.Date.fromYMD(y, CFG.months[1] + 1, 1);
  var col = L89.filterDate(start, end).filterBounds(region).map(function (img) {
    var qa = img.select('QA_PIXEL');
    var bad = qa.bitwiseAnd(1 << 3).neq(0)
      .or(qa.bitwiseAnd(1 << 4).neq(0))
      .or(qa.bitwiseAnd(1 << 5).neq(0));
    var lst = img.select('ST_B10').multiply(0.00341802).add(149.0)
      .subtract(273.15).rename('LST_C');
    var valid = bad.not().toFloat().rename('VALID');
    return lst.addBands(valid).updateMask(bad.not());
  });
  var safe = ee.Image.constant([0, 0])
    .rename(['LST_C', 'VALID']).toFloat().updateMask(0);
  var reduced = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), safe));
  return reduced.clip(region).set('system:time_start', start.millis());
}

function exportWindow(w) {
  var grid = windowGrid(w);
  var region = grid.region;
  for (var y = CFG.start; y <= CFG.end; y++) {
    var s2 = s2Year(y, region);
    var lst = lstYear(y, region);
    var valid = s2.select('VALID').min(lst.select('VALID')).rename('VALID');
    var img = s2.select('NDVI')
      .addBands(lst.select('LST_C'))
      .addBands(s2.select('NDWI'))
      .addBands(valid)
      .select(['NDVI', 'LST_C', 'NDWI', 'VALID'])
      .clip(region);
    Export.image.toDrive({
      image: img.toFloat(),
      description: grid.id + '_' + y + '_S2_LST_NDWI',
      folder: CFG.folder,
      fileNamePrefix: grid.id + '_' + y,
      region: region,
      crs: CFG.epsg,
      scale: CFG.scale,
      maxPixels: 1e10,
      fileFormat: 'GeoTIFF'
    });
  }

  // A fixed mask: pixels with at least half of the years carrying a valid
  // Sentinel-2 vegetation observation. It is not a river vector.
  // Build this list client-side. Passing an ee.Number into s2Year makes
  // `y >= 2017` an EE object rather than a JavaScript boolean and can produce
  // the misleading "Can't encode object: undefined" error.
  var maskImages = [];
  for (var my = CFG.start; my <= CFG.end; my++) {
    var s2Valid = s2Year(my, region).select('VALID');
    var lstValid = lstYear(my, region).select('VALID');
    maskImages.push(s2Valid.min(lstValid).unmask(0).toFloat());
  }
  var validCount = ee.ImageCollection(maskImages).sum();
  var minCount = (CFG.end - CFG.start + 1) * 0.5;
  var mask = validCount.gte(minCount).rename('MASK').toByte();
  Export.image.toDrive({
    image: mask,
    description: grid.id + '_WINDOW_MASK',
    folder: CFG.folder,
    fileNamePrefix: grid.id + '_MASK',
    region: region,
    crs: CFG.epsg,
    scale: CFG.scale,
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });

  // Keep the table properties to primitive client-side values.  EE objects
  // inside ee.Feature properties can be serialised as undefined and trigger
  // "Can't encode object: undefined" on some Code Editor builds.
  var meta = ee.Feature(null, {
    window: String(grid.id),
    lon: Number(grid.lon),
    lat: Number(grid.lat),
    epsg: CFG.epsg,
    scale: CFG.scale,
    size_px: CFG.sizePx,
    width: CFG.sizePx,
    height: CFG.sizePx,
    start_year: CFG.start,
    end_year: CFG.end
  });
  Export.table.toDrive({
    collection: ee.FeatureCollection([meta]),
    description: grid.id + '_GRID',
    folder: CFG.folder,
    fileNamePrefix: grid.id + '_GRID',
    fileFormat: 'CSV'
  });
  print(grid.id + ' grid metadata', meta);
}

CFG.windows.forEach(exportWindow);

// Expected Drive filenames after running the tasks:
//   W1_2015.tif ... W3_2023.tif, W1_MASK.tif ... W3_MASK.tif,
//   W1_GRID.csv ... W3_GRID.csv
// Feed the downloaded folder to:
//   python spatial_pinn.py --stack <folder> --window W1 --outdir results_spatial_pinn/W1
