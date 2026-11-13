# Raw 10 m window inputs

The GeoTIFF files are deliberately not committed because they are generated from Google Earth Engine and are large relative to the code and tabular data.

Run basins/sentinel2_pattern_export_v2.js in the Earth Engine Code Editor and place the outputs here:

~~~text
W1_2015.tif ... W1_2023.tif
W1_MASK.tif
W1_GRID.csv
W2_2015.tif ... W2_2023.tif
W2_MASK.tif
W2_GRID.csv
W3_2015.tif ... W3_2023.tif
W3_MASK.tif
W3_GRID.csv
~~~

The primary analysis uses 2017--2021 for calibration and holds out 2022 and 2023. 2015 has no usable Sentinel-2 optical bands and is excluded automatically. 2016 uses TOA and is low-reliability supporting data only.

Check the files before training:

~~~powershell
python check_spatial_windows.py --folder data/raw_spatial_windows
~~~

The expected fixed lower-left coordinates are listed in docs/REPRODUCIBILITY.md.
