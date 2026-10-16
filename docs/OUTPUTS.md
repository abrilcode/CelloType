# Outputs

Every result CelloType produces is a plain file: a CSV table, a JSON record of how the table was
made, the final label masks and PNG plots. This page describes each one.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/screen-measure-dark.png">
  <img src="assets/screen-measure-light.png" width="100%" alt="Measure step with the per-object table, sizes in µm from the CZI pixel size, and the download buttons.">
</picture>
<p align="center"><sub>Measure step. The pixel size (0.293 µm) comes from the CZI metadata, so areas are reported in µm² next to the pixel values.</sub></p>

## Files

| File | Contents |
|---|---|
| `measurements.csv` | One row per object and channel. Columns: `image`, `object_id`, `channel`, `at_edge` (Single image) or `image`, `condition`, `technical_replicate`, `object_id`, `channel`, `at_edge` (Multiple images), then `background` when background subtraction is on, then the selected measurements. |
| `measurements_parameters.json` | The parameter record of the table: input files and planes, segmentation channel, object type, model and settings, size reduction, border rule, mask edits, pixel size, background method and software versions. It states whether the result can be reproduced from the parameters alone (not when masks were edited by hand). |
| `<image>_masks.tif`, `masks.zip` | The final label image as a 16-bit TIFF (32-bit when a label number exceeds 65535): one TIFF for one image, a ZIP with one TIFF per image for several. |
| `cells_positive.csv` | Positivity per object and marker (Multiple images), with its own parameter record, `cells_positive_parameters.json`. |
| Plots | PNG figures: `distributions.png` (Single image); `comparison.png`, `scatter.png` and `positivity.png` (Multiple images). |

"Save to folder" writes a table and its parameter record together into a folder chosen in the
browser (Chrome or Edge; in other browsers the Download buttons save each file).

## Measurements

The measurement set follows ImageJ/Fiji "Set Measurements". Geometry does not depend on the
channel; intensity measurements are computed on every channel at native bit depth. The table is
tidy: one row per object and channel, with only the selected columns. Area, centroid, mean
gray value and integrated density are selected by default; both flows offer the full list in their
Measure step.

| Measurement | Type | Columns |
|---|---|---|
| Area | geometry | `area_px`, `area_um2` |
| Mean gray value | intensity | `mean_intensity` |
| Standard deviation | intensity | `std_intensity` |
| Modal gray value | intensity | `mode_intensity` |
| Min and max gray value | intensity | `min_intensity`, `max_intensity` |
| Centroid | geometry | `centroid_x`, `centroid_y` |
| Center of mass | intensity | `com_x`, `com_y` |
| Perimeter | geometry | `perimeter`, `perimeter_um` |
| Bounding rectangle | geometry | `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` |
| Fit ellipse | geometry | `ellipse_major`, `ellipse_minor`, `ellipse_angle`, `ellipse_major_um`, `ellipse_minor_um` |
| Shape descriptors | geometry | `circularity`, `aspect_ratio`, `roundness`, `solidity` |
| Feret's diameter (max) | geometry | `feret_max`, `feret_max_um` |
| Integrated density (sum of pixel values, ImageJ RawIntDen) | intensity | `intden_sum` |
| Median | intensity | `median_intensity` |
| Skewness | intensity | `skewness` |
| Kurtosis | intensity | `kurtosis` |
| Area fraction (above Otsu) | intensity | `area_fraction_pct` |
| Stack position (channel index) | per channel | `stack_position` |

## Channel names

The `channel` column holds the name of each channel: the name typed in the interface (Channels
step, or next to each channel in the Segment step of a single image), else the name read from the
file (CZI metadata, or the marker part of the file name). Two channels cannot share a name. The
parameter record lists, per image, the name in the file and the name used for every channel.

## Units

With a known pixel size (from the CZI, OME-TIFF or ImageJ TIFF metadata, or entered manually when
the file has none), sizes are also reported in µm and µm², next to the pixel columns. Without one,
the µm columns are empty. Positions (centroid, center of mass, bounding box) stay in pixels because
they locate the object in the image. Lengths need square pixels; with non-square pixels only
`area_um2` is filled.

## Border objects

Objects touching the image border (any pixel in the outermost row or column) are excluded by
default: their rows are left out of the table, plots and positivity, and they stay visible with an
orange outline. With the rule off they are kept and marked with `at_edge` = 1. The rule can be
changed after segmentation without segmenting again.

## Background subtraction

Optional and off by default; estimated per image and channel. The median of the pixels outside
every object, beyond a 5 px margin around the objects (all pixels outside the objects if the margin
leaves none), is subtracted from each object's mean and integrated density (area × background),
without clipping, so values can be negative. The subtracted value is written in a `background`
column. The agreement study used no background subtraction.
