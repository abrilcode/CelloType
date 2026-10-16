# Changelog

Changes to CelloType since the version described and evaluated in the thesis. Most of them answer
the reviewers' observations; each links to its pull request.

## Unreleased

### Results that can differ from the evaluated version

- **Border objects.** Objects touching the image border are excluded by default, so object counts
  and table rows can be lower than before. The rule can be turned off before or after
  segmentation. ([#3](https://github.com/abrilcode/CelloType/pull/3))
- **Background subtraction** (off by default) is applied at the object level: the background is
  subtracted from each object's mean and integrated density, without clipping pixels at zero, and
  with a 5 px guard band around the objects. ([#2](https://github.com/abrilcode/CelloType/pull/2))
- **Reduced images.** When a field exceeds the size limit (longest side 2048 px on GPU, 512 px on
  CPU), Cellpose rescales it itself and the masks come back at full size. The reduction factor is
  reported. ([#7](https://github.com/abrilcode/CelloType/pull/7))

### Added

- JSON record of the run parameters next to every table, and download of the final label masks.
  ([#4](https://github.com/abrilcode/CelloType/pull/4))
- Sizes in µm and µm² from the pixel size in the CZI, OME or ImageJ metadata, with a manual
  fallback that never overrides the metadata. ([#5](https://github.com/abrilcode/CelloType/pull/5))
- Detection of multi-plane files (Z, T, scenes) with the plane used reported; CZI tile scans are
  rejected instead of being read in part. ([#6](https://github.com/abrilcode/CelloType/pull/6))
- StarDist (`2D_versatile_fluo`) as an optional second model for nuclei. Any channel can be
  segmented, as nuclei or as cells and other objects; the single-image flow previously always
  segmented nuclei. ([#12](https://github.com/abrilcode/CelloType/pull/12))

### Changed

- Output columns: `at_edge` after `channel`; `background` when background subtraction is on; µm
  columns (`area_um2`, `perimeter_um`, `ellipse_major_um`, `ellipse_minor_um`, `feret_max_um`)
  next to the pixel columns; `replicate` renamed `technical_replicate`, since it is the field
  number and not a biological replicate. ([#3](https://github.com/abrilcode/CelloType/pull/3),
  [#2](https://github.com/abrilcode/CelloType/pull/2),
  [#5](https://github.com/abrilcode/CelloType/pull/5),
  [#8](https://github.com/abrilcode/CelloType/pull/8))
- The interface notes that analyses are descriptive: cells and fields are not independent
  samples. ([#8](https://github.com/abrilcode/CelloType/pull/8))
- The validation is described as an agreement study between two automatic methods, not as
  accuracy. ([#9](https://github.com/abrilcode/CelloType/pull/9))
- Interface: redesigned screens and an output folder chosen in the browser
  ([#1](https://github.com/abrilcode/CelloType/pull/1)); image viewer pinned beside each step and
  explanations behind info icons ([#11](https://github.com/abrilcode/CelloType/pull/11));
  consistent headings, labels and buttons
  ([#13](https://github.com/abrilcode/CelloType/pull/13)); workflow sidebar layout
  ([#14](https://github.com/abrilcode/CelloType/pull/14)).

### Removed

- An unused file-saving endpoint, an unused import and an unused dependency (`fpdf2`).
  ([#10](https://github.com/abrilcode/CelloType/pull/10))

## v1.0.0, evaluated version (31 August 2026)

The version described in the thesis and read by the reviewers: commit
[`fa3803c`](https://github.com/abrilcode/CelloType/tree/fa3803c).
