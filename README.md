<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.png">
    <img src="docs/assets/banner-light.png" width="100%" alt="CelloType. From fluorescence images to per-cell data, in one browser tab. A 16-bit, three-channel CZI field with 178 nuclei segmented by Cellpose-SAM on the DAPI channel.">
  </picture>
</p>

<p align="center">
  <img alt="Python 3.10 to 3.12" src="https://img.shields.io/badge/python-3.10%20to%203.12-5AAAFF?style=flat-square&labelColor=0B1019">
  <img alt="Segmentation: Cellpose-SAM" src="https://img.shields.io/badge/segmentation-Cellpose--SAM-50EB8C?style=flat-square&labelColor=0B1019">
  <img alt="Backend: FastAPI" src="https://img.shields.io/badge/backend-FastAPI-FFE246?style=flat-square&labelColor=0B1019">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-FF506E?style=flat-square&labelColor=0B1019"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> &nbsp;·&nbsp;
  <a href="#workflow">Workflow</a> &nbsp;·&nbsp;
  <a href="#interface">Interface</a> &nbsp;·&nbsp;
  <a href="#outputs">Outputs</a> &nbsp;·&nbsp;
  <a href="#agreement-with-a-reference-workflow">Agreement study</a> &nbsp;·&nbsp;
  <a href="#architecture">Architecture</a>
</p>

CelloType is a local web application for the quantitative analysis of multichannel fluorescence
microscopy images. It brings dataset organisation, segmentation with Cellpose-SAM, review and
correction of every mask, per-object quantification on every channel and descriptive analysis
into one guided workflow. A FastAPI backend and a plain HTML and JavaScript frontend run on
localhost with no build step, and images never leave the machine.

Mechanical operations are automated. Experimental decisions (conditions, segmentation channel
and model, which objects are kept, the measurement set, positivity thresholds) stay with the
researcher, and each one is recorded next to the results.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/screen-segment-dark.png">
  <img src="docs/assets/screen-segment-light.png" width="100%" alt="Segment step of the single-image flow. Channel, object type and segmentation model on the left; segmentation overlay on the right.">
</picture>
<p align="center"><sub>Single-image flow, Segment step. 178 nuclei on the DAPI channel with Cellpose-SAM on an Apple GPU in 23 s; the 20 that touch the image border are excluded and 158 are measured.</sub></p>

## Workflow

Two flows share the same engine: **Single image** for one field, and **Multiple images** for a
dataset organised into experimental conditions.

| | Step | What happens | Decided by the researcher |
|---|---|---|---|
| 01 | **Raw data** | Files are read at their native bit depth, with channel names from the metadata. Separate single-channel files of one field are grouped. In batch mode each field is assigned to a condition, and its field (Foto) number is a technical replicate. | Files and conditions |
| 02 | **Channels** | Marker names, then the channel to segment, the object type (nuclei, or cells and other objects) and the model, for the whole batch. | Segmentation channel, object type, model |
| 03 | **Segment** | Cellpose-SAM, or StarDist for nuclei, on the chosen channel of every field, with progress and timing. Objects touching the image border are excluded by default. Masks are reviewed and edited before anything is measured. | Border rule, mask edits |
| 04 | **Measure** | The same masks are applied to every channel. Sizes are reported in µm when the pixel size is known. Background subtraction is optional. | Measurement set, background, pixel size |
| 05 | **Analysis** | Distributions per condition, co-expression scatter, and positivity against a control condition. | Plot settings, positivity rule and control |

The single-image flow follows the same order in four steps (Raw data, Segment, Measure, Plot),
with the channel, object type and model chosen inside the Segment step.

## Interface

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/assets/screen-display.jpg" alt="Display and contrast viewer with per-channel histograms and segmentation contours over the merged channels.">
      <br><sub><b>Display and contrast.</b> Colour, histogram and numeric minimum and maximum per channel, as in ImageJ, with contours or filled masks over any channel or the merged view. Display settings never change the measured values.</sub>
    </td>
    <td width="50%" valign="top">
      <img src="docs/assets/screen-editor.jpg" alt="Mask editor with every segmented nucleus filled in a different colour and the editing toolbar.">
      <br><sub><b>Mask editor.</b> Remove spurious objects, draw missing ones, pan, fit and undo. The corrected mask is the one that gets measured.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="docs/assets/screen-batch-segment-dark.png">
        <img src="docs/assets/screen-batch-segment-light.png" alt="Batch Segment step with a per-field table of objects and a pinned image viewer.">
      </picture>
      <br><sub><b>Batch segmentation.</b> Every field in one pass, with objects detected, objects at the border and objects measured per field, next to a pinned viewer.</sub>
    </td>
    <td width="50%" valign="top">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="docs/assets/screen-analysis-dark.png">
        <img src="docs/assets/screen-analysis-light.png" alt="Analysis step with violin plots of mean intensity per condition for two markers.">
      </picture>
      <br><sub><b>Analysis.</b> Violin or box plots of any measurement by condition, by field or both, co-expression scatter and positivity. Every plot reuses the stored measurements.</sub>
    </td>
  </tr>
</table>

### Capabilities

- **Input.** CZI, TIFF (single and multichannel, with OME or ImageJ metadata) and PNG/JPEG, kept
  at native bit depth so intensities stay quantitative. Multi-plane files are detected and the
  plane used is reported; CZI tile scans are rejected instead of being read in part.
- **Segmentation.** Cellpose-SAM (`cpsam`) for nuclei, whole cells or other objects on any
  channel, on NVIDIA (CUDA) or Apple Silicon (MPS) GPUs, with a CPU fallback. Optional StarDist
  (`2D_versatile_fluo`) for nuclei. Fields above the size limit (longest side 2048 px on GPU,
  512 px on CPU) are rescaled by Cellpose itself, masks come back at full size, and the
  reduction factor is reported.
- **Review.** Viewer and mask editor in the browser. Masks are corrected before measurement, and
  the border rule can be changed afterwards without segmenting again.
- **Quantification.** 18 ImageJ-style measurement options per object and channel: intensity
  statistics, integrated density, area fraction, centroid and centre of mass, area, perimeter,
  bounding rectangle, fitted ellipse, shape descriptors and maximum Feret diameter. Optional
  background subtraction per channel: the median of the pixels outside every object, beyond a
  5 px guard band, is subtracted from each object's mean and integrated density.
- **Analysis.** Violin, box, co-expression scatter and positivity (mean + k × SD of a control
  condition, a control percentile, or an absolute value), with palette, grouping and log2
  options. Results are descriptive: cells and fields are not independent samples.
- **Reproducibility.** Every table can be saved with a JSON record of how it was produced and
  with the final label masks.

## Quick start

### Requirements

- Python 3.10 to 3.12.
- A GPU is optional but strongly recommended: NVIDIA (CUDA) or Apple Silicon (MPS). On CPU the
  app runs, but Cellpose segmentation is slow.
- About 1.2 GB of disk for the Cellpose-SAM weights, downloaded automatically on first use to
  `~/.cellpose`.

### Install

```bash
git clone https://github.com/abrilcode/CelloType.git
cd CelloType

python3 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of PyTorch that matches the driver first (see
https://pytorch.org/get-started/locally/), then run the `pip install -r requirements.txt` above.
On Apple Silicon the default install already uses the GPU through MPS.

### Run

```bash
uvicorn app:app --host 127.0.0.1 --port 8010
```

Then open http://127.0.0.1:8010 in a browser. The first segmentation downloads the Cellpose-SAM
weights (about 1.2 GB); later runs start immediately.

### Optional components

StarDist for nuclei:

```bash
pip install stardist tensorflow
```

With StarDist installed, the Segment step offers "Cellpose-SAM" or "StarDist" when the objects
are nuclei. StarDist runs the `2D_versatile_fluo` model at full resolution with the settings of
the reference workflow (normalisation to the 1st to 99.8th percentiles, probability 0.5,
overlap 0.4); TensorFlow uses the CPU unless it has its own GPU support. StarDist is not offered
for cells or other objects. Its masks can be corrected in the web editor like any other.

## Outputs

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/screen-measure-dark.png">
  <img src="docs/assets/screen-measure-light.png" width="100%" alt="Measure step with the per-object table, sizes in µm from the CZI pixel size, and the download buttons.">
</picture>
<p align="center"><sub>Measure step. The pixel size (0.293 µm) comes from the CZI metadata, so areas are reported in µm² next to the pixel values.</sub></p>

| File | Contents |
|---|---|
| `measurements.csv` | One row per object and channel: image, condition and technical replicate (batch), object id, channel, `at_edge`, and the selected measurements. |
| `<table>_parameters.json` | How the table was produced: input files and planes, segmentation channel, object type, model and settings, size reduction, border rule, mask edits, pixel size, background method and software versions. It states whether the result can be reproduced from the parameters alone (not when masks were edited by hand). |
| Masks | The final label image as TIFF (single image), or one TIFF per field in a ZIP (batch). |
| `cells_positive.csv` | Positivity per object and marker (batch), with its own `_parameters.json`. |
| Plots | PNG figures of the comparison, co-expression and positivity views. |

The "Save to folder" option writes the CSV and its parameter record together into a folder
chosen in the browser.

### Changes since the evaluated version (31 August 2026)

- `at_edge` (0 or 1) after `channel`. Objects touching the image border are left out by default.
- `background` (the value subtracted) when background subtraction is on.
- Sizes in µm next to the pixel columns (`area_um2`, `perimeter_um`, `ellipse_major_um`,
  `ellipse_minor_um`, `feret_max_um`) when the pixel size is known; empty otherwise.
- `replicate` is now `technical_replicate` (batch `measurements.csv` and `cells_positive.csv`).
  It is the field (Foto) number, not a biological replicate.
- Every table can be saved with a `<table>_parameters.json` file and the final masks.

## Agreement with a reference workflow

CelloType's nuclear segmentation and per-nucleus readouts were compared with an independent,
published StarDist-based workflow, Multi-Channel Nuclear Analysis (Waisman, 2025; StarDist
`2D_versatile_fluo` in ImageJ/FIJI), on the same images: five datasets and 78 fields. Both mask
sets were measured with the same function, so differences reflect the segmented regions only.
All runs used the GPU at full resolution, no background subtraction and nuclear segmentation.

<p align="center">
  <img src="docs/assets/fig-agreement.png" width="100%" alt="Three panels: segmentation agreement per dataset (Dice, F1 and matched IoU against the 0.85 threshold), nuclei counted by CelloType against the reference, and per-marker intensity correlation.">
</p>
<p align="center"><sub>(a) Dice, F1 at IoU ≥ 0.5 and matched IoU per dataset, against the 0.85 threshold. (b) Nuclei detected by CelloType against the StarDist reference; the band is ±5 %. (c) Pearson r of per-nucleus mean intensity for each marker, on matched nuclei. Thesis Fig. 5.1.</sub></p>

| Dataset | Fields | Nuclei, reference / CelloType | Dice | F1 at IoU ≥ 0.5 | Count r | Time, reference / CelloType |
|---|---:|---:|---:|---:|---:|---:|
| Public example set | 16 | 3862 / 3768 | 0.950 | 0.949 | 0.9994 | 8.6 s / 178 s |
| FN/R1 D40 | 11 | 1218 / 1233 | 0.956 | 0.965 | 0.9978 | 4.6 s / 121 s |
| ACTH D40 | 15 | 3173 / 3151 | 0.961 | 0.962 | 0.9973 | 8.5 s / 164 s |
| IF FOXA2/NANOG D0, D7, D15 | 21 | 10729 / 10663 | 0.948 | 0.942 | 0.9996 | 15 s / 230 s |
| EpCAM day 60, 16-bit CZI | 15 | 2403 / 2304 | 0.964 | 0.949 | 0.9914 | 12.8 s / 472 s |

Every dataset met the thresholds set before the comparison (Dice ≥ 0.85, count r ≥ 0.90,
F1 ≥ 0.70, intensity r ≥ 0.90), and per-nucleus marker intensities correlated with r between
0.997 and 0.9996. This is agreement between two automatic methods, not accuracy against manual
annotation. Cellpose-SAM generalises to other morphologies, but on these nuclear images it was
15 to 37 times slower than StarDist. For the comparison, objects with any pixel within 3 px of
the border were removed from both label sets; the app itself excludes objects that touch the
border. Methods, per-dataset results and timing are in [docs/VALIDATION.md](docs/VALIDATION.md).

## Architecture

Interface, endpoints and scientific modules are separated, so changing how the interface works
never requires rewriting how images are read, segmented, measured or analysed. The segmenter is
one module: another model can be added as an option without redesigning the flow, as StarDist was.

| Layer | Implementation |
|---|---|
| Frontend | `web/index.html`, `web/app.css`, `web/editor.js`: HTML, CSS and JavaScript, no build step |
| API | `app.py`: FastAPI endpoints for the single-image and batch flows, rendering and exports |
| Input | `loaders.py` (CZI, TIFF, PNG/JPEG; channel names, pixel size, planes), `organize.py` (fields and conditions), `single.py` (single-image session) |
| Segmentation | `segment.py`: Cellpose-SAM (`cpsam`, PyTorch) by default, optional StarDist (TensorFlow) for nuclei |
| Measurement | `measure.py`: per-object measurements, background, border rule and µm conversion (NumPy, SciPy, scikit-image) |
| Rendering and analysis | `render.py` (channels, composites, overlays, histograms), `analysis.py` (plots and positivity; Matplotlib, seaborn) |
| Run record | `runlog.py`: the JSON record saved next to every result |

`cells.py` holds an earlier whole-cell approach anchored to nuclei; the current interface does
not call it. Session state is kept on the server, so moving between steps does not recompute a
segmentation.

## Data privacy

CelloType runs entirely on the local machine and does not upload images anywhere. Microscopy
images and patient-derived data must not be committed to this repository; the `.gitignore`
excludes common microscopy formats as a safety net.

## Citation

CelloType was developed by Abril Herrlein as the final project of the Bioengineering degree at
the Instituto Tecnológico de Buenos Aires (ITBA), supervised by Dra. María Andrea Camilletti.

> A. Herrlein, "Desarrollo de una plataforma web para segmentación y cuantificación de imágenes
> de fluorescencia multicanal," Proyecto final de carrera, Bioingeniería, Instituto Tecnológico
> de Buenos Aires, 2026.

Work that uses CelloType should also cite the segmentation models behind the results:

- Cellpose-SAM: M. Pachitariu, M. Rariden and C. Stringer, "Cellpose-SAM: superhuman
  generalization for cellular segmentation," bioRxiv, 2025, doi:10.1101/2025.04.28.651001.
- Cellpose: C. Stringer, T. Wang, M. Michaelos and M. Pachitariu, "Cellpose: a generalist
  algorithm for cellular segmentation," Nature Methods 18, 100 to 106, 2021,
  doi:10.1038/s41592-020-01018-x.
- StarDist, when results were produced with the StarDist option: U. Schmidt, M. Weigert,
  C. Broaddus and G. Myers, "Cell Detection with Star-Convex Polygons," MICCAI 2018,
  doi:10.1007/978-3-030-00934-2_30; and M. Weigert et al., WACV 2020.

Reference workflow of the agreement study: A. Waisman, "Multi-Channel Nuclear Analysis: An
ImageJ/FIJI Plugin for Automated Nuclear Segmentation and Multi-Channel Fluorescence Analysis,"
microPublication Biology, 2025, doi:10.17912/micropub.biology.001590.

## License

MIT. See [LICENSE](LICENSE).
