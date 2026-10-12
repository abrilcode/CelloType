# CelloType

Fluorescence microscopy nuclear analysis in your browser, running locally on your
own machine and GPU. Upload multichannel images (CZI, TIFF, PNG), segment nuclei
or whole cells with Cellpose-SAM, measure per-nucleus marker intensities
(ImageJ style), explore the result in an interactive viewer, and make
publication style plots. No build step: a FastAPI backend plus a plain HTML/JS
frontend that you run on localhost.

## Features

- Reads CZI, TIFF (single and multichannel) and PNG/JPEG, keeping native bit
  depth so intensities stay quantitative.
- Cellpose-SAM segmentation of nuclei, cells or other objects on any channel,
  using your GPU (NVIDIA CUDA or Apple Silicon MPS) when available, otherwise CPU.
- Optional StarDist (`2D_versatile_fluo`) for nuclei, with the settings of the
  reference workflow used in the validation. Cellpose-SAM stays the default and
  is the only option for cells and other objects.
- ImageJ style "Set Measurements": area, centroid, mean, integrated density,
  shape descriptors, Feret, and more, per object and per channel.
- Interactive viewer: per channel colour and brightness/contrast with a
  histogram and numeric min/max (like ImageJ), plus the segmentation shown as
  contours, filled masks, or both, over any channel or the merged composite.
- Batch mode: organize many fields (technical replicates) into conditions, segment in one
  pass, and pool the measurements.
- Plots: violin and box distributions, co-expression scatter, and positivity
  (percent positive), with palette, grouping and log2 options.
- CSV export of the measurement table, and an optional round trip to the Cellpose
  GUI for manual mask correction.

## Requirements

- Python 3.10 to 3.12.
- A GPU is optional but strongly recommended: NVIDIA (CUDA) or Apple Silicon
  (MPS). On CPU it runs, but Cellpose segmentation is slow.
- About 1.2 GB of disk for the Cellpose-SAM model, downloaded automatically on
  first use to `~/.cellpose`.

## Install

```bash
git clone <your-repo-url>
cd CelloType

python3 -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

GPU note: for an NVIDIA GPU, install the CUDA build of PyTorch that matches your
driver first (see https://pytorch.org/get-started/locally/), then run the
`pip install -r requirements.txt` above. On Apple Silicon the default install
already uses the GPU through MPS.

## Run

```bash
uvicorn app:app --host 127.0.0.1 --port 8010
```

Then open http://127.0.0.1:8010 in your browser. The first segmentation downloads
the Cellpose-SAM weights (about 1.2 GB); later runs are fast.

## Quick start

Single image:

1. Upload one multichannel file, or the separate single channel files of one field.
2. Check the channels, pick the channel to segment (a nuclear stain or any other
   marker), choose Nuclei or Cells or other objects, and click Segment.
3. Explore in the viewer, review the per nucleus table, and export the CSV.

Batch:

1. Upload many files and organize them into conditions (each field, or Foto number, is a technical replicate).
2. Pick the segmentation channel and segment all fields.
3. Measure, then build violin, scatter or positivity plots and export.

## Optional: manual mask editing in the Cellpose GUI

To correct masks by hand, install the Cellpose GUI extra, then use the export and
import buttons in the app:

```bash
pip install "cellpose[gui]"
```

## Optional: StarDist for nuclei

```bash
pip install stardist tensorflow
```

With StarDist installed, the Segment step offers "Cellpose-SAM" or "StarDist" when
segmenting nuclei. StarDist runs the `2D_versatile_fluo` model at full resolution with
the settings of the reference workflow (normalisation to the 1st to 99.8th percentiles,
probability 0.5, overlap 0.4); TensorFlow uses the CPU unless it has its own GPU support.
StarDist is not offered for cells or other objects. Its masks can be corrected in the web editor;
the Cellpose GUI round trip needs a Cellpose-SAM segmentation.

## Exported tables: changes after the thesis review

- `at_edge` (0 or 1) after `channel`. Objects touching the image border are left out by default.
- `background` (the value subtracted) when background subtraction is on.
- Sizes in µm next to the pixel columns (`area_um2`, `perimeter_um`, `ellipse_major_um`,
  `ellipse_minor_um`, `feret_max_um`) when the pixel size is known; empty otherwise.
- `replicate` is now `technical_replicate` (batch `measurements.csv` and `cells_positive.csv`).
  It is the field (Foto) number, not a biological replicate.
- Every table can be saved with a `<table>_parameters.json` file (how it was produced, including
  the segmentation model and its settings) and the final masks.

## Validation

CelloType's nuclear segmentation and per-nucleus readouts were compared with
an independent, published StarDist-based nuclei-analysis workflow (StarDist model
`2D_versatile_fluo`, ImageJ/FIJI) on five datasets. Agreement was high:
foreground Dice about 0.95, instance F1 at IoU >= 0.5 between 0.94 and 0.97, and
per-nucleus intensity Pearson r about 0.997 to 0.9996. This is agreement between
two automatic methods, not accuracy against manual annotation. Details, including
the datasets and timing, are in [docs/VALIDATION.md](docs/VALIDATION.md).

## Data privacy

This tool runs entirely on your machine and does not upload your images anywhere.
Do not commit microscopy images or patient derived data to the repository; the
`.gitignore` already excludes common image formats as a safety net.

## Acknowledgements and citation

- Segmentation: Cellpose-SAM (Stringer and Pachitariu). Please cite Cellpose if
  you use this tool in your work.
- Optional nuclei segmentation: StarDist (Schmidt et al., MICCAI 2018; Weigert et al.,
  WACV 2020); cite it when results were produced with the StarDist option.

## License

MIT. See [LICENSE](LICENSE).
