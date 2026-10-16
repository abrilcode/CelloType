<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.png">
    <img src="docs/assets/banner-light.png" width="100%" alt="CelloType. From fluorescence images to per-cell data, in one browser tab. A 16-bit, three-channel CZI field with 178 nuclei segmented by Cellpose-SAM on the DAPI channel.">
  </picture>
</p>

<p align="center">
  <img alt="Python 3.10 to 3.12" src="https://img.shields.io/badge/python-3.10%20to%203.12-4c6fa5?style=flat-square">
  <img alt="Segmentation: Cellpose-SAM and StarDist" src="https://img.shields.io/badge/segmentation-Cellpose--SAM%20%7C%20StarDist-4c6fa5?style=flat-square">
  <img alt="Runs locally" src="https://img.shields.io/badge/runs-locally-4c6fa5?style=flat-square">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-4c6fa5?style=flat-square"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#usage">Usage</a> &nbsp;·&nbsp;
  <a href="#outputs">Outputs</a> &nbsp;·&nbsp;
  <a href="#agreement-study">Agreement study</a> &nbsp;·&nbsp;
  <a href="#documentation">Documentation</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

**CelloType** is a local web application for the quantitative analysis of multichannel
fluorescence microscopy images. It takes a dataset from raw files to per-object measurements in a
single guided workflow, and runs entirely on the researcher's machine.

## Overview

<p align="center">
  <img src="docs/assets/fig-workflow.png" width="100%" alt="CelloType workflow in four phases. Input: select images, read files at native bit depth, organise the dataset. Segmentation: choose the channel, object type and model, segment with Cellpose-SAM or StarDist, review and edit masks, with a loop back to segmentation. Quantification: choose measurements, measure every channel with the same masks, export the CSV, parameters JSON and masks. Analysis: analysis settings, then plots and positivity. Orange outlines mark researcher decisions; grey outlines mark automated steps.">
</p>
<p align="center"><sub><b>Workflow.</b> Orange outlines mark decisions taken by the researcher; grey outlines, automated steps. Masks are reviewed and edited before anything is measured (dashed red: back to segmentation).</sub></p>

Mechanical steps are automated; experimental decisions stay with the researcher and are recorded
next to the results.

- **Input.** CZI, TIFF and PNG/JPEG, read at native bit depth (PNG and JPEG are flagged as not
  quantitative); channel names from the CZI metadata or the file names, editable in the
  interface; pixel size from CZI, OME-TIFF or ImageJ TIFF metadata.
- **Segmentation.** Cellpose-SAM for nuclei, cells or other objects on any channel, on CUDA or
  Apple Silicon GPUs with a CPU fallback; optional StarDist for nuclei.
- **Review.** Every mask can be inspected and edited in the browser before measurement.
- **Quantification.** 18 ImageJ-style measurements per object on every channel with the same
  masks, sizes in µm, optional background subtraction, and border objects excluded by default.
- **Analysis.** Distributions, co-expression and positivity against a control condition.
- **Reproducibility.** Each table can be exported with a JSON record of its parameters and the
  final label masks.

<details>
<summary><b>Architecture</b></summary>
<br>
<p align="center">
  <img src="docs/assets/fig-architecture.png" width="100%" alt="CelloType architecture. Six stages (import, inspect, segment, review, measure, analyse) across four layers: interface controls, FastAPI endpoints in app.py, Python modules, and the outputs of each stage, from the channel stack to the measurements CSV, the parameters JSON, the masks and the figures.">
</p>
<p align="center"><sub>At each stage the interface sends a request to a FastAPI endpoint (under <code>/api</code>), a Python module does the work, and its output feeds the next stage. Details in <a href="docs/ARCHITECTURE.md">docs/ARCHITECTURE.md</a>.</sub></p>
</details>

## Installation

Tested with Python 3.12; 3.10 to 3.12 are supported. A GPU is optional but strongly recommended (NVIDIA CUDA or Apple
Silicon MPS). The Cellpose-SAM weights (about 1.2 GB) are downloaded to `~/.cellpose` on first
use.

```bash
git clone https://github.com/abrilcode/CelloType.git
cd CelloType
python3 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

On an NVIDIA GPU, install the [CUDA build of PyTorch](https://pytorch.org/get-started/locally/)
that matches the driver before `requirements.txt`. StarDist is optional:
`pip install stardist tensorflow`.

## Usage

```bash
uvicorn app:app --host 127.0.0.1 --port 8010
```

Open http://127.0.0.1:8010 and choose **Single image** for one field, or **Multiple images** for a
dataset organised into experimental conditions. Both flows lead through Raw data, Segment, Measure
and Analysis; Multiple images adds a Channels step after Raw data.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/screen-segment-dark.png">
  <img src="docs/assets/screen-segment-light.png" width="100%" alt="Segment step of the single-image flow. Channel, object type and segmentation model on the left; segmentation overlay on the right.">
</picture>
<p align="center"><sub>Segment step: 178 nuclei on the DAPI channel with Cellpose-SAM; the 20 that touch the image border are excluded and 158 are measured.</sub></p>

## Outputs

| File | Contents |
|---|---|
| `measurements.csv` | One row per object and channel, with the selected measurements. |
| `measurements_parameters.json` | Inputs, settings, mask edits and software versions, and whether the result can be reproduced from the parameters alone. |
| `<image>_masks.tif`, `masks.zip` | The final label image as TIFF; several images come as one TIFF each in a ZIP. |
| `cells_positive.csv` | Positivity per object and marker (multiple images). |
| Plots | PNG figures of each analysis view. |

Columns and the parameter record are described in [docs/OUTPUTS.md](docs/OUTPUTS.md).

## Agreement study

Nuclear segmentation and per-nucleus readouts were compared with a published StarDist workflow,
Multi-Channel Nuclear Analysis (Waisman, 2025), on five datasets: 78 fields and 21,385 reference
nuclei, both mask sets measured with the same code.

<p align="center">
  <img src="docs/assets/fig-agreement.png" width="100%" alt="Three panels: segmentation agreement per dataset (Dice, F1 and matched IoU, with the Dice threshold of 0.85), nuclei counted by CelloType against the reference, and per-marker intensity correlation.">
</p>
<p align="center"><sub>(a) Dice, F1 at IoU ≥ 0.5 and matched IoU per dataset; the dashed line is the Dice threshold (0.85; the F1 threshold was 0.70). (b) Nuclei detected by CelloType against the reference; the band is ±5 %. (c) Pearson r of per-nucleus mean intensity for each marker, on matched nuclei.</sub></p>

Every dataset met the thresholds set before the comparison (Dice 0.948 to 0.964, F1 0.942 to
0.965, count r ≥ 0.991, intensity r ≥ 0.997). This is agreement between two automatic methods,
not accuracy against manual annotation; on these images Cellpose-SAM (on the GPU) was 15 to 37
times slower than StarDist (on the CPU). Method, per-dataset results and timing are in
[docs/VALIDATION.md](docs/VALIDATION.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/VALIDATION.md](docs/VALIDATION.md) | Agreement study: method, thresholds, per-dataset results and timing |
| [docs/OUTPUTS.md](docs/OUTPUTS.md) | Output files, columns, measurements and the parameter record |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Modules by layer and design notes |
| [CHANGELOG.md](CHANGELOG.md) | Changes since the version evaluated in the thesis ([v1.0.0](https://github.com/abrilcode/CelloType/releases/tag/v1.0.0)) |

## Data and privacy

Images never leave the machine: the browser sends them to the local server, which keeps copies in
the system temporary folder (`pfcv2_upload_*`) until the operating system clears it. Microscopy
images and patient-derived data must not be committed to this repository; the `.gitignore`
excludes common microscopy formats.
CelloType is research software and is not intended for diagnostic use.

## Citation

CelloType was developed by Abril Herrlein as the final project of the Bioengineering degree at
the Instituto Tecnológico de Buenos Aires (ITBA), supervised by Dra. María Andrea Camilletti.
Citation metadata is also available through "Cite this repository" ([CITATION.cff](CITATION.cff)).

```bibtex
@thesis{herrlein2026celloType,
  author      = {Herrlein, Abril},
  title       = {Desarrollo de una plataforma web para segmentaci{\'o}n y cuantificaci{\'o}n
                 de im{\'a}genes de fluorescencia multicanal},
  type        = {Proyecto final de carrera, Bioingenier{\'i}a},
  institution = {Instituto Tecnol{\'o}gico de Buenos Aires},
  year        = {2026}
}
```

Results produced with CelloType should also cite the segmentation model used:
Cellpose-SAM (Pachitariu, Rariden and Stringer, bioRxiv 2025,
[doi:10.1101/2025.04.28.651001](https://doi.org/10.1101/2025.04.28.651001)) and Cellpose
(Stringer et al., Nature Methods 2021,
[doi:10.1038/s41592-020-01018-x](https://doi.org/10.1038/s41592-020-01018-x)), or StarDist
(Schmidt et al., MICCAI 2018,
[doi:10.1007/978-3-030-00934-2_30](https://doi.org/10.1007/978-3-030-00934-2_30)). The reference
workflow of the agreement study is Waisman, microPublication Biology 2025,
[doi:10.17912/micropub.biology.001590](https://doi.org/10.17912/micropub.biology.001590).

## License

Released under the [MIT License](LICENSE).
