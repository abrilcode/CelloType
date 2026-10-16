# Architecture

CelloType is a FastAPI backend and a plain HTML, CSS and JavaScript frontend served on localhost,
with no build step. The interface, the endpoints and the scientific modules are separated, so a
change to how the interface works never requires rewriting how images are read, segmented,
measured or analysed.

<p align="center">
  <img src="assets/fig-architecture.png" width="100%" alt="CelloType architecture. Six stages (import, inspect, segment, review, measure, analyse) across four layers: interface controls, FastAPI endpoints in app.py, Python modules, and the outputs of each stage.">
</p>
<p align="center"><sub>Columns are the stages of the analysis; bands are the layers. Orange outlines mark researcher input; grey, automated processing. Endpoint paths are under <code>/api</code>. Dashed red: after review, a field can be segmented again.</sub></p>

## Modules by layer

| Layer | Implementation |
|---|---|
| Frontend | `web/index.html`, `web/app.css`, `web/editor.js`: HTML, CSS and JavaScript, no build step |
| API | `app.py`: FastAPI endpoints for the Single image and Multiple images flows, rendering and exports |
| Input | `loaders.py` (CZI, TIFF, PNG/JPEG; channel names, pixel size, planes), `organize.py` (fields and conditions), `single.py` (one field from its files: channel order, names and pixel size; used by both flows) |
| Segmentation | `segment.py`: Cellpose-SAM by default (PyTorch; the `cpsam_v2` weights, default model of the tested Cellpose 4.2.1.1), optional StarDist (TensorFlow) for nuclei |
| Measurement | `measure.py`: per-object measurements, background, border rule and µm conversion (NumPy, SciPy, scikit-image) |
| Rendering and analysis | `render.py` (channels, composites, overlays, histograms), `analysis.py` (plots and positivity; Matplotlib, seaborn) |
| Run record | `runlog.py`: the JSON parameter record offered with every table |

Code the current interface does not call: `cells.py` (an earlier whole-cell approach anchored to
nuclei) with `measure.measure_regions`, and the endpoints `/api/upload`, `/api/config` and the
Cellpose GUI hand-off (`/api/single/export_cellpose`, `launch_cellpose`, `reload_cellpose`,
`import_cellpose`). Masks are edited in the web editor (`web/editor.js`).

## Design notes

- **One segmenter module.** Every model is called through `segment.py`, so another model can be
  added as an option without redesigning the flow, as StarDist was.
- **Same masks for every channel.** Segmentation runs on one chosen channel; the resulting label
  image is measured on every channel, so all markers of an object share one region.
- **Session state on the server.** Moving between steps does not recompute a segmentation, and the
  border rule can be changed without segmenting again.
- **Native bit depth.** Images are kept at their original bit depth from loading to measurement.
  Display settings (colour, contrast) only affect rendering.
- **Local by design.** The documented command binds the server to 127.0.0.1, and images never
  leave the machine. The only external requests are the interface's web fonts (Google Fonts and
  Fontshare) and the one-time download of the model weights.

## Dependencies

Core dependencies and the versions tested are listed in [`requirements.txt`](../requirements.txt).
StarDist and TensorFlow are optional and not installed by default.
