# Architecture

CelloType is a FastAPI backend and a plain HTML, CSS and JavaScript frontend served on localhost,
with no build step. The interface, the endpoints and the scientific modules are separated, so a
change to how the interface works never requires rewriting how images are read, segmented,
measured or analysed.

## Modules by layer

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
not call it.

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
