"""
CelloType backend API + web UI.

Endpoints:
  POST /api/upload   multiple files -> organized dataset (fields + rejected)
  GET  /             minimal upload page

Run:
  .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8010
"""

import base64
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import uuid
from typing import List

import numpy as np

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import analysis
import cells
import measure
import render
import segment
from organize import organize
from single import inspect_single, field_channels


def _measure_items(entries, subtract_bg=False, measurements=None, bg_method="object", bg_infos=None):
    """Measure a list of {token, image, condition, replicate} into tidy rows (one row
    per segmented object per channel). measurements selects the set (None -> basics).
    The background (when subtract_bg) is estimated per image and channel; if bg_infos is
    a list, one {image, ...} entry per image is appended (see measure.measure_field)."""
    allrows = []
    for it in entries:
        session = _SINGLE_SESSIONS.get(it.get("token"))
        if not session or session.get("masks") is None:
            continue
        channels = field_channels(session["paths"])
        info = {}
        rows = measure.measure_field(session["masks"], channels,
                                     subtract_bg=subtract_bg, measurements=measurements,
                                     bg_method=bg_method, bg_info=info)
        if bg_infos is not None and info:
            bg_infos.append({"image": it.get("image", ""), **info})
        for r in rows:
            r["image"] = it.get("image", "")
            r["condition"] = it.get("condition") or ""
            r["replicate"] = "" if it.get("replicate") is None else it.get("replicate")
        allrows += rows
    return allrows


def _bg_params(subtract_bg: str, bg_method: str):
    """Parse the background form fields. Returns (on, method, error_response_or_None)."""
    on = (subtract_bg == "true")
    if bg_method not in measure.BG_METHODS:
        return on, bg_method, JSONResponse(
            {"error": f"bg_method must be one of {', '.join(measure.BG_METHODS)}."}, status_code=400)
    return on, bg_method, None


def _bg_warnings(infos, on: bool):
    """Warnings about the background estimate, one per affected image."""
    out = []
    if not on:
        return out
    for inf in infos:
        if not inf.get("n_px"):
            out.append(f"Background subtraction was skipped for {inf.get('image', 'an image')}: "
                       "no pixels outside the objects.")
        elif inf.get("fraction", 1) < measure.BG_MIN_FRACTION:
            out.append(f"Few background pixels in {inf.get('image', 'an image')} "
                       f"({100 * inf['fraction']:.0f}% of the image): the background estimate is less reliable.")
    return out


def _rows_csv(rows, columns) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(columns)
    for r in rows:
        w.writerow([r.get(c, "") for c in columns])
    return buf.getvalue()

# token -> {field_key: [full file paths]} so a large preview can be built on demand
_BATCH_SESSIONS: dict = {}
# token -> {"paths": [...], "masks": <label array or None>} for the single-image flow
_SINGLE_SESSIONS: dict = {}
# result_token -> measured rows, so plotting/positivity reuse them without re-measuring
# (this is what makes the live plot preview fast and independent of segmentation sessions)
_MEASURE_RESULTS: dict = {}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

WEB = os.path.join(os.path.dirname(__file__), "web")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def no_store(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


async def _save(files: List[UploadFile]) -> List[str]:
    tmp = tempfile.mkdtemp(prefix="pfcv2_upload_")
    paths = []
    for f in files:
        # keep the original filename so marker/condition parsing works
        dest = os.path.join(tmp, os.path.basename(f.filename))
        with open(dest, "wb") as out:
            out.write(await f.read())
        paths.append(dest)
    return paths


@app.post("/api/save_file")
async def save_file(out_dir: str = Form(...), filename: str = Form(...),
                    content: str = Form(...), encoding: str = Form("text")):
    """Save one result artifact (CSV text or a base64 image) into a folder on this
    machine that the user chose. The app runs locally, so it writes to the user's own
    filesystem. Offered alongside the browser Download buttons, not instead of them."""
    folder = os.path.expanduser((out_dir or "").strip())
    if not folder:
        return JSONResponse({"error": "No output folder set."}, status_code=400)
    try:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, os.path.basename(filename))   # basename: no path traversal
        if encoding == "base64":
            payload = content.split(",", 1)[1] if content.startswith("data:") else content
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(payload))
        else:
            with open(path, "w") as fh:
                fh.write(content)
        return JSONResponse({"saved": path})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/single/inspect")
async def single_inspect(files: List[UploadFile] = File(...)):
    paths = await _save(files)
    result = inspect_single(paths, with_preview=True)
    if result.get("field"):
        token = uuid.uuid4().hex
        _SINGLE_SESSIONS[token] = {"paths": paths, "masks": None}
        result["token"] = token
    return JSONResponse(result)


@app.post("/api/single/editor")
async def single_editor(token: str = Form(...)):
    """Return the base image + encoded label map for interactive editing."""
    session = _SINGLE_SESSIONS.get(token)
    if not session or session.get("masks") is None:
        return JSONResponse({"error": "No segmentation to edit. Segment first."}, status_code=410)
    masks = session["masks"]
    arr = field_channels(session["paths"])[session["seg_channel"]][1]
    return JSONResponse({
        "base": segment.base_png_b64(arr),
        "idmap": segment.encode_idmap_png_b64(masks),
        "w": int(masks.shape[1]),
        "h": int(masks.shape[0]),
        "count": segment.count_objects(masks),
    })


@app.post("/api/single/commit")
async def single_commit(token: str = Form(...), idmap: str = Form(...)):
    """Store the user-edited label map and return a fresh overlay + count."""
    session = _SINGLE_SESSIONS.get(token)
    if not session:
        return JSONResponse({"error": "This session expired. Upload the image again."}, status_code=410)
    masks = segment.decode_idmap_dataurl(idmap)
    session["masks"] = masks
    arr = field_channels(session["paths"])[session["seg_channel"]][1]
    return JSONResponse({
        "count": segment.count_objects(masks),
        "overlay": segment.overlay_png_b64(arr, masks, kind=session.get("seg_type", "nuclei")),
    })


# measurement CSV columns are built dynamically from the chosen measurement set
# (see the measure endpoints); no fixed column list is needed here.


@app.post("/api/single/measure")
async def single_measure(token: str = Form(...), subtract_bg: str = Form("false"),
                         measurements: str = Form(""), bg_method: str = Form("object")):
    session = _SINGLE_SESSIONS.get(token)
    if not session or session.get("masks") is None:
        return JSONResponse({"error": "Segment first, then measure."}, status_code=410)
    bg_on, bg_method, err = _bg_params(subtract_bg, bg_method)
    if err:
        return err
    sel = json.loads(measurements) if measurements.strip() else None
    channels = field_channels(session["paths"])
    info = {}
    rows = measure.measure_field(session["masks"], channels,
                                 subtract_bg=bg_on, measurements=sel, bg_method=bg_method, bg_info=info)
    image = os.path.splitext(os.path.basename(session["paths"][0]))[0]
    bg_infos = [{"image": image, **info}] if info else []
    for r in rows:
        r["image"] = image
        r["condition"] = image     # one "condition" so the per-channel plot can group
        r["replicate"] = 1
    cols = ["image", "object_id", "channel"] + measure.columns_for(sel, background=bg_on)
    rtok = uuid.uuid4().hex
    _MEASURE_RESULTS[rtok] = rows
    if len(_MEASURE_RESULTS) > 8:
        _MEASURE_RESULTS.pop(next(iter(_MEASURE_RESULTS)))
    chans = list(dict.fromkeys(r["channel"] for r in rows))
    return JSONResponse({
        "result_token": rtok, "channels": chans,
        "columns": cols, "preview": rows[:100], "nrows": len(rows),
        "csv": _rows_csv(rows, cols),
        "params": {"subtract_bg": bg_on, "bg_method": bg_method if bg_on else None, "background": bg_infos},
        "warnings": _bg_warnings(bg_infos, bg_on),
    })


@app.post("/api/batch/measure")
async def batch_measure(items: str = Form(...), subtract_bg: str = Form("false"),
                        measurements: str = Form(""), bg_method: str = Form("object")):
    # items = JSON list of {token, image, condition, replicate}
    bg_on, bg_method, err = _bg_params(subtract_bg, bg_method)
    if err:
        return err
    sel = json.loads(measurements) if measurements.strip() else None
    bg_infos = []
    allrows = _measure_items(json.loads(items), subtract_bg=bg_on, measurements=sel, bg_method=bg_method,
                             bg_infos=bg_infos)
    if not allrows:
        return JSONResponse({"error": "No measurements. Segment the images first, then Measure (the session may have expired)."}, status_code=400)
    chans = list(dict.fromkeys(r["channel"] for r in allrows))
    conds = list(dict.fromkeys(r["condition"] for r in allrows))
    cols = ["image", "condition", "replicate", "object_id", "channel"] + measure.columns_for(sel, background=bg_on)
    rtok = uuid.uuid4().hex
    _MEASURE_RESULTS[rtok] = allrows
    if len(_MEASURE_RESULTS) > 8:                         # keep only recent results
        _MEASURE_RESULTS.pop(next(iter(_MEASURE_RESULTS)))
    return JSONResponse({
        "result_token": rtok,
        "columns": cols, "preview": allrows[:100], "nrows": len(allrows),
        "csv": _rows_csv(allrows, cols), "channels": chans, "conditions": conds,
        "params": {"subtract_bg": bg_on, "bg_method": bg_method if bg_on else None, "background": bg_infos},
        "warnings": _bg_warnings(bg_infos, bg_on),
    })


@app.post("/api/batch/plot")
async def batch_plot(result_token: str = Form(...), value: str = Form("mean_intensity"),
                     log2: str = Form("false"), x: str = Form("condition"),
                     conditions: str = Form(""), markers: str = Form(""),
                     focus_condition: str = Form(""), style: str = Form("violin"),
                     palette: str = Form("vivid")):
    rows = _MEASURE_RESULTS.get(result_token)
    if rows is None:
        return JSONResponse({"error": "Measurements expired. Run Measure again."}, status_code=410)
    try:
        conds = json.loads(conditions) if conditions.strip() else None
        mks = json.loads(markers) if markers.strip() else None
        fig = analysis.comparison_figure(
            rows, value=value, log2=(log2 == "true"), x=x,
            conditions=conds, markers=mks,
            focus_condition=focus_condition or None, style=style, palette=palette)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"figure": fig})


@app.post("/api/batch/scatter")
async def batch_scatter(result_token: str = Form(...), x_channel: str = Form(...),
                        y_channel: str = Form(...), log2: str = Form("true"),
                        conditions: str = Form(""), x_thr: str = Form(""), y_thr: str = Form(""),
                        palette: str = Form("vivid")):
    rows = _MEASURE_RESULTS.get(result_token)
    if rows is None:
        return JSONResponse({"error": "Measurements expired. Run Measure again."}, status_code=410)
    try:
        conds = json.loads(conditions) if conditions.strip() else None
        fig = analysis.scatter_figure(
            rows, x_channel, y_channel, log2=(log2 == "true"), conditions=conds,
            x_thr=(float(x_thr) if x_thr.strip() else None),
            y_thr=(float(y_thr) if y_thr.strip() else None), palette=palette)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"figure": fig})


@app.post("/api/batch/positivity")
async def batch_positivity(result_token: str = Form(...), channel: str = Form(...),
                           control: str = Form(""), rule: str = Form("mean_ksd"),
                           k: str = Form("2"), percentile: str = Form("99"),
                           absolute: str = Form(""), palette: str = Form("vivid")):
    rows = _MEASURE_RESULTS.get(result_token)
    if rows is None:
        return JSONResponse({"error": "Measurements expired. Run Measure again."}, status_code=410)
    value = "mean_intensity"   # positivity classifies on mean intensity
    if not any(value in r for r in rows[:1]):
        return JSONResponse({"error": "Include 'Mean gray value' in Set Measurements to compute positivity."}, status_code=400)
    try:
        res = analysis.positivity(
            rows, channel, control=control or None, rule=rule,
            k=float(k or 2), percentile=float(percentile or 99),
            absolute=(float(absolute) if absolute.strip() else None), value=value, palette=palette)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # per-cell classification CSV for the chosen channel (optional download)
    thr = res["threshold"]
    cell_cols = ["image", "condition", "replicate", "object_id", "channel", value, "positive"]
    cellrows = []
    for r in rows:
        if str(r["channel"]) == str(channel):
            cellrows.append({**{c: r.get(c, "") for c in cell_cols if c != "positive"},
                             "positive": 1 if float(r[value]) > thr else 0})
    res["cell_csv"] = _rows_csv(cellrows, cell_cols)
    res["n_cells"] = len(cellrows)
    return JSONResponse(res)


@app.post("/api/batch/positivity_all")
async def batch_positivity_all(result_token: str = Form(...), control: str = Form(""),
                               rule: str = Form("mean_ksd"), k: str = Form("2"),
                               percentile: str = Form("99"), absolute: str = Form(""),
                               conditions: str = Form(""), palette: str = Form("vivid")):
    """Positivity for EVERY channel at once (own threshold per channel), so markers can
    be compared side by side. Also returns one per-cell row with a column per marker."""
    rows = _MEASURE_RESULTS.get(result_token)
    if rows is None:
        return JSONResponse({"error": "Measurements expired. Run Measure again."}, status_code=410)
    value = "mean_intensity"
    if not any(value in r for r in rows[:1]):
        return JSONResponse({"error": "Include 'Mean gray value' in Set Measurements to compute positivity."}, status_code=400)
    try:
        res = analysis.positivity_all(
            rows, control=control or None, rule=rule,
            k=float(k or 2), percentile=float(percentile or 99),
            absolute=(float(absolute) if absolute.strip() else None),
            conditions=(json.loads(conditions) if conditions.strip() else None),
            value=value, palette=palette)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # one row per object, one positive/negative column per marker
    chans, thr = res["channels"], res["thresholds"]
    cells = {}
    for r in rows:
        ch = str(r["channel"])
        if ch not in thr:
            continue
        key = (r.get("image", ""), r.get("condition", ""), r.get("replicate", ""), r.get("object_id", ""))
        cell = cells.setdefault(key, {"image": key[0], "condition": key[1],
                                      "replicate": key[2], "object_id": key[3]})
        cell[ch] = r[value]
        cell[ch + "_positive"] = 1 if float(r[value]) > thr[ch] else 0
    cols = ["image", "condition", "replicate", "object_id"]
    for ch in chans:
        cols += [ch, ch + "_positive"]
    res["cell_csv"] = _rows_csv(list(cells.values()), cols)
    res["n_cells"] = len(cells)
    return JSONResponse(res)


@app.get("/api/palettes")
def palette_list():
    """Plot palettes (name + group) in the order they should appear in the UI."""
    return JSONResponse({"palettes": analysis.palette_list(),
                         "default": analysis._DEFAULT_PALETTE})


@app.get("/api/device")
def device_info():
    d = segment.probe_device()
    return JSONResponse({"device": d, "gpu": d != "cpu"})


@app.post("/api/measure_table")
async def measure_table(result_token: str = Form(...), columns: str = Form(...),
                        sort_by: str = Form(""), desc: str = Form("false")):
    """Re-sort the cached measurement rows by a column; returns sorted preview + CSV."""
    rows = _MEASURE_RESULTS.get(result_token)
    if rows is None:
        return JSONResponse({"error": "Measurements expired. Run Measure again."}, status_code=410)
    cols = json.loads(columns)
    order = json.loads(sort_by) if sort_by.strip() else []

    def key(r):
        out = []
        for c in order:
            v = r.get(c, "")
            try:
                out.append((0, float(v)))
            except (TypeError, ValueError):
                out.append((1, str(v)))
        return out

    srt = sorted(rows, key=key, reverse=(desc == "true")) if order else list(rows)
    return JSONResponse({"preview": srt[:100], "csv": _rows_csv(srt, cols)})


@app.get("/api/measurements")
def measurement_list():
    """The measurement set for the Set Measurements checklist (ImageJ-style)."""
    return JSONResponse({"measurements": measure.measurement_options()})


@app.post("/api/render")
async def render_view(token: str = Form(...), spec: str = Form("{}")):
    """Compose a display image: channels in chosen colours + optional segmentation
    overlay (contours or filled masks). Works for any segmented session (single or
    a batch field), since both are single-session tokens."""
    session = _SINGLE_SESSIONS.get(token)
    if not session:
        return JSONResponse({"error": "Session expired. Segment again."}, status_code=410)
    channels = field_channels(session["paths"])
    n = len(channels)
    s = json.loads(spec) if spec.strip() else {}
    colors = list(s.get("colors") or [])
    while len(colors) < n:
        colors.append([255, 255, 255])
    visible = s.get("visible")
    if not isinstance(visible, list) or len(visible) != n:
        visible = [True] * n
    ranges = s.get("ranges") if isinstance(s.get("ranges"), list) else None
    seg = s.get("seg", "off")
    masks = session.get("masks") if seg in ("contours", "masks", "both") else None
    try:
        img = render.compose(channels, colors, visible, masks=masks, seg_mode=seg, ranges=ranges)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"image": img, "channels": [nm for nm, _ in channels],
                         "has_masks": session.get("masks") is not None})


@app.post("/api/field_hist")
async def field_hist(token: str = Form(...)):
    """Per-channel histogram + native min/max + suggested display range (for the
    viewer's brightness/contrast controls)."""
    session = _SINGLE_SESSIONS.get(token)
    if not session:
        return JSONResponse({"error": "Session expired. Segment again."}, status_code=410)
    channels = field_channels(session["paths"])
    return JSONResponse({"hists": render.channel_hists(channels)})


@app.get("/api/config")
def config():
    # local: full features incl. Cellpose-GUI handoff; deploy: hide handoff, use basic web editor
    mode = "deploy" if os.environ.get("PFCV2_DEPLOY") else "local"
    return JSONResponse({"mode": mode})


EDIT_DIR = os.environ.get("PFCV2_EDIT_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cellpose_edits")


@app.post("/api/single/export_cellpose")
async def export_cellpose(token: str = Form(...)):
    session = _SINGLE_SESSIONS.get(token)
    if not session or session.get("masks") is None:
        return JSONResponse({"error": "Segment first, then export."}, status_code=410)
    arr = field_channels(session["paths"])[session["seg_channel"]][1]
    base = os.path.splitext(os.path.basename(session["paths"][0]))[0]
    out_dir = os.path.join(EDIT_DIR, token[:8])
    image_path, mask_path = segment.export_cellpose(arr, session["masks"], out_dir, base)
    return JSONResponse({
        "dir": out_dir,
        "image": os.path.basename(image_path),
        "mask": os.path.basename(mask_path),
    })


@app.post("/api/single/launch_cellpose")
async def launch_cellpose(token: str = Form(...)):
    """Write <base>.tif + <base>_seg.npy and open the real Cellpose GUI (local, full version)."""
    session = _SINGLE_SESSIONS.get(token)
    if not session or session.get("masks") is None:
        return JSONResponse({"error": "Segment first, then open in Cellpose."}, status_code=410)
    if session.get("flows") is None:
        return JSONResponse({"error": "Missing segmentation flows; re-run segmentation."}, status_code=400)
    arr = field_channels(session["paths"])[session["seg_channel"]][1]
    base = os.path.splitext(os.path.basename(session["paths"][0]))[0]
    out_dir = os.path.join(EDIT_DIR, token[:8])
    image_path, _seg_path = segment.write_seg_for_gui(arr, session["masks"], session["flows"], out_dir, base)
    session["edit_dir"] = out_dir
    session["edit_base"] = base
    try:
        subprocess.Popen([sys.executable, "-m", "cellpose"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        launched = True
    except Exception:
        launched = False
    return JSONResponse({"launched": launched, "dir": out_dir, "image": os.path.basename(image_path)})


@app.post("/api/single/reload_cellpose")
async def reload_cellpose(token: str = Form(...)):
    """Read the masks the user saved in the Cellpose GUI (<base>_seg.npy) back into the session."""
    session = _SINGLE_SESSIONS.get(token)
    if not session or not session.get("edit_dir"):
        return JSONResponse({"error": "Open in Cellpose first."}, status_code=410)
    seg_path = os.path.join(session["edit_dir"], session["edit_base"] + "_seg.npy")
    if not os.path.exists(seg_path):
        return JSONResponse({"error": "No saved edits found yet. In Cellpose: open the image, edit, and save (Ctrl+S)."}, status_code=404)
    masks = segment.read_seg_masks(seg_path)
    arr = np.asarray(field_channels(session["paths"])[session["seg_channel"]][1])
    if masks.shape != arr.shape:
        return JSONResponse({"error": f"Edited mask size {masks.shape} does not match the image {arr.shape}."}, status_code=400)
    session["masks"] = masks
    return JSONResponse({
        "count": segment.count_objects(masks),
        "overlay": segment.overlay_png_b64(arr, masks, kind=session.get("seg_type", "nuclei")),
    })


@app.post("/api/single/import_cellpose")
async def import_cellpose(token: str = Form(...), file: UploadFile = File(...)):
    session = _SINGLE_SESSIONS.get(token)
    if not session:
        return JSONResponse({"error": "This session expired. Upload the image again."}, status_code=410)
    data = await file.read()
    try:
        masks = segment.read_imported_masks(file.filename, data)
    except Exception as e:
        return JSONResponse({"error": f"Could not read masks: {e}"}, status_code=400)
    arr = np.asarray(field_channels(session["paths"])[session["seg_channel"]][1])
    if masks.shape != arr.shape:
        return JSONResponse({"error": f"Mask size {masks.shape} does not match the image {arr.shape}."}, status_code=400)
    session["masks"] = masks
    return JSONResponse({
        "count": segment.count_objects(masks),
        "overlay": segment.overlay_png_b64(arr, masks, kind=session.get("seg_type", "nuclei")),
    })


@app.post("/api/batch/segment_one")
async def batch_segment_one(token: str = Form(...), key: str = Form(...),
                            channel: int = Form(...), type: str = Form("nuclei"),
                            diameter: str = Form("")):
    """Segment ONE batch field on the chosen channel (nuclei OR whole cells). Called
    once per image. `type` is 'nuclei' or 'cells' and only affects the overlay colour."""
    mapping = _BATCH_SESSIONS.get(token)
    if not mapping or key not in mapping:
        return JSONResponse({"error": "Batch session expired. Upload again."}, status_code=410)
    channels = field_channels(mapping[key])
    if channel < 0 or channel >= len(channels):
        return JSONResponse({"error": f"Channel {channel + 1} not present in {key}."}, status_code=400)
    arr = channels[channel][1]
    diam = float(diameter) if diameter.strip() else None
    masks, flows = segment.segment_channel(arr, diameter=diam, return_flows=True)
    # each segmented field gets its own single-session so the editor/measurement/viewer reuse works per image
    field_token = uuid.uuid4().hex
    _SINGLE_SESSIONS[field_token] = {"paths": mapping[key], "masks": masks, "flows": flows,
                                     "seg_channel": channel, "seg_type": type}
    return JSONResponse({"key": key, "count": segment.count_objects(masks),
                         "overlay": segment.overlay_png_b64(arr, masks, kind=type),
                         "token": field_token})


@app.post("/api/single/segment")
async def single_segment(
    token: str = Form(...),
    channel: int = Form(...),
    type: str = Form("nuclei"),
    diameter: str = Form(""),
):
    session = _SINGLE_SESSIONS.get(token)
    if not session:
        return JSONResponse({"error": "This session expired. Upload the image again."}, status_code=410)

    channels = field_channels(session["paths"])
    if channel < 0 or channel >= len(channels):
        return JSONResponse({"error": "Invalid channel."}, status_code=400)

    arr = channels[channel][1]
    diam = float(diameter) if diameter.strip() else None
    masks, flows = segment.segment_channel(arr, diameter=diam, return_flows=True)
    session["masks"] = masks
    session["flows"] = flows
    session["seg_channel"] = channel
    session["seg_type"] = type

    return JSONResponse({
        "count": segment.count_objects(masks),
        "overlay": segment.overlay_png_b64(arr, masks, kind=type),
        "channel": channel,
        "type": type,
        "device": segment.device_name(),
    })


_QUANT_EXTS = {".czi", ".tif", ".tiff"}
_DISPLAY_EXTS = {".png", ".jpg", ".jpeg"}


@app.post("/api/batch/inspect")
async def batch_inspect(files: List[UploadFile] = File(...)):
    paths = await _save(files)

    # Block mixing quantitative (CZI/TIFF) with display images (PNG/JPEG) in one set:
    # comparing 16-bit real intensities against 8-bit adjusted ones is not valid.
    exts = {os.path.splitext(p)[1].lower() for p in paths}
    if (exts & _QUANT_EXTS) and (exts & _DISPLAY_EXTS):
        return JSONResponse({
            "fields": [], "conditions": [], "rejected": [], "blocked": True,
            "error": (
                "This upload mixes quantitative images (CZI/TIFF) with display images "
                "(PNG/JPEG). For a valid intensity comparison, upload one consistent set. "
                "Remove one group and try again."
            ),
        })

    # one composite thumbnail per field, no heavy per-channel previews
    ds = organize(paths, with_preview=False, thumbnails=True)

    # remember each field's source files so we can build a big preview on click
    tmp = os.path.dirname(paths[0]) if paths else ""
    token = uuid.uuid4().hex
    _BATCH_SESSIONS[token] = {
        f["key"]: [os.path.join(tmp, b) for b in f["files"]] for f in ds["fields"]
    }
    ds["token"] = token
    return JSONResponse(ds)


@app.post("/api/batch/preview")
async def batch_preview(token: str = Form(...), key: str = Form(...)):
    """Return one field's full breakdown on demand (composite + each channel preview + names)."""
    mapping = _BATCH_SESSIONS.get(token)
    if not mapping or key not in mapping:
        return JSONResponse({"error": "This upload session expired. Upload again."}, status_code=410)
    r = inspect_single(mapping[key], with_preview=True)   # composite ~1200 px + channel previews ~1000 px
    field = r.get("field")
    if not field:
        return JSONResponse({"error": r.get("error", "No preview available.")}, status_code=404)
    return JSONResponse({"field": field})


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    paths = await _save(files)
    return JSONResponse(organize(paths, with_preview=True))


@app.get("/")
def index():
    return FileResponse(
        os.path.join(WEB, "index.html"),
        headers={"Cache-Control": "no-store"},
    )
