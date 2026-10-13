"""
Stage 2 segmentation with Cellpose-SAM.

Segments ONE channel (the one the user chose) into a label image, and renders
an overlay (mask outlines drawn on that channel) for the viewer. The Cellpose
model is loaded once and reused.
"""

from __future__ import annotations

import base64
import io

import numpy as np

_MODEL = None
_DEVICE = None   # "mps" | "cuda" | "cpu", detected on first model load

# Resolution cap for segmentation (longest side). Cellpose-SAM is fast on a GPU
# (incl. Apple MPS) but very slow on CPU, so we cap harder on CPU. Measured on an
# M4 Pro: full 1920x1216 ~23s on MPS (n=179) vs 5+ min on CPU.
GPU_MAX_SIDE = 2048   # effectively full-res for typical microscopy fields
CPU_MAX_SIDE = 512    # keep CPU usable (lower quality, but it runs)

# outline colors by what is being segmented
OUTLINE_COLORS = {"nuclei": (0, 255, 255), "membrane": (255, 230, 0), "cells": (255, 230, 0)}


def _gpu_available() -> bool:
    import torch
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def probe_device() -> str:
    """Report the device segmentation WOULD use, without loading the model."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model():
    """Load Cellpose-SAM once, on GPU (CUDA/Apple MPS) if available else CPU."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        from cellpose import models
        use_gpu = _gpu_available()
        _MODEL = models.CellposeModel(gpu=use_gpu)
        _DEVICE = str(getattr(_MODEL, "device", "cpu"))
    return _MODEL


def default_max_side() -> int:
    return GPU_MAX_SIDE if _gpu_available() else CPU_MAX_SIDE


def device_name() -> str:
    get_model()
    return _DEVICE or "cpu"


def segment_channel(arr, diameter=None, flow_threshold: float = 0.4,
                    cellprob_threshold: float = 0.0, max_side: int = None,
                    return_flows: bool = False, info: dict = None):
    """
    Run Cellpose-SAM on a single 2D channel. Returns an int label image (0=bg)
    at the ORIGINAL resolution.

    For speed (mainly on CPU) the network runs on a reduced copy whose longest side is at
    most max_side. The reduction is done by Cellpose itself (reviewer Tomás, obs. 5): the
    image is rescaled for the network, the flows are resampled to the original size and the
    masks are computed at full resolution, which keeps contours and shape descriptors close
    to a full-resolution run. A diameter given by the user is in original pixels. Without
    reduction the call is the same as always (no rescale).
    If return_flows=True, also returns the eval flows for writing a Cellpose `_seg.npy`.
    If info is a dict, it is filled with the settings actually used (sizes, downscale
    factor, size limit, model, device, thresholds).
    """
    if max_side is None:
        max_side = default_max_side()
    a = np.asarray(arr)
    h, w = a.shape[:2]
    cap = min(1.0, max_side / max(h, w))
    model = get_model()
    if cap < 1.0:
        # Cellpose rescales by 30 / diameter; pick the diameter that gives the needed factor
        # (or the user's diameter, if it already implies a stronger reduction)
        scale = min(cap, 30.0 / diameter) if diameter else cap
        eval_diameter = 30.0 / scale
    else:
        scale, eval_diameter = 1.0, diameter
    masks, flows, _styles = model.eval(
        a,
        diameter=eval_diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    masks = np.asarray(masks)
    if info is not None:
        import os
        info.update(original_size_px=[int(h), int(w)],
                    processed_size_px=[max(1, int(round(h * scale))), max(1, int(round(w * scale)))],
                    downscale_factor=round(float(scale), 4), max_side_limit=int(max_side),
                    downscale_method=("cellpose_rescale" if scale < 1.0 else None),
                    model=os.path.basename(str(getattr(model, "pretrained_model", "cpsam"))),
                    device=_DEVICE or "cpu", flow_threshold=float(flow_threshold),
                    cellprob_threshold=float(cellprob_threshold),
                    diameter=(None if diameter is None else float(diameter)))
    return (masks, flows) if return_flows else masks


def downscale_note(info: dict):
    """User-facing note when segmentation ran on a reduced copy, else None."""
    f = (info or {}).get("downscale_factor") or 1.0
    if f >= 1.0:
        return None
    ph, pw = info["processed_size_px"]
    why = "no GPU was found" if (info.get("device") or "cpu") == "cpu" else "the image is larger than the size limit"
    return (f"Segmented on a reduced copy (x{f:.2f}, {pw} x {ph} px) because {why}. Small nuclei may be "
            "missed and close nuclei may merge; this setting was not part of the validation.")


def write_seg_for_gui(image, masks, flows, out_dir: str, base: str):
    """
    Write <base>.tif + <base>_seg.npy so the real Cellpose GUI auto-loads the masks
    when the image is opened. Returns (image_path, seg_path).
    """
    import os
    import tifffile
    from cellpose import io as cpio

    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, base + ".tif")
    tifffile.imwrite(image_path, np.asarray(image))
    # masks_flows_to_seg writes file_name + "_seg.npy" and resizes flows to mask size
    cpio.masks_flows_to_seg(np.asarray(image), np.asarray(masks), flows, os.path.join(out_dir, base))
    return image_path, os.path.join(out_dir, base + "_seg.npy")


def read_seg_masks(seg_path: str) -> np.ndarray:
    """Read the 'masks' from a Cellpose `_seg.npy` the GUI saved."""
    d = np.load(seg_path, allow_pickle=True).item()
    return np.asarray(d["masks"]).astype(np.int32)


def count_objects(masks) -> int:
    """Number of objects (distinct labels > 0). Not masks.max(): after editing, labels can
    have gaps (a deleted object leaves its number unused)."""
    m = np.asarray(masks)
    return int(np.count_nonzero(np.unique(m) > 0)) if m.size else 0


def _stretch8(arr) -> np.ndarray:
    a = np.asarray(arr).astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    return (a * 255).astype(np.uint8)


def base_png_b64(arr) -> str:
    """Full-resolution grayscale PNG of the channel (background for the editor)."""
    from PIL import Image
    im = Image.fromarray(_stretch8(arr))  # mode 'L'
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def encode_idmap_png_b64(masks) -> str:
    """Encode an int label map as an RGB PNG: label = R + G*256 + B*65536 (round-trips exactly)."""
    from PIL import Image
    m = np.asarray(masks).astype(np.int64)
    rgb = np.stack([(m & 255), ((m >> 8) & 255), ((m >> 16) & 255)], axis=-1).astype(np.uint8)
    im = Image.fromarray(rgb, "RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def decode_idmap_dataurl(dataurl: str) -> np.ndarray:
    """Decode an RGB-encoded idmap data-uri back into an int32 label map."""
    from PIL import Image
    b64 = dataurl.split(",", 1)[1]
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    a = np.asarray(im).astype(np.int32)
    return a[..., 0] + (a[..., 1] << 8) + (a[..., 2] << 16)


def export_cellpose(arr, masks, out_dir: str, base: str):
    """
    Write the channel image + a label mask image for editing in the Cellpose GUI.
    The user opens <base>.tif and uses File > Load masks on <base>_masks.tif.
    Returns (image_path, mask_path).
    """
    import os
    import tifffile
    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, base + ".tif")
    mask_path = os.path.join(out_dir, base + "_masks.tif")
    tifffile.imwrite(image_path, np.asarray(arr))
    m = np.asarray(masks)
    dtype = np.uint16 if int(m.max()) < 65535 else np.uint32
    tifffile.imwrite(mask_path, m.astype(dtype))
    return image_path, mask_path


def read_imported_masks(filename: str, data: bytes) -> np.ndarray:
    """
    Read corrected masks from a Cellpose `_seg.npy` (dict with 'masks') or from a
    label image (tif/png). Returns an int32 label map.
    """
    name = filename.lower()
    if name.endswith(".npy"):
        arr = np.load(io.BytesIO(data), allow_pickle=True)   # user's own local file
        obj = arr.item() if arr.dtype == object else arr
        masks = obj["masks"] if isinstance(obj, dict) else obj
        return np.asarray(masks).astype(np.int32)
    if name.endswith((".tif", ".tiff")):
        import tifffile
        return np.asarray(tifffile.imread(io.BytesIO(data))).astype(np.int32)
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data))).astype(np.int32)


EDGE_OUTLINE = (255, 140, 0)   # objects excluded because they touch the image border


def overlay_png_b64(arr, masks, kind: str = "nuclei", max_side: int = 1200, edge_ids=None) -> str:
    """Draw mask outlines on the (stretched) channel and return a base64 PNG. Objects in
    edge_ids (excluded at the image border) are outlined in EDGE_OUTLINE instead."""
    from PIL import Image
    from cellpose import utils

    base = _stretch8(arr)
    rgb = np.stack([base] * 3, axis=-1)
    m = np.asarray(masks)
    outlines = utils.masks_to_outlines(m)
    rgb[outlines] = OUTLINE_COLORS.get(kind, (0, 255, 255))
    if edge_ids:
        rgb[outlines & np.isin(m, list(edge_ids))] = EDGE_OUTLINE

    im = Image.fromarray(rgb, "RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
