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
                    return_flows: bool = False):
    """
    Run Cellpose-SAM on a single 2D channel. Returns an int label image (0=bg)
    at the ORIGINAL resolution. For speed on CPU the model runs on a downscaled
    copy (longest side <= max_side); labels are mapped back with nearest-neighbor.
    If return_flows=True, also returns the eval flows (at the downscaled size) for
    writing a Cellpose `_seg.npy`.
    """
    from scipy.ndimage import zoom

    if max_side is None:
        max_side = default_max_side()
    a = np.asarray(arr)
    h, w = a.shape[:2]
    scale = min(1.0, max_side / max(h, w))

    small = zoom(a.astype(np.float32), scale, order=1) if scale < 1.0 else a
    model = get_model()
    masks, flows, _styles = model.eval(
        small,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    masks = np.asarray(masks)

    if scale < 1.0:
        # upscale labels to full resolution with nearest-neighbor (preserves ids)
        from PIL import Image
        masks = np.asarray(
            Image.fromarray(masks.astype(np.int32), mode="I").resize((w, h), Image.NEAREST)
        ).astype(np.int32)
    return (masks, flows) if return_flows else masks


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
    return int(np.asarray(masks).max())


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


def overlay_png_b64(arr, masks, kind: str = "nuclei", max_side: int = 1200) -> str:
    """Draw mask outlines on the (stretched) channel and return a base64 PNG."""
    from PIL import Image
    from cellpose import utils

    base = _stretch8(arr)
    rgb = np.stack([base] * 3, axis=-1)
    outlines = utils.masks_to_outlines(np.asarray(masks))
    rgb[outlines] = OUTLINE_COLORS.get(kind, (0, 255, 255))

    im = Image.fromarray(rgb, "RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
