"""
On-demand rendering for the post-segmentation viewer.

Compose a display image from the field's channels: each visible channel is
contrast-stretched to 8-bit, tinted with a user-chosen colour, and added
(screen/additive blend) into an RGB image. Optionally overlay the segmentation
as CONTOURS (object outlines) or as filled MASKS (each object its own colour),
on top of whatever channels are shown. This lets the user see the ROI outlines
over any single channel, any combination, or the merged composite.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from scipy.ndimage import zoom


def _to01(a, rng=None):
    """Contrast-stretch a channel to [0,1]. rng = [lo, hi] display range (brightness/
    contrast, like ImageJ min/max); if not given, use the 1-99 percentiles."""
    a = np.asarray(a).astype(np.float32)
    if rng and rng[0] is not None and rng[1] is not None:
        lo, hi = float(rng[0]), float(rng[1])
    else:
        lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def channel_hists(channels, bins=64):
    """Per-channel histogram + native min/max + suggested display range (1-99 pct),
    for the viewer's brightness/contrast controls."""
    out = []
    for name, arr in channels:
        a = np.asarray(arr).astype(np.float32)
        vmin, vmax = float(a.min()), float(a.max())
        hi_edge = vmax if vmax > vmin else vmin + 1.0
        counts, edges = np.histogram(a, bins=bins, range=(vmin, hi_edge))
        out.append({
            "name": name,
            "vmin": round(vmin, 2), "vmax": round(vmax, 2),
            "lo": round(float(np.percentile(a, 1)), 2), "hi": round(float(np.percentile(a, 99)), 2),
            "counts": counts.astype(int).tolist(),
            "edges": [round(float(e), 2) for e in edges],
        })
    return out


def _hex_to_rgb(c):
    if isinstance(c, (list, tuple)):
        return tuple(float(x) for x in c[:3])
    s = str(c).lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _label_boundaries(m):
    """Boolean mask of object outlines (where a label meets a different label/bg)."""
    b = np.zeros(m.shape, dtype=bool)
    b[:-1, :] |= m[:-1, :] != m[1:, :]
    b[1:, :] |= m[1:, :] != m[:-1, :]
    b[:, :-1] |= m[:, :-1] != m[:, 1:]
    b[:, 1:] |= m[:, 1:] != m[:, :-1]
    return b & (m > 0) | (b & _dilate_labels_present(m))


def _dilate_labels_present(m):
    """True where a labelled pixel is a 4-neighbour (so outlines show on the bg side too)."""
    lab = m > 0
    near = np.zeros_like(lab)
    near[:-1, :] |= lab[1:, :]
    near[1:, :] |= lab[:-1, :]
    near[:, :-1] |= lab[:, 1:]
    near[:, 1:] |= lab[:, :-1]
    return near


def _label_colors(m):
    """Deterministic per-label colour (same scheme as the editor)."""
    ml = m.astype(np.int64)
    r = (60 + (ml * 67) % 196).astype(np.uint8)
    g = (60 + (ml * 97) % 196).astype(np.uint8)
    b = (60 + (ml * 137) % 196).astype(np.uint8)
    return r, g, b


def compose(channels, colors, visible, masks=None, seg_mode="off", ranges=None,
            contour_rgb=(255, 235, 0), mask_alpha=0.5, max_side=1200, edge_ids=None,
            edge_rgb=(255, 140, 0)) -> str:
    """
    channels: list of (name, 2D array). colors: list of RGB (0-255) or hex per channel.
    visible: list of bool per channel. ranges: optional per-channel [lo, hi] display range
    (brightness/contrast). masks: label image or None. seg_mode: off|contours|masks.
    edge_ids: labels excluded at the image border; drawn with edge_rgb contours and a
    grey, lighter fill so they stay visible but stand apart.
    Returns a base64 PNG data URI.
    """
    if not channels:
        raise ValueError("No channels to display.")
    ranges = ranges or [None] * len(channels)
    h, w = np.asarray(channels[0][1]).shape[:2]
    scale = min(1.0, max_side / max(h, w))

    acc = None
    for i, (_name, arr) in enumerate(channels):
        if not visible[i]:
            continue
        a = _to01(arr, ranges[i] if i < len(ranges) else None)
        if scale < 1.0:
            a = zoom(a, scale, order=1)
        if acc is None:
            acc = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.float32)
        col = np.array(_hex_to_rgb(colors[i]), dtype=np.float32) / 255.0
        acc[:a.shape[0], :a.shape[1], :] += a[..., None] * col
    if acc is None:  # nothing visible -> black canvas at the channel size
        oh = max(1, int(round(h * scale)))
        ow = max(1, int(round(w * scale)))
        acc = np.zeros((oh, ow, 3), dtype=np.float32)
    rgb = (np.clip(acc, 0.0, 1.0) * 255).astype(np.uint8)

    if masks is not None and seg_mode in ("contours", "masks", "both"):
        m = np.asarray(masks)
        if scale < 1.0:
            m = zoom(m.astype(np.int32), scale, order=0)
        m = m[:rgb.shape[0], :rgb.shape[1]]
        edge = np.isin(m, list(edge_ids)) if edge_ids else np.zeros(m.shape, dtype=bool)
        if seg_mode in ("masks", "both"):
            lr, lg, lb = _label_colors(m)
            sel = m > 0
            col = np.stack([lr, lg, lb], axis=-1).astype(np.float32)
            col[edge] = 128.0                                  # excluded objects: grey fill
            base = rgb.astype(np.float32)
            base[sel] = base[sel] * (1 - mask_alpha) + col[sel] * mask_alpha
            rgb = np.clip(base, 0, 255).astype(np.uint8)
        if seg_mode in ("contours", "both"):
            bnd = _label_boundaries(m)
            rgb[bnd] = np.array(contour_rgb, dtype=np.uint8)
            if edge.any():
                rgb[_label_boundaries(np.where(edge, m, 0))] = np.array(edge_rgb, dtype=np.uint8)

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
