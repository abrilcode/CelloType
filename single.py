"""
Single-image flow (Stage 1, simple branch).

One image = one field. No conditions, no replicates, no grouping across
images. The user uploads either one multichannel file, or the separate
single-channel files of that one field. We produce a consistent structure:
a list of channels (marker + preview + dims) so the user can see what they
uploaded and pick the one channel to segment.
"""

from __future__ import annotations

import os
from typing import List

from loaders import (
    LoadError,
    composite_png_b64,
    load_file,
    preview_png_b64,
    resolve_color,
)


def marker_from_filename(filename: str) -> str:
    """Marker = the part after the last '-' in the base name (e.g. sample-DAPI -> DAPI)."""
    base = os.path.splitext(os.path.basename(filename))[0]
    if "-" in base:
        tail = base.rsplit("-", 1)[1].strip()
        return tail or base
    return base


def _channel(cid: int, name: str, arr, quantitative: bool, source: str,
             with_preview: bool) -> dict:
    d = {
        "id": cid,
        "name": name,
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "bit_depth": 8 if arr.dtype == "uint8" else (16 if arr.dtype == "uint16" else arr.itemsize * 8),
        "quantitative": quantitative,
        "source": source,
    }
    if with_preview:
        d["preview"] = preview_png_b64(arr, 1000)   # large enough for the main viewer
    return d


def field_channels(paths: List[str]):
    """
    Return [(name, 2D array)] for the field, in the SAME order/ids as inspect_single,
    so a channel id chosen in the UI maps to the right pixel data for segmentation.
    """
    loaded = [(os.path.basename(p), load_file(p)) for p in paths]
    multich = [(n, i) for n, i in loaded if i.n_channels > 1]
    singles = [(n, i) for n, i in loaded if i.n_channels == 1]

    if len(multich) == 1 and not singles:
        img = multich[0][1]
        return [(img.names[i], img.channels[i]) for i in range(img.n_channels)]
    if not multich and singles:
        return [(marker_from_filename(n), img.channels[0]) for n, img in singles]
    raise LoadError("Ambiguous single-image upload for segmentation.")


def inspect_single(paths: List[str], with_preview: bool = True) -> dict:
    """
    Build one field from the uploaded file(s).

    Returns {"field": {...}} on success or {"field": None, "error": ...}.
    field: channels[], n_channels, quantitative, formats[], status, flags[].
    """
    loaded, rejected = [], []
    for p in paths:
        name = os.path.basename(p)
        try:
            loaded.append((name, load_file(p)))
        except LoadError as e:
            rejected.append({"file": name, "reason": str(e)})

    if not loaded:
        return {"field": None, "rejected": rejected, "error": "No readable image."}

    multich = [(n, i) for n, i in loaded if i.n_channels > 1]
    singles = [(n, i) for n, i in loaded if i.n_channels == 1]

    channels, flags = [], []
    arrays, colors = [], []          # kept to build the merged "original" preview
    quant, formats, status = True, set(), "ok"

    if len(multich) == 1 and not singles:
        # One multichannel file: channels come from inside it.
        name, img = multich[0]
        quant = img.quantitative
        formats.add(img.fmt)
        for i in range(img.n_channels):
            col = resolve_color(img.colors[i] if i < len(img.colors) else None, img.names[i], i)
            channels.append(_channel(i, img.names[i], img.channels[i],
                                     img.quantitative, f"channel {i + 1}", with_preview))
            arrays.append(img.channels[i])
            colors.append(col)

    elif not multich and singles:
        # Separate single-channel files: each file is one channel of this field.
        dims = set()
        for idx, (name, img) in enumerate(singles):
            arr = img.channels[0]
            dims.add((arr.shape[1], arr.shape[0]))
            mk = marker_from_filename(name)
            col = resolve_color(img.colors[0] if img.colors else None, mk, idx)
            channels.append(_channel(idx, mk, arr, img.quantitative, name, with_preview))
            arrays.append(arr)
            colors.append(col)
            quant = quant and img.quantitative
            formats.add(img.fmt)
        if len(dims) > 1:
            flags.append("The channel files have different dimensions, so they cannot be one image.")
            status = "needs_attention"

    else:
        # Multiple multichannel files, or a mix: ambiguous for a single image.
        return {
            "field": None,
            "rejected": rejected,
            "error": (
                "For a single image, upload one multichannel file, "
                "or the separate single-channel files of one field. "
                "Not multiple multichannel files or a mix."
            ),
        }

    if not quant:
        flags.append("Display image (8-bit/lossy): intensities are not quantitative.")

    field = {
        "channels": channels,
        "n_channels": len(channels),
        "quantitative": quant,
        "formats": sorted(formats),
        "status": status,
        "flags": flags,
    }

    # Merged "original" view (same colors as ZEN/Fiji). Only when all channels share size.
    if with_preview and arrays and len({(c["width"], c["height"]) for c in channels}) == 1:
        field["composite"] = composite_png_b64(arrays, colors, 1200)

    return {"field": field, "rejected": rejected}
