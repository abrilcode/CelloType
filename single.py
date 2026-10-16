"""
Single-image flow (Stage 1, simple branch). field_channels, field_calibration and
inspect_single also build each field of the multiple-image flow.

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
    read_pixel_size,
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


def plane_sources(loaded) -> List[dict]:
    """Per file: the extra axes found and the plane used (for the run-parameters record)."""
    return [{"file": n, "axes_found": dict(i.dims), "plane_used": dict(i.plane)} for n, i in loaded]


def field_channels(paths: List[str], sources: list = None):
    """
    Return [(name, 2D array)] for the field, in the SAME order/ids as inspect_single,
    so a channel id chosen in the UI maps to the right pixel data for segmentation.
    If sources is a list, it is extended with plane_sources() of the files.
    """
    loaded = [(os.path.basename(p), load_file(p)) for p in paths]
    if sources is not None:
        sources.extend(plane_sources(loaded))
    multich = [(n, i) for n, i in loaded if i.n_channels > 1]
    singles = [(n, i) for n, i in loaded if i.n_channels == 1]

    if len(multich) == 1 and not singles:
        img = multich[0][1]
        return [(img.names[i], img.channels[i]) for i in range(img.n_channels)]
    if not multich and singles:
        return [(marker_from_filename(n), img.channels[0]) for n, img in singles]
    raise LoadError("Ambiguous single-image upload for segmentation.")


def field_calibration(paths: List[str]) -> dict:
    """Pixel size of one field from its files' metadata: {"x", "y", "source"} in µm.
    Separate single-channel files must agree (within 0.1 %); otherwise there is no
    calibration and "conflict" is True."""
    cals = [read_pixel_size(p) for p in paths]
    if not cals:
        return {"x": None, "y": None, "source": "none"}
    first = cals[0]
    for c in cals[1:]:
        if (c["source"] == "none") != (first["source"] == "none") or (
                first["x"] is not None and (abs(c["x"] - first["x"]) > 1e-3 * first["x"]
                                            or abs(c["y"] - first["y"]) > 1e-3 * first["y"])):
            return {"x": None, "y": None, "source": "none", "conflict": True}
    return dict(first)


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
        flags += img.warnings
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
            flags += img.warnings
        if len(dims) > 1:
            flags.append("The channel files have different dimensions, so they cannot be combined into one field.")
            status = "needs_attention"

    else:
        # Multiple multichannel files, or a mix: ambiguous for a single image.
        return {
            "field": None,
            "rejected": rejected,
            "error": (
                "A single image is one multichannel file or the separate single-channel "
                "files of one field, not several multichannel files or a mix of both. "
                "For a dataset, use Multiple images."
            ),
        }

    if not quant:
        flags.append("Display image (PNG or JPEG): intensities are not quantitative.")
    if any(i.needs_attention for _, i in loaded):
        status = "needs_attention"

    field = {
        "channels": channels,
        "n_channels": len(channels),
        "quantitative": quant,
        "formats": sorted(formats),
        "status": status,
        "flags": list(dict.fromkeys(flags)),
        "source": plane_sources(loaded),
        "pixel_size": field_calibration([p for p in paths if os.path.basename(p) in {n for n, _ in loaded}]),
    }

    # Merged "original" view (same colors as ZEN/Fiji). Only when all channels share size.
    if with_preview and arrays and len({(c["width"], c["height"]) for c in channels}) == 1:
        field["composite"] = composite_png_b64(arrays, colors, 1200)

    return {"field": field, "rejected": rejected}
