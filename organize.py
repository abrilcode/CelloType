"""
Multi-file organization (Stage 1, flow B).

Takes many uploaded files and turns them into a validated dataset of
"fields" (one field = one imaged position with its channels), grouped by
condition. Handles:
- two input modes: multichannel file (1 file = 1 field) vs separate
  single-channel files grouped by base name;
- deterministic filename parsing of condition / replicate / marker;
- a control-vs-experimental guess per field (immunofluorescence negative /
  secondary-only controls), which the user confirms;
- per-field validation with explicit flags.

Nothing is guessed silently. Anything ambiguous is flagged for the user.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional

from loaders import (
    LoadError,
    composite_png_b64,
    load_file,
    preview_png_b64,
    resolve_color,
)

# Anchor on a "Foto<number>" token (their convention for the field number; shown and exported
# as the technical replicate: another imaged position of the sample, not a biological replicate).
_FOTO_RE = re.compile(r"[-_ ]?foto\s*0*(\d+)", re.IGNORECASE)

# Substrings that suggest a control condition (user confirms).
_CONTROL_HINTS = (
    "control", "ctrl", "cneg", "c-neg", "cn-", "neg-", "-neg", "secondary",
    "secundario", "2rio", "unstained", "noprimary", "no-primary", "sinprim", "blank",
)


def parse_filename(filename: str) -> dict:
    """Parse `condition-Foto<replicate>[-marker].ext` deterministically."""
    base = os.path.splitext(os.path.basename(filename))[0]
    m = _FOTO_RE.search(base)
    if not m:
        return {"condition": None, "replicate": None, "marker": None,
                "field_key": base, "matched": False}
    condition = base[: m.start()].strip(" -_")
    replicate = int(m.group(1))
    marker = base[m.end():].strip(" -_") or None
    field_key = base[: m.end()].strip(" -_")
    return {"condition": condition or None, "replicate": replicate,
            "marker": marker, "field_key": field_key, "matched": True}


def guess_control(text: str) -> bool:
    """Heuristic: does this condition name look like a control? (user confirms)."""
    t = (text or "").lower()
    return any(h in t for h in _CONTROL_HINTS)


def _channel_dict(name: str, marker: Optional[str], arr, quantitative: bool,
                  with_preview: bool) -> dict:
    d = {
        "name": name,
        "marker": marker,
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "bit_depth": 8 if arr.dtype == "uint8" else (16 if arr.dtype == "uint16" else arr.itemsize * 8),
        "quantitative": quantitative,
    }
    if with_preview:
        d["preview"] = preview_png_b64(arr)
    return d


def organize(paths: List[str], with_preview: bool = True, thumbnails: bool = False) -> dict:
    """
    Build the dataset from uploaded file paths.

    Returns {"fields": [...], "rejected": [...], "conditions": [...]}.
    Each field: key, mode, files, condition, replicate, control, channels[],
    quantitative, status, flags[], and thumbnail (when thumbnails=True).
    """
    rejected: List[dict] = []
    multichannel_fields: List[dict] = []
    singles: dict = {}   # field_key -> list of (path, parsed, LoadedImage)

    for path in sorted(paths, key=lambda p: os.path.basename(p).lower()):
        name = os.path.basename(path)
        try:
            img = load_file(path)
        except LoadError as e:
            rejected.append({"file": name, "reason": str(e)})
            continue

        parsed = parse_filename(name)

        if img.n_channels > 1:
            channels, arrays, colors = [], [], []
            for i in range(img.n_channels):
                channels.append(_channel_dict(img.names[i], img.names[i], img.channels[i],
                                              img.quantitative, with_preview))
                arrays.append(img.channels[i])
                colors.append(resolve_color(img.colors[i] if i < len(img.colors) else None,
                                            img.names[i], i))
            multichannel_fields.append(_make_field(
                key=parsed["field_key"], mode="multichannel", files=[name], parsed=parsed,
                channels=channels, quantitative=img.quantitative, warnings=list(img.warnings),
                arrays=arrays, colors=colors, thumbnails=thumbnails, attention=img.needs_attention,
            ))
        else:
            singles.setdefault(parsed["field_key"], []).append((name, parsed, img))

    single_fields: List[dict] = []
    for key, members in singles.items():
        members.sort(key=lambda t: t[0].lower())
        channels, arrays, colors, quant, warns, attn = [], [], [], True, [], False
        for idx, (name, parsed, img) in enumerate(members):
            arr = img.channels[0]
            ch_name = parsed["marker"] or img.names[0]
            channels.append(_channel_dict(ch_name, parsed["marker"], arr,
                                          img.quantitative, with_preview))
            arrays.append(arr)
            colors.append(resolve_color(img.colors[0] if img.colors else None, ch_name, idx))
            quant = quant and img.quantitative
            warns += img.warnings
            attn = attn or img.needs_attention
        any_parsed = next((p for _, p, _ in members if p["matched"]), members[0][1])
        single_fields.append(_make_field(
            key=key, mode="single-channel", files=[n for n, _, _ in members], parsed=any_parsed,
            channels=channels, quantitative=quant, warnings=list(dict.fromkeys(warns)),
            arrays=arrays, colors=colors, thumbnails=thumbnails, attention=attn,
        ))

    fields = multichannel_fields + single_fields
    fields.sort(key=lambda f: f["key"].lower())

    # flag identical images (same pixel content) uploaded more than once
    by_hash: dict = {}
    for f in fields:
        by_hash.setdefault(f.get("_hash"), []).append(f)
    for hsh, group in by_hash.items():
        if hsh and len(group) > 1:
            keys = [g["key"] for g in group]
            for g in group:
                others = [k for k in keys if k != g["key"]]
                g["flags"].append("Identical image to: " + ", ".join(others))
                g["status"] = "needs_attention"
    for f in fields:
        f.pop("_hash", None)

    return {"fields": fields, "rejected": rejected, "conditions": _summarize(fields)}


def _summarize(fields: List[dict]) -> List[dict]:
    """One row per distinct condition: name, field count, control flag."""
    out: dict = {}
    for f in fields:
        cond = f["condition"] or "(unassigned)"
        row = out.setdefault(cond, {"condition": cond, "n_fields": 0, "control": False})
        row["n_fields"] += 1
        row["control"] = row["control"] or f["control"]
    return list(out.values())


def _make_field(key, mode, files, parsed, channels, quantitative, warnings,
                arrays=None, colors=None, thumbnails=False, attention=False) -> dict:
    flags = list(warnings)
    # several planes (only one is used) or unlabelled TIFF planes: shown, never blocking
    status = "needs_attention" if attention else "ok"

    dims = {(c["width"], c["height"]) for c in channels}
    if len(dims) > 1:
        flags.append("Channels have different dimensions; cannot combine into one field.")
        status = "needs_attention"

    if not parsed.get("condition"):
        flags.append("Condition not detected from filename. Enter it manually.")
        status = "needs_attention"

    if parsed.get("replicate") is None:
        flags.append("Technical replicate (Foto number) not detected. Set it manually (required).")
        if status == "ok":
            status = "needs_attention"

    if not parsed.get("matched"):
        flags.append(
            "Filename does not match 'condition-Foto<n>[-marker]'. "
            "Set condition and technical replicate manually before processing."
        )
        status = "needs_attention"

    field = {
        "key": key,
        "mode": mode,
        "files": files,
        "condition": parsed.get("condition"),
        "replicate": parsed.get("replicate"),
        "control": guess_control(parsed.get("condition") or key),
        "channels": channels,
        "quantitative": quantitative,
        "status": status,
        "flags": list(dict.fromkeys(flags)),
    }

    if thumbnails and arrays and len(dims) == 1:
        field["thumbnail"] = composite_png_b64(arrays, colors, 240)

    # content hash (pixel data) to detect identical images across the dataset
    if arrays:
        h = hashlib.md5()
        for a in arrays:
            h.update(a.tobytes())
        field["_hash"] = h.hexdigest()

    return field
