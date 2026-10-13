"""
Run-parameters record: how each exported result was produced (reviewer Tomás, obs. 7).

One JSON record per measurement result (and a copy per positivity result), offered as
`<table>_parameters.json` next to the exported table. Every key is always present; values
that do not apply are null, never missing. A run with manual mask edits cannot be
reproduced from parameters alone, and the record says so explicitly.
"""

from __future__ import annotations

import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone

import numpy as np

SCHEMA_VERSION = "1.1"   # 1.1: pixel size filled (P-03a/b), measurement.manual_pixel_size_um
APP_NAME = "CelloType"
APP_VERSION = "2026.10-post-review"
NOT_REPRODUCIBLE_NOTE = "Masks were edited manually; this run cannot be reproduced from these parameters alone."

_PACKAGES = ("cellpose", "torch", "numpy", "scipy", "scikit-image", "aicspylibczi", "tifffile", "fastapi")
_SOFTWARE = None


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _git(*args):
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", here, *args], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def software_info() -> dict:
    """App version, git commit (None outside a git checkout), Python and package versions."""
    global _SOFTWARE
    if _SOFTWARE is None:
        from importlib import metadata
        pk = {}
        for name in _PACKAGES:
            try:
                pk[name] = metadata.version(name)
            except Exception:
                pk[name] = None
        commit = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
        _SOFTWARE = {"app_name": APP_NAME, "app_version": APP_VERSION, "git_commit": commit,
                     "git_dirty": (bool(dirty) if commit is not None and dirty is not None else None),
                     "python": platform.python_version(), "packages": pk}
    return _SOFTWARE


def edit_stats(auto_masks, final_masks) -> dict:
    """Compare the automatic segmentation with the final (edited) masks, independent of label
    numbers: objects are paired one-to-one by IoU >= 0.5 (greedy, highest IoU first).
    n_removed = automatic objects with no partner, n_added = final objects with no partner,
    n_modified = paired objects whose pixels are not identical."""
    if auto_masks is None or final_masks is None:
        return {"n_added": None, "n_removed": None, "n_modified": None}
    a = np.asarray(auto_masks).astype(np.int64)
    b = np.asarray(final_masks).astype(np.int64)
    a_ids = set(np.unique(a[a > 0]).tolist())
    b_ids = set(np.unique(b[b > 0]).tolist())
    a_area = dict(zip(*np.unique(a[a > 0], return_counts=True))) if a_ids else {}
    b_area = dict(zip(*np.unique(b[b > 0], return_counts=True))) if b_ids else {}
    both = (a > 0) & (b > 0)
    pairs = []
    if both.any():
        key = a[both] * 10_000_000 + b[both]
        vals, inter = np.unique(key, return_counts=True)
        for v, c in zip(vals.tolist(), inter.tolist()):
            i, j = v // 10_000_000, v % 10_000_000
            u = a_area[i] + b_area[j] - c
            pairs.append((c / u, i, j, c))
    pairs.sort(reverse=True)
    used_a, used_b, matched = set(), set(), []
    for iou, i, j, c in pairs:
        if iou < 0.5:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); matched.append((i, j, c))
    modified = sum(1 for i, j, c in matched if not (c == a_area[i] == b_area[j]))
    return {"n_added": len(b_ids - used_b), "n_removed": len(a_ids - used_a), "n_modified": modified}


def field_record(image: str, session: dict, condition=None, replicate=None, background=None, edge=None,
                 calibration=None) -> dict:
    """Per-image block of the record, from a segmentation session."""
    seg = session.get("seg_info") or {}
    masks = session.get("masks")
    auto = session.get("auto_masks")
    st = edit_stats(auto, masks) if auto is not None else {"n_added": None, "n_removed": None, "n_modified": None}
    changed = bool(st["n_added"] or st["n_removed"] or st["n_modified"])
    paths = session.get("paths") or []
    n_final = int(np.count_nonzero(np.unique(np.asarray(masks)) > 0)) if masks is not None else None
    bg = background or {}
    return {
        "image": image,
        "input_files": [os.path.basename(p) for p in paths],
        "input_format": (os.path.splitext(paths[0])[1].lstrip(".").lower() if paths else None),
        "condition": condition, "replicate": replicate,
        "original_size_px": seg.get("original_size_px"), "processed_size_px": seg.get("processed_size_px"),
        "downscale_factor": seg.get("downscale_factor"),
        "segmentation_channel_index": session.get("seg_channel"),
        "segmentation_channel_name": seg.get("channel_name"),
        "object_type": session.get("seg_type"),
        "diameter_px": seg.get("diameter"),
        "segmented_at": seg.get("segmented_at"),
        "n_objects_automatic": seg.get("n_objects"),
        "masks": {"edited": changed, "edit_sources": list(session.get("edit_sources") or []),
                  "n_objects_final": n_final, **st},
        "edge_objects": ({"excluded": edge.get("exclude"), "rule": edge.get("rule"),
                          "n_at_edge": edge.get("at_edge"), "n_excluded": edge.get("excluded"),
                          "n_measured": edge.get("kept")} if edge else
                         {"excluded": None, "rule": None, "n_at_edge": None, "n_excluded": None, "n_measured": None}),
        "pixel_size_um": ({"x": calibration.get("x"), "y": calibration.get("y"), "source": calibration.get("source")}
                          if calibration else {"x": None, "y": None, "source": None}),
        "background": ({"method": bg.get("method"), "guard_px": bg.get("guard_px"), "n_px": bg.get("n_px"),
                        "fraction": bg.get("fraction"), "per_channel": bg.get("per_channel")} if bg else None),
    }


def build_record(workflow: str, fields: list, measurements, columns, subtract_bg: bool, bg_method,
                 n_rows: int, model_info: dict, output: str, manual_pixel_size_um=None) -> dict:
    """Whole record for one measurement result."""
    edited = any(f["masks"]["edited"] for f in fields)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "created_at": now_iso(),
        "reproducible_from_parameters": not edited,
        "reproducibility_note": NOT_REPRODUCIBLE_NOTE if edited else None,
        "software": software_info(),
        "segmentation_model": {"engine": "cellpose", "model": model_info.get("model"),
                               "device": model_info.get("device"),
                               "flow_threshold": model_info.get("flow_threshold"),
                               "cellprob_threshold": model_info.get("cellprob_threshold"),
                               "max_side_limit": model_info.get("max_side_limit")},
        "workflow": workflow,
        "fields": fields,
        "measurement": {"measurements": list(measurements) if measurements else None, "columns": list(columns),
                        "background_subtraction": bool(subtract_bg),
                        "background_method": bg_method if subtract_bg else None,
                        "manual_pixel_size_um": manual_pixel_size_um,
                        "n_objects": int(sum((f["edge_objects"]["n_measured"] or 0) for f in fields)) if fields else 0,
                        "n_rows": int(n_rows)},
        "positivity": None,
        "outputs": [output],
    }


def with_positivity(record: dict, mode: str, value: str, channel, control, rule: str, k, percentile, absolute,
                    thresholds: dict, rule_expression, output: str) -> dict:
    """Copy of a measurement record with the positivity block filled in."""
    import copy
    rec = copy.deepcopy(record)
    rec["run_id"] = uuid.uuid4().hex
    rec["created_at"] = now_iso()
    rec["positivity"] = {"mode": mode, "value": value, "channel": channel, "control_condition": control or None,
                         "rule": rule, "k": k if rule == "mean_ksd" else None,
                         "percentile": percentile if rule == "percentile" else None,
                         "absolute": absolute if rule == "absolute" else None,
                         "threshold_per_channel": thresholds, "rule_expression": rule_expression}
    rec["outputs"] = rec.get("outputs", []) + [output]
    return rec
