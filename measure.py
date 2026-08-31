"""
Per-object measurement (the "redirect"), with a selectable measurement set
mirroring ImageJ/Fiji "Set Measurements".

For each segmented object (nucleus / cytoplasm ring / cell) in a label image,
measure the requested set on every channel's NATIVE bit depth. Geometry measures
(area, centroid, perimeter, bbox, fit ellipse, shape, Feret) are per object and
channel-independent; intensity measures (mean, std, mode, min/max, center of mass,
integrated density, median, skewness, kurtosis, area fraction) are per object per
channel. Produces tidy rows (one row per object x channel), each with only the
selected columns.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, stats

# Registry (ordered): key -> (label, category, emitted columns). The order here is
# also the canonical CSV column order. "geometry" = per object; "intensity" = per channel.
MEASUREMENTS = {
    "area":               ("Area", "geometry", ["area_px"]),
    "mean":               ("Mean gray value", "intensity", ["mean_intensity"]),
    "std":                ("Standard deviation", "intensity", ["std_intensity"]),
    "mode":               ("Modal gray value", "intensity", ["mode_intensity"]),
    "minmax":             ("Min & max gray value", "intensity", ["min_intensity", "max_intensity"]),
    "centroid":           ("Centroid", "geometry", ["centroid_x", "centroid_y"]),
    "center_of_mass":     ("Center of mass", "intensity", ["com_x", "com_y"]),
    "perimeter":          ("Perimeter", "geometry", ["perimeter"]),
    "bounding_rect":      ("Bounding rectangle", "geometry", ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]),
    "fit_ellipse":        ("Fit ellipse", "geometry", ["ellipse_major", "ellipse_minor", "ellipse_angle"]),
    "shape":              ("Shape descriptors", "geometry", ["circularity", "aspect_ratio", "roundness", "solidity"]),
    "feret":              ("Feret's diameter (max)", "geometry", ["feret_max"]),
    "integrated_density": ("Integrated density", "intensity", ["intden_sum"]),
    "median":             ("Median", "intensity", ["median_intensity"]),
    "skewness":           ("Skewness", "intensity", ["skewness"]),
    "kurtosis":           ("Kurtosis", "intensity", ["kurtosis"]),
    "area_fraction":      ("Area fraction (>Otsu)", "intensity", ["area_fraction_pct"]),
    "stack_position":     ("Stack position (channel index)", "geometry", ["stack_position"]),
}

# The basics are always available and checked by default.
DEFAULT_MEASUREMENTS = ["area", "centroid", "mean", "integrated_density"]

_GEOM_SK = {"perimeter", "bounding_rect", "fit_ellipse", "shape", "feret"}   # need scikit-image


def measurement_options():
    """List for the UI: key, label, category, columns, default(checked)."""
    return [{"key": k, "label": v[0], "category": v[1], "columns": v[2],
             "default": k in DEFAULT_MEASUREMENTS} for k, v in MEASUREMENTS.items()]


def columns_for(measurements):
    """Ordered measure-column names for a selection (canonical registry order)."""
    sel = set(measurements or DEFAULT_MEASUREMENTS)
    cols = []
    for k, v in MEASUREMENTS.items():
        if k in sel:
            cols += v[2]
    return cols


def _mode(v):
    vv = np.rint(np.asarray(v)).astype(np.int64)
    if vv.size == 0:
        return 0.0
    vals, cnts = np.unique(vv, return_counts=True)
    return float(vals[int(np.argmax(cnts))])


def _skew(v):
    v = np.asarray(v, dtype=np.float64)
    return float(stats.skew(v)) if v.size > 2 and v.std() > 0 else 0.0


def _kurt(v):
    v = np.asarray(v, dtype=np.float64)
    return float(stats.kurtosis(v)) if v.size > 3 and v.std() > 0 else 0.0


def measure_field(masks, channels, subtract_bg: bool = False, measurements=None):
    """
    masks: int label image (0 = background). channels: list of (name, 2D array).
    measurements: list of keys from MEASUREMENTS (None -> DEFAULT_MEASUREMENTS).
    Returns tidy rows (object x channel), each carrying object_id, channel and the
    selected measure columns (geometry repeated across a channel's rows).

    subtract_bg: if True, subtract a per-channel background (median of the pixels
    outside all objects) from that channel before measuring (clamped at 0).
    """
    selset = set(measurements) if measurements else set(DEFAULT_MEASUREMENTS)
    masks = np.asarray(masks)
    ids = np.unique(masks)
    ids = ids[ids > 0]
    if ids.size == 0:
        return []
    id_list = [int(x) for x in ids]
    n = len(id_list)

    ones = np.ones(masks.shape, dtype=np.float64)
    area = np.atleast_1d(ndimage.sum(ones, masks, index=ids)).astype(float)

    # ---- geometry (per object, channel-independent) ----
    geom = [dict() for _ in range(n)]
    if "area" in selset:
        for k in range(n):
            geom[k]["area_px"] = int(round(area[k]))
    if "centroid" in selset:
        cents = ndimage.center_of_mass(ones, masks, index=ids)
        if not isinstance(cents, list):
            cents = [cents]
        for k in range(n):
            cy, cx = cents[k]
            geom[k]["centroid_x"] = round(float(cx), 1)
            geom[k]["centroid_y"] = round(float(cy), 1)
    if selset & _GEOM_SK:
        from skimage.measure import regionprops
        props = {p.label: p for p in regionprops(masks.astype(np.int32))}
        for k, i in enumerate(id_list):
            p = props.get(i)
            if p is None:
                continue
            if "perimeter" in selset:
                geom[k]["perimeter"] = round(float(p.perimeter), 2)
            if "bounding_rect" in selset:
                minr, minc, maxr, maxc = p.bbox
                geom[k].update(bbox_x=int(minc), bbox_y=int(minr),
                               bbox_w=int(maxc - minc), bbox_h=int(maxr - minr))
            if "fit_ellipse" in selset:
                geom[k]["ellipse_major"] = round(float(p.axis_major_length), 2)
                geom[k]["ellipse_minor"] = round(float(p.axis_minor_length), 2)
                geom[k]["ellipse_angle"] = round(float(np.degrees(p.orientation)), 2)
            if "shape" in selset:
                per = float(p.perimeter) or 1.0
                a = float(p.area)
                major = float(p.axis_major_length) or 1.0
                minor = float(p.axis_minor_length)
                geom[k]["circularity"] = round(min(1.0, 4 * np.pi * a / (per * per)), 3)
                geom[k]["aspect_ratio"] = round((major / minor) if minor else 0.0, 3)
                geom[k]["roundness"] = round(4 * a / (np.pi * major * major), 3)
                geom[k]["solidity"] = round(float(p.solidity), 3)
            if "feret" in selset:
                geom[k]["feret_max"] = round(float(p.feret_diameter_max), 2)

    bg_pixels = masks == 0

    # ---- intensity (per object per channel) ----
    per_ch = []
    for ci, (name, arr) in enumerate(channels):
        a = np.asarray(arr).astype(np.float64)
        if subtract_bg and bg_pixels.any():
            bg = float(np.median(a[bg_pixels]))
            a = np.clip(a - bg, 0.0, None)
        cols = {}
        if "mean" in selset:
            cols["mean_intensity"] = np.atleast_1d(ndimage.mean(a, masks, index=ids))
        if "integrated_density" in selset:
            cols["intden_sum"] = np.atleast_1d(ndimage.sum(a, masks, index=ids))
        if "std" in selset:
            cols["std_intensity"] = np.atleast_1d(ndimage.standard_deviation(a, masks, index=ids))
        if "minmax" in selset:
            cols["min_intensity"] = np.atleast_1d(ndimage.minimum(a, masks, index=ids))
            cols["max_intensity"] = np.atleast_1d(ndimage.maximum(a, masks, index=ids))
        if "median" in selset:
            cols["median_intensity"] = np.atleast_1d(ndimage.median(a, masks, index=ids))
        if "center_of_mass" in selset:
            coms = ndimage.center_of_mass(a, masks, index=ids)
            if not isinstance(coms, list):
                coms = [coms]
            cols["com_x"] = np.array([c[1] for c in coms])
            cols["com_y"] = np.array([c[0] for c in coms])
        if "mode" in selset:
            cols["mode_intensity"] = np.atleast_1d(ndimage.labeled_comprehension(a, masks, ids, _mode, float, 0.0))
        if "skewness" in selset:
            cols["skewness"] = np.atleast_1d(ndimage.labeled_comprehension(a, masks, ids, _skew, float, 0.0))
        if "kurtosis" in selset:
            cols["kurtosis"] = np.atleast_1d(ndimage.labeled_comprehension(a, masks, ids, _kurt, float, 0.0))
        if "area_fraction" in selset:
            from skimage.filters import threshold_otsu
            try:
                thr = float(threshold_otsu(a))
            except Exception:
                thr = float(a.mean())
            cols["area_fraction_pct"] = np.atleast_1d(ndimage.mean((a > thr).astype(np.float64), masks, index=ids)) * 100.0
        per_ch.append((name, ci, cols))

    _round1 = {"intden_sum", "com_x", "com_y", "area_fraction_pct"}
    rows = []
    for k, i in enumerate(id_list):
        for name, ci, cols in per_ch:
            row = {"object_id": i, "channel": name}
            row.update(geom[k])
            for col, arr in cols.items():
                v = float(arr[k])
                row[col] = round(v, 1) if col in _round1 else round(v, 3)
            if "stack_position" in selset:
                row["stack_position"] = ci + 1
            rows.append(row)
    return rows


def measure_regions(nuclei_masks, cell_labels, channels, subtract_bg: bool = False, measurements=None):
    """
    Measure the selected set in three matched regions per object: the nucleus, the
    cytoplasm ring (cell minus nucleus), and the whole cell. Each row gets a "region"
    field. Objects with an empty cytoplasm ring simply have no "cytoplasm" rows.
    """
    nuclei = np.asarray(nuclei_masks)
    cells = np.asarray(cell_labels)
    ring = cells.copy()
    ring[nuclei > 0] = 0
    out = []
    for region, lab in (("nucleus", nuclei), ("cytoplasm", ring), ("cell", cells)):
        for r in measure_field(lab, channels, subtract_bg=subtract_bg, measurements=measurements):
            r["region"] = region
            out.append(r)
    return out
