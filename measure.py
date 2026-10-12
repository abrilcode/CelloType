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

# Background subtraction methods (only used when subtract_bg is True).
#   "object":     B = median of the background pixels; subtracted from each object's
#                 mean (mean - B) and integrated density (IntDen - area * B), not clipped.
#                 Every other intensity measurement uses the native pixel values.
#   "pixel_clip": legacy behaviour (thesis version, Aug 2026): B is subtracted from every
#                 pixel and each pixel is clipped at 0 before ALL intensity measurements.
#                 Kept only to reproduce earlier results; not offered in the UI.
BG_METHODS = ("object", "pixel_clip")

# Guard band for the "object" method: pixels within this distance of any object are left
# out of the background estimate, so blur halos and unsegmented signal next to the objects
# (for example cytoplasm or membrane around nuclear masks) do not raise B (Waters 2009).
BG_GUARD_PX = 5
BG_MIN_FRACTION = 0.10   # warn when fewer background pixels than this fraction remain


def background_mask(masks, guard_px: int = BG_GUARD_PX):
    """Pixels used to estimate the background: outside every object, minus a guard band of
    guard_px pixels around the objects. If the guard band leaves no pixel (very crowded
    field), fall back to all pixels outside the objects."""
    fg = np.asarray(masks) > 0
    bg = ~fg
    if guard_px and guard_px > 0 and fg.any():
        guarded = ~ndimage.binary_dilation(fg, iterations=int(guard_px))
        if guarded.any():
            bg = guarded
    return bg

_GEOM_SK = {"perimeter", "bounding_rect", "fit_ellipse", "shape", "feret"}   # need scikit-image


def measurement_options():
    """List for the UI: key, label, category, columns, default(checked)."""
    return [{"key": k, "label": v[0], "category": v[1], "columns": v[2],
             "default": k in DEFAULT_MEASUREMENTS} for k, v in MEASUREMENTS.items()]


def columns_for(measurements, background: bool = False):
    """Ordered measure-column names for a selection (canonical registry order).
    background=True puts the "background" column (value subtracted) first."""
    sel = set(measurements or DEFAULT_MEASUREMENTS)
    cols = ["background"] if background else []
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


def measure_field(masks, channels, subtract_bg: bool = False, measurements=None,
                  bg_method: str = "object", bg_mask=None, guard_px: int = BG_GUARD_PX,
                  bg_info: dict = None):
    """
    masks: int label image (0 = background). channels: list of (name, 2D array).
    measurements: list of keys from MEASUREMENTS (None -> DEFAULT_MEASUREMENTS).
    Returns tidy rows (object x channel), each carrying object_id, channel and the
    selected measure columns (geometry repeated across a channel's rows).

    subtract_bg: if True, estimate a background B per channel as the median of the
    pixels in bg_mask and correct with bg_method (see BG_METHODS). Default bg_mask:
    background_mask(masks, guard_px) for "object"; all pixels outside the objects for
    the legacy "pixel_clip" (unchanged). Each row then carries "background" = B (None when
    no background pixel exists, in which case nothing is subtracted).
    bg_info: optional dict filled with guard_px, n_px, fraction and B per channel.
    """
    if bg_method not in BG_METHODS:
        raise ValueError(f"bg_method must be one of {BG_METHODS}, got {bg_method!r}")
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

    if bg_mask is not None:
        bg_pixels = np.asarray(bg_mask, dtype=bool)
        used_guard = None
    elif bg_method == "object":
        bg_pixels = background_mask(masks, guard_px)
        used_guard = int(guard_px) if (bg_pixels.sum() < (masks == 0).sum()) else 0
    else:
        bg_pixels = masks == 0
        used_guard = 0
    if bg_info is not None and subtract_bg:
        bg_info.update(method=bg_method, guard_px=used_guard, n_px=int(bg_pixels.sum()),
                       fraction=round(float(bg_pixels.mean()), 4), per_channel={})

    # ---- intensity (per object per channel) ----
    per_ch = []
    for ci, (name, arr) in enumerate(channels):
        a = np.asarray(arr).astype(np.float64)
        bg = None
        if subtract_bg and bg_pixels.any():
            bg = float(np.median(a[bg_pixels]))
            if bg_method == "pixel_clip":
                a = np.clip(a - bg, 0.0, None)
        if bg_info is not None and subtract_bg:
            bg_info["per_channel"][name] = None if bg is None else round(bg, 1)
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
        if bg is not None and bg_method == "object":
            # object-level correction, before rounding; not clipped (may be negative)
            if "mean_intensity" in cols:
                cols["mean_intensity"] = cols["mean_intensity"] - bg
            if "intden_sum" in cols:
                cols["intden_sum"] = cols["intden_sum"] - area * bg
        per_ch.append((name, ci, cols, bg))

    _round1 = {"intden_sum", "com_x", "com_y", "area_fraction_pct"}
    rows = []
    for k, i in enumerate(id_list):
        for name, ci, cols, bg in per_ch:
            row = {"object_id": i, "channel": name}
            if subtract_bg:
                row["background"] = None if bg is None else round(bg, 1)
            row.update(geom[k])
            for col, arr in cols.items():
                v = float(arr[k])
                row[col] = round(v, 1) if col in _round1 else round(v, 3)
            if "stack_position" in selset:
                row["stack_position"] = ci + 1
            rows.append(row)
    return rows


def measure_regions(nuclei_masks, cell_labels, channels, subtract_bg: bool = False, measurements=None,
                    bg_method: str = "object"):
    """
    Measure the selected set in three matched regions per object: the nucleus, the
    cytoplasm ring (cell minus nucleus), and the whole cell. Each row gets a "region"
    field. Objects with an empty cytoplasm ring simply have no "cytoplasm" rows.
    The background is estimated once, outside both nuclei and cells, and shared by the
    three regions (estimating it per region would treat cytoplasm as background).
    """
    nuclei = np.asarray(nuclei_masks)
    cells = np.asarray(cell_labels)
    ring = cells.copy()
    ring[nuclei > 0] = 0
    bg_mask = background_mask((nuclei > 0) | (cells > 0), BG_GUARD_PX if bg_method == "object" else 0)
    out = []
    for region, lab in (("nucleus", nuclei), ("cytoplasm", ring), ("cell", cells)):
        for r in measure_field(lab, channels, subtract_bg=subtract_bg, measurements=measurements,
                               bg_method=bg_method, bg_mask=bg_mask):
            r["region"] = region
            out.append(r)
    return out
