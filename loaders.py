"""
Stage 1 file loading.

Reads one microscopy file into a normalized in-memory representation:
a list of 2D channel arrays (native bit depth) plus metadata. Supports
CZI, TIFF (single- and multi-channel), and PNG/JPEG.

Design rules:
- Never guess silently. If a file cannot be read as a valid image, raise
  LoadError with an actionable message so the caller can reject it.
- Channels are kept at native bit depth (measurement happens on these later).
- PNG/JPEG are accepted but marked non-quantitative (8-bit / lossy).
- One 2D plane per file (reviewer Pérez, obs. 4): if a file has several Z planes, time
  points, scenes or series, index 0 of each is used and a warning says so; CZI tile scans
  (mosaics) are rejected; TIFF stacks whose planes are not labelled are flagged.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

SUPPORTED_EXTS = {".czi", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class LoadError(Exception):
    """Raised when a file cannot be read as a valid image."""


@dataclass
class LoadedImage:
    fmt: str                       # "czi" | "tiff" | "png" | "jpeg"
    channels: List[np.ndarray]     # each a 2D array, native dtype
    names: List[str]               # channel names (metadata or generic)
    width: int
    height: int
    dtype: str
    bit_depth: int
    quantitative: bool             # False for 8-bit/lossy display formats
    colors: List = field(default_factory=list)   # per-channel display color (hex) or None
    warnings: List[str] = field(default_factory=list)
    dims: dict = field(default_factory=dict)    # size of every non-YX, non-channel axis found (+ "series")
    plane: dict = field(default_factory=dict)   # index used on each of those axes
    multi_plane: bool = False                   # some axis had more than one plane: only one is used
    unlabelled_planes: bool = False             # TIFF planes without axis labels, taken as channels

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def needs_attention(self) -> bool:
        return self.multi_plane or self.unlabelled_planes


def bit_depth_of(dtype: np.dtype) -> int:
    dt = np.dtype(dtype)
    if dt == np.uint8:
        return 8
    if dt == np.uint16:
        return 16
    if dt == np.uint32:
        return 32
    if dt.kind == "f":
        return 32
    return dt.itemsize * 8


def ext_of(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def load_file(path: str, plane: dict = None) -> LoadedImage:
    """Read a file into a LoadedImage. Raises LoadError on anything unreadable.
    plane: index to use per axis, e.g. {"Z": 2, "T": 0, "S": 0, "series": 0}; every axis
    defaults to 0 (hook for a future plane selector; the app always uses the defaults)."""
    ext = ext_of(path)
    if ext not in SUPPORTED_EXTS:
        raise LoadError(f"Unsupported format '{ext}'. Accepted: {sorted(SUPPORTED_EXTS)}")
    plane = dict(plane or {})
    try:
        if ext == ".czi":
            return _load_czi(path, plane)
        if ext in (".tif", ".tiff"):
            return _load_tiff(path, plane)
        return _load_pil(path)  # png / jpg / jpeg
    except LoadError:
        raise
    except Exception as e:  # any decoder failure -> actionable reject
        raise LoadError(f"Could not read file: {e}")


# ---------------------------------------------------------------- planes

_AXIS_NAMES = {"Z": ("Z plane", "Z planes"), "T": ("time point", "time points"),
               "S": ("scene", "scenes"), "series": ("series", "series")}


def plane_warning(axis: str, n: int, used: int) -> str:
    """Plain-language warning for an axis with n planes of which one (index `used`) is kept."""
    one, many = _AXIS_NAMES.get(axis, (f"plane along axis {axis}", f"planes along axis {axis}"))
    msg = (f"This file has {n} {many}. CelloType analyses one 2D plane: {one} {used + 1} of {n} "
           f"(index {used}) is used")
    if axis == "Z":
        msg += "; it may not be the in-focus plane."
    elif axis in ("S", "series"):
        msg += ". To analyse the others, save each one as its own file."
    else:
        msg += "."
    return msg


# ---------------------------------------------------------------- CZI

def _czi_sizes(czi) -> dict:
    """Size of every axis across all scenes (scenes may differ in shape)."""
    lo, hi = {}, {}
    for block in czi.get_dims_shape():
        for d, (a, b) in block.items():
            lo[d] = min(lo.get(d, a), a)
            hi[d] = max(hi.get(d, b), b)
    return {d: hi[d] - lo[d] for d in hi}


def _load_czi(path: str, plane: dict = None) -> LoadedImage:
    from aicspylibczi import CziFile

    plane = plane or {}
    czi = CziFile(path)
    sizes = _czi_sizes(czi)
    if sizes.get("M", 1) > 1:
        raise LoadError(
            f"This CZI is a tile scan (mosaic) of {sizes['M']} tiles. CelloType analyses one field of "
            "view: export the stitched image or a single tile from ZEN and upload that.")
    # read only the plane that is used (index 0 of every non-C/Y/X axis by default)
    want = {d: int(plane.get(d, 0)) for d in sizes if d not in ("C", "Y", "X", "A", "M")}
    for d, i in want.items():
        if not 0 <= i < sizes[d]:
            raise LoadError(f"Plane {i} on axis {d} does not exist (the file has {sizes[d]}).")
    data, shape = czi.read_image(**want)   # ndarray + [(dim_char, size), ...]
    dims = [d for d, _ in shape]

    # Collapse every non-CYX axis by taking index 0 (single scene/tile/z).
    arr = data
    keep = {"C", "Y", "X"}
    axis = 0
    for d in list(dims):
        if d in keep:
            axis += 1
            continue
        arr = np.take(arr, 0, axis=axis)
        dims.remove(d)
    # arr now has axes among C, Y, X in original order
    if "C" in dims:
        c_axis = dims.index("C")
        arr = np.moveaxis(arr, c_axis, 0)
        channels = [np.asarray(arr[i]) for i in range(arr.shape[0])]
    else:
        channels = [np.asarray(arr)]

    if all(int(c.max()) == 0 for c in channels):
        raise LoadError(
            "CZI decoded to all zeros (compression not decodable here). "
            "Re-export as OME-TIFF from ZEN/Fiji and upload that."
        )

    names, colors = _czi_channels_meta(czi, len(channels))
    h, w = channels[0].shape[:2]
    found = {d: sizes[d] for d in want}
    warns = [plane_warning(d, n, want[d]) for d, n in found.items() if n > 1]
    return LoadedImage(
        fmt="czi",
        channels=channels,
        names=names,
        width=int(w),
        height=int(h),
        dtype=str(channels[0].dtype),
        bit_depth=bit_depth_of(channels[0].dtype),
        quantitative=True,
        colors=colors,
        warnings=warns,
        dims=found,
        plane=want,
        multi_plane=any(n > 1 for n in found.values()),
    )


def _czi_channels_meta(czi, n: int):
    """Pull dye names and display colors from CZI metadata, aligned per channel."""
    names: List[str] = []
    colors: List = []
    seen = set()
    try:
        for ch in czi.meta.iter("Channel"):
            dye = ch.findtext("DyeName") or ch.findtext("Fluor") or ch.get("Name")
            col = ch.findtext("Color")  # e.g. "#FFFF0000" (AARRGGBB)
            key = (dye, col)
            if dye and key not in seen:
                seen.add(key)
                names.append(dye.strip())
                colors.append(col)
    except Exception:
        names, colors = [], []
    while len(names) < n:
        names.append(f"Channel {len(names) + 1}")
        colors.append(None)
    return names[:n], colors[:n]


# ---------------------------------------------------------------- TIFF

def _load_tiff(path: str, plane: dict = None) -> LoadedImage:
    import tifffile

    plane = plane or {}
    with tifffile.TiffFile(path) as tf:
        n_series = len(tf.series)
        si = int(plane.get("series", 0))
        if not 0 <= si < n_series:
            raise LoadError(f"Series {si} does not exist (the file has {n_series}).")
        series = tf.series[si]
        axes = series.axes            # e.g. "YX", "CYX", "YXC", "ZCYX"
        arr = series.asarray()

    info = {}
    channels = _split_by_axes(arr, axes, plane, info)
    found, used = dict(info["collapsed"]), {d: int(plane.get(d, 0)) for d in info["collapsed"]}
    warns = [plane_warning(d, n, used[d]) for d, n in found.items() if n > 1]
    if n_series > 1:
        found["series"], used["series"] = n_series, si
        warns.insert(0, plane_warning("series", n_series, si))
    if info.get("unlabelled"):
        warns.append(f"The {len(channels)} planes of this TIFF are not labelled, so they are treated as "
                     "channels. If they are Z planes or time points, re-save the file from Fiji as a hyperstack.")
    h, w = channels[0].shape[:2]
    return LoadedImage(
        fmt="tiff",
        channels=channels,
        names=[f"Channel {i + 1}" for i in range(len(channels))],
        width=int(w),
        height=int(h),
        dtype=str(channels[0].dtype),
        bit_depth=bit_depth_of(channels[0].dtype),
        quantitative=True,
        colors=[None] * len(channels),
        warnings=warns,
        dims=found,
        plane=used,
        multi_plane=any(n > 1 for n in found.values()),
        unlabelled_planes=bool(info.get("unlabelled")),
    )


def _split_by_axes(arr: np.ndarray, axes: str, plane: dict = None, info: dict = None) -> List[np.ndarray]:
    """Reduce a TIFF array to a list of 2D channel planes using its axes string.
    Every labelled non-channel axis (Z, T, ...) is reduced to one plane (index 0 unless
    `plane` says otherwise); info["collapsed"] gets {axis: size}. One unlabelled axis
    (tifffile I or Q: no metadata) is kept as channels and info["unlabelled"] is set;
    two or more unlabelled axes cannot be interpreted and raise LoadError."""
    plane = plane or {}
    info = info if info is not None else {}
    info.setdefault("collapsed", {})
    axes = axes.upper()
    if arr.ndim == 2:
        return [np.asarray(arr)]

    unl = [c for c in axes if c in ("I", "Q")]
    if len(unl) > 1 or (unl and "C" in axes):
        raise LoadError(
            f"This TIFF has {arr.ndim - 2} dimensions besides the image and not all of them are labelled "
            f"(shape {tuple(arr.shape)}), so its channels cannot be identified. Re-save it from Fiji as a "
            "hyperstack (with channels, Z and time labelled) or as one file per channel.")

    # Collapse any Z/T/etc. (not C/S/Y/X) to one plane.
    a = arr
    ax = list(axes)
    i = 0
    while i < len(ax):
        if ax[i] in ("Y", "X", "C", "S", "I", "Q"):
            i += 1
            continue
        idx = int(plane.get(ax[i], 0))
        if not 0 <= idx < a.shape[i]:
            raise LoadError(f"Plane {idx} on axis {ax[i]} does not exist (the file has {a.shape[i]}).")
        info["collapsed"][ax[i]] = int(a.shape[i])
        a = np.take(a, idx, axis=i)
        ax.pop(i)

    # Find the channel-like axis.
    ch_axis = None
    for cand in ("C", "S", "I", "Q"):
        if cand in ax:
            ch_axis = ax.index(cand)
            break
    if ch_axis is None:
        if a.ndim == 2:
            return [np.asarray(a)]
        # 3D with no labeled channel axis: assume smallest non-YX axis is channels
        ch_axis = 0
    if ax[ch_axis] in ("I", "Q") and a.shape[ch_axis] > 1:
        info["unlabelled"] = True
    a = np.moveaxis(a, ch_axis, 0)
    return [np.asarray(a[i]) for i in range(a.shape[0])]


# ---------------------------------------------------------------- PNG / JPEG

def _load_pil(path: str) -> LoadedImage:
    from PIL import Image

    im = Image.open(path)
    im.load()
    ext = ext_of(path)
    fmt = "jpeg" if ext in (".jpg", ".jpeg") else "png"

    if im.mode in ("L", "I", "I;16", "F"):
        arr = np.asarray(im)
        channels = [arr]
        names = ["Gray"]
        colors = [None]
    else:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb)  # H,W,3
        channels = [arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]]
        names = ["Red", "Green", "Blue"]
        colors = ["#FF0000", "#00FF00", "#0000FF"]

    h, w = channels[0].shape[:2]
    return LoadedImage(
        fmt=fmt,
        channels=[np.asarray(c) for c in channels],
        names=names,
        width=int(w),
        height=int(h),
        dtype=str(channels[0].dtype),
        bit_depth=bit_depth_of(channels[0].dtype),
        quantitative=False,   # 8-bit / lossy: not measurement-grade
        colors=colors,
        warnings=["Display image (8-bit/lossy): intensities are not quantitative."],
    )


# ---------------------------------------------------------------- pixel size (spatial calibration)

# Physical length units accepted from metadata, in micrometres per unit. Resolutions in dots
# per inch (image editors, 72 or 300 dpi) and plain TIFF resolution tags are NOT a calibration.
_UNIT_UM = {"m": 1e6, "cm": 1e4, "mm": 1e3, "um": 1.0, "µm": 1.0, "μm": 1.0, "\\u00b5m": 1.0,
            "micron": 1.0, "microns": 1.0, "micrometer": 1.0, "micrometre": 1.0, "nm": 1e-3}
PIXEL_SIZE_RANGE_UM = (1e-3, 1e3)    # values outside this range are treated as no calibration


def _no_calibration():
    return {"x": None, "y": None, "source": "none"}


def _calibration(x_um, y_um, source):
    lo, hi = PIXEL_SIZE_RANGE_UM
    try:
        x, y = float(x_um), float(y_um)
    except (TypeError, ValueError):
        return _no_calibration()
    if not (lo <= x <= hi and lo <= y <= hi):
        return _no_calibration()
    return {"x": float(f"{x:.6g}"), "y": float(f"{y:.6g}"), "source": source}


def read_pixel_size(path: str) -> dict:
    """Pixel size in µm from the file's metadata, without reading the pixel data.
    Returns {"x", "y", "source"}; source is "metadata:czi", "metadata:ome", "metadata:imagej"
    or "none" (x and y None). PNG/JPEG and TIFFs without a physical unit give "none"."""
    ext = ext_of(path)
    try:
        if ext == ".czi":
            return _czi_pixel_size(path)
        if ext in (".tif", ".tiff"):
            return _tiff_pixel_size(path)
    except Exception:
        pass
    return _no_calibration()


def _czi_pixel_size(path: str) -> dict:
    """CZI: Scaling/Items/Distance[@Id=X|Y]/Value, stored in metres."""
    from aicspylibczi import CziFile

    vals = {}
    for d in CziFile(path).meta.iter("Distance"):
        axis = d.get("Id")
        v = d.findtext("Value")
        if axis in ("X", "Y") and v and axis not in vals:
            vals[axis] = float(v) * 1e6
    if "X" not in vals:
        return _no_calibration()
    return _calibration(vals["X"], vals.get("Y", vals["X"]), "metadata:czi")


def _tiff_pixel_size(path: str) -> dict:
    """OME-TIFF: Pixels PhysicalSizeX/Y with their unit (default µm).
    ImageJ TIFF: X/YResolution (pixels per unit) with a physical unit in the ImageJ metadata."""
    import tifffile
    import xml.etree.ElementTree as ET

    with tifffile.TiffFile(path) as tf:
        if tf.is_ome and tf.ome_metadata:
            root = ET.fromstring(tf.ome_metadata)
            px = next((e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "Pixels"), None)
            if px is None or px.get("PhysicalSizeX") is None:
                return _no_calibration()
            out = []
            for ax in ("X", "Y"):
                v = px.get(f"PhysicalSize{ax}") or px.get("PhysicalSizeX")
                unit = (px.get(f"PhysicalSize{ax}Unit") or "µm").strip().lower()
                if unit not in _UNIT_UM:
                    return _no_calibration()
                out.append(float(v) * _UNIT_UM[unit])
            return _calibration(out[0], out[1], "metadata:ome")
        if tf.is_imagej:
            unit = str((tf.imagej_metadata or {}).get("unit") or "").strip().lower()
            if unit not in _UNIT_UM:
                return _no_calibration()          # uncalibrated ImageJ files say "pixel" or nothing
            tags = tf.pages[0].tags
            out = []
            for name in ("XResolution", "YResolution"):
                t = tags.get(name) or tags.get("XResolution")
                if t is None:
                    return _no_calibration()
                num, den = t.value
                if not num:
                    return _no_calibration()
                out.append(den / num * _UNIT_UM[unit])   # resolution = pixels per unit
            return _calibration(out[0], out[1], "metadata:imagej")
    return _no_calibration()


# ---------------------------------------------------------------- previews

def preview_png_b64(arr: np.ndarray, max_side: int = 512) -> str:
    """Percentile-stretch a channel to 8-bit and return a small base64 PNG."""
    from PIL import Image

    a = np.asarray(arr).astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    a8 = (a * 255).astype(np.uint8)

    im = Image.fromarray(a8)
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- colors

_DEFAULT_PALETTE = [
    (0, 90, 255), (0, 220, 0), (255, 40, 0),
    (255, 0, 220), (0, 220, 220), (255, 210, 0),
]


def hex_to_rgb(h):
    if not h:
        return None
    h = str(h).lstrip("#")
    if len(h) == 8:      # AARRGGBB -> drop alpha
        h = h[2:]
    if len(h) != 6:
        return None
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def dye_color(name):
    """Conventional microscopy color for a fluorophore/marker name, or None."""
    n = (name or "").upper()

    def has(*ks):
        return any(k in n for k in ks)

    if has("DAPI", "HOECHST"):
        return (0, 90, 255)
    if has("647", "CY5", "APC", "640", "633"):
        return (255, 0, 120)
    if has("594", "555", "568", "CY3", "TRITC", "MCHERRY", "RFP"):
        return (255, 60, 0)
    if has("PE"):
        return (255, 170, 0)
    if has("488", "FITC", "GFP"):
        return (0, 220, 0)
    return None


def resolve_color(hex_color, name, idx):
    """Pick an RGB color: metadata hex first, then dye-name, then default palette."""
    rgb = hex_to_rgb(hex_color)
    if rgb:
        return rgb
    d = dye_color(name)
    if d:
        return d
    return _DEFAULT_PALETTE[idx % len(_DEFAULT_PALETTE)]


def composite_png_b64(arrays, rgb_colors, max_side: int = 640) -> str:
    """Merge channels into one RGB image using each channel's color (the 'original' view)."""
    from PIL import Image

    h, w = np.asarray(arrays[0]).shape[:2]
    acc = np.zeros((h, w, 3), np.float32)
    for arr, (r, g, b) in zip(arrays, rgb_colors):
        a = np.asarray(arr).astype(np.float32)
        if np.asarray(arr).dtype == np.uint8:
            norm = a / 255.0                      # true 8-bit display (RGB photos)
        else:
            lo, hi = np.percentile(a, 1), np.percentile(a, 99)
            if hi <= lo:
                hi = lo + 1.0
            norm = np.clip((a - lo) / (hi - lo), 0, 1)
        acc[..., 0] += norm * (r / 255.0)
        acc[..., 1] += norm * (g / 255.0)
        acc[..., 2] += norm * (b / 255.0)

    acc = np.clip(acc, 0, 1)
    im = Image.fromarray((acc * 255).astype(np.uint8), "RGB")
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
