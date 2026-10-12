"""
Comparison plots from the per-object measurement rows.

- violin_figure: per-marker (channel) violin of a metric across conditions,
  optional log2 y, with a points overlay (Maru / Waisman style).
- positivity: % positive per condition using a control-derived threshold
  (mean + k*SD of the control), plus a bar chart.
"""

from __future__ import annotations

import base64
import io
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns            # noqa: E402


def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- colour utilities
# Palettes come in two kinds and cannot be used the same way:
#   "cat" = several distinct hues  -> one hue per marker panel / per series entry.
#   "seq" = one hue, light to dark -> the ramp itself orders the categories, and
#           every marker panel reuses it (that is the point of a monochrome palette).
# Pale palettes need their own ramp: seaborn's white->base ramp is invisible when the
# base is already near white, so ramps here span light tint -> base -> slight shade.

def _hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb2hex(c):
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(v * 255))) for v in c)


def _mix(a, b, t):
    ca, cb = _hex2rgb(a), _hex2rgb(b)
    return _rgb2hex(tuple(ca[i] + (cb[i] - ca[i]) * t for i in range(3)))


def _lum(c):
    r, g, b = _hex2rgb(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _readable(c, cap=0.62):
    """Darken a colour until a thin mark (point, line) in it is visible on white."""
    for _ in range(12):
        if _lum(c) <= cap:
            break
        c = _mix(c, "#000000", 0.18)
    return c


def _ramp(stops, n):
    """n colours spread evenly along a list of colour stops."""
    if n <= 0:
        return []
    stops = list(stops)
    if len(stops) == 1:
        stops = [_mix(stops[0], "#ffffff", 0.55), stops[0]]
    if n == 1:
        return [stops[-1]]
    out = []
    for i in range(n):
        p = i / (n - 1) * (len(stops) - 1)
        j = min(int(p), len(stops) - 2)
        out.append(_mix(stops[j], stops[j + 1], p - j))
    return out


# name -> colours, kind, and the group shown in the UI dropdown
_PALETTES = {
    # soft multi-hue
    "quartz":     {"colors": ["#FBD9E4", "#FDECC8", "#CFEAD9", "#D3DCF7"], "kind": "cat", "group": "Soft"},
    "opal":       {"colors": ["#F7D9C4", "#FAEDCB", "#C9E4DE", "#DBCDF0"], "kind": "cat", "group": "Soft"},
    "agate":      {"colors": ["#FFDAC6", "#FCE8B6", "#C6E7D4", "#C3D9F0"], "kind": "cat", "group": "Soft"},
    # single hue, light to dark
    "rose":       {"colors": ["#FDE7EF", "#F9C6D7", "#F3A5BE", "#E87EA1"], "kind": "seq", "group": "Sequential"},
    "azure":      {"colors": ["#E3F0FB", "#BBD9F2", "#8FBEE8", "#5F9FD9"], "kind": "seq", "group": "Sequential"},
    "moss":       {"colors": ["#EAF2E7", "#CDE3C4", "#A9CE9C", "#83B573"], "kind": "seq", "group": "Sequential"},
    "iris":       {"colors": ["#F0EBF7", "#D8CBEC", "#BCA8DD", "#9D82C8"], "kind": "seq", "group": "Sequential"},
    "ochre":      {"colors": ["#F6F1E9", "#E7DBC8", "#D4C2A3", "#BCA57E"], "kind": "seq", "group": "Sequential"},
    # saturated
    "vivid":      {"colors": ["#dd00fa", "#02f26e", "#ff2e63", "#00a2ff", "#ffb300", "#00e5d0"], "kind": "cat", "group": "Saturated"},
    "neon":       {"colors": ["#ff006e", "#3a86ff", "#8338ec", "#fb5607", "#ffbe0b", "#00f5d4"], "kind": "cat", "group": "Saturated"},
    "bold":       {"colors": ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#20a3a3"], "kind": "cat", "group": "Saturated"},
    "berry":      {"colors": ["#c2255c", "#9c36b5", "#6741d9", "#3b5bdb", "#0c8599", "#0ca678"], "kind": "cat", "group": "Saturated"},
    "ocean":      {"colors": ["#0b7285", "#1098ad", "#0ca678", "#66a80f", "#f59f00", "#e8590c"], "kind": "cat", "group": "Saturated"},
    "warm":       {"colors": ["#e03131", "#f76707", "#f59f00", "#c2255c", "#ae3ec9", "#7048e8"], "kind": "cat", "group": "Saturated"},
    # kept so earlier figures stay reproducible
    "fresh":      {"colors": ["#e15c8a", "#4fb06a", "#4b8fd6", "#f0a24b", "#a06cd5", "#2bb3ad"], "kind": "cat", "group": "Standard"},
    "pastel":     {"colors": ["#ef9fbd", "#9fd6a6", "#a8c8ef", "#f3cfa0", "#c9b2e6", "#9dd8d3"], "kind": "cat", "group": "Standard"},
    "colorblind": {"colors": ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"], "kind": "cat", "group": "Standard"},
}
_DEFAULT_PALETTE = "vivid"
_OUTLINE = "#2b2b2b"     # so pale fills still read as shapes on white


def _pal(name):
    return _PALETTES.get(name) or _PALETTES[_DEFAULT_PALETTE]


def _palette(name):
    return _pal(name)["colors"]


def _series_colors(name, n):
    """n colours for a categorical series (one per condition, or one per marker)."""
    p = _pal(name)
    if p["kind"] == "seq":
        return _ramp(p["colors"], n)
    return [p["colors"][i % len(p["colors"])] for i in range(n)]


def _panel_colors(name, i, n_slots):
    """Colours for the categories drawn inside marker panel i."""
    p = _pal(name)
    if p["kind"] == "seq":
        return _ramp(p["colors"], n_slots)      # every panel shares the one ramp
    base = p["colors"][i % len(p["colors"])]
    if n_slots <= 1:
        return [base]
    return _ramp([_mix(base, "#ffffff", 0.55), base, _mix(base, "#000000", 0.22)], n_slots)


def palette_names():
    return list(_PALETTES.keys())


def palette_list():
    """Palettes for the UI dropdown, grouped."""
    return [{"name": n, "group": p["group"]} for n, p in _PALETTES.items()]


def _natkey(s):
    """Natural sort key so photo labels order 1,2,10 (not 1,10,2)."""
    s = str(s)
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def comparison_figure(rows, value: str = "mean_intensity", log2: bool = False,
                      x: str = "condition", conditions=None, markers=None,
                      focus_condition: str = None, style: str = "violin",
                      palette: str = _DEFAULT_PALETTE) -> str:
    """
    One panel per marker; a violin (or box) + strip of a per-object metric.

    x (what goes on the x-axis of each panel):
      "condition"       -> one group per condition, pooling all its photos (the default).
      "photo"           -> one group per photo, WITHIN a single condition (focus_condition).
                           Use this to compare the fields/photos of one condition.
      "condition_photo" -> x = condition, split (hue) by photo, so every photo shows separately.

    conditions -> optional list of conditions to include (None = all).
    markers    -> optional list of channels to plot (None = all).
    style      -> "violin" or "box".
    """
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No data to plot.")
    df["condition"] = df["condition"].astype(str)
    df["channel"] = df["channel"].astype(str)
    # photo label = the replicate if the user set one, else the field/image key (always present)
    def _photo(r):
        rep = str(r.get("replicate", "")).strip()
        return rep if rep not in ("", "None") else str(r.get("image", ""))
    df["_photo"] = df.apply(_photo, axis=1)

    all_markers = list(dict.fromkeys(df["channel"]))
    if markers is None:          # None = not specified (all); [] = explicitly none (error)
        markers = all_markers
    markers = [m for m in markers if m in set(all_markers)]
    if not markers:
        raise ValueError("Select at least one marker to plot.")

    if x != "photo" and conditions is not None and len(conditions) == 0:
        raise ValueError("Select at least one condition to plot.")

    if x == "photo":
        if not focus_condition:
            raise ValueError("Choose the condition whose photos you want to compare.")
        df = df[df["condition"] == str(focus_condition)]
        if df.empty:
            raise ValueError("That condition has no data.")
        xcol, hue = "_photo", None
        order = sorted(df["_photo"].unique(), key=_natkey)
        hue_order = None
        subtitle = f"{value} by photo in {focus_condition}"
    elif x == "condition_photo":
        if conditions:
            df = df[df["condition"].isin([str(c) for c in conditions])]
        xcol, hue = "condition", "_photo"
        order = sorted(df["condition"].unique())
        hue_order = sorted(df["_photo"].unique(), key=_natkey)
        subtitle = f"{value} by condition and photo"
    else:  # "condition"
        if conditions:
            df = df[df["condition"].isin([str(c) for c in conditions])]
        xcol, hue = "condition", None
        order = sorted(df["condition"].unique())
        hue_order = None
        subtitle = f"{value} by condition"

    if df.empty:
        raise ValueError("No data after the chosen filters.")

    n = len(markers)
    # a real hue only when we actually split by photo AND there is more than one photo;
    # otherwise we colour each x category with the panel's own hue (no legend).
    real_hue = (hue is not None) and (hue_order is not None) and (len(hue_order) > 1)
    per_w = 3.6 + (1.2 if real_hue else 0.0)
    # sharey: the marker panels share one y scale so channels are directly comparable
    fig, axes = plt.subplots(1, n, figsize=(max(3.8, per_w * n), 4.6), squeeze=False, sharey=True)
    ylab = value + (" (log2)" if log2 else "")
    labels_long = any(len(str(o)) > 8 for o in order)
    rot = 30 if (len(order) > 1 and labels_long) else 0   # never slant a single label
    n_slots = len(hue_order) if real_hue else len(order)
    for i, ch in enumerate(markers):
        ax = axes[0][i]
        sub = df[df["channel"] == ch].copy()
        v = sub[value].astype(float)
        sub["_y"] = np.log2(v.clip(lower=1)) if log2 else v
        colors = _panel_colors(palette, i, n_slots)
        if real_hue:                       # colour by photo, keep a compact legend
            common = dict(data=sub, x=xcol, y="_y", order=order, hue=hue, hue_order=hue_order,
                          palette=colors, ax=ax, linewidth=1, linecolor=_OUTLINE)
        else:                              # one colour per x category, no legend
            common = dict(data=sub, x=xcol, y="_y", order=order, hue=xcol, hue_order=order,
                          palette=colors, legend=False, ax=ax, linewidth=1, linecolor=_OUTLINE)
        if style == "box":
            sns.boxplot(**common, showfliers=False, width=0.6)
        else:
            # default cut (>0) so the violins taper to a point instead of a flat cut
            sns.violinplot(**common, inner="box", density_norm="width", width=0.8)
        ax.set_title(str(ch), fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel(ylab if i == 0 else "", fontsize=11)   # one y label (shared axis)
        ax.tick_params(axis="x", labelrotation=rot, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        sns.despine(ax=ax)
        leg = ax.get_legend()
        if leg:
            if real_hue and i == n - 1:
                leg.set_title("photo"); leg.set_frame_on(False)
            else:
                leg.remove()
    fig.tight_layout()
    return _fig_b64(fig)


def violin_figure(rows, value: str = "mean_intensity", log2: bool = False) -> str:
    """Backward-compatible condition-level violin (thin wrapper over comparison_figure)."""
    return comparison_figure(rows, value=value, log2=log2, x="condition")


def scatter_figure(rows, x_channel: str, y_channel: str, value: str = "mean_intensity",
                   log2: bool = True, conditions=None, x_thr=None, y_thr=None,
                   palette: str = _DEFAULT_PALETTE) -> str:
    """
    Co-expression scatter: one panel per condition, each cell a point (x = x_channel,
    y = y_channel), optional log2 axes and dashed threshold guides. Matches the
    reference figure's bottom row.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No data to plot.")
    if str(x_channel) == str(y_channel):
        raise ValueError("Pick two different markers for the scatter.")
    df["condition"] = df["condition"].astype(str)
    df["channel"] = df["channel"].astype(str)
    keys = ["condition", "image", "object_id"]
    piv = df[df["channel"].isin([str(x_channel), str(y_channel)])].pivot_table(
        index=keys, columns="channel", values=value, aggfunc="first").reset_index()
    if str(x_channel) not in piv.columns or str(y_channel) not in piv.columns:
        raise ValueError("Both markers must have been measured.")
    piv = piv.dropna(subset=[str(x_channel), str(y_channel)])
    if conditions:
        piv = piv[piv["condition"].isin([str(c) for c in conditions])]
    if piv.empty:
        raise ValueError("No data after the chosen filters.")

    tx = (lambda v: np.log2(np.clip(np.asarray(v, dtype=float), 1, None))) if log2 else (lambda v: np.asarray(v, dtype=float))
    conds = sorted(piv["condition"].unique())
    n = len(conds)
    # points are small and semi transparent, so pale palette colours are darkened here
    PAL = [_readable(c, 0.55) for c in _series_colors(palette, n)]
    fig, axes = plt.subplots(1, n, figsize=(max(3.8, 3.8 * n), 4.2), squeeze=False, sharex=True, sharey=True)
    for i, c in enumerate(conds):
        ax = axes[0][i]
        g = piv[piv["condition"] == c]
        ax.scatter(tx(g[str(x_channel)]), tx(g[str(y_channel)]), s=10, alpha=0.45,
                   color=PAL[i], edgecolors="none")
        if x_thr is not None:
            ax.axvline(float(np.log2(max(1.0, float(x_thr)))) if log2 else float(x_thr), ls="--", color="#888", lw=1)
        if y_thr is not None:
            ax.axhline(float(np.log2(max(1.0, float(y_thr)))) if log2 else float(y_thr), ls="--", color="#888", lw=1)
        ax.set_title(str(c), fontsize=13)
        ax.set_xlabel(("log2 " if log2 else "") + str(x_channel), fontsize=11)
        if i == 0:
            ax.set_ylabel(("log2 " if log2 else "") + str(y_channel), fontsize=11)
        ax.tick_params(labelsize=10)
        sns.despine(ax=ax)
    fig.tight_layout()
    return _fig_b64(fig)


def _threshold(values, rule: str, control: str, k: float, percentile: float, absolute):
    """Threshold on one channel + a readable expression of how it was obtained.

    Returns (threshold, expression). `values` is the control condition's values
    (ignored for the absolute rule).
    """
    if rule == "absolute":
        if absolute is None:
            raise ValueError("An absolute threshold value is required.")
        return float(absolute), "fixed value"
    if not control:
        raise ValueError("A control condition is required for this rule.")
    if values is None or len(values) == 0:
        raise ValueError("The chosen control condition has no data.")
    if rule == "percentile":
        return float(np.percentile(values, float(percentile))), f"P{percentile:g} of control"
    thr = float(np.mean(values) + float(k) * np.std(values, ddof=1))   # sample SD
    return thr, f"mean + {k:g} × SD of control"


def positivity(rows, channel: str, control: str = None, rule: str = "mean_ksd",
               k: float = 2.0, percentile: float = 99.0, absolute: float = None,
               value: str = "mean_intensity", palette: str = _DEFAULT_PALETTE) -> dict:
    """
    % positive per condition using a threshold on the marker channel.

    rule:
      "mean_ksd"   -> threshold = mean + k*SD of the control condition (sample SD, ddof=1).
                      k=2 ~ 97.7th percentile of a Gaussian background (~2.3% false-positive);
                      k=3 ~ 99.87th (~0.13%). Standard negative-control cutoff.
      "percentile" -> threshold = the p-th percentile of the control condition (robust to skew).
      "absolute"   -> threshold = a fixed value (no control needed).
    """
    df = pd.DataFrame(rows)
    sub = df[df["channel"].astype(str) == str(channel)].copy()
    if sub.empty:
        raise ValueError("Channel not found in the data.")
    sub["_v"] = sub[value].astype(float)
    sub["_cond"] = sub["condition"].astype(str)

    ctrl = sub[sub["_cond"] == str(control)]["_v"] if control else None
    thr, expr = _threshold(ctrl, rule, control, k, percentile, absolute)

    table, counts = {}, {}
    for c, g in sub.groupby("_cond"):
        pos = int((g["_v"] > thr).sum())
        table[c] = round(pos / len(g) * 100, 1)
        counts[c] = [pos, int(len(g))]
    conds = sorted(table.keys())

    fig, ax = plt.subplots(figsize=(max(3.8, 1.15 * len(conds) + 1), 4.0))
    ax.bar(conds, [table[c] for c in conds], color=_series_colors(palette, len(conds)),
           width=0.7, edgecolor=_OUTLINE, linewidth=0.8)
    ax.set_ylabel("% positive", fontsize=11)
    ax.set_title(f"{channel} positive", fontsize=13, pad=16)
    ax.text(0.5, 1.015, f"threshold {thr:.1f}  ({expr})", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#666")
    ax.tick_params(axis="x", labelrotation=(30 if any(len(str(c)) > 8 for c in conds) else 0), labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    return {"threshold": round(thr, 2), "rule": expr, "expr": expr, "control": control or "",
            "percent_positive": table, "counts": counts, "figure": _fig_b64(fig)}


def positivity_all(rows, control: str = None, rule: str = "mean_ksd", k: float = 2.0,
                   percentile: float = 99.0, absolute: float = None,
                   value: str = "mean_intensity", palette: str = _DEFAULT_PALETTE,
                   channels=None, conditions=None) -> dict:
    """
    Same classification as `positivity`, applied to EVERY channel so the markers can be
    compared side by side. Each channel gets its OWN threshold from the control condition
    (backgrounds differ per channel), then % positive is computed per condition.

    Returns per-channel thresholds, a channel x condition table of % positive and
    positive/total counts, and one grouped bar figure.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No data.")
    df = df.copy()
    df["_v"] = df[value].astype(float)
    df["_cond"] = df["condition"].astype(str)
    df["_ch"] = df["channel"].astype(str)

    chans = [str(c) for c in (channels or list(dict.fromkeys(df["_ch"])))]
    chans = [c for c in chans if c in set(df["_ch"])]
    if not chans:
        raise ValueError("Select at least one marker.")
    if conditions:
        df = df[df["_cond"].isin([str(c) for c in conditions])]
        if df.empty:
            raise ValueError("No data after the chosen filters.")

    thresholds, pct, counts, expr = {}, {}, {}, ""
    for ch in chans:
        sub = df[df["_ch"] == ch]
        ctrl = sub[sub["_cond"] == str(control)]["_v"] if control else None
        thr, expr = _threshold(ctrl, rule, control, k, percentile, absolute)
        thresholds[ch] = round(thr, 2)
        pct[ch], counts[ch] = {}, {}
        for c, g in sub.groupby("_cond"):
            pos = int((g["_v"] > thr).sum())
            pct[ch][c] = round(pos / len(g) * 100, 1)
            counts[ch][c] = [pos, int(len(g))]

    conds = sorted({c for ch in chans for c in pct[ch]})
    PAL = _series_colors(palette, len(chans))
    width = 0.8 / len(chans)
    fig, ax = plt.subplots(figsize=(max(4.2, 1.5 * len(conds) * len(chans) * 0.55 + 2), 4.2))
    base = np.arange(len(conds), dtype=float)
    for j, ch in enumerate(chans):
        xs = base + (j - (len(chans) - 1) / 2) * width
        ax.bar(xs, [pct[ch].get(c, 0) for c in conds], width=width * 0.92,
               color=PAL[j], label=ch, edgecolor=_OUTLINE, linewidth=0.8)
    ax.set_xticks(base)
    ax.set_xticklabels(conds, rotation=(30 if any(len(str(c)) > 8 for c in conds) else 0))
    ax.set_ylabel("% positive", fontsize=11)
    ax.set_title("Positive cells per marker", fontsize=13)
    ax.tick_params(labelsize=10)
    leg = ax.legend(frameon=False, fontsize=9, title="marker")
    leg.get_title().set_fontsize(9)
    sns.despine(ax=ax)
    fig.tight_layout()
    return {"channels": chans, "conditions": conds, "thresholds": thresholds,
            "percent_positive": pct, "counts": counts, "rule": expr, "expr": expr,
            "control": control or "", "figure": _fig_b64(fig)}
