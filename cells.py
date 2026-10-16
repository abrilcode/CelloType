"""
Whole-cell segmentation anchored to nuclei. Not called by the current interface (see
docs/ARCHITECTURE.md); kept as an earlier approach.

The user picks the nuclei channel AND a cytoplasm/membrane channel. We segment
whole cells with Cellpose on the cytoplasm channel, then MATCH each cell to the
nucleus it contains, so every cell carries its nucleus's id. This gives three
matched regions per object for measurement: the nucleus, the cytoplasm ring
(cell minus nucleus), and the whole cell.

Rules (documented, deliberately simple and honest about the edge cases):
- A Cellpose cell that overlaps NO nucleus is dropped (not a counted cell in a
  nucleus-anchored analysis).
- Each nucleus keeps at most one cell: if two cells map to the same nucleus, the
  larger one wins. If a nucleus has no overlapping cell, its "cell" is the nucleus
  itself (empty cytoplasm ring).
- If one Cellpose cell contains two nuclei (undersegmentation), the cell goes to
  the nucleus with the largest overlap; the other nucleus falls back to nucleus-only.
"""

from __future__ import annotations

import numpy as np

import segment


def match_cells_to_nuclei(raw_cells, nuclei):
    """
    Relabel raw Cellpose cell labels so each cell carries the id of the nucleus it
    contains (max overlap). Returns an int32 label image aligned to the nuclei ids:
    cell label == nucleus label. Every nucleus is guaranteed to be a subset of its
    own cell (cell >= nucleus).
    """
    raw_cells = np.asarray(raw_cells)
    nuclei = np.asarray(nuclei)
    out = np.zeros(nuclei.shape, dtype=np.int32)

    cell_ids = np.unique(raw_cells)
    cell_ids = cell_ids[cell_ids > 0]
    sizes = {int(c): int((raw_cells == c).sum()) for c in cell_ids}

    used = set()
    # large cells first so the biggest cell wins a shared nucleus
    for c in sorted((int(x) for x in cell_ids), key=lambda c: -sizes[c]):
        region = raw_cells == c
        here = nuclei[region]
        here = here[here > 0]
        if here.size == 0:
            continue  # cell with no nucleus -> drop
        vals, counts = np.unique(here, return_counts=True)
        n = int(vals[int(np.argmax(counts))])
        if n in used:
            continue  # that nucleus already has a (larger) cell
        used.add(n)
        out[region] = n

    # guarantee cell >= nucleus: force each nucleus's pixels onto its own cell id
    nz = nuclei > 0
    out[nz] = nuclei[nz].astype(np.int32)
    return out


def cytoplasm_ring(cell_labels, nuclei):
    """Cytoplasm ring = the cell region minus the nucleus, per matched id."""
    ring = np.array(cell_labels, dtype=np.int32)
    ring[np.asarray(nuclei) > 0] = 0
    return ring


def segment_cells(cyto_arr, nuclei_masks, diameter=None, max_side: int = None):
    """
    Segment whole cells on the cytoplasm/membrane channel and match them to the
    given nuclei. Returns cell labels whose ids equal the matching nuclei ids.
    """
    raw = segment.segment_channel(cyto_arr, diameter=diameter, max_side=max_side)
    return match_cells_to_nuclei(raw, nuclei_masks)
