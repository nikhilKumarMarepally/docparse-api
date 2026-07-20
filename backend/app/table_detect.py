"""Detect tabular layout from OCR word geometry (no keyword heuristics)."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

from app.paths import ensure_script_path

ensure_script_path()

from ocr_word_to_line_boxes import Box, Word, group_into_rows  # noqa: E402


@dataclass(frozen=True)
class TableLayout:
    row_count: int
    column_count: int
    aligned_column_count: int


def _words_in_bounds(words: Sequence[Word], bounds: dict[str, Any] | None) -> list[Word]:
    if not bounds:
        return list(words)
    x0 = float(bounds.get("min_x", 0))
    y0 = float(bounds.get("min_y", 0))
    x1 = float(bounds.get("max_x", 0))
    y1 = float(bounds.get("max_y", 0))
    kept: list[Word] = []
    for w in words:
        cx, cy = w.box.centroid_x, w.box.centroid_y
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            kept.append(w)
    return kept


def _split_row_into_cells(row: list[Word]) -> list[list[Word]]:
    if not row:
        return []
    ordered = sorted(row, key=lambda w: w.box.min_x)
    if len(ordered) == 1:
        return [ordered]

    gaps = [ordered[i].box.min_x - ordered[i - 1].box.max_x for i in range(1, len(ordered))]
    widths = [w.box.width for w in ordered]
    med_w = median(widths) if widths else 12.0
    med_gap = median(gaps) if gaps else 0.0
    # Large relative horizontal whitespace → new column cell.
    threshold = max(med_gap * 2.5, med_w * 1.25, 10.0)

    cells: list[list[Word]] = [[ordered[0]]]
    for word, gap in zip(ordered[1:], gaps):
        if gap > threshold:
            cells.append([word])
        else:
            cells[-1].append(word)
    return cells


def _cluster_1d(values: list[float], tolerance: float) -> list[list[float]]:
    if not values:
        return []
    ordered = sorted(values)
    clusters: list[list[float]] = [[ordered[0]]]
    for val in ordered[1:]:
        if val - clusters[-1][-1] <= tolerance:
            clusters[-1].append(val)
        else:
            clusters.append([val])
    return clusters


def _cell_center_x(cell: list[Word]) -> float:
    box = cell[0].box
    for w in cell[1:]:
        box = box.union(w.box)
    return box.centroid_x


def _row_center_y(row: list[Word]) -> float:
    return sum(w.box.centroid_y for w in row) / len(row)


def detect_table_layout(
    words: Sequence[Word],
    *,
    bounds: dict[str, Any] | None = None,
) -> TableLayout | None:
    """Return layout stats when word positions form a multi-column grid."""
    section_words = _words_in_bounds(words, bounds)
    if len(section_words) < 4:
        return None

    rows = group_into_rows(section_words)
    if len(rows) < 2:
        return None

    widths = [w.box.width for w in section_words]
    x_tol = max(8.0, median(widths) * 1.4) if widths else 12.0

    row_cells: list[list[list[Word]]] = []
    cell_centers_per_row: list[list[float]] = []
    for row in rows:
        cells = _split_row_into_cells(row)
        if len(cells) < 2:
            continue
        row_cells.append(cells)
        cell_centers_per_row.append([_cell_center_x(c) for c in cells])

    if len(row_cells) < 2:
        return None

    max_cols = max(len(r) for r in row_cells)
    if max_cols < 2:
        return None

    all_centers = [x for row in cell_centers_per_row for x in row]
    col_clusters = _cluster_1d(all_centers, x_tol)
    if len(col_clusters) < 2:
        return None

    # Column band must appear in at least two different rows.
    aligned_cols = 0
    for cluster in col_clusters:
        cluster_mid = sum(cluster) / len(cluster)
        rows_hit = 0
        for centers in cell_centers_per_row:
            if any(abs(cx - cluster_mid) <= x_tol for cx in centers):
                rows_hit += 1
        if rows_hit >= 2:
            aligned_cols += 1

    if aligned_cols < 2:
        return None

    # Row spacing should be more uniform than arbitrary prose blocks.
    row_ys = [_row_center_y(row) for row in rows]
    y_gaps = [row_ys[i] - row_ys[i - 1] for i in range(1, len(row_ys))]
    if len(y_gaps) >= 2:
        med_y = median(y_gaps)
        if med_y > 0:
            uniform_rows = sum(1 for g in y_gaps if abs(g - med_y) <= med_y * 0.75)
            if uniform_rows < max(1, len(y_gaps) // 2):
                return None

    multi_cell_rows = sum(1 for cells in row_cells if len(cells) >= 2)
    if multi_cell_rows < 2:
        return None

    rows_with_3plus = sum(1 for cells in row_cells if len(cells) >= 3)
    rows_with_2plus = sum(1 for cells in row_cells if len(cells) >= 2)

    # Two-row grids with matching column counts are label + value rows, not data tables.
    if len(row_cells) == 2:
        col_counts = [len(c) for c in row_cells]
        if col_counts[0] >= 3 and col_counts[0] == col_counts[1]:
            return None

    # Header + data row(s) + optional subtotal row.
    if len(row_cells) < 3:
        return None
    if rows_with_3plus < 2 and rows_with_2plus < 3:
        return None

    # Pure two-column layouts (label | value) are forms, not tables.
    if max(len(c) for c in row_cells) == 2:
        return None

    if aligned_cols < 2 or max_cols < 2:
        return None

    return TableLayout(
        row_count=len(rows),
        column_count=max_cols,
        aligned_column_count=aligned_cols,
    )


def looks_like_table(
    words: Sequence[Word] | None = None,
    *,
    bounds: dict[str, Any] | None = None,
) -> bool:
    if not words:
        return False
    return detect_table_layout(words, bounds=bounds) is not None
