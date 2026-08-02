"""Classify horizontal OCR sections as table vs prose from word geometry.

Repeated multi-column x anchors across rows → table; else prose/unknown.
Clusters word min_x per row; needs ≥3 aligned columns on several rows + uniform y spacing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import Any

from ocr_word_to_line_boxes import (
    Line,
    Word,
    aligned_column_valid_for_vertical,
    detect_aligned_text_column,
    estimate_page_gutter_x,
    group_into_rows,
)

_AMOUNT_LIKE = re.compile(r"^\s*\$?\s*[\d,]+\.\d{2}\s*$")


def _text_is_amount_like(text: str) -> bool:
    return bool(_AMOUNT_LIKE.match(text.strip()))


@dataclass(frozen=True)
class VerticalPartitionAnalysis:
    """Whether a detected vertical gutter splits one row grid or parallel regions."""

    has_column: bool
    is_unified_grid: bool
    split_x: float | None
    row_coupling_ratio: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_column": self.has_column,
            "is_unified_grid": self.is_unified_grid,
            "split_x": round(self.split_x, 1) if self.split_x is not None else None,
            "row_coupling_ratio": round(self.row_coupling_ratio, 3),
            "reasons": list(self.reasons),
        }

    def allows_column_split(self) -> bool:
        return self.has_column and not self.is_unified_grid


def analyze_vertical_partition(lines: list[Line], page_width: float) -> VerticalPartitionAnalysis:
    """Row-level coupling across a vertical gutter: grid column vs independent sidebar."""
    if not lines or page_width <= 0:
        return VerticalPartitionAnalysis(
            False, True, None, 0.0, ("empty",)
        )
    gutter_x = estimate_page_gutter_x(lines, page_width)
    column = detect_aligned_text_column(
        lines, page_width=page_width, gutter_x=gutter_x
    )
    if column is None or not aligned_column_valid_for_vertical(lines, column):
        return VerticalPartitionAnalysis(
            False, True, None, 0.0, ("no_valid_column",)
        )

    split_x = column.split_x
    margin = max(14.0, page_width * 0.012)
    words = [w for ln in lines for w in ln.words]
    if len(words) < 12:
        return VerticalPartitionAnalysis(
            True, True, split_x, 0.0, ("sparse_words",)
        )

    widths = [w.box.width for w in words]
    med_w = median(widths) if widths else 12.0
    _ = max(8.0, med_w * 0.85)  # reserved for future mass thresholds

    rows = group_into_rows(words)
    coupled = 0
    left_only = 0
    right_only = 0
    right_only_amount = 0

    for row in rows:
        cells = _split_row_into_cells(row)
        left_cells = [
            c
            for c in cells
            if max(w.box.centroid_x for w in c) < split_x - margin
        ]
        right_cells = [
            c
            for c in cells
            if min(w.box.centroid_x for w in c) > split_x + margin
        ]
        has_left = bool(left_cells)
        has_right = bool(right_cells)
        if has_left and has_right:
            coupled += 1
        elif has_left:
            left_only += 1
        elif has_right:
            right_only += 1
            row_text = " ".join(w.text for c in right_cells for w in c)
            if _text_is_amount_like(row_text):
                right_only_amount += 1

    total_rows = coupled + left_only + right_only
    if total_rows < 4:
        return VerticalPartitionAnalysis(
            True, True, split_x, 0.0, ("few_rows",)
        )

    coupling = coupled / total_rows
    amount_right_frac = (
        right_only_amount / right_only if right_only else 0.0
    )
    independent_right_rows = right_only - right_only_amount

    reasons: list[str] = [
        f"row_coupling:{coupling:.2f}",
        f"right_only_rows:{right_only}",
        f"independent_right:{independent_right_rows}",
        f"amount_right_frac:{amount_right_frac:.2f}",
    ]

    is_unified: bool
    if right_only == 0 and total_rows >= 6:
        is_unified = True
        reasons.append("inline_table_cells")
    elif right_only >= 4 and amount_right_frac >= 0.30:
        is_unified = True
        reasons.append("amount_column_grid")
    elif independent_right_rows >= 5 and amount_right_frac < 0.15:
        is_unified = False
        reasons.append("prose_sidebar")
    elif coupling >= 0.38:
        is_unified = True
        reasons.append("row_grid_coupling")
    elif independent_right_rows >= 4 and coupling < 0.34:
        is_unified = False
        reasons.append("parallel_right_stream")
    else:
        is_unified = coupling >= 0.28 or amount_right_frac >= 0.22

    if is_unified:
        reasons.append("unified_grid")
    else:
        reasons.append("parallel_regions")

    return VerticalPartitionAnalysis(
        True,
        is_unified,
        split_x,
        coupling,
        tuple(reasons),
    )


@dataclass(frozen=True)
class SectionLayoutResult:
    layout_kind: str  # table | section_table | prose | unknown
    confidence: float
    table_column_x: list[float]
    multi_column_line_indices: list[int]
    aligned_column_count: int
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_kind": self.layout_kind,
            "confidence": round(self.confidence, 3),
            "table_column_x": [round(x, 1) for x in self.table_column_x],
            "multi_column_line_indices": self.multi_column_line_indices,
            "aligned_column_count": self.aligned_column_count,
            "row_count": self.row_count,
        }


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


def _cell_min_x(cell: list[Word]) -> float:
    return min(w.box.min_x for w in cell)


def _row_center_y(row: list[Word]) -> float:
    return sum(w.box.centroid_y for w in row) / len(row)


def _is_pure_table_grid(
    row_cells: list[list[list[Word]]],
    *,
    max_cols: int,
    is_two_col_only: bool,
) -> bool:
    """True when the band is primarily a uniform grid (invoice line items), not a composite block (TIL)."""
    n = len(row_cells)
    if n < 2:
        return False
    single = sum(1 for cells in row_cells if len(cells) == 1)
    two_plus = sum(1 for cells in row_cells if len(cells) >= 2)
    three_plus = sum(1 for cells in row_cells if len(cells) >= 3)
    frac_single = single / n
    frac_3 = three_plus / n
    frac_2 = two_plus / n

    if max_cols >= 4 and frac_3 >= 0.45 and frac_single <= 0.25:
        return True
    if max_cols >= 3 and frac_3 >= 0.5 and frac_single <= 0.22:
        return True
    if (
        is_two_col_only
        and max_cols == 2
        and frac_2 >= 0.92
        and frac_single <= 0.08
        and n >= 6
    ):
        return True
    return False


def _layout_is_table_family(layout: SectionLayoutResult) -> bool:
    return layout.layout_kind in ("table", "section_table")


def classify_section_layout(lines: list[Line]) -> SectionLayoutResult:
    """Repeated multi-column x anchors across rows → table; single-stream lines → prose."""
    empty = SectionLayoutResult("unknown", 0.0, [], [], 0, 0)
    if not lines:
        return empty

    words: list[Word] = []
    line_by_word_id: dict[int, int] = {}
    for line in lines:
        for w in line.words:
            words.append(w)
            line_by_word_id[id(w)] = line.index

    if len(words) < 4:
        return SectionLayoutResult("prose", 0.55, [], [], 0, len(lines))

    rows = group_into_rows(words)
    if len(rows) < 2:
        return SectionLayoutResult("prose", 0.6, [], [], 0, len(lines))

    widths = [w.box.width for w in words]
    x_tol = max(8.0, median(widths) * 1.4) if widths else 12.0

    row_cells: list[list[list[Word]]] = []
    cell_min_x_per_row: list[list[float]] = []
    row_line_indices: list[set[int]] = []
    for row in rows:
        cells = _split_row_into_cells(row)
        row_cells.append(cells)
        cell_min_x_per_row.append([_cell_min_x(c) for c in cells])
        row_line_indices.append({line_by_word_id.get(id(w), -1) for w in row})

    multi_col_rows = [i for i, cells in enumerate(row_cells) if len(cells) >= 2]
    if len(multi_col_rows) < 2:
        return SectionLayoutResult("prose", 0.65, [], [], 0, len(rows))

    all_min_x = [x for row in cell_min_x_per_row for x in row]
    col_clusters = _cluster_1d(all_min_x, x_tol)
    col_mids = [sum(c) / len(c) for c in col_clusters]
    aligned_cols = 0
    for mid in col_mids:
        rows_hit = sum(
            1
            for mins in cell_min_x_per_row
            if any(abs(x - mid) <= x_tol for x in mins)
        )
        if rows_hit >= 2:
            aligned_cols += 1

    triggered_lines: set[int] = set()
    rows_with_3plus = 0
    rows_with_2plus = 0
    min_rows_for_table = max(3, len(row_cells) // 5)
    for i, cells in enumerate(row_cells):
        if len(cells) < 2:
            continue
        anchors = cell_min_x_per_row[i]
        hits = sum(
            1
            for mid in col_mids
            if any(abs(x - mid) <= x_tol for x in anchors)
        )
        if hits >= 2:
            rows_with_2plus += 1
            if len(cells) < 3:
                triggered_lines.update(row_line_indices[i])
        if len(cells) < 3:
            continue
        if hits >= 3:
            rows_with_3plus += 1
            triggered_lines.update(row_line_indices[i])

    row_ys = [_row_center_y(row) for row in rows]
    y_gaps = [row_ys[i] - row_ys[i - 1] for i in range(1, len(row_ys))]
    spacing_uniform = True
    if len(y_gaps) >= 2:
        med_y = median(y_gaps)
        if med_y > 0:
            uniform_rows = sum(1 for g in y_gaps if abs(g - med_y) <= med_y * 0.85)
            spacing_uniform = uniform_rows >= max(1, len(y_gaps) // 3)

    max_cols = max(len(c) for c in row_cells)
    col_span = (col_mids[-1] - col_mids[0]) if len(col_mids) >= 2 else 0.0
    min_rows_2col = max(3, len(row_cells) // 6)
    is_two_col_table = (
        aligned_cols >= 2
        and rows_with_2plus >= min_rows_2col
        and max_cols >= 2
        and spacing_uniform
        and len(multi_col_rows) >= 3
        and col_span >= x_tol * 2.5
    )
    is_three_col_table = (
        aligned_cols >= 3
        and rows_with_3plus >= min_rows_for_table
        and max_cols >= 3
        and spacing_uniform
        and len(multi_col_rows) >= 3
    )
    is_table = is_three_col_table or is_two_col_table

    if is_table:
        row_score = rows_with_3plus if is_three_col_table else rows_with_2plus
        conf = min(
            0.98,
            0.5
            + 0.08 * min(row_score, 10)
            + 0.05 * min(aligned_cols, 8)
            + (0.1 if spacing_uniform else 0.0),
        )
        pure = _is_pure_table_grid(
            row_cells,
            max_cols=max_cols,
            is_two_col_only=is_two_col_table and not is_three_col_table,
        )
        kind = "table" if pure else "section_table"
        return SectionLayoutResult(
            kind,
            conf,
            col_mids,
            sorted(triggered_lines),
            aligned_cols,
            len(rows),
        )

    if aligned_cols >= 2 and len(multi_col_rows) >= 2:
        return SectionLayoutResult(
            "unknown",
            0.45,
            col_mids[:aligned_cols],
            sorted(triggered_lines),
            aligned_cols,
            len(rows),
        )

    return SectionLayoutResult("prose", 0.7, [], [], aligned_cols, len(rows))


TABLE_LAYOUT_MIN_CONFIDENCE = 0.85


def page_layout_from_horizontal_bands(
    section_line_counts: list[int],
    layouts: list[SectionLayoutResult],
    *,
    confidence_threshold: float = TABLE_LAYOUT_MIN_CONFIDENCE,
) -> str:
    """Dominant horizontal band → page_layout table vs prose vs mixed."""
    if not section_line_counts or not layouts:
        return "unknown"
    dominant_i = max(range(len(section_line_counts)), key=lambda i: section_line_counts[i])
    dom = layouts[dominant_i]
    total = sum(section_line_counts)
    if _layout_is_table_family(dom) and dom.confidence >= confidence_threshold:
        # Dominant table-family band covering most of the page → pure table.
        # section_table with only a tiny header/footer stub is still one table
        # (e.g. RISC itemization); "mixed" is for real section + table pages.
        non_table = sum(
            n
            for n, lay in zip(section_line_counts, layouts)
            if not (
                _layout_is_table_family(lay) and lay.confidence >= confidence_threshold
            )
        )
        if dom.layout_kind == "table" or (total and non_table / total <= 0.15):
            return "table"
        return "mixed"
    kinds = {lay.layout_kind for lay in layouts if lay.confidence >= 0.5}
    if kinds == {"prose"}:
        return "prose"
    if kinds <= {"prose", "unknown"}:
        return "prose"
    return "mixed"


def table_layout_skips_vertical(
    section_line_counts: list[int],
    layouts: list[SectionLayoutResult],
    *,
    all_lines: list[Line] | None = None,
    page_width: float | None = None,
    confidence_threshold: float = TABLE_LAYOUT_MIN_CONFIDENCE,
    multi_band_table_line_fraction: float = 0.85,
) -> bool:
    """True when horizontal content is predominantly a table — skip column sectioning."""
    if not section_line_counts or not layouts:
        return False
    total = sum(section_line_counts)
    if total == 0:
        return False
    table_lines = sum(
        n
        for n, layout in zip(section_line_counts, layouts)
        if _layout_is_table_family(layout)
        and layout.confidence >= confidence_threshold
    )
    if len(section_line_counts) == 1:
        if not (table_lines > 0 and table_lines >= total):
            return False
        if all_lines is not None and page_width is not None:
            return analyze_vertical_partition(all_lines, page_width).is_unified_grid
        return True
    if table_lines > 0 and table_lines / total >= multi_band_table_line_fraction:
        return True
    # Do not gate the whole page from classify_section_layout(all_lines): mixed invoices
    # have table bands plus header blocks; per-band band_skips_vertical_column_split
    # is the right lever. Page-level all_lines table was splitting line-item / totals
    # columns while skipping splits on non-table bands incorrectly when re-run stale.
    return False


def band_skips_vertical_column_split(
    layout: SectionLayoutResult,
    *,
    band_lines: list[Line] | None = None,
    page_width: float | None = None,
) -> bool:
    """Per horizontal band: never column-split table-like or prose bands."""
    if (
        _layout_is_table_family(layout)
        and layout.confidence >= TABLE_LAYOUT_MIN_CONFIDENCE
    ):
        if (
            band_lines is not None
            and page_width is not None
            and analyze_vertical_partition(band_lines, page_width).allows_column_split()
        ):
            return False
        return True
    # Full-width paragraphs: vertical peel creates invalid strips (last word per line).
    if layout.layout_kind == "prose":
        return True
    return False
