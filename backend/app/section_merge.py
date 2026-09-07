"""Merge adjacent OCR sections that belong to one table band (header + rows + subtotal)."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

from app.paths import ensure_script_path

ensure_script_path()

from ocr_word_to_line_boxes import (  # noqa: E402
    Line,
    Word,
    estimate_page_gutter_x,
    group_into_rows,
)
from section_table_layout import classify_section_layout, analyze_vertical_partition  # noqa: E402


def _line_spans_lr_split(line: Line, split_x: float, *, margin: float = 20.0) -> bool:
    return (
        line.content_box.min_x < split_x - margin
        and line.content_box.max_x > split_x + margin
    )


def _line_lane_lr_counts(
    lines: list[Line],
    page_width: float,
) -> tuple[int, int, int, float | None]:
    """Exclusive left/right OCR lines vs lines crossing the page gutter."""
    if page_width <= 0 or len(lines) < 2:
        return 0, 0, 0, None
    split_x = estimate_page_gutter_x(lines, page_width)
    margin = max(20.0, page_width * 0.012)
    left_only = 0
    right_only = 0
    spanning = 0
    for ln in lines:
        if ln.content_box.max_x < split_x - margin:
            left_only += 1
        elif ln.content_box.min_x > split_x + margin:
            right_only += 1
        elif _line_spans_lr_split(ln, split_x, margin=margin):
            spanning += 1
    return left_only, right_only, spanning, split_x


def _line_lookup(lines: list[Line] | dict[int, Line]) -> dict[int, Line]:
    if isinstance(lines, dict):
        return lines
    return {ln.index: ln for ln in lines}


def _section_words(
    section: dict[str, Any],
    lines: list[Line] | dict[int, Line],
) -> list[Word]:
    by_index = _line_lookup(lines)
    words: list[Word] = []
    for line_idx in section.get("line_indices") or []:
        idx = int(line_idx)
        if idx in by_index:
            words.extend(by_index[idx].words)

    bounds = section.get("bounds") or {}
    if not bounds:
        return words

    min_y = float(bounds.get("min_y", 0))
    max_y = float(bounds.get("max_y", 0))
    if max_y <= min_y:
        return words
    margin = 12.0
    in_bounds = [
        w for w in words if min_y - margin <= w.box.centroid_y <= max_y + margin
    ]
    if len(in_bounds) >= max(3, len(words) * 0.4):
        return in_bounds

    line_list = list(by_index.values()) if by_index else (
        list(lines.values()) if isinstance(lines, dict) else lines
    )
    from_bounds: list[Word] = []
    for ln in line_list:
        if ln.content_box.max_y < min_y - margin or ln.content_box.min_y > max_y + margin:
            continue
        for w in ln.words:
            if min_y - margin <= w.box.centroid_y <= max_y + margin:
                from_bounds.append(w)
    if len(from_bounds) >= 3:
        return from_bounds
    return in_bounds if in_bounds else words


def _section_opens_with_heading(words: list[Word]) -> bool:
    """Short single-line opener with title-like capitalization — geometry only."""
    rows = group_into_rows(words)
    if not rows:
        return False
    first = rows[0]
    cells = _split_row_into_cells(first)
    if len(cells) >= 3:
        return False
    text = " ".join((w.text or "") for w in first).strip()
    if not text or len(first) > 16 or len(text) > 110:
        return False
    title_like = sum(
        1
        for w in first
        if (w.text or "").isupper() or ((w.text or "")[:1].isupper())
    )
    return title_like >= max(2, len(first) * 0.4) and len(cells) <= 1


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


def _cluster_1d(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    clusters: list[list[float]] = [[ordered[0]]]
    for val in ordered[1:]:
        if val - clusters[-1][-1] <= tolerance:
            clusters[-1].append(val)
        else:
            clusters.append([val])
    return [sum(c) / len(c) for c in clusters]


@dataclass(frozen=True)
class ColumnGridStats:
    col_mids: list[float]
    row_hits: list[int]
    row_count: int
    x_tolerance: float


def _column_grid_stats(words: list[Word]) -> ColumnGridStats | None:
    """Column x anchors that repeat across visual rows (same x, different y)."""
    if len(words) < 2:
        return None

    rows = group_into_rows(words)
    if not rows:
        return None

    widths = [w.box.width for w in words]
    x_tol = max(8.0, median(widths) * 1.4) if widths else 12.0

    cell_min_x_per_row: list[list[float]] = []
    for row in rows:
        cells = _split_row_into_cells(row)
        cell_min_x_per_row.append([min(w.box.min_x for w in c) for c in cells])

    all_min_x = [x for row in cell_min_x_per_row for x in row]
    if not all_min_x:
        return None

    col_mids = _cluster_1d(all_min_x, x_tol)
    row_hits = [
        sum(
            1
            for mins in cell_min_x_per_row
            if any(abs(x - mid) <= x_tol for x in mins)
        )
        for mid in col_mids
    ]
    return ColumnGridStats(
        col_mids=col_mids,
        row_hits=row_hits,
        row_count=len(rows),
        x_tolerance=x_tol,
    )


def _dominant_column_mids(stats: ColumnGridStats, *, min_rows: int = 3) -> list[float]:
    return [mid for mid, hits in zip(stats.col_mids, stats.row_hits) if hits >= min_rows]


def _has_repeated_column_grid(stats: ColumnGridStats | None) -> bool:
    """≥2 column x anchors each appearing on ≥3 visual rows (same x, different y)."""
    if stats is None or stats.row_count < 2:
        return False
    return len(_dominant_column_mids(stats, min_rows=3)) >= 2


def _row_has_multi_column_header(words: list[Word]) -> bool:
    rows = group_into_rows(words)
    for row in rows:
        cells = _split_row_into_cells(row)
        if len(cells) >= 3:
            return True
    return False


def _words_structurally_table(words: list[Word], page_width: float) -> bool:
    """Geometry-only: repeated columns, wide numeric rows, or multi-cell grid."""
    if not words:
        return False
    stats = _column_grid_stats(words)
    if stats and _has_repeated_column_grid(stats):
        return True
    if _is_wide_table_row_band(words, page_width):
        return True
    if _row_has_multi_column_header(words):
        return True
    rows = group_into_rows(words)
    if len(rows) == 1:
        row = rows[0]
        if len(_split_row_into_cells(row)) >= 3:
            return True
        if _words_look_numeric_row(row):
            return True
    if len(rows) < 2:
        return False
    multi_cell_rows = sum(
        1 for row in rows if len(_split_row_into_cells(row)) >= 2
    )
    if multi_cell_rows >= 2:
        return True
    numeric_rows = sum(1 for row in rows if _words_look_numeric_row(row))
    if numeric_rows >= 2:
        return True
    return False


def section_dict_structurally_table(
    section: dict[str, Any],
    lines: list[Line] | None = None,
    page_width: float = 0.0,
) -> bool:
    """True when section metadata or word geometry indicates a table grid."""
    if section.get("table_band"):
        return True
    kind = _layout_kind_from_section(section)
    if kind in ("table", "section_table"):
        return True
    for key in ("layout", "layout_detect"):
        lay = section.get(key) or {}
        if not isinstance(lay, dict):
            continue
        if lay.get("layout_kind") in ("table", "section_table"):
            return True
        if int(lay.get("aligned_column_count") or 0) >= 2:
            return True
        if lay.get("multi_column_line_indices"):
            return True
    if lines is not None and page_width > 0:
        line_indices = section.get("line_indices") or []
        by_index = _line_lookup(lines)
        sec_lines: list[Line] = []
        for line_idx in line_indices:
            idx = int(line_idx)
            if idx in by_index:
                sec_lines.append(by_index[idx])
        if sec_lines:
            lay = classify_section_layout(sec_lines)
            if lay.layout_kind in ("table", "section_table"):
                return True
            if lay.aligned_column_count >= 2 or lay.multi_column_line_indices:
                return True
        section_words = _section_words(section, lines)
        if section_words and _words_structurally_table(section_words, page_width):
            return True
    return False


def _is_wrapped_prose_block(words: list[Word], page_width: float, stats: ColumnGridStats) -> bool:
    """Full-width paragraph lines sharing one left margin — not a multi-column grid."""
    if _words_structurally_table(words, page_width):
        return False
    if _has_repeated_column_grid(stats):
        return False
    rows = group_into_rows(words)
    if len(rows) < 2:
        return False

    min_xs: list[float] = []
    widths: list[float] = []
    for row in rows:
        cells = _split_row_into_cells(row)
        if len(cells) != 1:
            return False
        min_xs.append(min(w.box.min_x for w in cells[0]))
        widths.append(max(w.box.max_x for w in row) - min(w.box.min_x for w in row))

    if max(min_xs) - min(min_xs) > stats.x_tolerance:
        return False
    # Flowing paragraph: several lines, shared left margin, full width.
    if (
        len(rows) >= 3
        and page_width > 0
        and median(widths) >= page_width * 0.38
    ):
        return True
    if _is_wide_table_row_band(words, page_width):
        return False
    if len(stats.col_mids) == 1 and stats.row_hits[0] >= 3:
        if any(len(_split_row_into_cells(row)) >= 2 for row in rows):
            return False
        if sum(1 for row in rows if _words_look_numeric_row(row)) >= 2:
            return False
        return True
    if page_width <= 0:
        return False
    return median(widths) >= page_width * 0.38


def _median_row_gap(words: list[Word]) -> float:
    rows = group_into_rows(words)
    if len(rows) < 2:
        return 24.0
    ys = [sum(w.box.centroid_y for w in row) / len(row) for row in rows]
    gaps = [ys[i] - ys[i - 1] for i in range(1, len(ys))]
    return float(median(gaps)) if gaps else 24.0


def _word_hits_column_mids(words: list[Word], col_mids: list[float], tolerance: float) -> int:
    hits = 0
    for w in words:
        if any(abs(w.box.min_x - mid) <= tolerance for mid in col_mids):
            hits += 1
    return hits


def _is_wide_table_row_band(words: list[Word], page_width: float) -> bool:
    """Single-line numeric/grid rows: many tokens spanning most of the table width."""
    if page_width <= 0 or len(words) < 4:
        return False
    for row in group_into_rows(words):
        if len(row) < 4:
            continue
        span = max(w.box.max_x for w in row) - min(w.box.min_x for w in row)
        if span >= page_width * 0.32:
            return True
    return False


def _continues_table_columns(
    left_words: list[Word],
    right_words: list[Word],
    *,
    tolerance: float,
) -> bool:
    left_stats = _column_grid_stats(left_words)
    if left_stats is None:
        return False

    dominant = _dominant_column_mids(left_stats, min_rows=3)
    if len(dominant) < 2:
        dominant = [
            mid
            for mid, hits in zip(left_stats.col_mids, left_stats.row_hits)
            if hits >= 2
        ]
    if len(dominant) < 2:
        return False

    aligned_multi_cell_rows = 0
    for row in group_into_rows(right_words):
        cells = _split_row_into_cells(row)
        if len(cells) < 2:
            continue
        mins = [min(w.box.min_x for w in c) for c in cells]
        col_hits = sum(
            1 for mid in dominant if any(abs(x - mid) <= tolerance for x in mins)
        )
        if col_hits >= 2:
            aligned_multi_cell_rows += 1
    if aligned_multi_cell_rows >= 2:
        return True

    right_stats = _column_grid_stats(right_words)
    return _column_grids_share_anchors(left_stats, right_stats)


def _column_grids_share_anchors(
    left_stats: ColumnGridStats | None,
    right_stats: ColumnGridStats | None,
    *,
    min_shared: int = 2,
    min_hits: int = 2,
) -> bool:
    """True when stacked bands reuse the same column x-anchors (continuation, not two grids)."""
    if left_stats is None or right_stats is None:
        return False
    left_dom = [
        mid
        for mid, hits in zip(left_stats.col_mids, left_stats.row_hits)
        if hits >= min_hits
    ]
    right_dom = [
        mid
        for mid, hits in zip(right_stats.col_mids, right_stats.row_hits)
        if hits >= min_hits
    ]
    if len(left_dom) < min_shared or len(right_dom) < min_shared:
        return False
    tolerance = max(left_stats.x_tolerance, right_stats.x_tolerance)
    used: set[int] = set()
    shared = 0
    for left_mid in left_dom:
        best_j: int | None = None
        best_d: float | None = None
        for j, right_mid in enumerate(right_dom):
            if j in used:
                continue
            dist = abs(left_mid - right_mid)
            if dist <= tolerance and (best_d is None or dist < best_d):
                best_d = dist
                best_j = j
        if best_j is not None:
            used.add(best_j)
            shared += 1
    smaller = min(len(left_dom), len(right_dom))
    return shared >= min_shared and shared >= smaller


def _rows_share_column_starts(
    row_a: list[Word],
    row_b: list[Word],
    *,
    tolerance: float,
    min_matches: int = 2,
) -> bool:
    """≥2 cells in each row whose left edges (min_x) align — stacked table rows."""
    cells_a = _split_row_into_cells(row_a)
    cells_b = _split_row_into_cells(row_b)
    if len(cells_a) < 2 or len(cells_b) < 2:
        return False
    mins_a = [min(w.box.min_x for w in c) for c in cells_a]
    mins_b = [min(w.box.min_x for w in c) for c in cells_b]
    matches = sum(
        1
        for left_min in mins_a
        for right_min in mins_b
        if abs(left_min - right_min) <= tolerance
    )
    return matches >= min_matches


def _numeric_rows_share_word_column_starts(
    row_a: list[Word],
    row_b: list[Word],
    *,
    tolerance: float,
    min_matches: int = 3,
) -> bool:
    """Stacked numeric table rows: word left-edges repeat at the same x anchors."""
    if not _words_look_numeric_row(row_a) or not _words_look_numeric_row(row_b):
        return False
    mins_a = [w.box.min_x for w in row_a]
    mins_b = [w.box.min_x for w in row_b]
    matches = sum(
        1
        for left_min in mins_a
        for right_min in mins_b
        if abs(left_min - right_min) <= tolerance
    )
    return matches >= min_matches


def _prose_last_line_blocks_table_merge(
    left_words: list[Word],
    right_words: list[Word],
    page_width: float,
    *,
    tolerance: float,
) -> bool:
    """Full-width prose line above a multi-column row is not a table stack."""
    if _is_leading_title_band(left_words, page_width):
        return False
    left_rows = group_into_rows(left_words)
    right_rows = group_into_rows(right_words)
    if not left_rows or not right_rows:
        return False
    last = left_rows[-1]
    first = right_rows[0]
    last_cells = _split_row_into_cells(last)
    first_cells = _split_row_into_cells(first)
    if len(last_cells) != 1 or len(first_cells) < 2:
        return False
    span = max(w.box.max_x for w in last) - min(w.box.min_x for w in last)
    if page_width <= 0 or span < page_width * 0.35:
        return False
    return not _rows_share_column_starts(last, first, tolerance=tolerance)


def _stacked_vertical_table_continuity(
    left_words: list[Word],
    right_words: list[Word],
    page_width: float,
    *,
    tolerance: float,
) -> bool:
    """Vertically stacked sections merge only when cell start-x anchors line up across rows."""
    if _is_leading_title_band(left_words, page_width):
        if _band_looks_like_table_body(right_words, page_width):
            return True
        if _row_has_multi_column_header(right_words):
            return True

    left_rows = group_into_rows(left_words)
    right_rows = group_into_rows(right_words)
    if left_rows and right_rows:
        last_row = left_rows[-1]
        first_row = right_rows[0]
        if _numeric_rows_share_word_column_starts(
            last_row, first_row, tolerance=tolerance
        ):
            return True
        tail = left_rows[-2:] if len(left_rows) >= 2 else [left_rows[-1]]
        head = right_rows[:2]
        for left_row in tail:
            if len(_split_row_into_cells(left_row)) < 2:
                continue
            for right_row in head:
                if _rows_share_column_starts(
                    left_row, right_row, tolerance=tolerance
                ):
                    return True

    return _continues_table_columns(left_words, right_words, tolerance=tolerance)


def _column_centers(
    section: dict[str, Any],
    lines: list[Line] | dict[int, Line],
) -> list[float]:
    """Best-effort column x-centroids for the section's widest line."""
    by_index = _line_lookup(lines)
    best: list[float] = []
    for line_idx in section.get("line_indices") or []:
        idx = int(line_idx)
        if idx not in by_index:
            continue
        cells = _split_row_into_cells(by_index[idx].words)
        if len(cells) > len(best):
            best = [_cell_center(c) for c in cells]
    return best


def _cell_center(cell: list[Word]) -> float:
    box = cell[0].box
    for w in cell[1:]:
        box = box.union(w.box)
    return box.centroid_x


def _vertical_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    ab = a.get("bounds") or {}
    bb = b.get("bounds") or {}
    return max(0.0, float(bb.get("min_y", 0)) - float(ab.get("max_y", 0)))


def _horizontal_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    ab = a.get("bounds") or {}
    bb = b.get("bounds") or {}
    left = max(float(ab.get("min_x", 0)), float(bb.get("min_x", 0)))
    right = min(float(ab.get("max_x", 0)), float(bb.get("max_x", 0)))
    overlap = max(0.0, right - left)
    narrower = min(
        float(ab.get("max_x", 0)) - float(ab.get("min_x", 0)),
        float(bb.get("max_x", 0)) - float(bb.get("min_x", 0)),
    )
    if narrower <= 0:
        return 0.0
    return overlap / narrower


def _columns_align(centers_a: list[float], centers_b: list[float], *, tolerance: float) -> bool:
    if len(centers_a) < 2 or len(centers_b) < 2:
        return False
    matches = 0
    for ca in centers_a:
        if any(abs(ca - cb) <= tolerance for cb in centers_b):
            matches += 1
    return matches >= 2


def _merge_bounds(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    ab = a.get("bounds") or {}
    bb = b.get("bounds") or {}
    return {
        "min_x": min(float(ab.get("min_x", 0)), float(bb.get("min_x", 0))),
        "min_y": min(float(ab.get("min_y", 0)), float(bb.get("min_y", 0))),
        "max_x": max(float(ab.get("max_x", 0)), float(bb.get("max_x", 0))),
        "max_y": max(float(ab.get("max_y", 0)), float(bb.get("max_y", 0))),
    }


def _merge_two(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    line_indices = list(a.get("line_indices") or []) + list(b.get("line_indices") or [])
    texts = [t for t in (a.get("text") or "", b.get("text") or "") if t.strip()]
    gap = _vertical_gap(a, b)
    kind = _layout_kind_from_section(a) or _layout_kind_from_section(b)
    merged: dict[str, Any] = {
        "index": a.get("index", 0),
        "line_indices": line_indices,
        "line_count": len(line_indices),
        "gap_above": a.get("gap_above"),
        "text": "\n".join(texts),
        "bounds": _merge_bounds(a, b),
        "merged_gap_below": round(gap, 2),
        "table_band": True,
    }
    if kind:
        merged["layout_kind"] = kind
    return merged


def _layout_kind_from_section(section: dict[str, Any]) -> str:
    kind = section.get("layout_kind")
    if isinstance(kind, str) and kind:
        return kind
    layout = section.get("layout") or section.get("layout_detect") or {}
    if isinstance(layout, dict):
        nested = layout.get("layout_kind")
        if isinstance(nested, str) and nested:
            return nested
    return ""


def _is_leading_title_band(words: list[Word], page_width: float = 0.0) -> bool:
    """Short band above a grid (caption/title line), not a multi-row table itself."""
    if not words:
        return False
    rows = group_into_rows(words)
    if len(rows) > 3:
        return False
    stats = _column_grid_stats(words)
    if stats and _has_repeated_column_grid(stats):
        return False
    if page_width > 0 and stats and _is_wrapped_prose_block(words, page_width, stats):
        return False
    multi_cell_rows = sum(
        1 for row in rows if len(_split_row_into_cells(row)) >= 3
    )
    return multi_cell_rows == 0


def _is_follow_on_prose_after_table(
    left_words: list[Word],
    right_words: list[Word],
    page_width: float,
) -> bool:
    """Wrapped full-width paragraph after a table band — geometry only."""
    if _words_structurally_table(right_words, page_width):
        return False
    left_stats = _column_grid_stats(left_words)
    if not _has_repeated_column_grid(left_stats):
        return False
    right_stats = _column_grid_stats(right_words)
    if right_stats is None:
        return False
    return _is_wrapped_prose_block(right_words, page_width, right_stats)


def _words_look_numeric_row(words: list[Word]) -> bool:
    if len(words) < 6:
        return False
    numeric = sum(1 for w in words if any(ch.isdigit() for ch in w.text))
    return numeric >= len(words) * 0.35


def _words_in_y_band(words: list[Word], min_y: float, max_y: float) -> list[Word]:
    return [w for w in words if min_y <= w.box.centroid_y <= max_y]


def _same_row_lr_table_shards(
    left: dict[str, Any],
    right: dict[str, Any],
    left_words: list[Word],
    right_words: list[Word],
    page_width: float,
) -> bool:
    """Left/right OCR shards of one table row (same y, disjoint x)."""
    ab = left.get("bounds") or {}
    bb = right.get("bounds") or {}
    y_overlap = min(float(ab.get("max_y", 0)), float(bb.get("max_y", 0))) - max(
        float(ab.get("min_y", 0)), float(bb.get("min_y", 0))
    )
    row_h = min(
        float(ab.get("max_y", 0)) - float(ab.get("min_y", 0)),
        float(bb.get("max_y", 0)) - float(bb.get("min_y", 0)),
    )
    if row_h <= 0 or y_overlap / row_h < 0.5:
        return False
    if _vertical_gap(left, right) > 8.0:
        return False
    overlap_min_y = max(float(ab.get("min_y", 0)), float(bb.get("min_y", 0)))
    overlap_max_y = min(float(ab.get("max_y", 0)), float(bb.get("max_y", 0)))
    left_band = _words_in_y_band(left_words, overlap_min_y, overlap_max_y)
    right_band = _words_in_y_band(right_words, overlap_min_y, overlap_max_y)
    if not left_band or not right_band:
        return False
    left_max_x = max(w.box.max_x for w in left_band)
    right_min_x = min(w.box.min_x for w in right_band)
    if right_min_x - left_max_x < 20.0:
        return False
    row_words = left_band + right_band
    return (
        _is_wide_table_row_band(row_words, page_width)
        and _words_look_numeric_row(row_words)
    )


def _section_lines(
    section: dict[str, Any],
    lines: list[Line] | dict[int, Line],
) -> list[Line]:
    by_index = _line_lookup(lines)
    sec_lines: list[Line] = []
    for line_idx in section.get("line_indices") or []:
        idx = int(line_idx)
        if idx in by_index:
            sec_lines.append(by_index[idx])
    return sec_lines


def _section_is_parallel_lr_block(
    section: dict[str, Any],
    lines: list[Line],
    page_width: float,
) -> bool:
    """Side-by-side lanes (address | hours), not a unified table grid."""
    sec_lines = _section_lines(section, lines)
    if len(sec_lines) < 2 or page_width <= 0:
        return False
    part = analyze_vertical_partition(sec_lines, page_width)
    if part.is_unified_grid:
        return False
    left_only, right_only, spanning, _ = _line_lane_lr_counts(sec_lines, page_width)
    min_side = max(2, len(sec_lines) // 6)
    return left_only >= min_side and right_only >= min_side and spanning <= 1


def _band_looks_like_table_body(words: list[Word], page_width: float) -> bool:
    """Numeric or multi-cell rows — table data under a caption/header band."""
    if not words:
        return False
    stats = _column_grid_stats(words)
    if stats and _has_repeated_column_grid(stats):
        return True
    rows = group_into_rows(words)
    if sum(1 for row in rows if _words_look_numeric_row(row)) >= 2:
        return True
    multi_cell_rows = sum(
        1 for row in rows if len(_split_row_into_cells(row)) >= 3
    )
    if multi_cell_rows >= 2:
        return True
    return False


def _prose_section_continues_table_rows(
    left_words: list[Word],
    right_words: list[Word],
    page_width: float,
    *,
    x_tol: float,
) -> bool:
    """Allow prose-labeled HV shards to merge when column x anchors align across rows."""
    if not _continues_table_columns(left_words, right_words, tolerance=x_tol):
        return False
    right_rows = group_into_rows(right_words)
    if any(_words_look_numeric_row(row) for row in right_rows):
        return True
    if _row_has_multi_column_header(right_words):
        return True
    multi_cell_rows = sum(
        1 for row in right_rows if len(_split_row_into_cells(row)) >= 4
    )
    if multi_cell_rows >= 2:
        return True
    if any(len(_split_row_into_cells(row)) >= 6 for row in right_rows):
        return True
    return False


def _prose_caption_introduces_table(
    left_words: list[Word],
    right_words: list[Word],
    page_width: float,
    *,
    x_tol: float,
) -> bool:
    if not _is_leading_title_band(left_words, page_width):
        return False
    return _stacked_vertical_table_continuity(
        left_words, right_words, page_width, tolerance=x_tol
    )


def should_merge_table_fragments(
    left: dict[str, Any],
    right: dict[str, Any],
    lines: list[Line],
    *,
    gap_threshold: float,
    page_width: float,
) -> bool:
    """True when adjacent sections share a table column grid (same x anchors, different y)."""
    left_words = _section_words(left, lines)
    right_words = _section_words(right, lines)
    if _section_opens_with_heading(right_words):
        return False
    left_stats = _column_grid_stats(left_words)
    right_stats = _column_grid_stats(right_words)
    x_tol_early = right_stats.x_tolerance if right_stats else max(12.0, gap_threshold)
    if _prose_last_line_blocks_table_merge(
        left_words, right_words, page_width, tolerance=x_tol_early
    ):
        return False

    left_kind = _layout_kind_from_section(left)
    right_kind = _layout_kind_from_section(right)
    left_struct = _words_structurally_table(left_words, page_width)
    right_struct = _words_structurally_table(right_words, page_width)
    share_grid = _column_grids_share_anchors(left_stats, right_stats)
    if left_kind == "prose" and right_kind == "prose" and not (
        left_struct or right_struct or share_grid
    ):
        if not (
            _prose_section_continues_table_rows(
                left_words, right_words, page_width, x_tol=x_tol_early
            )
            or _prose_caption_introduces_table(
                left_words, right_words, page_width, x_tol=x_tol_early
            )
            or _stacked_vertical_table_continuity(
                left_words, right_words, page_width, tolerance=x_tol_early
            )
        ):
            return False

    gap = _vertical_gap(left, right)
    left_rows = int((left.get("layout_detect") or {}).get("row_count") or 0)
    right_rows = int((right.get("layout_detect") or {}).get("row_count") or 0)
    if (
        min(left_rows, right_rows) >= 3
        and gap >= gap_threshold
        and left_kind in ("table", "section_table")
        and right_kind in ("table", "section_table")
    ):
        return False
    left_is_table = (
        bool(left.get("table_band"))
        or _has_repeated_column_grid(left_stats)
        or left_kind in ("table", "section_table")
        or left_struct
    )

    if _section_is_parallel_lr_block(right, lines, page_width):
        return False
    if (
        _layout_kind_from_section(left) == "prose"
        and _layout_kind_from_section(right) in ("table", "section_table")
        and len(group_into_rows(left_words)) >= 2
    ):
        return False

    if _is_follow_on_prose_after_table(left_words, right_words, page_width):
        return False
    if _same_row_lr_table_shards(left, right, left_words, right_words, page_width):
        return True
    if (
        not left_is_table
        and right_stats
        and _is_wrapped_prose_block(right_words, page_width, right_stats)
    ):
        return False

    overlap = _horizontal_overlap_ratio(left, right)
    max_gap = max(gap_threshold * 2.5, 42.0)
    if gap > max_gap or overlap < 0.42:
        return False

    left_cols = int((left.get("layout_detect") or {}).get("aligned_column_count") or 0)
    right_cols = int((right.get("layout_detect") or {}).get("aligned_column_count") or 0)
    if left_cols >= 2 and right_cols < 2:
        return False
    if left_cols >= 2 and right_cols >= 2 and abs(left_cols - right_cols) >= 2:
        return False

    x_tol = right_stats.x_tolerance if right_stats else max(12.0, gap_threshold)

    if gap > _median_row_gap(left_words) * 1.35:
        if not share_grid and not (
            _has_repeated_column_grid(left_stats)
            and _has_repeated_column_grid(right_stats)
        ):
            return False

    left_rows = int((left.get("layout_detect") or {}).get("row_count") or 0)
    if (
        not _has_repeated_column_grid(left_stats)
        and _has_repeated_column_grid(right_stats)
        and left_rows <= 5
    ):
        return False

    if left_is_table:
        row_gap = _median_row_gap(left_words)
        grid_continues = share_grid or _continues_table_columns(
            left_words, right_words, tolerance=x_tol
        )
        if row_gap > 0 and gap > row_gap * 1.65 and not grid_continues:
            return False
        if not grid_continues and not _stacked_vertical_table_continuity(
            left_words, right_words, page_width, tolerance=x_tol
        ):
            return False
        combined_words = left_words + right_words
        combined_stats = _column_grid_stats(combined_words)
        if combined_stats and _is_wrapped_prose_block(
            combined_words, page_width, combined_stats
        ):
            return False
        return True

    if _is_leading_title_band(left_words, page_width):
        return _stacked_vertical_table_continuity(
            left_words, right_words, page_width, tolerance=x_tol
        )

    if left.get("table_band") or _has_repeated_column_grid(_column_grid_stats(left_words)):
        if not _stacked_vertical_table_continuity(
            left_words, right_words, page_width, tolerance=x_tol
        ):
            return False
        combined_stats = _column_grid_stats(left_words + right_words)
        if combined_stats is None:
            return False
        return not _is_wrapped_prose_block(
            left_words + right_words, page_width, combined_stats
        )

    if _stacked_vertical_table_continuity(
        left_words, right_words, page_width, tolerance=x_tol
    ):
        combined_words = left_words + right_words
        combined_stats = _column_grid_stats(combined_words)
        if combined_stats and _is_wrapped_prose_block(
            combined_words, page_width, combined_stats
        ):
            return False
        return True

    combined_words = left_words + right_words
    combined_stats = _column_grid_stats(combined_words)
    if combined_stats is None:
        return False
    if not _has_repeated_column_grid(combined_stats):
        return False
    if _is_wrapped_prose_block(combined_words, page_width, combined_stats):
        return False
    return overlap >= 0.48


def merge_table_fragments(
    sections: list[dict[str, Any]],
    lines: list[Line],
    *,
    gap_threshold: float,
    page_width: float,
) -> list[dict[str, Any]]:
    if len(sections) < 2:
        return sections

    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(sections):
        current = dict(sections[i])
        j = i + 1
        while j < len(sections) and should_merge_table_fragments(
            current,
            sections[j],
            lines,
            gap_threshold=gap_threshold,
            page_width=page_width,
        ):
            current = _merge_two(current, sections[j])
            j += 1
        current["index"] = len(merged)
        merged.append(current)
        i = j
    return merged
