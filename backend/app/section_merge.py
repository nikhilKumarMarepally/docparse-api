"""Merge adjacent OCR sections that belong to one table band (header + rows + subtotal)."""

from __future__ import annotations

from statistics import median
from typing import Any

from app.paths import ensure_script_path

ensure_script_path()

from ocr_word_to_line_boxes import Line, Word  # noqa: E402


def _section_words(section: dict[str, Any], lines: list[Line]) -> list[Word]:
    words: list[Word] = []
    for line_idx in section.get("line_indices") or []:
        idx = int(line_idx)
        if 0 <= idx < len(lines):
            words.extend(lines[idx].words)
    return words


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


def _column_centers(section: dict[str, Any], lines: list[Line]) -> list[float]:
    """Best-effort column x-centroids for the section's widest line."""
    best: list[float] = []
    for line_idx in section.get("line_indices") or []:
        idx = int(line_idx)
        if not (0 <= idx < len(lines)):
            continue
        cells = _split_row_into_cells(lines[idx].words)
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
    return {
        "index": a.get("index", 0),
        "line_indices": line_indices,
        "line_count": len(line_indices),
        "gap_above": a.get("gap_above"),
        "text": "\n".join(texts),
        "bounds": _merge_bounds(a, b),
        "merged_gap_below": round(gap, 2),
        "table_band": True,
    }


def should_merge_table_fragments(
    left: dict[str, Any],
    right: dict[str, Any],
    lines: list[Line],
    *,
    gap_threshold: float,
    page_width: float,
) -> bool:
    """True when two adjacent sections are one table band split by line-gap sectioning."""
    left_n = len(left.get("line_indices") or [])
    right_n = len(right.get("line_indices") or [])
    # Only glue thin shards (header row, data row, subtotal row) — not multi-line blocks.
    if right_n > 2 or left_n > 4:
        return False

    gap = _vertical_gap(left, right)
    if gap > gap_threshold * 1.45:
        return False

    if _horizontal_overlap_ratio(left, right) < 0.72:
        return False

    left_cols = _column_centers(left, lines)
    right_cols = _column_centers(right, lines)
    widths = [w.box.width for w in _section_words(left, lines) + _section_words(right, lines)]
    x_tol = max(12.0, median(widths) * 1.5) if widths else 20.0

    if _columns_align(left_cols, right_cols, tolerance=x_tol):
        return True

    # Wide bands with multi-column lines and table-row spacing (not a new section).
    left_w = float((left.get("bounds") or {}).get("max_x", 0)) - float(
        (left.get("bounds") or {}).get("min_x", 0)
    )
    right_w = float((right.get("bounds") or {}).get("max_x", 0)) - float(
        (right.get("bounds") or {}).get("min_x", 0)
    )
    if page_width <= 0:
        return False
    wide_band = left_w / page_width >= 0.82 and right_w / page_width >= 0.82
    multi_col = max(len(left_cols), len(right_cols)) >= 3
    return wide_band and multi_col and gap <= gap_threshold * 1.45


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
