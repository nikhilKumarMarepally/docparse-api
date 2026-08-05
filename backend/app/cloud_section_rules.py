"""Render-only section split rules (geometry only — no text heuristics)."""

from __future__ import annotations

from typing import Any

from app.paths import ensure_script_path
from app.section_merge import (
    _words_structurally_table,
    section_dict_structurally_table,
)

ensure_script_path()

from ocr_line_to_sections import _make_section, Section  # noqa: E402
from ocr_word_to_line_boxes import Line  # noqa: E402
from section_opencv_boxes import _reindex_sections  # noqa: E402


def is_stacked_table_section(
    section: Section | dict[str, Any],
    lines: list[Line],
    page_width: float,
) -> bool:
    """True when x/y geometry shows a stacked table grid (not parallel prose lanes)."""
    if page_width <= 0:
        return False
    if isinstance(section, dict):
        return section_dict_structurally_table(section, lines, page_width)
    sec_lines = section.lines
    if not sec_lines:
        return False
    words = [w for ln in sec_lines for w in ln.words]
    if words and _words_structurally_table(words, page_width):
        return True
    indices: list[int] = []
    line_ids = {id(ln) for ln in sec_lines}
    for idx, ln in enumerate(lines):
        if id(ln) in line_ids:
            indices.append(idx)
    if indices:
        stub = {"line_indices": indices}
        return section_dict_structurally_table(stub, lines, page_width)
    return False


def _line_lookup(lines: list[Line] | dict[int, Line]) -> dict[int, Line]:
    if isinstance(lines, dict):
        return lines
    return {ln.index: ln for ln in lines}


def sections_from_dicts(
    raw_sections: list[dict[str, Any]],
    lines: list[Line] | dict[int, Line],
    *,
    pad: float = 6.0,
) -> list[Section]:
    by_index = _line_lookup(lines)
    out: list[Section] = []
    for sec in raw_sections:
        indices = sec.get("line_indices") or []
        sec_lines: list[Line] = []
        seen: set[int] = set()
        for line_idx in indices:
            idx = int(line_idx)
            if idx in by_index and idx not in seen:
                sec_lines.append(by_index[idx])
                seen.add(idx)
        if not sec_lines:
            continue
        out.append(_make_section(len(out), sec_lines, None, pad))
    return out


def split_sections_at_large_vertical_gaps(
    sections: list[Section],
    lines: list[Line],
    page_width: float,
    *,
    min_gap_px: float = 18.0,
    pad: float = 6.0,
) -> tuple[list[Section], dict[str, Any]]:
    """
    Split when inter-line gap is taller than the remaining block below and section is not a table.
    """
    out: list[Section] = []
    split_count = 0
    for sec in sections:
        if is_stacked_table_section(sec, lines, page_width):
            out.append(sec)
            continue
        ordered = sorted(
            sec.lines,
            key=lambda ln: (ln.content_box.min_y, ln.content_box.min_x),
        )
        if len(ordered) < 2:
            out.append(sec)
            continue

        groups: list[list[Line]] = []
        current: list[Line] = [ordered[0]]
        for i in range(1, len(ordered)):
            prev, curr = ordered[i - 1], ordered[i]
            gap = max(0.0, curr.content_box.min_y - prev.content_box.max_y)
            below = ordered[i:]
            if below:
                remaining_height = (
                    max(ln.content_box.max_y for ln in below)
                    - min(ln.content_box.min_y for ln in below)
                )
            else:
                remaining_height = 0.0
            if gap >= min_gap_px and below and gap > remaining_height:
                groups.append(current)
                current = [curr]
            else:
                current.append(curr)
        groups.append(current)

        if len(groups) > 1:
            split_count += 1
            for group in groups:
                out.append(_make_section(len(out), group, sec.gap_above, pad))
        else:
            out.append(sec)

    meta = {
        "vertical_gap_splits": split_count,
        "section_count": len(out),
    }
    return _reindex_sections(out), meta
