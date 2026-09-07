#!/usr/bin/env python3
"""Cluster line-level OCR into vertical sections using inter-line gap analysis.

Uses average/median line spacing to detect larger gaps (section breaks), then
consolidates all lines in each section into one bounding box (leftmost→rightmost
of contained OCR) and a merged text block.

Examples:
  python ocr_line_to_sections.py \\
    --vision page.json --image page.png --out /tmp/ocr_sections

  # Reuse prior line JSON instead of re-parsing vision:
  python ocr_line_to_sections.py \\
    --lines-json wa577_gallery/vin_ticket/87c842be_p0_lines/87c842be_p0_lines.json \\
    --image wa577_gallery/vin_ticket/87c842be_p0.png \\
    --out wa577_gallery/vin_ticket/87c842be_p0_sections
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from section_table_layout import (  # noqa: E402
    analyze_vertical_partition,
    band_skips_vertical_column_split,
    classify_section_layout,
    page_layout_from_horizontal_bands,
    table_layout_skips_vertical,
)
from ocr_word_to_line_boxes import (  # noqa: E402
    Box,
    Line,
    Word,
    aligned_column_valid_for_vertical,
    detect_aligned_text_column,
    estimate_page_gutter_x,
    is_vertical_angle,
    line_from_words,
    line_is_vertical,
    load_font,
    load_vision,
    load_words,
    median_positive_word_gap_from_lines,
    column_gap_between_lines,
    partition_lines_at_column,
    resolve_paths,
    split_lines_by_orientation_spacing,
    word_is_vertical,
    words_to_lines,
)

_COLUMN_WATERMARK_STAMP = re.compile(
    r"^(?:ACCURATE|TRUE|AND|COPY|NON|UCC|COMPLETED|AUTHORITATIVE|-{1,2})$",
    re.I,
)
_STAMP_WORD = re.compile(
    r"^(?:ACCURATE|TRUE|AND|COPY|NON|UCC|COMPLETED|AUTHORITATIVE)$",
    re.I,
)

_LEFT_HEADER_BREAK = re.compile(
    r"(?i)^(?:"
    r"option\s*:|"
    r"optional\s+gap\s+contract|"
    r"returned check charge"
    r")"
)


@dataclass
class GapStats:
    gaps: list[float]
    mean: float
    median: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gaps": [round(g, 2) for g in self.gaps],
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "threshold": round(self.threshold, 2),
        }


@dataclass
class Section:
    index: int
    lines: list[Line]
    text: str
    box: Box
    gap_above: float | None

    def to_dict(self, *, layout: dict[str, Any] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": self.index,
            "line_indices": [line.index for line in self.lines],
            "line_count": len(self.lines),
            "gap_above": None if self.gap_above is None else round(self.gap_above, 2),
            "text": self.text,
            "bounds": self.box.to_dict(),
        }
        if layout:
            out["layout_kind"] = layout.get("layout_kind", "unknown")
            if layout.get("table_column_x"):
                out["table_column_x"] = layout["table_column_x"]
            out["layout_detect"] = layout
        return out


def line_gaps(lines: list[Line]) -> list[float]:
    gaps: list[float] = []
    for prev, curr in zip(lines, lines[1:]):
        gap = curr.content_box.min_y - prev.content_box.max_y
        gaps.append(max(0.0, gap))
    return gaps


def gap_stats(
    lines: list[Line],
    *,
    multiplier: float = 2.0,
    min_gap_px: float = 18.0,
) -> GapStats:
    gaps = line_gaps(lines)
    if not gaps:
        return GapStats(gaps=[], mean=0.0, median=0.0, threshold=min_gap_px)

    positive = [g for g in gaps if g > 0]
    if not positive:
        positive = gaps

    mean = statistics.fmean(positive)
    median = statistics.median(positive)
    threshold = max(min_gap_px, median * multiplier, mean * 1.25)
    return GapStats(gaps=gaps, mean=mean, median=median, threshold=threshold)


def lines_to_sections(
    lines: list[Line],
    *,
    multiplier: float = 2.0,
    min_gap_px: float = 18.0,
    pad: float = 6.0,
) -> tuple[list[Section], GapStats]:
    if not lines:
        return [], GapStats([], 0.0, 0.0, min_gap_px)

    ordered = sorted(lines, key=lambda ln: (ln.content_box.min_y, ln.content_box.min_x))
    stats = gap_stats(ordered, multiplier=multiplier, min_gap_px=min_gap_px)
    sections: list[Section] = []
    current: list[Line] = [ordered[0]]
    gap_above: float | None = None

    for i, line in enumerate(ordered[1:], start=1):
        gap = stats.gaps[i - 1]
        if gap > stats.threshold:
            sections.append(_make_section(len(sections), current, gap_above, pad))
            current = [line]
            gap_above = gap
        else:
            current.append(line)

    sections.append(_make_section(len(sections), current, gap_above, pad))
    sections = merge_adjacent_table_row_bands(sections, pad=pad)
    return sections, stats


TABLE_ROW_MERGE_MAX_LINES_PER_BAND = 2
TABLE_ROW_MERGE_MAX_GAP_PX = 130.0
TABLE_ROW_MERGE_MIN_X_OVERLAP = 0.52
TABLE_ROW_MERGE_MIN_COMBINED_CONF = 0.72
TABLE_ROW_MERGE_MIN_ROW_WIDTH_PX = 180.0


def _box_x_overlap_frac(a: Box, b: Box) -> float:
    overlap = max(0.0, min(a.max_x, b.max_x) - max(a.min_x, b.min_x))
    narrower = min(a.width, b.width)
    return overlap / narrower if narrower > 0 else 0.0


def _section_looks_like_table_row_band(section: Section) -> bool:
    """Single gap bands that are one table header/data/summary row (not a title line)."""
    lay = classify_section_layout(section.lines)
    if lay.layout_kind in ("table", "section_table"):
        return True
    if lay.aligned_column_count >= 2 or lay.multi_column_line_indices:
        return True
    for line in section.lines:
        n = len(line.words)
        if n >= 3 and line.content_box.width >= TABLE_ROW_MERGE_MIN_ROW_WIDTH_PX:
            return True
        if n >= 2 and line.content_box.width >= TABLE_ROW_MERGE_MIN_ROW_WIDTH_PX * 2:
            return True
    return False


def _split_section_by_horizontal_lanes(section: Section, *, pad: float) -> list[Section]:
    """Split a prose section when lines occupy disjoint horizontal lanes (e.g. side-by-side charts)."""
    lines = sorted(section.lines, key=lambda ln: ln.content_box.min_y)
    if len(lines) < 2:
        return [section]

    lanes: list[list[Line]] = []
    for line in lines:
        best_idx = -1
        best_overlap = 0.0
        for idx, lane in enumerate(lanes):
            overlap = max(
                _box_x_overlap_frac(line.content_box, ln.content_box) for ln in lane
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        if best_idx >= 0 and best_overlap >= 0.12:
            lanes[best_idx].append(line)
        else:
            lanes.append([line])

    if len(lanes) <= 1:
        return [section]

    lane_boxes: list[Box] = []
    for lane in lanes:
        box = lane[0].content_box
        for ln in lane[1:]:
            box = box.union(ln.content_box)
        lane_boxes.append(box)
    lane_boxes.sort(key=lambda b: b.min_x)

    separations = [
        lane_boxes[i + 1].min_x - lane_boxes[i].max_x
        for i in range(len(lane_boxes) - 1)
    ]
    min_sep = max(24.0, section.box.width * 0.035)
    if not separations or max(separations) < min_sep:
        return [section]

    return [_make_section(0, lane, section.gap_above, pad) for lane in lanes]


def _split_section_by_word_x_clusters(section: Section, *, page_width: float, pad: float) -> list[Section]:
    """Split a wide band when word x-centroids form two separated clusters (geometry only)."""
    margin_x = page_width * 0.08
    words = [
        word
        for line in section.lines
        for word in line.words
        if word.box.centroid_x >= margin_x
    ]
    if len(words) < 12:
        return [section]

    xs = [word.box.centroid_x for word in words]
    centroids = [xs[len(xs) // 4], xs[(3 * len(xs)) // 4]]
    for _ in range(24):
        left_xs = [x for x in xs if abs(x - centroids[0]) <= abs(x - centroids[1])]
        right_xs = [x for x in xs if abs(x - centroids[0]) > abs(x - centroids[1])]
        if not left_xs or not right_xs:
            return [section]
        centroids = [sum(left_xs) / len(left_xs), sum(right_xs) / len(right_xs)]

    separation = abs(centroids[1] - centroids[0])
    min_sep = max(72.0, page_width * 0.08)
    if separation < min_sep:
        return [section]

    split_x = (centroids[0] + centroids[1]) * 0.5
    left_count = sum(1 for x in xs if x < split_x)
    right_count = len(xs) - left_count
    if min(left_count, right_count) / len(xs) < 0.18:
        return [section]

    left_lines: list[Line] = []
    right_lines: list[Line] = []
    line_index = 0
    for line in section.lines:
        left_words = [word for word in line.words if word.box.centroid_x < split_x]
        right_words = [word for word in line.words if word.box.centroid_x >= split_x]
        if left_words:
            chunk = line_from_words(left_words, index=line_index)
            if chunk is not None:
                left_lines.append(chunk)
                line_index += 1
        if right_words:
            chunk = line_from_words(right_words, index=line_index)
            if chunk is not None:
                right_lines.append(chunk)
                line_index += 1

    if len(left_lines) < 2 or len(right_lines) < 2:
        return [section]

    return [
        _make_section(0, left_lines, section.gap_above, pad),
        _make_section(1, right_lines, None, pad),
    ]


def split_prose_sections_by_lanes(sections: list[Section], *, pad: float) -> list[Section]:
    """Break non-table sections apart when content sits in separated horizontal lanes."""
    out: list[Section] = []
    for sec in sections:
        if _section_looks_like_table_row_band(sec):
            out.append(sec)
            continue
        lay = classify_section_layout(sec.lines)
        if lay.layout_kind in ("table", "section_table"):
            out.append(sec)
            continue
        out.extend(_split_section_by_horizontal_lanes(sec, pad=pad))
    return [_make_section(i, s.lines, s.gap_above, pad) for i, s in enumerate(out)]


def merge_adjacent_table_row_bands(sections: list[Section], *, pad: float) -> list[Section]:
    """Glue consecutive thin gap bands when combined geometry is a table grid."""
    if len(sections) < 2:
        return sections
    out: list[Section] = []
    i = 0
    while i < len(sections):
        if not _section_looks_like_table_row_band(sections[i]):
            out.append(_make_section(len(out), sections[i].lines, sections[i].gap_above, pad))
            i += 1
            continue

        chunk: list[Section] = [sections[i]]
        j = i + 1
        while j < len(sections):
            nxt = sections[j]
            if len(nxt.lines) > TABLE_ROW_MERGE_MAX_LINES_PER_BAND:
                break
            if not _section_looks_like_table_row_band(nxt):
                break
            gap = nxt.box.min_y - chunk[-1].box.max_y
            if gap > TABLE_ROW_MERGE_MAX_GAP_PX:
                break
            if _box_x_overlap_frac(chunk[-1].box, nxt.box) < TABLE_ROW_MERGE_MIN_X_OVERLAP:
                break
            chunk.append(nxt)
            j += 1

        if len(chunk) >= 2:
            merged_lines = [ln for s in chunk for ln in s.lines]
            lay = classify_section_layout(merged_lines)
            if (
                lay.layout_kind in ("table", "section_table")
                and lay.confidence >= TABLE_ROW_MERGE_MIN_COMBINED_CONF
            ):
                out.append(_make_section(len(out), merged_lines, chunk[0].gap_above, pad))
                i = j
                continue

        out.append(_make_section(len(out), sections[i].lines, sections[i].gap_above, pad))
        i += 1
    return out


def section_layout_payload(sections: list[Section]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-section layout_kind + page-level section_meta from word-column geometry."""
    section_dicts: list[dict[str, Any]] = []
    section_meta: dict[str, Any] = {}
    for section in sections:
        layout = classify_section_layout(section.lines).to_dict()
        section_meta[f"S{section.index}"] = layout
        section_dicts.append(section.to_dict(layout=layout))
    return section_dicts, section_meta


def _bounds_covering_lines(lines: list[Line], *, pad: float) -> Box:
    """Tight box around every OCR word in ``lines`` (+ pad). Never shrink below text."""
    if not lines:
        return Box(0.0, 0.0, 1.0, 1.0)
    box = lines[0].content_box
    for line in lines[1:]:
        if line.words:
            box = box.union(line.content_box)
    # Prefer full word boxes when present (content_box already unions words, but
    # keep an explicit pass so clipped/stale section.box cannot cut glyphs).
    for line in lines:
        for word in line.words:
            box = box.union(word.box)
    return Box(
        box.min_x - pad,
        box.min_y - pad,
        box.max_x + pad,
        box.max_y + pad,
    )


def _word_ids(lines: list[Line]) -> set[int]:
    return {id(w) for ln in lines for w in ln.words}


def _nudge_box_off_foreign_words(
    box: Box,
    *,
    own_ids: set[int],
    all_words: list[Word],
    gap: float = 1.0,
) -> Box:
    """Keep own glyphs fully inside; pull edges out of foreign glyphs.

    Prefer shedding pad over cutting a neighbor. If own and foreign OCR boxes
    overlap, own coverage wins (cannot satisfy both).
    """
    own_boxes = [w.box for w in all_words if id(w) in own_ids]
    if not own_boxes:
        return box
    ox0 = min(b.min_x for b in own_boxes)
    oy0 = min(b.min_y for b in own_boxes)
    ox1 = max(b.max_x for b in own_boxes)
    oy1 = max(b.max_y for b in own_boxes)

    # Start from requested box but never inside own text.
    min_x = min(box.min_x, ox0)
    min_y = min(box.min_y, oy0)
    max_x = max(box.max_x, ox1)
    max_y = max(box.max_y, oy1)

    foreign = [w.box for w in all_words if id(w) not in own_ids]
    for wb in foreign:
        if (
            wb.min_x + 0.5 < max_x < wb.max_x - 0.5
            and min(max_y, wb.max_y) - max(min_y, wb.min_y) > 1.0
        ):
            # Only shed pad — do not cut own text.
            max_x = max(ox1, min(max_x, wb.min_x - gap))
        if (
            wb.min_x + 0.5 < min_x < wb.max_x - 0.5
            and min(max_y, wb.max_y) - max(min_y, wb.min_y) > 1.0
        ):
            min_x = min(ox0, max(min_x, wb.max_x + gap))
        if (
            wb.min_y + 0.5 < max_y < wb.max_y - 0.5
            and min(max_x, wb.max_x) - max(min_x, wb.min_x) > 1.0
        ):
            max_y = max(oy1, min(max_y, wb.min_y - gap))
        if (
            wb.min_y + 0.5 < min_y < wb.max_y - 0.5
            and min(max_x, wb.max_x) - max(min_x, wb.min_x) > 1.0
        ):
            min_y = min(oy0, max(min_y, wb.max_y + gap))

    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        max_y = min_y + 1.0
    return Box(min_x, min_y, max_x, max_y)


def _bounds_covering_lines_safe(
    lines: list[Line],
    *,
    pad: float,
    all_words: list[Word] | None = None,
) -> Box:
    box = _bounds_covering_lines(lines, pad=pad)
    if not all_words:
        return box
    return _nudge_box_off_foreign_words(
        box, own_ids=_word_ids(lines), all_words=all_words
    )


def _recompute_section_box(
    section: Section,
    *,
    pad: float,
    all_words: list[Word] | None = None,
) -> Section:
    """Force section.box to fully cover assigned text — bounds must never cut glyphs."""
    return Section(
        index=section.index,
        lines=section.lines,
        text=section.text,
        box=_bounds_covering_lines_safe(section.lines, pad=pad, all_words=all_words),
        gap_above=section.gap_above,
    )


def _make_section(
    index: int,
    lines: list[Line],
    gap_above: float | None,
    pad: float,
    *,
    all_words: list[Word] | None = None,
) -> Section:
    box = _bounds_covering_lines_safe(lines, pad=pad, all_words=all_words)
    text = "\n".join(line.text for line in lines if line.text.strip())
    return Section(index=index, lines=lines, text=text, box=box, gap_above=gap_above)


def _flatten_words(lines: list[Line]) -> list[Word]:
    out: list[Word] = []
    for ln in lines:
        out.extend(ln.words)
    return out


def _orientation_spacing_threshold(
    lines: list[Line],
    *,
    min_gap_px: float = 18.0,
    gap_multiplier: float = 1.0,
) -> float:
    """Break threshold = average (median) horizontal word spacing on the page."""
    ref = median_positive_word_gap_from_lines(lines)
    return max(min_gap_px, ref * gap_multiplier)


def _prepare_orientation_lines(
    lines: list[Line],
    fw: list[Line],
    *,
    min_gap_px: float = 18.0,
    gap_multiplier: float = 1.0,
    apply_spacing_peel: bool,
) -> tuple[list[Line], list[Line], list[Line], list[Line], list[Line], float]:
    """Split mixed rows; optionally peel vertical strips far from horizontal text."""
    lines = split_lines_by_orientation_spacing(
        lines, min_gap_px=min_gap_px, gap_multiplier=gap_multiplier
    )
    fw = split_lines_by_orientation_spacing(
        fw, min_gap_px=min_gap_px, gap_multiplier=gap_multiplier
    )
    threshold = _orientation_spacing_threshold(
        lines, min_gap_px=min_gap_px, gap_multiplier=gap_multiplier
    )
    isolated_vertical: list[Line] = []
    if apply_spacing_peel:
        body_lines, isolated_vertical = _partition_isolated_vertical_lines(
            lines, min_gap_px=min_gap_px, gap_multiplier=gap_multiplier
        )
        fw_body, _ = _partition_isolated_vertical_lines(
            fw, min_gap_px=min_gap_px, gap_multiplier=gap_multiplier
        )
        body_lines = _strip_isolated_vertical_words_from_lines(
            body_lines, lines, threshold
        )
        fw_body = _strip_isolated_vertical_words_from_lines(fw_body, fw, threshold)
    else:
        body_lines, fw_body = lines, fw
    return lines, fw, body_lines, fw_body, isolated_vertical, threshold


def _line_is_isolated_vertical_strip(
    line: Line,
    pool: list[Line],
    threshold: float,
) -> bool:
    """Vertical OCR row separated from horizontal neighbors by wide column gap."""
    if not line_is_vertical(line):
        return False
    for other in pool:
        if other is line or line_is_vertical(other):
            continue
        if not line.content_box.intersects_horizontally(other.content_box, slack=4.0):
            continue
        if column_gap_between_lines(line, other) < threshold:
            return False
    return True


def _partition_isolated_vertical_lines(
    lines: list[Line],
    *,
    min_gap_px: float = 18.0,
    gap_multiplier: float = 1.0,
) -> tuple[list[Line], list[Line]]:
    threshold = _orientation_spacing_threshold(
        lines, min_gap_px=min_gap_px, gap_multiplier=gap_multiplier
    )
    body: list[Line] = []
    isolated: list[Line] = []
    for ln in lines:
        if _line_is_isolated_vertical_strip(ln, lines, threshold):
            isolated.append(ln)
        else:
            body.append(ln)
    return body, isolated


def _word_is_isolated_vertical(
    word: Word,
    horizontal_lines: list[Line],
    threshold: float,
) -> bool:
    if not word_is_vertical(word):
        return False
    wb = word.box
    row_lines = [
        ln
        for ln in horizontal_lines
        if ln.content_box.intersects_horizontally(wb, slack=4.0)
    ]
    if not row_lines:
        return True
    for ln in row_lines:
        if wb.max_x <= ln.content_box.min_x:
            gap = ln.content_box.min_x - wb.max_x
        elif ln.content_box.max_x <= wb.min_x:
            gap = wb.min_x - ln.content_box.max_x
        else:
            return False
        if gap < threshold:
            return False
    return True


def _strip_isolated_vertical_words_from_lines(
    lines: list[Line],
    pool: list[Line],
    threshold: float,
) -> list[Line]:
    h_pool = [ln for ln in pool if not line_is_vertical(ln)]
    out: list[Line] = []
    idx = 0
    for ln in lines:
        if line_is_vertical(ln):
            out.append(ln)
            continue
        kept = [
            w
            for w in ln.words
            if not _word_is_isolated_vertical(w, h_pool, threshold)
        ]
        if not kept:
            continue
        if len(kept) == len(ln.words):
            out.append(ln)
            continue
        chunk = line_from_words(kept, index=idx)
        if chunk is not None:
            out.append(chunk)
            idx += 1
    return out


def _is_vertical_margin_word(word: Word, page_width: float) -> bool:
    """Left-margin stamp word — narrow strip and near-vertical OCR orientation."""
    if page_width <= 0:
        return False
    text = (word.text or "").strip()
    if _STAMP_WORD.match(text) or text in {"-", "–", "—"}:
        return False
    margin_hi = page_width * 0.12
    if word.box.max_x > margin_hi:
        return False
    if word.angle_deg is not None and is_vertical_angle(word.angle_deg):
        return True
    cb = word.box
    aspect = cb.height / max(1.0, cb.width)
    return aspect >= 1.2 and cb.max_x <= margin_hi


def _is_margin_vertical_line(line: Line, page_width: float) -> bool:
    """Narrow margin OCR from vertical stamps (arXiv sidebar, not diagonal watermarks)."""
    if _is_column_watermark_line(line, page_width):
        return False
    if line.words:
        v_count = sum(
            1 for w in line.words if _is_vertical_margin_word(w, page_width)
        )
        if v_count >= max(1, len(line.words) * 0.5):
            return True
    cb = line.content_box
    margin_hi = page_width * 0.065
    if cb.min_x > margin_hi:
        return False
    if cb.max_x > max(margin_hi * 1.5, page_width * 0.10):
        return False
    aspect = cb.height / max(1.0, cb.width)
    if aspect >= 1.0:
        return True
    text = (line.text or "").strip()
    if len(line.words) == 1 and cb.width <= page_width * 0.05 and text:
        if text in {"[", "]", ":"}:
            return True
    return False


def _partition_margin_vertical_lines(
    lines: list[Line],
    page_width: float,
) -> tuple[list[Line], list[Line]]:
    body: list[Line] = []
    margin: list[Line] = []
    for ln in lines:
        if _is_margin_vertical_line(ln, page_width):
            margin.append(ln)
        else:
            body.append(ln)
    return body, margin


def _strip_margin_words_from_lines(
    lines: list[Line],
    page_width: float,
) -> list[Line]:
    """Drop margin-strip words merged into wide body lines by column splitting."""
    from ocr_word_to_line_boxes import line_from_words

    if page_width <= 0:
        return lines
    margin_hi = page_width * 0.14
    out: list[Line] = []
    idx = 0
    for ln in lines:
        kept = [
            w
            for w in ln.words
            if w.box.max_x >= margin_hi and not _is_vertical_margin_word(w, page_width)
        ]
        if not kept:
            continue
        chunk = line_from_words(kept, index=idx)
        if chunk is not None:
            out.append(chunk)
            idx += 1
    return out


def _is_left_margin_vertical_word(word: Word, page_width: float, margin_frac: float = 0.12) -> bool:
    """OCR-rotated glyph parked in the left margin (arXiv spine). Angle only — no aspect."""
    if page_width <= 0 or not word_is_vertical(word):
        return False
    return word.box.max_x <= page_width * margin_frac


def _is_left_margin_stamp_word(word: Word, page_width: float) -> bool:
    """Tall narrow left-edge stamp when OCR omits rotation angle (RISC watermark strip)."""
    if page_width <= 0:
        return False
    text = (word.text or "").strip()
    if not text or text in {"-", "–", "—"}:
        return False
    cb = word.box
    narrow_margin = page_width * 0.055
    if cb.max_x > narrow_margin or cb.width > 16.0:
        return False
    aspect = cb.height / max(1.0, cb.width)
    return aspect >= 2.5 and cb.height >= 30.0


def partition_left_margin_vertical_words(
    lines: list[Line],
    page_width: float,
    *,
    margin_frac: float = 0.12,
) -> tuple[list[Line], list[Line]]:
    """Split OCR-vertical left-margin words from horizontal body words."""
    if page_width <= 0:
        return lines, []
    body: list[Line] = []
    margin: list[Line] = []
    idx_b = 0
    idx_m = 0
    for ln in lines:
        vert = [
            w
            for w in ln.words
            if _is_left_margin_vertical_word(w, page_width, margin_frac)
            or _is_left_margin_stamp_word(w, page_width)
        ]
        horiz = [
            w
            for w in ln.words
            if not (
                _is_left_margin_vertical_word(w, page_width, margin_frac)
                or _is_left_margin_stamp_word(w, page_width)
            )
        ]
        if vert:
            chunk = line_from_words(vert, index=idx_m)
            if chunk is not None:
                margin.append(chunk)
                idx_m += 1
        if not horiz:
            continue
        if len(horiz) == len(ln.words):
            body.append(ln)
            continue
        chunk = line_from_words(horiz, index=idx_b)
        if chunk is not None:
            body.append(chunk)
            idx_b += 1
    return body, margin


def strip_left_margin_vertical_words(
    lines: list[Line],
    page_width: float,
    *,
    margin_frac: float = 0.12,
) -> list[Line]:
    """Remove left-margin vertical-angle words from lines; keep horizontal left-column text."""
    body, _margin = partition_left_margin_vertical_words(
        lines, page_width, margin_frac=margin_frac
    )
    return body


def split_left_margin_vertical_from_sections(
    sections: list[Section],
    page_width: float,
    *,
    pad: float = 6.0,
    margin_frac: float = 0.12,
) -> list[Section]:
    """Unmix a vertical left-margin stamp from body sections; keep it as its own section."""
    body_secs: list[Section] = []
    margin_lines: list[Line] = []
    for sec in sections:
        body_lines, vert_lines = partition_left_margin_vertical_words(
            sec.lines, page_width, margin_frac=margin_frac
        )
        margin_lines.extend(vert_lines)
        if body_lines:
            body_secs.append(_make_section(len(body_secs), body_lines, sec.gap_above, pad))
    return _attach_margin_vertical_sections(body_secs, margin_lines, pad=pad)


def peel_left_margin_vertical_from_sections(
    sections: list[Section],
    page_width: float,
    *,
    pad: float = 6.0,
    margin_frac: float = 0.12,
) -> list[Section]:
    """Alias: split vertical left-margin words into their own sections (do not drop)."""
    return split_left_margin_vertical_from_sections(
        sections, page_width, pad=pad, margin_frac=margin_frac
    )


def _cluster_margin_vertical_sections(
    margin_lines: list[Line],
    *,
    pad: float,
    max_join_gap_px: float = 100.0,
) -> list[Section]:
    """Cluster margin stamps; never bridge distant lines into one full-page bbox."""
    if not margin_lines:
        return []
    ordered = sorted(
        margin_lines, key=lambda ln: (ln.content_box.min_y, ln.content_box.min_x)
    )
    sections: list[Section] = []
    current: list[Line] = [ordered[0]]
    for line in ordered[1:]:
        gap = line.content_box.min_y - current[-1].content_box.max_y
        if gap > max_join_gap_px:
            sections.append(_make_section(len(sections), current, None, pad))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(_make_section(len(sections), current, None, pad))
    return sections


def _attach_margin_vertical_sections(
    sections: list[Section],
    margin_lines: list[Line],
    *,
    pad: float,
) -> list[Section]:
    if not margin_lines:
        return sections
    margin_secs = _cluster_margin_vertical_sections(margin_lines, pad=pad)
    merged = list(sections) + margin_secs
    merged.sort(key=lambda s: min(ln.content_box.min_y for ln in s.lines))
    return [
        Section(i, sec.lines, sec.text, sec.box, sec.gap_above)
        for i, sec in enumerate(merged)
    ]


def _cluster_gap_sections(
    lines: list[Line],
    *,
    multiplier: float,
    min_gap_px: float,
    pad: float,
) -> list[Section]:
    if not lines:
        return []
    ordered = sorted(lines, key=lambda ln: (ln.content_box.min_y, ln.content_box.min_x))
    stats = gap_stats(ordered, multiplier=multiplier, min_gap_px=min_gap_px)
    sections: list[Section] = []
    current: list[Line] = [ordered[0]]
    gap_above: float | None = None

    for i, line in enumerate(ordered[1:], start=1):
        gap = stats.gaps[i - 1]
        if gap > stats.threshold:
            sections.append(_make_section(len(sections), current, gap_above, pad))
            current = [line]
            gap_above = gap
        else:
            current.append(line)
    sections.append(_make_section(len(sections), current, gap_above, pad))
    return sections


def _line_without_stamp_words(line: Line) -> Line | None:
    """Remove diagonal stamp tokens merged onto the same OCR row as body copy."""
    kept = [w for w in line.words if not _STAMP_WORD.match((w.text or "").strip())]
    if not kept:
        return None
    if len(kept) == len(line.words):
        return line
    rebuilt = line_from_words(kept, index=line.index)
    return rebuilt


def _is_column_watermark_line(line: Line, page_width: float) -> bool:
    """Diagonal margin stamps between prose blocks — drop before vertical gap clustering."""
    t = re.sub(r"\s+", " ", (line.text or "").strip())
    if not t:
        return True
    if _COLUMN_WATERMARK_STAMP.match(t):
        return True
    words = t.split()
    if len(words) == 1 and t.isupper() and len(t) < 24:
        if any(c.isdigit() for c in t):
            return False
        letters = [c for c in t if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) >= 0.85:
            return True
    cx = line.content_box.centroid_x
    if len(words) <= 3 and (cx < page_width * 0.06 or cx > page_width * 0.94):
        letters = [c for c in t if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) >= 0.85:
            return True
    return False


def _filter_column_watermark_lines(lines: list[Line], page_width: float) -> list[Line]:
    out: list[Line] = []
    for ln in lines:
        stripped = _line_without_stamp_words(ln)
        if stripped is None:
            continue
        if not _is_column_watermark_line(stripped, page_width):
            out.append(stripped)
    return out


def _vertical_column_gap_threshold(gaps: list[float], *, floor: float = 14.0) -> float:
    """Paragraph-scale breaks in dense legal sidebars (median spacing is only a few px)."""
    positive = [g for g in gaps if g > 0]
    if not positive:
        return floor
    median = statistics.median(positive)
    p90 = sorted(positive)[int(0.9 * (len(positive) - 1))]
    return max(floor, median * 5.0, p90 * 0.85)


def _cluster_vertical_column_sections(
    lines: list[Line],
    *,
    page_width: float,
    pad: float,
    min_gap_px: float = 18.0,
    multiplier: float = 2.0,
) -> list[Section]:
    """Gap-cluster a single aligned column stream (y-sorted); whitespace-only breaks."""
    del multiplier  # vertical sidebar uses band-local threshold, not page min_gap floor
    if not lines:
        return []
    body = _filter_column_watermark_lines(lines, page_width)
    ordered = sorted(body, key=lambda ln: (ln.content_box.min_y, ln.content_box.min_x))
    if not ordered:
        ordered = sorted(lines, key=lambda ln: (ln.content_box.min_y, ln.content_box.min_x))
    gaps = line_gaps(ordered)
    floor = min(14.0, max(8.0, min_gap_px * 0.75))
    threshold = _vertical_column_gap_threshold(gaps, floor=floor)

    sections: list[Section] = []
    current: list[Line] = [ordered[0]]
    gap_above: float | None = None
    for i, line in enumerate(ordered[1:], start=1):
        gap = gaps[i - 1]
        if gap >= threshold:
            sections.append(_make_section(len(sections), current, gap_above, pad))
            current = [line]
            gap_above = gap
        else:
            current.append(line)
    sections.append(_make_section(len(sections), current, gap_above, pad))
    return sections


def _clip_section_box(
    section: Section,
    *,
    max_x: float | None,
    min_x: float | None,
    pad: float,
    all_words: list[Word] | None = None,
) -> Section:
    """Clip X for column gutters, then re-expand so no assigned text is cut."""
    text_box = _bounds_covering_lines_safe(
        section.lines, pad=pad, all_words=all_words
    )
    new_min_x = text_box.min_x if min_x is None else max(text_box.min_x, min_x)
    new_max_x = text_box.max_x if max_x is None else min(text_box.max_x, max_x)
    # Never clip inside the text — if gutter clip would cut glyphs, keep text span.
    new_min_x = min(new_min_x, text_box.min_x)
    new_max_x = max(new_max_x, text_box.max_x)
    if new_max_x <= new_min_x:
        new_max_x = new_min_x + 1
    box = Box(new_min_x, text_box.min_y, new_max_x, text_box.max_y)
    if all_words:
        box = _nudge_box_off_foreign_words(
            box, own_ids=_word_ids(section.lines), all_words=all_words
        )
    return Section(
        index=section.index,
        lines=section.lines,
        text=section.text,
        box=box,
        gap_above=section.gap_above,
    )


def _section_left_anchor(section: Section) -> float:
    return min(ln.content_box.min_x for ln in section.lines)


def _section_lane(section: Section, column_split_x: float | None) -> str:
    if column_split_x is None:
        return "main"
    if section.box.min_x >= column_split_x - 8:
        return "right"
    if section.box.max_x <= column_split_x + 12:
        return "left"
    return "span"


def _section_starts_band_header(section: Section) -> bool:
    for ln in section.lines:
        text = (ln.text or "").strip()
        if text:
            return bool(_LEFT_HEADER_BREAK.match(text))
    return False


def _merge_page_top_stub(sections: list[Section], *, pad: float) -> list[Section]:
    """Merge a short page-top header stub into the next left-column band."""
    if len(sections) < 2:
        return sections
    first, second = sections[0], sections[1]
    if len(first.lines) > 8:
        return sections
    y0, y1 = _section_line_y_span(first)
    if y0 > 140:
        return sections
    nxt_y0, _ = _section_line_y_span(second)
    if nxt_y0 - y1 > 60:
        return sections
    if _section_starts_band_header(second):
        return sections
    return [_merge_two_sections(first, second, pad=pad)] + sections[2:]


def _merge_page_bottom_stub(sections: list[Section], *, pad: float) -> list[Section]:
    """Merge a short footer stub into the dominant band above when the gap is small."""
    if len(sections) < 2:
        return sections
    prev, last = sections[-2], sections[-1]
    if len(last.lines) > 8:
        return sections
    _, prev_y1 = _section_line_y_span(prev)
    last_y0, _ = _section_line_y_span(last)
    if last_y0 - prev_y1 > 60:
        return sections
    return sections[:-2] + [_merge_two_sections(prev, last, pad=pad)]


def _clip_sections_to_column_bounds(
    sections: list[Section],
    *,
    col_bounds: Box,
    pad: float,
    all_words: list[Word] | None = None,
) -> list[Section]:
    """Widen to column X for layout, but Y/X must still fully cover section text."""
    rb = col_bounds
    out: list[Section] = []
    for sec in sections:
        text_box = _bounds_covering_lines_safe(sec.lines, pad=pad, all_words=all_words)
        # Prefer column X band, but never shrink inside text extents.
        min_x = min(rb.min_x, text_box.min_x)
        max_x = max(rb.max_x, text_box.max_x)
        box = Box(min_x, text_box.min_y, max_x, text_box.max_y)
        if all_words:
            box = _nudge_box_off_foreign_words(
                box, own_ids=_word_ids(sec.lines), all_words=all_words
            )
        out.append(
            Section(
                sec.index,
                sec.lines,
                sec.text,
                box,
                sec.gap_above,
            )
        )
    return out


def _merge_two_sections(a: Section, b: Section, *, pad: float) -> Section:
    lines = sorted(a.lines + b.lines, key=lambda ln: (ln.content_box.min_y, ln.content_box.min_x))
    return _make_section(a.index, lines, a.gap_above, pad)


def _section_line_y_span(section: Section) -> tuple[float, float]:
    return (
        min(ln.content_box.min_y for ln in section.lines),
        max(ln.content_box.max_y for ln in section.lines),
    )


def _section_member_ids(sections: list[Section]) -> set[int]:
    out: set[int] = set()
    for sec in sections:
        for ln in sec.lines:
            out.add(id(ln))
    return out


def _intervening_lines(
    prev: Section,
    nxt: Section,
    pool: list[Line],
    *,
    member_ids: set[int],
) -> bool:
    _, prev_y1 = _section_line_y_span(prev)
    nxt_y0, _ = _section_line_y_span(nxt)
    if nxt_y0 <= prev_y1:
        return False
    for ln in pool:
        if id(ln) in member_ids:
            continue
        cy = (ln.content_box.min_y + ln.content_box.max_y) * 0.5
        if prev_y1 < cy < nxt_y0:
            return True
    return False


def _clip_sections_to_column(
    sections: list[Section],
    *,
    col_bounds: Box,
    pad: float,
    all_words: list[Word] | None = None,
) -> list[Section]:
    return _clip_sections_to_column_bounds(
        sections, col_bounds=col_bounds, pad=pad, all_words=all_words
    )


def _band_column_overlap_frac(
    band_y0: float,
    band_y1: float,
    col: Any,
) -> float:
    col_y0, col_y1 = col.bounds.min_y, col.bounds.max_y
    overlap = min(band_y1, col_y1) - max(band_y0, col_y0)
    if overlap <= 0:
        return 0.0
    return overlap / max(1.0, band_y1 - band_y0)


def _band_lines_are_wide_label_value_rows(
    lines: list[Line],
    *,
    page_width: float,
    split_x: float,
) -> bool:
    """OCR lines already span left labels and right values — not a peelable margin column."""
    wide = [
        ln
        for ln in lines
        if (ln.content_box.max_x - ln.content_box.min_x) >= page_width * 0.25
    ]
    if len(wide) < 3:
        return False
    paired = 0
    for ln in wide:
        left = any(w.box.centroid_x < split_x - 12 for w in ln.words)
        right = any(w.box.centroid_x > split_x + 12 for w in ln.words)
        if left and right:
            paired += 1
    return paired >= max(3, int(len(wide) * 0.45))


def _split_horizontal_band_at_column(
    band_sec: Section,
    column: Any,
    *,
    page_width: float,
    gutter_x: float,
    multiplier: float,
    min_gap_px: float,
    pad: float,
) -> tuple[list[Section], list[str]]:
    """Within one horizontal gap band, keep left as one section; gap-cluster right column only."""
    band_lines = band_sec.lines
    band_y0, band_y1 = _section_line_y_span(band_sec)
    split_x = column.split_x
    if column.side == "right":
        left_clip_x = split_x - 4.0
        col_bounds = column.bounds
    else:
        left_clip_x = None
        col_bounds = column.bounds

    left_lines, right_lines = partition_lines_at_column(
        band_lines,
        column,
        page_width=page_width,
        split_x=split_x,
        gutter_x=gutter_x,
    )
    if column.side == "right":
        body_lines, col_lines = left_lines, right_lines
    else:
        body_lines, col_lines = right_lines, left_lines

    if body_lines:
        ordered_left = sorted(
            body_lines, key=lambda ln: (ln.content_box.min_y, ln.content_box.min_x)
        )
        left_secs = [_make_section(0, ordered_left, band_sec.gap_above, pad)]
        if left_clip_x is not None:
            left_secs = [
                _clip_section_box(s, max_x=left_clip_x, min_x=None, pad=pad)
                for s in left_secs
            ]
    else:
        left_secs = []

    col_secs = _cluster_vertical_column_sections(
        col_lines,
        page_width=page_width,
        pad=pad,
        min_gap_px=min_gap_px,
        multiplier=multiplier,
    )
    col_secs = _clip_sections_to_column(col_secs, col_bounds=col_bounds, pad=pad)

    out = left_secs + col_secs
    col_role = f"column_{column.side}"
    roles = ["horizontal"] * len(left_secs) + [col_role] * len(col_secs)
    return out, roles


def _is_full_height_column(column: Any, fw_lines: list[Line]) -> bool:
    page_y0 = min(ln.content_box.min_y for ln in fw_lines)
    page_y1 = max(ln.content_box.max_y for ln in fw_lines)
    page_h = max(1.0, page_y1 - page_y0)
    col_h = column.bounds.max_y - column.bounds.min_y
    top_margin = column.bounds.min_y - page_y0
    bot_margin = page_y1 - column.bounds.max_y
    if not (
        col_h >= page_h * 0.72
        and top_margin <= page_h * 0.06
        and bot_margin <= page_h * 0.06
    ):
        return False
    return aligned_column_valid_for_vertical(fw_lines, column)


def _column_split_safe(lines: list[Line], column: Any) -> bool:
    """False when the proposed gutter would cut through opposing OCR in column y-span."""
    y0, y1 = column.bounds.min_y, column.bounds.max_y
    span = [
        ln
        for ln in lines
        if ln.content_box.max_y >= y0 - 2.0 and ln.content_box.min_y <= y1 + 2.0
    ]
    return aligned_column_valid_for_vertical(span, column)


def _lines_to_sections_hv_full_column(
    lines: list[Line],
    page_col: Any,
    *,
    page_width: float,
    gutter_x: float,
    multiplier: float,
    min_gap_px: float,
    pad: float,
    all_words: list[Word] | None = None,
) -> tuple[list[Section], list[str], int]:
    """Gap-sectioned body stream alongside one full-height aligned column."""
    split_x = page_col.split_x
    combine_merges = 0
    left_lines, right_lines = partition_lines_at_column(
        lines,
        page_col,
        page_width=page_width,
        split_x=split_x,
        gutter_x=gutter_x,
    )
    if page_col.side == "right":
        body_lines, col_lines = left_lines, right_lines
        body_clip_max_x, body_clip_min_x = split_x - 4.0, None
        body_role = "column_left"
    else:
        body_lines, col_lines = right_lines, left_lines
        body_clip_max_x, body_clip_min_x = None, split_x + 4.0
        body_role = "column_right"

    body_sections = _cluster_gap_sections(
        body_lines,
        multiplier=multiplier,
        min_gap_px=max(min_gap_px, 22.0),
        pad=pad,
    )
    body_sections = [
        _clip_section_box(
            s,
            max_x=body_clip_max_x,
            min_x=body_clip_min_x,
            pad=pad,
            all_words=all_words,
        )
        for s in body_sections
    ]
    body_sections, combine_merges = combine_flowing_sections(
        body_sections,
        column_split_x=split_x,
        pad=pad,
        line_pool=body_lines,
    )
    before_stub = len(body_sections)
    body_sections = _merge_page_top_stub(body_sections, pad=pad)
    combine_merges += before_stub - len(body_sections)

    ordered_col = sorted(col_lines, key=lambda ln: ln.content_box.min_y)
    col_role = f"column_{page_col.side}"
    if ordered_col:
        # Do NOT clamp to column.bounds.min_y/max_y — that cuts header/footer glyphs
        # that belong to the column text (see 8b824d40_p1 S1 top cut).
        text = "\n".join(ln.text for ln in ordered_col if ln.text.strip())
        text_box = _bounds_covering_lines_safe(
            ordered_col, pad=pad, all_words=all_words
        )
        rb = page_col.bounds
        min_x = min(rb.min_x, text_box.min_x)
        max_x = max(rb.max_x, text_box.max_x)
        box = Box(min_x, text_box.min_y, max_x, text_box.max_y)
        if all_words:
            box = _nudge_box_off_foreign_words(
                box, own_ids=_word_ids(ordered_col), all_words=all_words
            )
        col_sections = [
            Section(
                index=0,
                lines=ordered_col,
                text=text,
                box=box,
                gap_above=None,
            )
        ]
    else:
        col_sections = []

    tagged: list[tuple[Section, str]] = [
        (s, body_role) for s in body_sections
    ] + [(s, col_role) for s in col_sections]
    tagged.sort(key=lambda t: min(ln.content_box.min_y for ln in t[0].lines))
    merged = [sec for sec, _ in tagged]
    roles = [role for _, role in tagged]
    return merged, roles, combine_merges


def _layout_meta(
    *,
    page_col: Any | None,
    gutter_x: float,
    **extra: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "aligned_column": page_col.to_dict() if page_col else None,
        "column_split_x": round(page_col.split_x, 1) if page_col else None,
        "gutter_x": round(gutter_x, 1),
    }
    base.update(extra)
    return base


def horizontal_gap_superboxes(
    lines: list[Line],
    *,
    page_width: float,
    full_width_lines: list[Line] | None = None,
    multiplier: float = 2.0,
    min_gap_px: float = 18.0,
    pad: float = 6.0,
) -> tuple[list[Section], GapStats, dict[str, Any]]:
    """
    Horizontal y-gap bands only — the first production superboxes before
    per-band column peel (lines_to_sections_hv_combined step 1).
    """
    fw = full_width_lines if full_width_lines is not None else lines
    preview_sections, preview_stats = lines_to_sections(
        fw, multiplier=multiplier, min_gap_px=min_gap_px, pad=pad
    )
    preview_layouts = [classify_section_layout(sec.lines) for sec in preview_sections]
    preview_line_counts = [len(sec.lines) for sec in preview_sections]
    skip_vertical_for_table = table_layout_skips_vertical(
        preview_line_counts,
        preview_layouts,
        all_lines=fw,
        page_width=page_width,
    )
    _MEGA_FW_BAND_LINES = 15
    has_mega_fw_section_table = any(
        n >= _MEGA_FW_BAND_LINES
        and lay.layout_kind in ("table", "section_table")
        for n, lay in zip(preview_line_counts, preview_layouts)
    )
    use_body_sl_bands = False
    body_preview_sections = preview_sections
    body_preview_stats = preview_stats
    if has_mega_fw_section_table:
        body_preview_sections, body_preview_stats = lines_to_sections(
            lines, multiplier=multiplier, min_gap_px=min_gap_px, pad=pad
        )
        use_body_sl_bands = len(body_preview_sections) > len(preview_sections)
    band_source = lines if use_body_sl_bands else fw
    if band_source is fw:
        h_sections, h_stats = preview_sections, preview_stats
    else:
        h_sections, h_stats = body_preview_sections, body_preview_stats
    if skip_vertical_for_table:
        h_sections = _merge_page_top_stub(h_sections, pad=pad)
        h_sections = _merge_page_bottom_stub(h_sections, pad=pad)
    snap_meta: dict[str, Any] = {
        "skip_vertical_for_table": skip_vertical_for_table,
        "use_body_sl_bands": use_body_sl_bands,
        "band_count": len(h_sections),
        "band_source": "body_sl" if use_body_sl_bands else "full_width",
    }
    return h_sections, h_stats, snap_meta


def apply_hv_column_peel(
    h_sections: list[Section],
    lines: list[Line],
    *,
    page_width: float,
    full_width_lines: list[Line],
    multiplier: float = 2.0,
    min_gap_px: float = 18.0,
    pad: float = 6.0,
    gutter_x: float | None = None,
    v_partition: Any | None = None,
) -> tuple[list[Section], dict[str, Any]]:
    """
    L/R column peel on horizontal gap bands (step 3 of production pipeline).
    Caller must skip when table_gate already handled superboxes in step 1.
    """
    fw = full_width_lines
    if gutter_x is None:
        gutter_x = estimate_page_gutter_x(fw, page_width)
    if v_partition is None:
        v_partition = analyze_vertical_partition(fw, page_width)

    band_layouts = [classify_section_layout(sec.lines) for sec in h_sections]
    band_line_counts = [len(sec.lines) for sec in h_sections]
    page_layout = page_layout_from_horizontal_bands(band_line_counts, band_layouts)

    page_col = detect_aligned_text_column(
        fw, page_width=page_width, gutter_x=gutter_x
    )
    page_col_ref = (
        page_col
        if page_col is not None and aligned_column_valid_for_vertical(fw, page_col)
        else None
    )

    # Full-height column peel is for multi-section pages, not a lone table grid.
    if (
        page_col_ref is not None
        and _is_full_height_column(page_col_ref, fw)
        and page_layout != "table"
    ):
        merged_secs, section_roles, combine_merges = _lines_to_sections_hv_full_column(
            lines,
            page_col_ref,
            page_width=page_width,
            gutter_x=gutter_x,
            multiplier=multiplier,
            min_gap_px=min_gap_px,
            pad=pad,
        )
        sections = [
            Section(idx, s.lines, s.text, s.box, s.gap_above)
            for idx, s in enumerate(merged_secs)
        ]
        sections = split_prose_sections_by_lanes(sections, pad=pad)
        meta = _layout_meta(
            page_col=page_col_ref,
            gutter_x=gutter_x,
            mode="hv_combined",
            horizontal_band_count=0,
            split_horizontal_indices=[],
            section_roles=section_roles,
            layout="full_height_column",
            combine_merges=combine_merges,
        )
        return sections, meta

    merged: list[tuple[Section, str]] = []
    split_band_indices: list[int] = []
    band_col_ref: Any | None = None

    for i, band_sec in enumerate(h_sections):
        if band_skips_vertical_column_split(
            band_layouts[i], band_lines=band_sec.lines, page_width=page_width
        ):
            merged.append((band_sec, "horizontal"))
            continue
        band_y0, band_y1 = _section_line_y_span(band_sec)
        band_col = detect_aligned_text_column(
            band_sec.lines,
            page_width=page_width,
            gutter_x=gutter_x,
            y_min=band_y0,
            y_max=band_y1,
        )
        if band_col is None:
            merged.append((band_sec, "horizontal"))
            continue
        if _band_column_overlap_frac(band_y0, band_y1, band_col) < 0.22:
            merged.append((band_sec, "horizontal"))
            continue
        if page_col_ref is not None and abs(
            band_col.anchor_min_x - page_col_ref.anchor_min_x
        ) > 90:
            merged.append((band_sec, "horizontal"))
            continue
        if not _column_split_safe(band_sec.lines, band_col):
            merged.append((band_sec, "horizontal"))
            continue
        if _band_lines_are_wide_label_value_rows(
            band_sec.lines,
            page_width=page_width,
            split_x=band_col.split_x,
        ) and not analyze_vertical_partition(
            band_sec.lines, page_width
        ).allows_column_split():
            merged.append((band_sec, "horizontal"))
            continue

        split_secs, roles = _split_horizontal_band_at_column(
            band_sec,
            band_col,
            page_width=page_width,
            gutter_x=gutter_x,
            multiplier=multiplier,
            min_gap_px=min_gap_px,
            pad=pad,
        )
        if not split_secs:
            merged.append((band_sec, "horizontal"))
            continue
        split_band_indices.append(i)
        band_col_ref = band_col
        merged.extend(zip(split_secs, roles))

    if not split_band_indices:
        h_sections = split_prose_sections_by_lanes(h_sections, pad=pad)
        return h_sections, _layout_meta(
            page_col=None,
            gutter_x=gutter_x,
            mode="gap",
            horizontal_band_count=len(h_sections),
            split_horizontal_indices=[],
            section_roles=["horizontal"] * len(h_sections),
        )

    merged.sort(key=lambda t: min(ln.content_box.min_y for ln in t[0].lines))
    section_roles = [role for _, role in merged]
    sections = [
        Section(idx, sec.lines, sec.text, sec.box, sec.gap_above)
        for idx, (sec, _) in enumerate(merged)
    ]
    sections = split_prose_sections_by_lanes(sections, pad=pad)

    meta = _layout_meta(
        page_col=page_col_ref or band_col_ref,
        gutter_x=gutter_x,
        mode="hv_combined",
        horizontal_band_count=len(h_sections),
        split_horizontal_indices=split_band_indices,
        section_roles=section_roles,
        vertical_partition=v_partition.to_dict(),
    )
    return sections, meta


def lines_to_sections_hv_combined(
    lines: list[Line],
    *,
    page_width: float,
    full_width_lines: list[Line] | None = None,
    multiplier: float = 2.0,
    min_gap_px: float = 18.0,
    pad: float = 6.0,
    apply_spacing_peel: bool = True,
    spacing_gap_multiplier: float = 1.0,
) -> tuple[list[Section], GapStats, dict[str, Any]]:
    """Horizontal gap bands first; subdivide only bands that contain a valid aligned column."""
    fw = full_width_lines if full_width_lines is not None else lines
    _, _, body_lines, fw_body, isolated_vertical, _ = _prepare_orientation_lines(
        lines,
        fw,
        min_gap_px=min_gap_px,
        gap_multiplier=spacing_gap_multiplier,
        apply_spacing_peel=apply_spacing_peel,
    )

    gutter_x = estimate_page_gutter_x(fw_body, page_width)
    v_partition = analyze_vertical_partition(fw_body, page_width)

    h_sections, h_stats, super_meta = horizontal_gap_superboxes(
        body_lines,
        page_width=page_width,
        full_width_lines=fw_body,
        multiplier=multiplier,
        min_gap_px=min_gap_px,
        pad=pad,
    )
    skip_vertical_for_table = bool(super_meta.get("skip_vertical_for_table"))
    band_layouts = [classify_section_layout(sec.lines) for sec in h_sections]
    band_line_counts = [len(sec.lines) for sec in h_sections]
    page_layout = page_layout_from_horizontal_bands(band_line_counts, band_layouts)
    if skip_vertical_for_table:
        h_sections = split_prose_sections_by_lanes(h_sections, pad=pad)
        if isolated_vertical:
            h_sections = _attach_margin_vertical_sections(
                h_sections, isolated_vertical, pad=pad
            )
        return h_sections, h_stats, _layout_meta(
            page_col=None,
            gutter_x=gutter_x,
            mode="gap",
            page_layout=page_layout,
            table_gate=True,
            vertical_partition=v_partition.to_dict(),
            horizontal_band_count=len(h_sections),
            split_horizontal_indices=[],
            section_roles=["horizontal"] * len(h_sections),
            gap_superboxes=super_meta,
            isolated_vertical_line_count=len(isolated_vertical),
            orientation_technique="spacing_peel",
        )

    sections, peel_meta = apply_hv_column_peel(
        h_sections,
        body_lines,
        page_width=page_width,
        full_width_lines=fw_body,
        multiplier=multiplier,
        min_gap_px=min_gap_px,
        pad=pad,
        gutter_x=gutter_x,
        v_partition=v_partition,
    )
    if isolated_vertical:
        sections = _attach_margin_vertical_sections(sections, isolated_vertical, pad=pad)
    peel_meta["gap_superboxes"] = super_meta
    peel_meta["isolated_vertical_line_count"] = len(isolated_vertical)
    peel_meta["orientation_technique"] = "spacing_peel"
    return sections, h_stats, peel_meta


def lines_to_sections_dual_cluster(
    lines: list[Line],
    *,
    page_width: float,
    full_width_lines: list[Line] | None = None,
    multiplier: float = 2.0,
    min_gap_px: float = 18.0,
    pad: float = 6.0,
    spacing_gap_multiplier: float = 1.0,
) -> tuple[list[Section], GapStats, dict[str, Any]]:
    """
    1) Cluster vertical-orientation lines into y-bands and redact them.
    2) Cluster remaining horizontal lines into y-bands.
    3) Merge both section lists by y.
    """
    fw = full_width_lines if full_width_lines is not None else lines
    lines, fw, _, _, _, threshold = _prepare_orientation_lines(
        lines,
        fw,
        min_gap_px=min_gap_px,
        gap_multiplier=spacing_gap_multiplier,
        apply_spacing_peel=False,
    )

    def _is_redacted_vertical(line: Line, pool: list[Line]) -> bool:
        return line_is_vertical(line) and _line_is_isolated_vertical_strip(
            line, pool, threshold
        )

    vertical_lines = [ln for ln in lines if _is_redacted_vertical(ln, lines)]
    horizontal_lines = [ln for ln in lines if ln not in vertical_lines]
    fw_horizontal = [ln for ln in fw if not _is_redacted_vertical(ln, fw)]

    horizontal_lines = _strip_isolated_vertical_words_from_lines(
        horizontal_lines, lines, threshold
    )
    fw_horizontal = _strip_isolated_vertical_words_from_lines(
        fw_horizontal, fw, threshold
    )

    vertical_sections = _cluster_margin_vertical_sections(vertical_lines, pad=pad)

    horizontal_sections, h_stats, meta = lines_to_sections_hv_combined(
        horizontal_lines,
        page_width=page_width,
        full_width_lines=fw_horizontal,
        multiplier=multiplier,
        min_gap_px=min_gap_px,
        pad=pad,
        apply_spacing_peel=False,
        spacing_gap_multiplier=spacing_gap_multiplier,
    )

    merged = list(horizontal_sections) + vertical_sections
    merged.sort(key=lambda s: min(ln.content_box.min_y for ln in s.lines))
    sections = [
        Section(i, sec.lines, sec.text, sec.box, sec.gap_above)
        for i, sec in enumerate(merged)
    ]

    meta = dict(meta)
    meta["orientation_technique"] = "dual_cluster_redact"
    meta["vertical_line_count"] = len(vertical_lines)
    meta["horizontal_line_count"] = len(horizontal_lines)
    meta["vertical_section_count"] = len(vertical_sections)
    meta["horizontal_section_count"] = len(horizontal_sections)
    return sections, h_stats, meta


def lines_to_sections_vertical_only(
    lines: list[Line],
    *,
    page_width: float,
    full_width_lines: list[Line] | None = None,
    multiplier: float = 2.0,
    min_gap_px: float = 18.0,
    pad: float = 6.0,
) -> tuple[list[Section], GapStats, dict[str, Any]]:
    """Gap-cluster only the detected aligned column (y-scoped to the column cluster)."""
    fw = full_width_lines if full_width_lines is not None else lines
    page_words = _flatten_words(fw)
    gutter_x = estimate_page_gutter_x(fw, page_width)
    column = detect_aligned_text_column(fw, page_width=page_width, gutter_x=gutter_x)
    if column is None or not aligned_column_valid_for_vertical(fw, column):
        return [], gap_stats([], multiplier=multiplier, min_gap_px=min_gap_px), {
            "mode": "vertical_only",
            "aligned_column": None,
            "column_split_x": None,
            "column_strategy": "none",
        }

    y0, y1 = column.bounds.min_y, column.bounds.max_y
    scoped = [
        ln
        for ln in lines
        if ln.content_box.max_y >= y0 - 4 and ln.content_box.min_y <= y1 + 4
    ]
    left_lines, right_lines = partition_lines_at_column(
        scoped,
        column,
        page_width=page_width,
        split_x=column.split_x,
        gutter_x=gutter_x,
    )
    col_lines = right_lines if column.side == "right" else left_lines

    col_secs = _cluster_vertical_column_sections(
        col_lines,
        page_width=page_width,
        pad=pad,
        min_gap_px=min_gap_px,
        multiplier=multiplier,
    )
    col_secs = _clip_sections_to_column(
        col_secs, col_bounds=column.bounds, pad=pad, all_words=page_words
    )
    sections = [
        _recompute_section_box(
            Section(idx, sec.lines, sec.text, sec.box, sec.gap_above),
            pad=pad,
            all_words=page_words,
        )
        for idx, sec in enumerate(col_secs)
    ]
    body = _filter_column_watermark_lines(
        sorted(col_lines, key=lambda ln: ln.content_box.min_y), page_width
    )
    stats = gap_stats(
        body or col_lines,
        multiplier=multiplier,
        min_gap_px=min_gap_px,
    )
    meta = {
        "mode": "vertical_only",
        "aligned_column": column.to_dict(),
        "column_split_x": round(column.split_x, 1),
        "column_strategy": "vertical_gap_cluster",
    }
    return sections, stats, meta


def combine_flowing_sections(
    sections: list[Section],
    *,
    column_split_x: float | None = None,
    max_vertical_gap_px: float = 52.0,
    x_align_tolerance_px: float = 32.0,
    pad: float = 6.0,
    line_pool: list[Line] | None = None,
) -> tuple[list[Section], int]:
    if len(sections) < 2:
        return sections, 0

    pool = line_pool or []
    member_ids = _section_member_ids(sections)

    by_lane: dict[str, list[Section]] = {"left": [], "right": [], "span": [], "main": []}
    for sec in sections:
        by_lane[_section_lane(sec, column_split_x)].append(sec)

    merges = 0

    def merge_lane(group: list[Section]) -> list[Section]:
        nonlocal merges
        if len(group) < 2:
            return group
        ordered = sorted(group, key=lambda s: _section_line_y_span(s)[0])
        out: list[Section] = [ordered[0]]
        for nxt in ordered[1:]:
            prev = out[-1]
            _, prev_y1 = _section_line_y_span(prev)
            nxt_y0, _ = _section_line_y_span(nxt)
            if nxt_y0 < prev_y1 - 4.0:
                out.append(nxt)
                continue
            gap = nxt_y0 - prev_y1
            aligned = abs(_section_left_anchor(prev) - _section_left_anchor(nxt)) <= x_align_tolerance_px
            blocked = pool and _intervening_lines(prev, nxt, pool, member_ids=member_ids)
            if (
                gap <= max_vertical_gap_px
                and aligned
                and not _section_starts_band_header(nxt)
                and not blocked
            ):
                out[-1] = _merge_two_sections(prev, nxt, pad=pad)
                member_ids.update(id(ln) for ln in nxt.lines)
                merges += 1
            else:
                out.append(nxt)
        return out

    combined: list[Section] = []
    combined.extend(merge_lane(by_lane["left"]))
    combined.extend(merge_lane(by_lane["main"]))
    combined.extend(by_lane["right"])
    combined.extend(by_lane["span"])
    combined.sort(key=lambda s: _section_line_y_span(s)[0])

    reindexed = [
        Section(idx, s.lines, s.text, s.box, s.gap_above) for idx, s in enumerate(combined)
    ]
    return reindexed, merges


def write_section_ranges_summary(
    path: Path,
    sections: list[Section],
    *,
    mode_label: str,
    column_split_x: float | None = None,
    section_roles: list[str] | None = None,
) -> None:
    lines_out = [f"mode: {mode_label}", f"section_count: {len(sections)}"]
    if column_split_x is not None:
        lines_out.append(f"column_split_x: {column_split_x:.1f}")
    lines_out.append("")
    lines_out.append("| index | y_range | x_range | lane | role |")
    lines_out.append("| --- | --- | --- | --- | --- |")
    for i, section in enumerate(sections):
        y0 = section.box.min_y
        y1 = section.box.max_y
        x0 = section.box.min_x
        x1 = section.box.max_x
        role = (
            section_roles[i]
            if section_roles is not None and i < len(section_roles)
            else ""
        )
        if column_split_x is not None:
            if x0 >= column_split_x - 8:
                lane = "right"
            elif x1 <= column_split_x + 12:
                lane = "left"
            else:
                lane = "span"
        else:
            lane = "span"
        lines_out.append(
            f"| S{section.index} | [{y0:.0f},{y1:.0f}] | [{x0:.0f},{x1:.0f}] | {lane} | {role} |"
        )
    lines_out.append("")
    for section in sections:
        y0 = section.box.min_y
        y1 = section.box.max_y
        lines_out.append(
            f"S{section.index}: y=[{y0:.1f}, {y1:.1f}] "
            f"x=[{section.box.min_x:.1f}, {section.box.max_x:.1f}]"
        )
    path.write_text("\n".join(lines_out) + "\n")


def lines_from_json(path: Path) -> list[Line]:
    data = json.loads(path.read_text())
    lines: list[Line] = []
    for row in data["lines"]:
        words = [
            Word(
                text=w["text"],
                box=Box(**w["bounds"]),
                index=i,
            )
            for i, w in enumerate(row.get("words") or [])
        ]
        if not words:
            bounds = row["bounds"]
            box = Box(**bounds)
            content = Box(**row.get("content_bounds", bounds))
            words = [Word(text=row["text"], box=content, index=0)]
        else:
            box = Box(**row["bounds"])
        lines.append(
            Line(
                index=row["index"],
                text=row["text"],
                box=box,
                words=words,
            )
        )
    return lines


SECTION_PALETTE: list[tuple[int, int, int]] = [
    (255, 99, 71),
    (50, 205, 50),
    (30, 144, 255),
    (255, 165, 0),
    (148, 0, 211),
    (0, 206, 209),
    (255, 20, 147),
    (128, 128, 0),
    (220, 40, 40),
    (100, 149, 237),
]


def _section_bounds(section: Section | dict[str, Any]) -> tuple[int, float, float, float, float]:
    if isinstance(section, dict):
        idx = int(section.get("index", 0))
        b = section["bounds"]
        return idx, b["min_x"], b["min_y"], b["max_x"], b["max_y"]
    return section.index, section.box.min_x, section.box.min_y, section.box.max_x, section.box.max_y


def draw_sections_overlay(
    full_img: Image.Image,
    sections: list[Section] | list[dict[str, Any]],
    picked_section_idx: int | None = None,
    crop_bounds: dict[str, float] | None = None,
    *,
    banner_title: str | None = None,
    column_guide_x: float | None = None,
    section_meta: dict[str, Any] | None = None,
) -> Image.Image:
    """Draw each OCR section in a unique color; highlight the picked VIN section.

    When ``crop_bounds`` is set, draw an inner green rectangle for the VIN crop
    sent to image enhancement (typically ``tight_vin_bounds`` + pad).
    """
    base = full_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(overlay)
    font = load_font(14)
    label_h = 18

    for section in sections:
        idx, min_x, min_y, max_x, max_y = _section_bounds(section)
        color = SECTION_PALETTE[idx % len(SECTION_PALETTE)]
        is_picked = picked_section_idx is not None and idx == picked_section_idx
        fill_alpha = int(255 * 0.30) if is_picked else int(255 * 0.20)
        fill_draw.rectangle(
            [(min_x, min_y), (max_x, max_y)],
            fill=(*color, fill_alpha),
        )
        if isinstance(section, dict):
            kind = section.get("layout_kind") or ""
        else:
            meta = (section_meta or {}).get(f"S{idx}")
            kind = (meta or {}).get("layout_kind") or ""
        if kind == "table":
            label = f"S{idx} T"
        elif kind == "section_table":
            label = f"S{idx} S+T"
        elif kind == "figure":
            label = f"S{idx} IMG"
        else:
            label = f"S{idx}"
        label_w = max(34, int(fill_draw.textlength(label, font=font)) + 8)
        fill_draw.rectangle(
            [(min_x, min_y - label_h), (min_x + label_w, min_y)],
            fill=(*color, 230),
        )
        fill_draw.text(
            (min_x + 3, min_y - label_h + 2),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    result = Image.alpha_composite(base, overlay)
    border_draw = ImageDraw.Draw(result)
    for section in sections:
        idx, min_x, min_y, max_x, max_y = _section_bounds(section)
        color = SECTION_PALETTE[idx % len(SECTION_PALETTE)]
        is_picked = picked_section_idx is not None and idx == picked_section_idx
        width = 5 if is_picked else 3
        border_draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=color, width=width)
        if is_picked:
            inset = 4
            border_draw.rectangle(
                [
                    (min_x + inset, min_y + inset),
                    (max_x - inset, max_y - inset),
                ],
                outline=(255, 255, 255),
                width=2,
            )

    if crop_bounds:
        cx0 = int(crop_bounds["min_x"])
        cy0 = int(crop_bounds["min_y"])
        cx1 = int(crop_bounds["max_x"])
        cy1 = int(crop_bounds["max_y"])
        crop_color = (20, 180, 80)
        border_draw.rectangle([(cx0, cy0), (cx1, cy1)], outline=crop_color, width=4)
        label = "crop"
        lw = border_draw.textlength(label, font=font)
        border_draw.rectangle(
            [(cx0, cy0 - label_h), (cx0 + lw + 8, cy0)],
            fill=(*crop_color, 255),
        )
        border_draw.text((cx0 + 4, cy0 - label_h + 2), label, fill=(255, 255, 255), font=font)

    if column_guide_x is not None:
        gx = int(column_guide_x)
        h = result.height
        for y in range(0, h, 12):
            border_draw.line([(gx, y), (gx, min(y + 6, h))], fill=(255, 255, 0), width=2)

    if banner_title:
        title_font = load_font(20)
        tw = border_draw.textlength(banner_title, font=title_font)
        pad_x, pad_y = 10, 6
        bar_h = 32
        border_draw.rectangle(
            [(8, 8), (8 + tw + pad_x * 2, 8 + bar_h)],
            fill=(0, 0, 0, 220),
        )
        border_draw.text((8 + pad_x, 8 + pad_y), banner_title, fill=(255, 255, 255), font=title_font)

    return result.convert("RGB")


def draw_sections(
    image_path: Path,
    sections: list[Section],
    *,
    show_lines: bool = False,
    lines: list[Line] | None = None,
    section_color: tuple[int, int, int] = (20, 150, 60),
    line_color: tuple[int, int, int] = (220, 40, 40),
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = load_font(12)

    if show_lines and lines:
        for line in lines:
            b = line.content_box
            draw.rectangle([(b.min_x, b.min_y), (b.max_x, b.max_y)], outline=line_color, width=1)

    for section in sections:
        b = section.box
        draw.rectangle([(b.min_x, b.min_y), (b.max_x, b.max_y)], outline=section_color, width=4)
        label = f"S{section.index}"
        draw.rectangle([(b.min_x, b.min_y - 16), (b.min_x + 34, b.min_y)], fill=section_color)
        draw.text((b.min_x + 3, b.min_y - 15), label, fill=(255, 255, 255), font=font)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", help="Path to vision page JSON")
    parser.add_argument("--lines-json", help="Reuse output from ocr_word_to_line_boxes.py")
    parser.add_argument("--image", help="Path to page PNG/JPG")
    parser.add_argument("--payload", help="mllm_test payload JSON (optional, for S3 vision key)")
    parser.add_argument("--out", required=True, help="Output directory or .png path")
    parser.add_argument("--gap-multiplier", type=float, default=2.0, help="Section break when gap > median * multiplier")
    parser.add_argument("--min-gap", type=float, default=18.0, help="Minimum px gap treated as a section break")
    parser.add_argument(
        "--section-mode",
        choices=("gap", "hv", "auto", "vertical-only"),
        default="auto",
        help="gap=single stream; hv=horizontal bands + valid aligned column; auto=hv when column passes validity; "
        "vertical-only=aligned column gap sections only",
    )
    parser.add_argument(
        "--overlay-title",
        default=None,
        help="Banner label on section overlay PNG (e.g. mode description)",
    )
    parser.add_argument("--show-lines", action="store_true", help="Overlay faint line boxes")
    parser.add_argument("--download-image", action="store_true")
    parser.add_argument("--aws-profile", default="prod")
    args = parser.parse_args()

    column_meta: dict[str, Any] | None = None
    section_mode = args.section_mode

    if args.lines_json:
        if not args.image:
            raise SystemExit("--image required with --lines-json")
        lines = lines_from_json(Path(args.lines_json))
        image_path = Path(args.image)
        vision_path = None
        out = Path(args.out)
        page_width = Image.open(image_path).width
        fw_lines = lines
        if section_mode in ("hv", "auto"):
            section_mode = "gap"
    else:
        vision_path, image_path, out = resolve_paths(args)
        vision = load_vision(vision_path)
        words = load_words(vision)
        with Image.open(image_path) as img:
            page_width = img.width
        fw_lines = words_to_lines(words, page_width=page_width, full_width=True)
        if section_mode in ("hv", "auto", "vertical-only"):
            lines = words_to_lines(
                words, page_width=page_width, full_width=False, split_columns=True
            )
        else:
            lines = fw_lines

    column_guide_x: float | None = None
    if section_mode == "vertical-only":
        sections, stats, column_meta = lines_to_sections_vertical_only(
            lines,
            page_width=float(page_width),
            full_width_lines=fw_lines,
            multiplier=args.gap_multiplier,
            min_gap_px=args.min_gap,
        )
        section_mode = column_meta.get("mode", "vertical_only")
        column_guide_x = column_meta.get("column_split_x") if column_meta.get("mode") != "gap" else None
    else:
        use_hv = section_mode in ("hv", "auto")
        if use_hv:
            sections, stats, column_meta = lines_to_sections_hv_combined(
                lines,
                page_width=float(page_width),
                full_width_lines=fw_lines,
                multiplier=args.gap_multiplier,
                min_gap_px=args.min_gap,
            )
            section_mode = column_meta.get("mode", "hv_combined")
            column_guide_x = column_meta.get("column_split_x") if column_meta.get("mode") != "gap" else None
        else:
            sections, stats = lines_to_sections(
                fw_lines,
                multiplier=args.gap_multiplier,
                min_gap_px=args.min_gap,
            )
            section_mode = "gap"
            column_meta = None

    if str(out).endswith((".png", ".jpg", ".jpeg")):
        annotated = out
        json_out = out.with_suffix(".json")
        txt_out = out.with_suffix(".txt")
        ranges_out = out.with_suffix(".ranges.txt")
    else:
        out.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        annotated = out / f"{stem}_sections.png"
        json_out = out / f"{stem}_sections.json"
        txt_out = out / f"{stem}_ocr_sections.txt"
        ranges_out = out / f"{stem}_section_ranges.txt"

    overlay_title = args.overlay_title
    if overlay_title is None:
        if section_mode == "gap" and column_meta and column_meta.get("table_gate"):
            overlay_title = "Table layout: horizontal gap sections only (no column split)"
        elif section_mode == "gap":
            overlay_title = "Horizontal gap sections (+ merged table rows)"
        elif section_mode == "vertical_only":
            overlay_title = "Vertical only: aligned column gap sections"
        elif column_meta and column_meta.get("split_horizontal_indices"):
            overlay_title = "HV combined: horizontal bands + aligned column splits"

    base = Image.open(image_path).convert("RGB")
    if args.show_lines and lines:
        line_draw = ImageDraw.Draw(base)
        for line in lines:
            b = line.content_box
            line_draw.rectangle(
                [(b.min_x, b.min_y), (b.max_x, b.max_y)],
                outline=(220, 40, 40),
                width=1,
            )
    section_dicts, layout_section_meta = section_layout_payload(sections)

    annotated_img = (
        draw_sections_overlay(
            base,
            section_dicts,
            banner_title=overlay_title,
            column_guide_x=column_guide_x,
            section_meta=layout_section_meta,
        )
        if sections
        else base
    )
    annotated_img.save(annotated)

    write_section_ranges_summary(
        ranges_out,
        sections,
        mode_label=section_mode,
        column_split_x=column_guide_x,
        section_roles=(column_meta or {}).get("section_roles"),
    )

    payload = {
        "image": str(image_path),
        "vision": str(vision_path) if vision_path else None,
        "lines_json": args.lines_json,
        "line_count": len(lines),
        "section_count": len(sections),
        "section_mode": section_mode,
        "gap_stats": stats.to_dict(),
        "section_meta": layout_section_meta,
        "sections": section_dicts,
    }
    if column_meta:
        payload["column_layout"] = column_meta
        if column_meta.get("page_layout"):
            payload["page_layout"] = column_meta["page_layout"]
    json_out.write_text(json.dumps(payload, indent=2))

    txt_blocks = []
    for section in sections:
        txt_blocks.append(f"=== Section {section.index} (lines {section.lines[0].index}-{section.lines[-1].index}) ===")
        txt_blocks.append(section.text)
    txt_out.write_text("\n\n".join(txt_blocks) + "\n")

    print(f"lines={len(lines)} sections={len(sections)} mode={section_mode}")
    print(f"gap mean={stats.mean:.1f}px median={stats.median:.1f}px threshold={stats.threshold:.1f}px")
    print(f"annotated: {annotated}")
    print(f"json:      {json_out}")
    print(f"text:      {txt_out}")
    print(f"ranges:    {ranges_out}")


if __name__ == "__main__":
    main()
