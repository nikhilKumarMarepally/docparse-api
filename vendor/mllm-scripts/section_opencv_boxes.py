"""Section OCR lines using OpenCV printed boxes on a text-redacted page image."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from form_box_detection import FormBox, detect_high_confidence_sections, _iou
from ocr_line_to_sections import _make_section, lines_to_sections_hv_combined, Section
from ocr_word_to_line_boxes import Box, Line, Word
from section_table_layout import analyze_vertical_partition

COVERAGE_BOX_ONLY = 0.80
COVERAGE_HYBRID_MIN = 0.40
SECTION_PAD = 6.0
COLUMN_PAIR_GAP_PX = 40.0
COLUMN_PAIR_MIN_PANEL_H_FRAC = 0.22
COLUMN_PAIR_Y_OVERLAP_FRAC = 0.85


def _lines_allow_opencv_lr_split(lines: list[Line], page_width: float) -> bool:
    """Parallel L/R regions only — not row-coupled label|amount grids (geometry)."""
    if not lines or page_width <= 0:
        return False
    part = analyze_vertical_partition(lines, page_width)
    return part.allows_column_split()


def _section_is_table_layout(section: Section, *, min_confidence: float = 0.72) -> bool:
    from section_table_layout import classify_section_layout

    lay = classify_section_layout(section.lines)
    return lay.layout_kind in ("table", "section_table") and lay.confidence >= min_confidence


def _section_allows_opencv_lr_split(section: Section, page_width: float) -> bool:
    if _section_is_table_layout(section):
        return False
    return _lines_allow_opencv_lr_split(section.lines, page_width)


def merge_form_boxes(
    primary: list[FormBox],
    secondary: list[FormBox],
    *,
    iou_thresh: float = 0.85,
) -> list[FormBox]:
    """Union box lists; prefer primary entries over near-duplicates in secondary."""
    merged = list(primary)
    for box in secondary:
        if any(_iou(box, kept) >= iou_thresh for kept in merged):
            continue
        merged.append(box)
    merged.sort(key=lambda b: (b.y, b.x))
    return merged


def detect_merged_high_confidence_sections(
    image_bgr: np.ndarray,
    words: list[Word],
    *,
    min_confidence: float = 0.9,
) -> list[FormBox]:
    """Detect on original + redacted so side panels survive text redaction."""
    redacted = redact_ocr_words(image_bgr, words)
    on_orig = detect_high_confidence_sections(image_bgr, min_confidence=min_confidence)
    on_red = detect_high_confidence_sections(redacted, min_confidence=min_confidence)
    return merge_form_boxes(on_orig, on_red)


def _form_column_pairs(
    boxes: list[FormBox],
    *,
    page_height: float,
    gap_px: float = COLUMN_PAIR_GAP_PX,
    min_panel_h_frac: float = COLUMN_PAIR_MIN_PANEL_H_FRAC,
    y_overlap_frac: float = COLUMN_PAIR_Y_OVERLAP_FRAC,
) -> list[tuple[FormBox, FormBox]]:
    """Tall adjacent side-by-side printed panels (form L|R columns, not chat bubbles)."""
    min_panel_h = page_height * min_panel_h_frac
    pairs: list[tuple[FormBox, FormBox]] = []
    ordered = sorted(boxes, key=lambda b: (b.x, b.y))
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            gap = right.x - left.x2
            if gap < 0 or gap > gap_px:
                continue
            lh = left.y2 - left.y
            rh = right.y2 - right.y
            min_h = min(lh, rh)
            if min_h < min_panel_h:
                continue
            y_overlap = min(left.y2, right.y2) - max(left.y, right.y)
            if y_overlap < min_h * y_overlap_frac:
                continue
            pairs.append((left, right))
    return pairs


def _section_spans_column_pair(
    section: Section,
    left_box: FormBox,
    right_box: FormBox,
) -> bool:
    sec_y0 = section.box.min_y
    sec_y1 = section.box.max_y
    pair_y0 = max(left_box.y, right_box.y)
    pair_y1 = min(left_box.y2, right_box.y2)
    overlap_y = min(sec_y1, pair_y1) - max(sec_y0, pair_y0)
    if overlap_y < (pair_y1 - pair_y0) * 0.35:
        return False
    split_x = (left_box.x2 + right_box.x) / 2.0
    left_lines = [ln for ln in section.lines if ln.box.centroid_x < split_x]
    right_lines = [ln for ln in section.lines if ln.box.centroid_x >= split_x]
    if not left_lines or not right_lines:
        return False
    if section.box.max_x <= left_box.x2 + 8:
        return False
    if section.box.min_x >= right_box.x - 8:
        return False
    return True


def _partition_lines_at_column_pair(
    lines: list[Line],
    left_box: FormBox,
    right_box: FormBox,
) -> list[list[Line]]:
    split_x = (left_box.x2 + right_box.x) / 2.0
    left, right = _partition_lines_at_lr_boundary(lines, split_x)
    groups: list[list[Line]] = []
    if left:
        groups.append(left)
    if right:
        groups.append(right)
    return groups


def _partition_lines_at_lr_boundary(
    lines: list[Line],
    split_x: float,
    *,
    line_index_start: int = 0,
) -> tuple[list[Line], list[Line]]:
    """Split lines at x using word centroids (full-width line boxes stay at page center)."""
    from ocr_word_to_line_boxes import line_from_words

    left_lines: list[Line] = []
    right_lines: list[Line] = []
    idx = line_index_start
    for ln in lines:
        left_words = [w for w in ln.words if w.box.centroid_x < split_x]
        right_words = [w for w in ln.words if w.box.centroid_x >= split_x]
        if left_words and not right_words:
            left_lines.append(ln)
        elif right_words and not left_words:
            right_lines.append(ln)
        elif left_words and right_words:
            lchunk = line_from_words(left_words, index=idx)
            if lchunk is not None:
                left_lines.append(lchunk)
                idx += 1
            rchunk = line_from_words(right_words, index=idx)
            if rchunk is not None:
                right_lines.append(rchunk)
                idx += 1
        elif ln.box.centroid_x < split_x:
            left_lines.append(ln)
        else:
            right_lines.append(ln)
    return left_lines, right_lines


def _section_has_lr_content_at_split(
    section: Section,
    left_box: FormBox,
    right_box: FormBox,
) -> bool:
    split_x = (left_box.x2 + right_box.x) / 2.0
    left, right = _partition_lines_at_lr_boundary(section.lines, split_x)
    return bool(left) and bool(right)


def opencv_boxes_enabled() -> bool:
    raw = os.environ.get("DOC_EXTRACT_OPENCV_BOXES", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def opencv_box_min_confidence() -> float:
    raw = os.environ.get("DOC_EXTRACT_OPENCV_BOX_MIN_CONF", "0.9").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.9


def redact_ocr_words(
    image_bgr: np.ndarray,
    words: list[Word],
    *,
    pad: int = 2,
) -> np.ndarray:
    """White-fill OCR word boxes so OpenCV sees printed frames, not glyph ink."""
    redacted = image_bgr.copy()
    h_img, w_img = redacted.shape[:2]
    for word in words:
        x0 = max(0, int(word.box.min_x) - pad)
        y0 = max(0, int(word.box.min_y) - pad)
        x1 = min(w_img, int(word.box.max_x) + pad)
        y1 = min(h_img, int(word.box.max_y) + pad)
        cv2.rectangle(redacted, (x0, y0), (x1, y1), (255, 255, 255), -1)
    return redacted


def _line_box_intersection_area(line: Line, box: FormBox) -> float:
    x0 = max(line.box.min_x, box.x)
    y0 = max(line.box.min_y, box.y)
    x1 = min(line.box.max_x, box.x2)
    y1 = min(line.box.max_y, box.y2)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _line_overlaps_box_vertically(line: Line, box: FormBox, slack: float = 4.0) -> bool:
    cy = line.box.centroid_y
    return box.y - slack <= cy <= box.y2 + slack


def _box_splits_ltr_band(lines: list[Line], box: FormBox, page_width: float) -> bool:
    """True when a horizontal text band spans L→R and this box sits in the middle."""
    band_slack = 8.0
    margin_frac = 0.12
    for anchor in lines:
        band = [
            ln
            for ln in lines
            if abs(ln.box.centroid_y - anchor.box.centroid_y) <= band_slack
        ]
        if len(band) < 2:
            continue
        min_x = min(ln.box.min_x for ln in band)
        max_x = max(ln.box.max_x for ln in band)
        span = max_x - min_x
        if span < page_width * 0.35:
            continue
        if min_x > page_width * margin_frac:
            continue
        if max_x < page_width * (1 - margin_frac):
            continue
        box_mid_x = (box.x + box.x2) / 2
        if min_x + span * 0.15 < box_mid_x < max_x - span * 0.15:
            left = any(ln.box.max_x < box.x for ln in band)
            right = any(ln.box.min_x > box.x2 for ln in band)
            if left and right:
                if _lines_allow_opencv_lr_split(band, page_width):
                    return True
    return False


def _line_in_box(line: Line, box: FormBox) -> bool:
    cx = line.box.centroid_x
    cy = line.box.centroid_y
    return box.x <= cx <= box.x2 and box.y <= cy <= box.y2


def _box_splits_horizontal_section(section: Section, box: FormBox, page_width: float) -> bool:
    """Wide horizontal section with an OpenCV frame between L→R content groups."""
    min_x = section.box.min_x
    max_x = section.box.max_x
    span = max_x - min_x
    margin_frac = 0.12
    if span < page_width * 0.35:
        return False
    if min_x > page_width * margin_frac:
        return False
    if max_x < page_width * (1 - margin_frac):
        return False
    if box.y2 < section.box.min_y - 4 or box.y > section.box.max_y + 4:
        return False
    box_mid_x = (box.x + box.x2) / 2
    if box_mid_x <= min_x + span * 0.15 or box_mid_x >= max_x - span * 0.15:
        return False
    outside_left = sum(1 for ln in section.lines if ln.box.centroid_x < box.x)
    outside_right = sum(1 for ln in section.lines if ln.box.centroid_x > box.x2)
    inside = sum(1 for ln in section.lines if box.x <= ln.box.centroid_x <= box.x2)
    if outside_left and outside_right:
        return True
    if outside_left and inside:
        return True
    if outside_right and inside:
        return True
    return False


def _box_splits_section_ltr(section: Section, box: FormBox, page_width: float) -> bool:
    """OpenCV frame in the middle of a wide horizontal section with L and R text."""
    if _box_splits_ltr_band(section.lines, box, page_width):
        return True
    smin_x = section.box.min_x
    smax_x = section.box.max_x
    span = smax_x - smin_x
    if span < page_width * 0.35:
        return False
    if box.y2 < section.box.min_y - 4 or box.y > section.box.max_y + 4:
        return False
    box_mid = (box.x + box.x2) / 2
    if box_mid <= smin_x + span * 0.12 or box_mid >= smax_x - span * 0.12:
        return False
    box_w = box.x2 - box.x
    if box_w > span * 0.55:
        return False
    left = any(ln.box.centroid_x < box.x for ln in section.lines)
    right = any(ln.box.centroid_x > box.x2 for ln in section.lines)
    if not (left and right):
        return False
    left_col = [ln for ln in section.lines if ln.box.centroid_x < smin_x + span * 0.42]
    right_col = [ln for ln in section.lines if ln.box.centroid_x > smin_x + span * 0.58]
    if len(left_col) >= 3 and len(right_col) >= 3:
        sec_h = max(1.0, section.box.max_y - section.box.min_y)
        left_h = max(ln.box.max_y for ln in left_col) - min(ln.box.min_y for ln in left_col)
        right_h = max(ln.box.max_y for ln in right_col) - min(ln.box.min_y for ln in right_col)
        if left_h > sec_h * 0.45 and right_h > sec_h * 0.45:
            return False
    return True


def _partition_lines_at_box(lines: list[Line], box: FormBox) -> list[list[Line]]:
    """Split lines into left / inside-box / right groups at an OpenCV frame."""
    left: list[Line] = []
    inside: list[Line] = []
    right: list[Line] = []
    for ln in lines:
        cx = ln.box.centroid_x
        if cx < box.x:
            left.append(ln)
        elif cx > box.x2:
            right.append(ln)
        else:
            inside.append(ln)
    groups: list[list[Line]] = []
    if left:
        groups.append(left)
    if inside:
        groups.append(inside)
    if right:
        groups.append(right)
    return groups


def split_sections_by_opencv_ltr_boxes(
    sections: list[Section],
    boxes: list[FormBox],
    *,
    page_width: float,
    page_height: float | None = None,
    pad: float = SECTION_PAD,
) -> tuple[list[Section], dict[str, Any]]:
    """
    Break a wide horizontal section only when a high-confidence OpenCV frame
    sits between parallel L/R lanes (not row-coupled label|value grids).
    """
    if not boxes:
        return _reindex_sections(list(sections)), {
            "split_section_count": 0,
            "ltr_split_box_count": 0,
            "column_pair_split_count": 0,
            "section_count": len(sections),
            "skipped": "no_boxes",
        }

    out: list[Section] = []
    split_section_count = 0
    ltr_split_box_count = 0
    for sec in sections:
        if not _section_allows_opencv_lr_split(sec, page_width):
            out.append(sec)
            continue
        split_boxes = [
            b
            for b in boxes
            if b.confidence >= 0.9 and _box_splits_ltr_band(sec.lines, b, page_width)
        ]
        ltr_split_box_count += len(split_boxes)
        if split_boxes:
            box = min(split_boxes, key=lambda b: b.x)
            groups = _partition_lines_at_box(sec.lines, box)
            if len(groups) > 1:
                split_section_count += 1
                for group in groups:
                    out.append(_make_section(len(out), group, sec.gap_above, pad))
                continue
        out.append(sec)
    meta: dict[str, Any] = {
        "split_section_count": split_section_count,
        "ltr_split_box_count": ltr_split_box_count,
        "column_pair_split_count": 0,
        "column_pair_count": 0,
        "section_count": len(out),
    }
    return _reindex_sections(out), meta


def _is_horizontal_section(section: Section, page_width: float) -> bool:
    if page_width <= 0:
        return False
    span = section.box.max_x - section.box.min_x
    return span >= page_width * 0.35


def _hi_conf_boxes_overlapping_section(
    section: Section,
    boxes: list[FormBox],
    *,
    min_confidence: float = 0.9,
    y_slack: float = 8.0,
) -> list[FormBox]:
    """OpenCV frames that overlap a section band vertically (confidence >= min)."""
    out: list[FormBox] = []
    for box in boxes:
        if box.confidence < min_confidence:
            continue
        if box.y2 < section.box.min_y - y_slack or box.y > section.box.max_y + y_slack:
            continue
        x0 = max(section.box.min_x, box.x)
        x1 = min(section.box.max_x, box.x2)
        if x1 - x0 < 24.0:
            continue
        out.append(box)
    return out


def _line_centroid_y(line: Line) -> float:
    if line.words:
        return sum(w.box.centroid_y for w in line.words) / len(line.words)
    return line.box.centroid_y


def _line_centroid_x(line: Line) -> float:
    if line.words:
        return sum(w.box.centroid_x for w in line.words) / len(line.words)
    return line.box.centroid_x


def _opencv_boxes_vertical_stack(boxes: list[FormBox]) -> list[FormBox]:
    """Side-by-side x overlap, sorted by y — stacked printed panels in one column."""
    if len(boxes) < 2:
        return []
    ordered = sorted(boxes, key=lambda b: (b.y, b.x))
    stack: list[FormBox] = []
    for box in ordered:
        if not stack:
            stack.append(box)
            continue
        ref = stack[0]
        x_overlap = min(ref.x2, box.x2) - max(ref.x, box.x)
        min_w = min(ref.x2 - ref.x, box.x2 - box.x)
        if min_w > 0 and x_overlap / min_w >= 0.45:
            stack.append(box)
        else:
            break
    return stack if len(stack) >= 2 else []


def _partition_lines_by_vertical_boxes(
    lines: list[Line],
    boxes: list[FormBox],
    *,
    y_slack: float = 10.0,
    x_slack: float = 40.0,
) -> list[list[Line]]:
    """Assign lines to vertically stacked OpenCV panels (word-centroid y/x)."""
    sorted_boxes = sorted(boxes, key=lambda b: b.y)
    groups: list[list[Line]] = [[] for _ in sorted_boxes]
    overflow: list[Line] = []
    for ln in lines:
        cy = _line_centroid_y(ln)
        cx = _line_centroid_x(ln)
        best_i: int | None = None
        best_score = -1.0
        for i, box in enumerate(sorted_boxes):
            if cx < box.x - x_slack or cx > box.x2 + x_slack:
                continue
            if cy < box.y - y_slack or cy > box.y2 + y_slack:
                continue
            y_overlap = min(cy + 7.0, box.y2) - max(cy - 7.0, box.y)
            if y_overlap > best_score:
                best_score = y_overlap
                best_i = i
        if best_i is not None:
            groups[best_i].append(ln)
        else:
            overflow.append(ln)
    if overflow:
        groups.append(overflow)
    return groups


def split_horizontal_sections_by_opencv_boxes(
    sections: list[Section],
    boxes: list[FormBox],
    *,
    page_width: float,
    page_height: float,
    min_confidence: float = 0.9,
    pad: float = SECTION_PAD,
) -> tuple[list[Section], dict[str, Any]]:
    """
    Break a wide horizontal section when one or more OpenCV boxes (>= min_confidence)
    overlap it — e.g. side-by-side printed panels (TILA | Insurance on RISC p1).
    """
    if not boxes:
        return _reindex_sections(list(sections)), {
            "split_section_count": 0,
            "column_pair_split_count": 0,
            "section_count": len(sections),
            "skipped": "no_boxes",
        }

    out: list[Section] = []
    split_section_count = 0
    column_pair_splits = 0
    vertical_stack_splits = 0

    for sec in sections:
        overlapping = _hi_conf_boxes_overlapping_section(
            sec, boxes, min_confidence=min_confidence
        )
        if not overlapping:
            out.append(sec)
            continue

        split_done = False

        stack = _opencv_boxes_vertical_stack(overlapping)
        if len(stack) >= 2:
            groups = _partition_lines_by_vertical_boxes(sec.lines, stack)
            non_empty = [g for g in groups if g]
            if len(non_empty) >= 2:
                split_section_count += 1
                vertical_stack_splits += 1
                gap = sec.gap_above
                for group in non_empty:
                    out.append(_make_section(len(out), group, gap, pad))
                    gap = None
                split_done = True

        if split_done:
            continue

        if not _is_horizontal_section(sec, page_width):
            out.append(sec)
            continue

        pairs = _form_column_pairs(overlapping, page_height=page_height)
        for left_box, right_box in pairs:
            if _section_has_lr_content_at_split(sec, left_box, right_box):
                groups = _partition_lines_at_column_pair(sec.lines, left_box, right_box)
                if len(groups) > 1:
                    split_section_count += 1
                    column_pair_splits += 1
                    gap = sec.gap_above
                    for group in groups:
                        out.append(_make_section(len(out), group, gap, pad))
                        gap = None
                    split_done = True
                    break

        if split_done:
            continue

        if len(overlapping) >= 2:
            left_box = min(overlapping, key=lambda b: b.x)
            right_box = max(overlapping, key=lambda b: b.x2)
            if right_box.x > left_box.x2 + 20.0:
                split_x = (left_box.x2 + right_box.x) / 2.0
                left_lines, right_lines = _partition_lines_at_lr_boundary(sec.lines, split_x)
                if left_lines and right_lines:
                    split_section_count += 1
                    gap = sec.gap_above
                    out.append(_make_section(len(out), left_lines, gap, pad))
                    out.append(_make_section(len(out), right_lines, None, pad))
                    continue

            groups, overflow = _partition_lines_by_boxes(
                sec.lines, overlapping, page_width=page_width
            )
            non_empty = [g for g in groups if g]
            if len(non_empty) > 1:
                split_section_count += 1
                gap = sec.gap_above
                for group in non_empty:
                    out.append(_make_section(len(out), group, gap, pad))
                    gap = None
                if overflow:
                    out.append(_make_section(len(out), overflow, None, pad))
                continue

        for box in sorted(overlapping, key=lambda b: b.x):
            if _box_splits_ltr_band(sec.lines, box, page_width):
                groups = _partition_lines_at_box(sec.lines, box)
                if len(groups) > 1:
                    split_section_count += 1
                    gap = sec.gap_above
                    for group in groups:
                        out.append(_make_section(len(out), group, gap, pad))
                        gap = None
                    split_done = True
                    break

        if not split_done:
            out.append(sec)

    meta: dict[str, Any] = {
        "split_section_count": split_section_count,
        "column_pair_split_count": column_pair_splits,
        "vertical_stack_split_count": vertical_stack_splits,
        "section_count": len(out),
        "min_confidence": min_confidence,
    }
    return _reindex_sections(out), meta


def _words_inside_box_count(line: Line, box: FormBox) -> int:
    return sum(
        1
        for w in line.words
        if box.x <= w.box.centroid_x <= box.x2 and box.y <= w.box.centroid_y <= box.y2
    )


def _section_with_opencv_bounds(
    section: Section,
    box: FormBox,
    *,
    pad: float,
    page_width: float,
    page_height: float,
) -> Section:
    bound = Box(
        max(0.0, box.x - pad),
        max(0.0, box.y - pad),
        min(float(page_width), box.x2 + pad),
        min(float(page_height), box.y2 + pad),
    )
    return Section(
        index=section.index,
        lines=section.lines,
        text=section.text,
        box=bound,
        gap_above=section.gap_above,
    )


def sections_from_opencv_boxes_direct(
    sections: list[Section],
    boxes: list[FormBox],
    *,
    page_width: float,
    page_height: float,
    min_confidence: float = 0.9,
    pad: float = SECTION_PAD,
) -> tuple[list[Section], dict[str, Any]]:
    """
    Snap section bounds to OpenCV printed frames (>= min_confidence) when those
    frames cover the page. Uncovered lines stay in their original sections.

    Never dump leftover lines into one page-wide section — that drops the
    prior boxes and hides text that already had a section.
    """
    hi_boxes = [b for b in boxes if b.confidence >= min_confidence]
    if not hi_boxes:
        return _reindex_sections(list(sections)), {
            "mode": "opencv_direct",
            "skipped": "no_hi_boxes",
            "section_count": len(sections),
        }

    all_lines: list[Line] = []
    seen: set[int] = set()
    for sec in sections:
        for ln in sec.lines:
            lid = id(ln)
            if lid not in seen:
                seen.add(lid)
                all_lines.append(ln)

    box_lines: list[list[Line]] = [[] for _ in hi_boxes]
    unassigned: list[Line] = []

    for ln in all_lines:
        best_i: int | None = None
        best_count = 0
        for i, box in enumerate(hi_boxes):
            count = _words_inside_box_count(ln, box)
            if count > best_count:
                best_count = count
                best_i = i
        if best_i is not None and best_count > 0:
            box_lines[best_i].append(ln)
        else:
            unassigned.append(ln)

    assigned_count = sum(len(group) for group in box_lines)
    coverage = assigned_count / len(all_lines) if all_lines else 0.0
    if coverage < COVERAGE_HYBRID_MIN:
        return _reindex_sections(list(sections)), {
            "mode": "opencv_direct",
            "skipped": "low_coverage",
            "min_confidence": min_confidence,
            "opencv_box_count": len(hi_boxes),
            "unassigned_lines": len(unassigned),
            "line_coverage": round(coverage, 4),
            "section_count": len(sections),
        }

    out: list[Section] = []
    boxes_used = 0
    order = sorted(range(len(hi_boxes)), key=lambda i: (hi_boxes[i].y, hi_boxes[i].x))
    for i in order:
        lines = box_lines[i]
        if not lines:
            continue
        boxes_used += 1
        lines = sorted(lines, key=lambda ln: ln.content_box.min_y)
        sec = _make_section(len(out), lines, None, pad)
        out.append(
            _section_with_opencv_bounds(
                sec,
                hi_boxes[i],
                pad=pad,
                page_width=page_width,
                page_height=page_height,
            )
        )

    assigned_ids = {id(ln) for group in box_lines for ln in group}
    for sec in sections:
        leftover = [ln for ln in sec.lines if id(ln) not in assigned_ids]
        if not leftover:
            continue
        leftover = sorted(leftover, key=lambda ln: ln.content_box.min_y)
        out.append(_make_section(len(out), leftover, sec.gap_above, pad))

    meta: dict[str, Any] = {
        "mode": "opencv_direct",
        "min_confidence": min_confidence,
        "opencv_box_count": len(hi_boxes),
        "sections_from_boxes": boxes_used,
        "unassigned_lines": len(unassigned),
        "line_coverage": round(coverage, 4),
        "overflow_policy": "keep_prior_sections",
        "section_count": len(out),
    }
    return _reindex_sections(out), meta


def _section_bounds_area(section: Section) -> float:
    b = section.box
    return max(0.0, (b.max_x - b.min_x) * (b.max_y - b.min_y))


def _opencv_box_covers_section(
    section: Section,
    box: FormBox,
    *,
    frac: float = 0.82,
) -> bool:
    """True when a high-confidence OpenCV frame already wraps this section."""
    sb = section.box
    x0 = max(sb.min_x, box.x)
    y0 = max(sb.min_y, box.y)
    x1 = min(sb.max_x, box.x2)
    y1 = min(sb.max_y, box.y2)
    if x1 <= x0 or y1 <= y0:
        return False
    inter = (x1 - x0) * (y1 - y0)
    area = _section_bounds_area(section)
    if area <= 0:
        return False
    return inter / area >= frac


def _is_row_coupled_lr_table_band(band: list[Line], page_width: float) -> bool:
    """Disjoint L/R clusters on shared rows — line-item grid, not parallel prose lanes."""
    if len(band) < 4 or page_width <= 0:
        return False
    split_x = page_width * 0.52
    left = [ln for ln in band if ln.box.centroid_x < split_x]
    right = [ln for ln in band if ln.box.centroid_x >= split_x]
    if len(left) < 3 or len(right) < 3:
        return False
    left_max_x = max(ln.box.max_x for ln in left)
    right_min_x = min(ln.box.min_x for ln in right)
    x_sep = right_min_x - left_max_x
    if x_sep < 72.0:
        return False
    coupled = 0
    for ll in left:
        cy = ll.box.centroid_y
        if any(abs(cy - rl.box.centroid_y) <= 14.0 for rl in right):
            coupled += 1
    return (coupled / len(left)) >= 0.38


def _word_x_two_cluster(
    lines: list[Line],
    *,
    page_width: float,
) -> tuple[float, float] | None:
    """Return (gutter_x, cluster_separation_px) from word x-centroids, or None."""
    margin = page_width * 0.06
    words = [
        w
        for ln in lines
        for w in ln.words
        if w.box.centroid_x >= margin and w.box.centroid_x <= page_width - margin
    ]
    if len(words) < 10:
        return None

    xs = [w.box.centroid_x for w in words]
    c0, c1 = xs[len(xs) // 4], xs[(3 * len(xs)) // 4]
    for _ in range(24):
        left_xs = [x for x in xs if abs(x - c0) <= abs(x - c1)]
        right_xs = [x for x in xs if abs(x - c0) > abs(x - c1)]
        if not left_xs or not right_xs:
            return None
        c0 = sum(left_xs) / len(left_xs)
        c1 = sum(right_xs) / len(right_xs)

    separation = abs(c1 - c0)
    if separation < max(48.0, page_width * 0.06):
        return None

    left_n = sum(1 for x in xs if x < (c0 + c1) * 0.5)
    right_n = len(xs) - left_n
    if min(left_n, right_n) / len(xs) < 0.15:
        return None

    return (c0 + c1) * 0.5, separation


def _word_spans_x_gutter(word: Word, gutter_x: float, gutter_pad: float) -> bool:
    """True when a word bbox straddles the gutter (must not be shard-split)."""
    return (
        word.box.min_x < gutter_x - gutter_pad
        and word.box.max_x > gutter_x + gutter_pad
    )


def _line_has_words_crossing_gutter(
    line: Line,
    gutter_x: float,
    gutter_pad: float,
) -> bool:
    return any(_word_spans_x_gutter(w, gutter_x, gutter_pad) for w in line.words)


def _line_spans_gutter_bbox(
    line: Line,
    gutter_x: float,
    gutter_pad: float,
) -> bool:
    box = line.content_box
    return box.min_x < gutter_x - gutter_pad and box.max_x > gutter_x + gutter_pad


def _line_preserve_in_prose_band(
    line: Line,
    gutter_x: float,
    gutter_pad: float,
    *,
    page_width: float,
    section: Section,
) -> bool:
    """Lines that must stay full-width prose — never shard-split at the gutter."""
    if _line_has_words_crossing_gutter(line, gutter_x, gutter_pad):
        return True
    box = line.content_box
    if page_width <= 0:
        return False
    height = box.height
    width_frac = box.width / page_width
    if height > 70.0:
        return True
    lower_anchor = section.box.min_y + section.box.height * 0.55
    if (
        width_frac >= 0.48
        and height <= 35.0
        and box.centroid_y < lower_anchor + 200.0
    ):
        return True
    return False


def _split_lines_at_x_gutter(
    lines: list[Line],
    gutter_x: float,
    *,
    pad: float,
    gap_above: float | None,
    gutter_pad: float | None = None,
    page_width: float = 0.0,
) -> tuple[list[Line], list[Line], list[Line]] | None:
    """
    Vertical peel at ``gutter_x``: words left of the gap → left column, else right.
    Lines whose words straddle the gutter stay full-width (e.g. centered URLs).
  Margin-strip tokens are never assigned to the left column.
    """
    from ocr_word_to_line_boxes import line_from_words

    if gutter_pad is None:
        gutter_pad = max(4.0, pad * 0.5)

    left_lines: list[Line] = []
    right_lines: list[Line] = []
    prose_lines: list[Line] = []
    idx = 0
    for line in lines:
        if _line_has_words_crossing_gutter(line, gutter_x, gutter_pad):
            prose_lines.append(line)
            continue
        left_candidates = [w for w in line.words if w.box.centroid_x < gutter_x]
        right_words = [w for w in line.words if w.box.centroid_x >= gutter_x]
        if left_candidates and right_words:
            body_left = [
                w for w in left_candidates
                if not _is_margin_strip_word(w, page_width)
            ]
            # Mixed row: drop margin OCR glued to chart/body shards; keep margin-only peels.
            left_words = body_left if body_left else left_candidates
        else:
            left_words = left_candidates
        if left_words:
            chunk = line_from_words(left_words, index=idx)
            if chunk is not None:
                left_lines.append(chunk)
                idx += 1
        if right_words:
            chunk = line_from_words(right_words, index=idx)
            if chunk is not None:
                right_lines.append(chunk)
                idx += 1

    if len(left_lines) < 2 or len(right_lines) < 2:
        return None
    return left_lines, right_lines, prose_lines


def _words_for_column_box(lines: list[Line], page_width: float) -> list[Word]:
    """Body words for bounds; fall back to all words for dedicated margin columns."""
    body = _body_words_from_lines(lines, page_width)
    if body:
        return body
    return [w for ln in lines for w in ln.words]


def _scrub_lr_split_lines(
    left_lines: list[Line],
    right_lines: list[Line],
    page_width: float,
) -> tuple[list[Line], list[Line]]:
    """Drop margin-strip words from wide chart shards, not dedicated margin columns."""
    from ocr_line_to_sections import _strip_margin_words_from_lines
    from ocr_word_to_line_boxes import line_from_words

    if page_width <= 0:
        return left_lines, right_lines

    margin_hi = page_width * 0.14
    wide_hi = page_width * 0.20

    def scrub_mixed(lines: list[Line]) -> list[Line]:
        out: list[Line] = []
        idx = 0
        for ln in lines:
            has_body = any(w.box.max_x >= margin_hi for w in ln.words)
            has_margin = any(w.box.max_x < margin_hi for w in ln.words)
            if has_body and has_margin and ln.content_box.max_x >= wide_hi:
                stripped = _strip_margin_words_from_lines([ln], page_width)
                out.extend(stripped)
                idx += len(stripped)
                continue
            chunk = line_from_words(ln.words, index=idx)
            if chunk is not None:
                out.append(chunk)
                idx += 1
        return out

    return scrub_mixed(left_lines), scrub_mixed(right_lines)


def _assemble_lr_gutter_sections(
    prose_lines: list[Line],
    left_lines: list[Line],
    right_lines: list[Line],
    *,
    sec: Section,
    page_width: float,
    pad: float,
    gutter_x: float,
    gutter_pad: float,
) -> list[Section] | None:
    """Build L/R (+ optional prose) sections with word-tight boxes and vertical peels."""
    left_lines, right_lines = _scrub_lr_split_lines(left_lines, right_lines, page_width)
    if len(left_lines) < 2 or len(right_lines) < 2:
        return None

    prose_box: Box | None = None
    if prose_lines:
        prose_words = _body_words_from_lines(prose_lines, page_width)
        prose_box = _box_from_words(prose_words, pad=pad)

    left_box = _box_from_words(_words_for_column_box(left_lines, page_width), pad=pad)
    right_box = _box_from_words(_words_for_column_box(right_lines, page_width), pad=pad)
    if left_box is None or right_box is None:
        return None

    prose_box, left_box, right_box = _enforce_disjoint_section_boxes(
        prose_box,
        left_box,
        right_box,
        pad=pad,
        gutter_x=gutter_x,
        gutter_pad=gutter_pad,
    )

    fitted = _fit_lr_gutter_boxes(
        prose_box,
        left_box,
        right_box,
        prose_lines=prose_lines,
        left_lines=left_lines,
        right_lines=right_lines,
        page_width=page_width,
        pad=pad,
        gutter_x=gutter_x,
        gutter_pad=gutter_pad,
    )
    if fitted is None:
        return None
    prose_box, left_box, right_box, prose_lines, left_lines, right_lines = fitted

    out: list[Section] = []
    if prose_lines and prose_box is not None:
        out.append(
            _make_section_with_box(
                len(out),
                sorted(prose_lines, key=lambda ln: ln.content_box.min_y),
                sec.gap_above,
                prose_box,
            )
        )
    gap_for_lr = sec.gap_above if not prose_lines else None
    if left_lines and left_box is not None:
        out.append(
            _make_section_with_box(
                len(out),
                sorted(left_lines, key=lambda ln: ln.content_box.min_y),
                gap_for_lr,
                left_box,
            )
        )
    if right_lines and right_box is not None:
        out.append(
            _make_section_with_box(
                len(out),
                sorted(right_lines, key=lambda ln: ln.content_box.min_y),
                None,
                right_box,
            )
        )
    return out if len(out) > 1 else None


def _lr_gap_typical_multiplier() -> float:
    raw = os.getenv("SECTION_LR_GAP_TYPICAL_MULTIPLIER", "30")
    try:
        return max(10.0, float(raw))
    except ValueError:
        return 30.0


def _page_typical_word_gap(lines: list[Line]) -> float:
    """IQR-trimmed mean inter-word gap — typical spacing with outliers removed."""
    import statistics

    from ocr_word_to_line_boxes import group_into_rows

    words = [w for ln in lines for w in ln.words]
    if not words:
        return 12.0
    gaps: list[float] = []
    for row in group_into_rows(words):
        ordered = sorted(row, key=lambda w: w.box.min_x)
        for i in range(1, len(ordered)):
            gap = ordered[i].box.min_x - ordered[i - 1].box.max_x
            if gap > 0:
                gaps.append(gap)
    if not gaps:
        return 12.0
    gaps.sort()
    q1 = gaps[len(gaps) // 4]
    q3 = gaps[(3 * len(gaps)) // 4]
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    trimmed = [g for g in gaps if lo <= g <= hi]
    if trimmed:
        return statistics.mean(trimmed)
    return statistics.median(gaps)


def _section_highest_lr_whitespace(lines: list[Line]) -> float:
    """Max whitespace between left and right word groups in any OCR row."""
    from ocr_word_to_line_boxes import group_into_rows

    words = [w for ln in lines for w in ln.words]
    best = 0.0
    for row in group_into_rows(words):
        ordered = sorted(row, key=lambda w: w.box.min_x)
        for split_i in range(len(ordered) - 1):
            left_max = max(w.box.max_x for w in ordered[: split_i + 1])
            right_min = min(w.box.min_x for w in ordered[split_i + 1 :])
            best = max(best, right_min - left_max)
    return best


def _detect_lr_whitespace_gutter(
    lines: list[Line],
    *,
    page_width: float,
    split_threshold: float,
    min_qualifying_rows: int = 2,
) -> tuple[float, float] | None:
    """
    Find a vertical gutter where OCR rows have whitespace between left and right
    word groups far wider than normal word spacing (includes margin tokens).
    """
    from ocr_word_to_line_boxes import group_into_rows

    words = [w for ln in lines for w in ln.words]
    if len(words) < 8:
        return None

    gutter_xs: list[float] = []
    gap_sizes: list[float] = []
    for row in group_into_rows(words):
        ordered = sorted(row, key=lambda w: w.box.min_x)
        if len(ordered) < 2:
            continue
        best_gap = 0.0
        best_i = 0
        for i in range(len(ordered) - 1):
            gap = ordered[i + 1].box.min_x - ordered[i].box.max_x
            if gap > best_gap:
                best_gap = gap
                best_i = i
        if best_gap < split_threshold:
            continue
        left_max = max(w.box.max_x for w in ordered[: best_i + 1])
        right_min = min(w.box.min_x for w in ordered[best_i + 1 :])
        whitespace = right_min - left_max
        if whitespace < split_threshold:
            continue
        gutter_xs.append((left_max + right_min) * 0.5)
        gap_sizes.append(whitespace)

    if len(gutter_xs) < min_qualifying_rows:
        return None
    gutter_xs.sort()
    gutter_x = gutter_xs[len(gutter_xs) // 2]
    return gutter_x, max(gap_sizes)


def _shard_lines_at_x_gutter(
    lines: list[Line],
    gutter_x: float,
    *,
    pad: float,
    gap_above: float | None,
    gutter_pad: float | None = None,
) -> tuple[list[Line], list[Line]] | None:
    """Shard every line at gutter_x; words left of gutter → left column, else right."""
    from ocr_word_to_line_boxes import line_from_words

    if gutter_pad is None:
        gutter_pad = max(4.0, pad * 0.5)

    left_lines: list[Line] = []
    right_lines: list[Line] = []
    idx = 0
    for line in lines:
        left_words = [
            w for w in line.words if w.box.centroid_x < gutter_x - gutter_pad * 0.5
        ]
        right_words = [
            w for w in line.words if w.box.centroid_x >= gutter_x - gutter_pad * 0.5
        ]
        if left_words:
            chunk = line_from_words(left_words, index=idx)
            if chunk is not None:
                left_lines.append(chunk)
                idx += 1
        if right_words:
            chunk = line_from_words(right_words, index=idx)
            if chunk is not None:
                right_lines.append(chunk)
                idx += 1

    if len(left_lines) < 2 or len(right_lines) < 2:
        return None
    return left_lines, right_lines


def _split_section_at_lr_whitespace(
    sec: Section,
    *,
    page_width: float,
    split_threshold: float,
    pad: float,
    min_lr_gap: float,
) -> list[Section]:
    """
    Split at a significant horizontal whitespace gap between left and right text.
    Finds the widest repeated gutter across OCR rows, then peels vertically at that x.
    """
    if page_width <= 0 or len(sec.lines) < 4:
        return [sec]

    if _is_row_coupled_lr_table_band(sec.lines, page_width):
        return [sec]

    body_lines = [
        ln for ln in sec.lines if not _is_margin_strip_line(ln, page_width)
    ]
    if len(body_lines) < 4:
        return [sec]

    gutter_info = _detect_lr_whitespace_gutter(
        body_lines,
        page_width=page_width,
        split_threshold=split_threshold,
    )
    if gutter_info is None:
        return [sec]

    gutter_x, gap_px = gutter_info
    if gap_px < min_lr_gap:
        return [sec]

    gutter_pad = max(4.0, page_width * 0.006)
    split = _split_lines_at_x_gutter(
        body_lines,
        gutter_x,
        pad=pad,
        gap_above=sec.gap_above,
        gutter_pad=gutter_pad,
        page_width=page_width,
    )
    if split is None:
        return [sec]

    left_lines, right_lines, prose_lines = split
    assembled = _assemble_lr_gutter_sections(
        prose_lines,
        left_lines,
        right_lines,
        sec=sec,
        page_width=page_width,
        pad=pad,
        gutter_x=gutter_x,
        gutter_pad=gutter_pad,
    )
    if assembled is None:
        return [sec]
    return assembled


def _page_row_word_gap_baseline(lines: list[Line]) -> float:
    """Median positive horizontal gap between words within OCR rows on this page."""
    from ocr_word_to_line_boxes import group_into_rows

    words = [w for ln in lines for w in ln.words]
    if not words:
        return 12.0
    gaps: list[float] = []
    for row in group_into_rows(words):
        ordered = sorted(row, key=lambda w: w.box.min_x)
        for i in range(1, len(ordered)):
            gap = ordered[i].box.min_x - ordered[i - 1].box.max_x
            if gap > 0:
                gaps.append(gap)
    if not gaps:
        return 12.0
    gaps.sort()
    return gaps[len(gaps) // 2]


def _page_x_gap_baseline(body_lines: list[Line], *, page_width: float) -> float:
    """Typical horizontal separation between x-clusters on this page."""
    if len(body_lines) < 4 or page_width <= 0:
        return max(24.0, page_width * 0.04)

    ordered = sorted(body_lines, key=lambda ln: ln.content_box.min_y)
    bands: list[list[Line]] = []
    for ln in ordered:
        cy = ln.content_box.centroid_y
        placed = False
        for band in bands:
            if abs(cy - band[0].content_box.centroid_y) <= 40.0:
                band.append(ln)
                placed = True
                break
        if not placed:
            bands.append([ln])

    separations: list[float] = []
    for band in bands:
        pair = _word_x_two_cluster(band, page_width=page_width)
        if pair is not None and pair[1] < 100.0:
            separations.append(pair[1])

    if not separations:
        return 32.0
    separations.sort()
    return separations[len(separations) // 2]


def _merge_x_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda t: t[0])
    merged: list[tuple[float, float]] = [ordered[0]]
    for x0, x1 in ordered[1:]:
        last = merged[-1]
        if x0 <= last[1] + 2.0:
            merged[-1] = (last[0], max(last[1], x1))
        else:
            merged.append((x0, x1))
    return merged


def _largest_x_gap(intervals: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Return (gutter_x, gap_px) for the widest whitespace between x-intervals."""
    if len(intervals) < 2:
        return None
    best_gap = 0.0
    best_gutter = 0.0
    for i in range(len(intervals) - 1):
        gap = intervals[i + 1][0] - intervals[i][1]
        if gap > best_gap:
            best_gap = gap
            best_gutter = (intervals[i][1] + intervals[i + 1][0]) * 0.5
    if best_gap <= 0:
        return None
    return best_gutter, best_gap


def _cluster_lines_into_y_bands(lines: list[Line], *, band_gap_px: float = 48.0) -> list[list[Line]]:
    ordered = sorted(lines, key=lambda ln: ln.content_box.min_y)
    if not ordered:
        return []
    bands: list[list[Line]] = []
    current: list[Line] = [ordered[0]]
    for line in ordered[1:]:
        cy = line.content_box.centroid_y
        ref = current[-1].content_box.centroid_y
        if cy - ref <= band_gap_px:
            current.append(line)
        else:
            bands.append(current)
            current = [line]
    if current:
        bands.append(current)
    return bands


def _is_margin_strip_line(line: Line, page_width: float) -> bool:
    box = line.content_box
    if page_width <= 0:
        return False
    return box.max_x < page_width * 0.14 and box.width < page_width * 0.12


def _is_margin_strip_word(word: Word, page_width: float) -> bool:
    if page_width <= 0:
        return False
    return word.box.max_x < page_width * 0.14


def _body_words_from_lines(lines: list[Line], page_width: float) -> list[Word]:
    return [
        w
        for ln in lines
        for w in ln.words
        if not _is_margin_strip_word(w, page_width)
    ]


def _cluster_words_into_y_bands(
    words: list[Word],
    *,
    band_gap_px: float = 22.0,
) -> list[list[Word]]:
    ordered = sorted(words, key=lambda w: w.box.centroid_y)
    if not ordered:
        return []
    bands: list[list[Word]] = []
    current: list[Word] = [ordered[0]]
    for word in ordered[1:]:
        cy = word.box.centroid_y
        ref = current[-1].box.centroid_y
        if cy - ref <= band_gap_px:
            current.append(word)
        else:
            bands.append(current)
            current = [word]
    if current:
        bands.append(current)
    return bands


def _lr_cluster_gap(words: list[Word], page_width: float) -> float | None:
    """Whitespace between left/right word clusters in a y-band."""
    margin = page_width * 0.12
    body = [w for w in words if w.box.min_x > margin]
    left_words = [w for w in body if w.box.centroid_x < page_width * 0.44]
    right_words = [w for w in body if w.box.centroid_x > page_width * 0.52]
    if not left_words or not right_words:
        return None
    left_max = max(w.box.max_x for w in left_words)
    right_min = min(w.box.min_x for w in right_words)
    return right_min - left_max


def _line_has_lr_structure(
    line: Line,
    page_width: float,
    *,
    min_centroid_y: float = 0.0,
) -> bool:
    if line.content_box.centroid_y < min_centroid_y:
        return False
    gap = _lr_cluster_gap(_body_words_from_lines([line], page_width), page_width)
    return gap is not None and gap >= 20.0


def _detect_chart_figure_y0(
    content_lines: list[Line],
    sec: Section,
    page_width: float,
) -> float | None:
    """
    Y where chart shards begin. Prose lines stay above the first L/R chart label
    band; centered link rows below that band still belong to prose.
    """
    words = _body_words_from_lines(content_lines, page_width)
    bands = _cluster_words_into_y_bands(words)
    sec_mid_y = sec.box.min_y + sec.box.height * 0.5

    label_band_y: float | None = None
    lr_gap_band_y: float | None = None

    for band in bands:
        band_min_y = min(w.box.min_y for w in band)
        band_cy = sum(w.box.centroid_y for w in band) / len(band)
        if band_cy < sec_mid_y:
            continue

        gap = _lr_cluster_gap(band, page_width)
        if gap is None:
            continue

        if label_band_y is None:
            label_band_y = band_min_y

        if gap >= 40.0:
            lr_gap_band_y = band_min_y
            break

    if label_band_y is None:
        return None

    chart_figure_y0 = max(sec.box.min_y, label_band_y)

    # Link rows between the label band and chart body — not side-column numeric shards.
    prose_tail_max = sec_mid_y
    tail_upper = lr_gap_band_y if lr_gap_band_y is not None else label_band_y + 120.0
    page_cx = page_width * 0.5
    for ln in content_lines:
        box = ln.content_box
        cy = box.centroid_y
        if cy < sec_mid_y:
            continue
        if cy >= tail_upper:
            continue
        if _line_has_lr_structure(ln, page_width, min_centroid_y=sec_mid_y):
            continue
        width_frac = box.width / page_width if page_width > 0 else 0.0
        centered = abs(box.centroid_x - page_cx) < page_width * 0.12
        if width_frac < 0.35 and not centered:
            continue
        prose_tail_max = max(prose_tail_max, box.max_y)

    return max(chart_figure_y0, prose_tail_max + 4.0)


def _section_has_prose_above_chart(
    content_lines: list[Line],
    chart_y0: float,
    page_width: float,
) -> bool:
    """True when full-width lines sit above the chart band (authors/abstract + charts)."""
    if page_width <= 0:
        return False
    for ln in content_lines:
        if ln.content_box.centroid_y >= chart_y0 - 20.0:
            continue
        if ln.content_box.width / page_width >= 0.45:
            return True
    return False


def _partition_line_words_by_y(
    line: Line,
    y0: float,
) -> tuple[list[Word], list[Word]]:
    above = [w for w in line.words if w.box.centroid_y < y0]
    below = [w for w in line.words if w.box.centroid_y >= y0]
    return above, below


def _estimate_lr_gutter(words: list[Word], page_width: float) -> float | None:
    """Midpoint between left/right word clusters in the chart band."""
    margin = page_width * 0.12
    body = [w for w in words if w.box.min_x > margin]
    left_words = [w for w in body if w.box.centroid_x < page_width * 0.44]
    right_words = [w for w in body if w.box.centroid_x > page_width * 0.52]
    if not left_words or not right_words:
        return None

    left_cx = sum(w.box.centroid_x for w in left_words) / len(left_words)
    right_cx = sum(w.box.centroid_x for w in right_words) / len(right_words)
    if right_cx - left_cx < page_width * 0.22:
        return None

    left_max = max(w.box.max_x for w in left_words)
    right_min = min(w.box.min_x for w in right_words)
    gutter_x = (left_cx + right_cx) * 0.5
    if right_min - left_max >= 40.0:
        gutter_x = (left_max + right_min) * 0.5
    return gutter_x


def _box_from_words(words: list[Word], *, pad: float) -> Box | None:
    if not words:
        return None
    min_x = min(w.box.min_x for w in words) - pad
    min_y = min(w.box.min_y for w in words) - pad
    max_x = max(w.box.max_x for w in words) + pad
    max_y = max(w.box.max_y for w in words) + pad
    return Box(min_x, min_y, max_x, max_y)


def _boxes_overlap_y(a: Box, b: Box) -> bool:
    return a.min_y < b.max_y and b.min_y < a.max_y


def _boxes_overlap_x(a: Box, b: Box) -> bool:
    return a.min_x < b.max_x and b.min_x < a.max_x


def _box_covers_lines(box: Box, lines: list[Line], *, pad: float) -> bool:
    """True when ``box`` fully contains every assigned word box (+ pad)."""
    from ocr_line_to_sections import _bounds_covering_lines

    if not lines:
        return True
    text_box = _bounds_covering_lines(lines, pad=pad)
    eps = 0.5
    return (
        box.min_x <= text_box.min_x + eps
        and box.min_y <= text_box.min_y + eps
        and box.max_x >= text_box.max_x - eps
        and box.max_y >= text_box.max_y - eps
    )


def _lr_column_box_usable(
    box: Box | None,
    lines: list[Line],
    sibling_boxes: list[Box | None],
    *,
    pad: float,
    gutter_x: float,
    gutter_pad: float,
    side: str,
) -> Box | None:
    """
    Return a box that covers assigned text, or None when gutter clipping would cut glyphs.
    """
    from ocr_line_to_sections import _bounds_covering_lines

    if box is None or not lines:
        return box
    if _box_covers_lines(box, lines, pad=pad):
        return box

    expanded = _bounds_covering_lines(lines, pad=pad)
    for sib in sibling_boxes:
        if sib is None:
            continue
        if _boxes_overlap_x(expanded, sib):
            return None

    if side == "left" and expanded.max_x > gutter_x - gutter_pad / 2:
        if any(
            sib is not None and sib.min_x < gutter_x + gutter_pad
            for sib in sibling_boxes
        ):
            return None
    if side == "right" and expanded.min_x < gutter_x + gutter_pad / 2:
        if any(
            sib is not None and sib.max_x > gutter_x - gutter_pad
            for sib in sibling_boxes
        ):
            return None
    return expanded


def _fit_lr_gutter_boxes(
    prose_box: Box | None,
    left_box: Box | None,
    right_box: Box | None,
    *,
    prose_lines: list[Line],
    left_lines: list[Line],
    right_lines: list[Line],
    page_width: float,
    pad: float,
    gutter_x: float,
    gutter_pad: float,
) -> tuple[Box | None, Box | None, Box | None, list[Line], list[Line], list[Line]] | None:
    """
    Ensure L/R column boxes cover assigned text; drop columns that would cut glyphs.
    Orphan lines from dropped columns merge into prose when possible.
    """
    left_box = _lr_column_box_usable(
        left_box,
        left_lines,
        [right_box, prose_box],
        pad=pad,
        gutter_x=gutter_x,
        gutter_pad=gutter_pad,
        side="left",
    )
    right_box = _lr_column_box_usable(
        right_box,
        right_lines,
        [left_box, prose_box],
        pad=pad,
        gutter_x=gutter_x,
        gutter_pad=gutter_pad,
        side="right",
    )

    orphan_left = left_box is None and bool(left_lines)
    orphan_right = right_box is None and bool(right_lines)
    if orphan_left and orphan_right:
        return None

    if orphan_left:
        if prose_lines or prose_box is not None:
            prose_lines = list(prose_lines) + left_lines
            left_lines = []
            prose_words = _body_words_from_lines(prose_lines, page_width)
            prose_box = _box_from_words(
                prose_words or _flatten_words(prose_lines),
                pad=pad,
            )
            left_box = None
        else:
            return None

    if orphan_right:
        if prose_lines or prose_box is not None:
            prose_lines = list(prose_lines) + right_lines
            right_lines = []
            prose_words = _body_words_from_lines(prose_lines, page_width)
            prose_box = _box_from_words(
                prose_words or _flatten_words(prose_lines),
                pad=pad,
            )
            right_box = None
        else:
            return None

    if left_box is None and right_box is None:
        return None

    if prose_lines and prose_box is not None and not _box_covers_lines(prose_box, prose_lines, pad=pad):
        from ocr_line_to_sections import _bounds_covering_lines

        prose_box = _bounds_covering_lines(prose_lines, pad=pad)

    return prose_box, left_box, right_box, prose_lines, left_lines, right_lines


def _flatten_words(lines: list[Line]) -> list[Word]:
    return [w for ln in lines for w in ln.words]


def _enforce_disjoint_section_boxes(
    prose_box: Box | None,
    left_box: Box | None,
    right_box: Box | None,
    *,
    pad: float,
    gutter_x: float,
    gutter_pad: float,
) -> tuple[Box | None, Box | None, Box | None]:
    """Ensure prose/chart section boxes do not overlap in y or x."""
    if left_box is not None:
        left_box = Box(
            left_box.min_x,
            left_box.min_y,
            min(left_box.max_x, gutter_x - gutter_pad),
            left_box.max_y,
        )
    if right_box is not None:
        right_box = Box(
            max(right_box.min_x, gutter_x + gutter_pad),
            right_box.min_y,
            right_box.max_x,
            right_box.max_y,
        )

    if prose_box is not None and left_box is not None and _boxes_overlap_y(prose_box, left_box):
        left_box = Box(
            left_box.min_x,
            prose_box.max_y + pad,
            left_box.max_x,
            left_box.max_y,
        )

    if prose_box is not None and right_box is not None and _boxes_overlap_y(prose_box, right_box):
        split_y = prose_box.max_y + pad
        if left_box is not None:
            split_y = max(split_y, left_box.min_y)
        right_box = Box(
            right_box.min_x,
            split_y,
            right_box.max_x,
            right_box.max_y,
        )

    if left_box is not None and right_box is not None and _boxes_overlap_x(left_box, right_box):
        left_box = Box(
            left_box.min_x,
            left_box.min_y,
            gutter_x - gutter_pad,
            left_box.max_y,
        )
        right_box = Box(
            gutter_x + gutter_pad,
            right_box.min_y,
            right_box.max_x,
            right_box.max_y,
        )

    return prose_box, left_box, right_box


def _make_section_with_box(
    index: int,
    lines: list[Line],
    gap_above: float | None,
    box: Box,
) -> Section:
    text = "\n".join(line.text for line in lines if line.text.strip())
    return Section(index=index, lines=lines, text=text, box=box, gap_above=gap_above)


def _split_section_prose_and_chart_columns(
    sec: Section,
    *,
    page_width: float,
    pad: float,
) -> list[Section]:
    """
    Keep full-width prose (authors, abstract, links) in the upper band; split the
    lower chart band at the x-gutter. Never shard-split lines whose words cross the gutter.
    """
    if sec.box.height < 200.0 or page_width <= 0:
        return [sec]

    content_lines = [
        ln
        for ln in sec.lines
        if not _is_margin_strip_line(ln, page_width)
        and ln.content_box.max_x >= page_width * 0.14
    ]
    gutter_pad = max(4.0, page_width * 0.006)

    chart_figure_y0 = _detect_chart_figure_y0(content_lines, sec, page_width)
    if chart_figure_y0 is None:
        return [sec]

    sec_mid_y = sec.box.min_y + sec.box.height * 0.5
    from ocr_word_to_line_boxes import line_from_words

    prose_lines: list[Line] = []
    chart_line_chunks: list[Line] = []
    line_idx = 0

    for ln in content_lines:
        if ln.content_box.centroid_y < chart_figure_y0:
            prose_lines.append(ln)
            continue
        if _line_has_lr_structure(ln, page_width, min_centroid_y=sec_mid_y):
            if ln.content_box.min_y < chart_figure_y0:
                above, below = _partition_line_words_by_y(ln, chart_figure_y0)
                if above:
                    chunk = line_from_words(above, index=line_idx)
                    if chunk is not None:
                        prose_lines.append(chunk)
                        line_idx += 1
                if below:
                    chunk = line_from_words(below, index=line_idx)
                    if chunk is not None:
                        chart_line_chunks.append(chunk)
                        line_idx += 1
            else:
                chart_line_chunks.append(ln)
            continue
        above, below = _partition_line_words_by_y(ln, chart_figure_y0)
        if above:
            chunk = line_from_words(above, index=line_idx)
            if chunk is not None:
                prose_lines.append(chunk)
                line_idx += 1
        if below:
            chunk = line_from_words(below, index=line_idx)
            if chunk is not None:
                chart_line_chunks.append(chunk)
                line_idx += 1

    chart_words = _body_words_from_lines(chart_line_chunks, page_width)
    gutter_x = _estimate_lr_gutter(chart_words, page_width)
    if gutter_x is None:
        return [sec]

    if len(chart_line_chunks) < 2:
        return [sec]

    split = _split_lines_at_x_gutter(
        chart_line_chunks,
        gutter_x,
        pad=pad,
        gap_above=None,
        gutter_pad=gutter_pad,
        page_width=page_width,
    )
    if split is None:
        return [sec]

    left_chart, right_chart, prose_overflow = split
    prose_ids = {id(ln) for ln in prose_lines}
    for ln in prose_overflow:
        if id(ln) not in prose_ids:
            prose_lines.append(ln)
            prose_ids.add(id(ln))

    if not left_chart or not right_chart:
        return [sec]

    left_words = _body_words_from_lines(left_chart, page_width)
    right_words = _body_words_from_lines(right_chart, page_width)
    if not left_words or not right_words:
        return [sec]

    prose_box: Box | None = None
    if prose_lines:
        prose_words = _body_words_from_lines(prose_lines, page_width)
        prose_box = _box_from_words(prose_words, pad=pad)
        if prose_box is None:
            prose_box = _make_section(0, prose_lines, sec.gap_above, pad).box

    left_box = _box_from_words(left_words, pad=pad)
    right_box = _box_from_words(right_words, pad=pad)
    if left_box is None or right_box is None:
        return [sec]

    prose_box, left_box, right_box = _enforce_disjoint_section_boxes(
        prose_box,
        left_box,
        right_box,
        pad=pad,
        gutter_x=gutter_x,
        gutter_pad=gutter_pad,
    )

    out: list[Section] = []
    if prose_lines and prose_box is not None:
        out.append(
            _make_section_with_box(
                0,
                sorted(prose_lines, key=lambda ln: ln.content_box.min_y),
                sec.gap_above,
                prose_box,
            )
        )

    out.append(
        _make_section_with_box(
            len(out),
            sorted(left_chart, key=lambda ln: ln.content_box.min_y),
            sec.gap_above if not prose_lines else None,
            left_box,
        )
    )
    out.append(
        _make_section_with_box(
            len(out),
            sorted(right_chart, key=lambda ln: ln.content_box.min_y),
            None,
            right_box,
        )
    )
    return out


def _split_section_by_line_x_columns(
    sec: Section,
    *,
    page_width: float,
    pad: float,
) -> list[Section]:
    """
    Split when some lines sit clearly left vs right (x-coordinates), while
    full-width prose lines (authors, abstract) stay in one band.
    """
    if page_width <= 0 or len(sec.lines) < 4:
        return [sec]

    left_only: list[Line] = []
    right_only: list[Line] = []
    full_width: list[Line] = []

    for ln in sec.lines:
        box = ln.content_box
        width_frac = box.width / page_width
        cx = box.centroid_x
        # Full-width prose / author rows span most of the page.
        if width_frac >= 0.52 or (box.min_x < page_width * 0.22 and box.max_x > page_width * 0.62):
            full_width.append(ln)
        elif box.max_x <= page_width * 0.46 and cx < page_width * 0.42:
            left_only.append(ln)
        elif box.min_x >= page_width * 0.50 and cx > page_width * 0.52:
            right_only.append(ln)
        else:
            full_width.append(ln)

    if not left_only or not right_only:
        return [sec]

    left_max = max(ln.content_box.max_x for ln in left_only)
    right_min = min(ln.content_box.min_x for ln in right_only)
    if right_min - left_max < 72.0:
        return [sec]

    out: list[Section] = []
    if full_width:
        out.append(
            _make_section(
                0,
                sorted(full_width, key=lambda ln: ln.content_box.min_y),
                sec.gap_above,
                pad,
            )
        )
    out.append(
        _make_section(
            len(out),
            sorted(left_only, key=lambda ln: ln.content_box.min_y),
            sec.gap_above if not full_width else None,
            pad,
        )
    )
    out.append(
        _make_section(
            len(out),
            sorted(right_only, key=lambda ln: ln.content_box.min_y),
            None,
            pad,
        )
    )
    return out


def _full_width_line_fraction(lines: list[Line], page_width: float) -> float:
    if not lines or page_width <= 0:
        return 0.0
    wide = sum(1 for ln in lines if ln.content_box.width / page_width >= 0.52)
    return wide / len(lines)


def _split_lines_by_vertical_gaps(
    lines: list[Line],
    *,
    pad: float,
    gap_above: float | None,
    page_width: float,
) -> list[Section]:
    """Split a column shard when lines form separate vertical bands."""
    if not lines:
        return []
    band_gap_px = max(72.0, page_width * 0.05)
    bands = _cluster_lines_into_y_bands(lines, band_gap_px=band_gap_px)
    if len(bands) <= 1:
        return [
            _make_section(
                0,
                sorted(lines, key=lambda ln: ln.content_box.min_y),
                gap_above,
                pad,
            )
        ]
    out: list[Section] = []
    for i, band in enumerate(bands):
        out.append(
            _make_section(
                i,
                sorted(band, key=lambda ln: ln.content_box.min_y),
                gap_above if i == 0 else None,
                pad,
            )
        )
    return out


def _section_lr_gutter(
    lines: list[Line],
    *,
    page_width: float,
) -> tuple[float, float] | None:
    """Return (gutter_x, separation_px) for parallel left/right lanes in a section."""
    cluster = _word_x_two_cluster(lines, page_width=page_width)
    if cluster is not None:
        return cluster

    words = _body_words_from_lines(lines, page_width)
    gutter_x = _estimate_lr_gutter(words, page_width)
    if gutter_x is None:
        return None

    left_words = [w for w in words if w.box.centroid_x < gutter_x]
    right_words = [w for w in words if w.box.centroid_x >= gutter_x]
    if not left_words or not right_words:
        return None

    left_max = max(w.box.max_x for w in left_words)
    right_min = min(w.box.min_x for w in right_words)
    whitespace_gap = right_min - left_max
    left_cx = sum(w.box.centroid_x for w in left_words) / len(left_words)
    right_cx = sum(w.box.centroid_x for w in right_words) / len(right_words)
    separation = max(whitespace_gap, right_cx - left_cx)
    return gutter_x, separation


def _split_section_at_lr_column_gap(
    sec: Section,
    *,
    page_width: float,
    split_threshold: float,
    pad: float,
) -> list[Section]:
    """
    Split a wide section at a document-relative horizontal gutter between
    left/right word clusters, then vertically band each column when warranted.
    """
    if page_width <= 0 or len(sec.lines) < 4:
        return [sec]

    if _is_row_coupled_lr_table_band(sec.lines, page_width):
        return [sec]

    body_lines = [
        ln for ln in sec.lines if not _is_margin_strip_line(ln, page_width)
    ]
    if len(body_lines) < 4:
        return [sec]

    gutter_info = _section_lr_gutter(body_lines, page_width=page_width)
    if gutter_info is None:
        return [sec]

    gutter_x, separation = gutter_info
    if separation < split_threshold:
        return [sec]

    full_frac = _full_width_line_fraction(body_lines, page_width)
    sep_frac = separation / page_width if page_width > 0 else 0.0
    # Prose-heavy bands (authors/abstract): skip unless gutter is a full page column break.
    if full_frac > 0.55 and sep_frac < 0.30:
        return [sec]

    gutter_pad = max(4.0, page_width * 0.006)
    crossing = sum(
        1
        for ln in body_lines
        if _line_has_words_crossing_gutter(ln, gutter_x, gutter_pad)
    )
    if crossing / len(body_lines) > 0.25:
        return [sec]

    split = _split_lines_at_x_gutter(
        body_lines,
        gutter_x,
        pad=pad,
        gap_above=sec.gap_above,
        gutter_pad=gutter_pad,
        page_width=page_width,
    )
    if split is None:
        return [sec]

    left_lines, right_lines, prose_lines = split
    assembled = _assemble_lr_gutter_sections(
        prose_lines,
        left_lines,
        right_lines,
        sec=sec,
        page_width=page_width,
        pad=pad,
        gutter_x=gutter_x,
        gutter_pad=gutter_pad,
    )
    if assembled is None:
        return [sec]
    return assembled


def _split_section_at_y_band_x_gaps(
    sec: Section,
    *,
    page_width: float,
    split_threshold: float,
    pad: float,
) -> list[Section]:
    """
    Within each y-band, split only when merged word x-intervals have a gap
  larger than the threshold (e.g. DER | tcpWER). Full-width prose bands stay intact.
    """
    bands = _cluster_lines_into_y_bands(sec.lines)
    full_lines: list[Line] = []
    left_lines: list[Line] = []
    right_lines: list[Line] = []
    split_count = 0

    for band in bands:
        if len(band) < 2 or _is_row_coupled_lr_table_band(band, page_width):
            full_lines.extend(band)
            continue

        intervals = _merge_x_intervals(
            [(w.box.min_x, w.box.max_x) for ln in band for w in ln.words]
        )
        gap_pair = _largest_x_gap(intervals)
        if gap_pair is None or gap_pair[1] < split_threshold:
            full_lines.extend(band)
            continue

        gutter_x, _ = gap_pair
        gutter_pad = max(4.0, page_width * 0.006)
        if any(
            _line_has_words_crossing_gutter(ln, gutter_x, gutter_pad) for ln in band
        ):
            full_lines.extend(band)
            continue
        split = _split_lines_at_x_gutter(
            band,
            gutter_x,
            pad=pad,
            gap_above=None,
            gutter_pad=gutter_pad,
            page_width=page_width,
        )
        if split is None:
            full_lines.extend(band)
            continue

        left_part, right_part, _prose_overflow = split
        split_count += 1
        left_lines.extend(left_part)
        right_lines.extend(right_part)

    out: list[Section] = []
    if full_lines:
        out.append(
            _make_section(0, sorted(full_lines, key=lambda ln: ln.content_box.min_y), sec.gap_above, pad)
        )
    if left_lines:
        out.append(
            _make_section(
                len(out),
                sorted(left_lines, key=lambda ln: ln.content_box.min_y),
                sec.gap_above if not full_lines else None,
                pad,
            )
        )
    if right_lines:
        out.append(
            _make_section(
                len(out),
                sorted(right_lines, key=lambda ln: ln.content_box.min_y),
                None,
                pad,
            )
        )

    if split_count == 0 or not out:
        return [sec]
    return out


def _x_gap_baseline_multiplier() -> float:
    raw = os.getenv("SECTION_X_GAP_BASELINE_MULTIPLIER", "2.5")
    try:
        return max(1.5, float(raw))
    except ValueError:
        return 2.5


def _is_data_table_section(sec: Section, page_width: float) -> bool:
    """Row-coupled label|value grids and high-confidence data tables — not figure pairs."""
    from section_table_layout import classify_section_layout, _layout_is_table_family

    if _is_row_coupled_lr_table_band(sec.lines, page_width):
        return True
    lay = classify_section_layout(sec.lines)
    if not _layout_is_table_family(lay) or lay.confidence < 0.72:
        return False
    if lay.layout_kind == "table":
        return True
    return lay.aligned_column_count >= 4


def _section_blocks_x_gap_split(sec: Section, page_width: float) -> bool:
    """Skip x-gap peel on multi-column grids and unified row-coupled tables."""
    from section_table_layout import analyze_vertical_partition, classify_section_layout, _layout_is_table_family

    if _is_row_coupled_lr_table_band(sec.lines, page_width):
        return True

    lay = classify_section_layout(sec.lines)
    if _layout_is_table_family(lay) and lay.confidence >= 0.72:
        if lay.layout_kind == "table":
            return True
        if lay.aligned_column_count >= 4:
            return True

    part = analyze_vertical_partition(sec.lines, page_width)
    if part.has_column and part.is_unified_grid:
        return True

    return False


def _lr_gutter_split_is_valid(
    parts: list[Section],
    parent: Section,
    *,
    page_width: float,
    pad: float,
) -> bool:
    """
    Reject L/R parts that would leave narrow column shards or cut assigned text.
    """
    del page_width  # reserved for future page-relative thresholds
    if len(parts) <= 1:
        return True

    parent_w = max(parent.box.width, 1.0)
    min_panel_w = max(parent_w * 0.20, 72.0)

    for part in parts:
        if not part.lines:
            continue
        if not _box_covers_lines(part.box, part.lines, pad=pad):
            return False

        for ln in part.lines:
            for w in ln.words:
                if (
                    w.box.min_x < part.box.min_x - 0.5
                    or w.box.max_x > part.box.max_x + 0.5
                    or w.box.min_y < part.box.min_y - 0.5
                    or w.box.max_y > part.box.max_y + 0.5
                ):
                    return False

        if part.box.width < min_panel_w:
            return False

    return True


def _prose_chart_split_is_valid(parts: list[Section], page_width: float) -> bool:
    """Reject splits that only shard abstract prose — need two tall side panels."""
    if len(parts) < 3 or page_width <= 0:
        return False
    min_chart_h = 80.0
    panels = [
        p
        for p in parts
        if p.box.min_y <= p.box.max_y
        and p.box.height >= min_chart_h
        and p.box.width <= page_width * 0.48
    ]
    return len(panels) >= 2


def _chart_region_has_wide_x_gap(
    sec: Section,
    chart_y0: float,
    page_width: float,
    split_threshold: float,
) -> bool:
    """True when chart/figure bands contain a whitespace gutter above the page baseline."""
    chart_lines = [
        ln for ln in sec.lines if ln.content_box.centroid_y >= chart_y0 - 24.0
    ]
    if not chart_lines:
        return False
    bands = _cluster_lines_into_y_bands(chart_lines, band_gap_px=48.0)
    for band in bands:
        if _is_row_coupled_lr_table_band(band, page_width) or len(band) < 2:
            continue
        intervals = _merge_x_intervals(
            [(w.box.min_x, w.box.max_x) for ln in band for w in ln.words]
        )
        gap_pair = _largest_x_gap(intervals)
        if gap_pair is not None and gap_pair[1] >= split_threshold:
            return True
    return False


def split_sections_by_horizontal_x_gaps(
    sections: list[Section],
    body_lines: list[Line],
    boxes: list[FormBox],
    *,
    page_width: float,
    pad: float = SECTION_PAD,
    lines: list[Line] | None = None,
) -> tuple[list[Section], dict[str, Any]]:
    """
    For each section: if multiple OCR rows show horizontal whitespace between left
    and right text far larger than typical word spacing, split into two columns.
  Skips data tables and OpenCV-boxed sections.
    """
    hi_boxes = [b for b in boxes if b.confidence >= 0.9]
    typical_gap = _page_typical_word_gap(body_lines)
    multiplier = _x_gap_baseline_multiplier()
    split_threshold = max(typical_gap * multiplier, 72.0)
    typical_mult = _lr_gap_typical_multiplier()
    min_lr_gap = max(split_threshold, typical_gap * typical_mult)
    out: list[Section] = []
    split_count = 0

    for sec in sections:
        if any(_opencv_box_covers_section(sec, b) for b in hi_boxes):
            out.append(sec)
            continue

        if _section_blocks_x_gap_split(sec, page_width):
            out.append(sec)
            continue

        if sec.box.width < page_width * 0.35 or sec.box.height < 80.0:
            out.append(sec)
            continue

        parts = _split_section_at_lr_whitespace(
            sec,
            page_width=page_width,
            split_threshold=split_threshold,
            pad=pad,
            min_lr_gap=min_lr_gap,
        )
        if len(parts) > 1 and _lr_gutter_split_is_valid(
            parts,
            sec,
            page_width=page_width,
            pad=pad,
        ):
            split_count += 1
            out.extend(parts)
        else:
            out.append(sec)

    meta: dict[str, Any] = {
        "mode": "lr_whitespace_gutter",
        "x_gap_split_section_count": split_count,
        "typical_word_gap_px": round(typical_gap, 1),
        "x_gap_baseline_multiplier": multiplier,
        "lr_gap_typical_multiplier": typical_mult,
        "split_threshold_px": round(split_threshold, 1),
        "min_lr_gap_px": round(min_lr_gap, 1),
        "section_count": len(out),
    }
    return _reindex_sections(out), meta


def _assign_line_to_box(
    line: Line,
    boxes: list[FormBox],
    *,
    page_width: float,
    all_lines: list[Line],
) -> int | None:
    """Assign a line to the box with strongest overlap; respect L→R middle-box splits."""
    if not boxes:
        return None

    overlapping = [
        i
        for i, b in enumerate(boxes)
        if _line_box_intersection_area(line, b) > 0 or _line_in_box(line, b)
    ]
    if overlapping:
        return max(overlapping, key=lambda i: _line_box_intersection_area(line, boxes[i]))

    cx = line.box.centroid_x
    vertical = [i for i, b in enumerate(boxes) if _line_overlaps_box_vertically(line, b)]

    # L→R band with a box in the middle: route side columns to neighboring panels.
    for i, box in enumerate(boxes):
        if not _box_splits_ltr_band(all_lines, box, page_width):
            continue
        if cx < box.x:
            left_boxes = [
                j
                for j in vertical
                if boxes[j].x2 <= box.x + 4 and _line_box_intersection_area(line, boxes[j]) > 0
            ]
            if left_boxes:
                return max(left_boxes, key=lambda j: _line_box_intersection_area(line, boxes[j]))
        if cx > box.x2:
            right_boxes = [
                j
                for j in vertical
                if boxes[j].x >= box.x2 - 4 and _line_box_intersection_area(line, boxes[j]) > 0
            ]
            if right_boxes:
                return max(right_boxes, key=lambda j: _line_box_intersection_area(line, boxes[j]))

    return None


def _partition_lines_by_boxes(
    lines: list[Line],
    boxes: list[FormBox],
    *,
    page_width: float,
) -> tuple[list[list[Line]], list[Line]]:
    sorted_boxes = sorted(boxes, key=lambda b: (b.y, b.x))
    groups: list[list[Line]] = [[] for _ in sorted_boxes]
    overflow: list[Line] = []
    for line in lines:
        idx = _assign_line_to_box(
            line,
            sorted_boxes,
            page_width=page_width,
            all_lines=lines,
        )
        if idx is None:
            overflow.append(line)
        else:
            groups[idx].append(line)
    return groups, overflow


def _sections_from_box_groups(groups: list[list[Line]]) -> list[Section]:
    sections: list[Section] = []
    for group in groups:
        if not group:
            continue
        group.sort(key=lambda ln: (ln.box.centroid_y, ln.box.min_x))
        sections.append(_make_section(len(sections), group, None, SECTION_PAD))
    return sections


def _reindex_sections(sections: list[Section]) -> list[Section]:
    return [
        Section(
            index=i,
            lines=s.lines,
            text=s.text,
            box=s.box,
            gap_above=s.gap_above,
        )
        for i, s in enumerate(sections)
    ]


def _overflow_full_width_lines(
    overflow: list[Line],
    full_width_lines: list[Line] | None,
) -> list[Line] | None:
    if not overflow or not full_width_lines:
        return full_width_lines
    overflow_ids = {id(ln) for ln in overflow}
    return [ln for ln in full_width_lines if id(ln) in overflow_ids]


def lines_to_sections_by_opencv_boxes(
    lines: list[Line],
    image_bgr: np.ndarray,
    words: list[Word],
    *,
    min_confidence: float = 0.9,
    page_width: float,
    full_width_lines: list[Line] | None = None,
    min_gap_px: float = 18.0,
) -> tuple[list[Section], dict[str, Any]]:
    meta: dict[str, Any] = {
        "mode": "opencv_boxes",
        "min_confidence": min_confidence,
        "use_opencv": False,
        "redacted": True,
    }

    redacted = redact_ocr_words(image_bgr, words)
    boxes = detect_merged_high_confidence_sections(
        image_bgr, words, min_confidence=min_confidence
    )
    meta["box_count"] = len(boxes)
    meta["box_confidences"] = [round(b.confidence, 3) for b in boxes]

    if not boxes:
        meta["reason"] = "no_boxes"
        return [], meta

    groups, overflow = _partition_lines_by_boxes(
        lines, boxes, page_width=page_width
    )
    assigned = sum(len(g) for g in groups)
    coverage = assigned / len(lines) if lines else 0.0
    meta["line_coverage"] = round(coverage, 4)
    meta["overflow_line_count"] = len(overflow)
    meta["ltr_band_splits"] = sum(
        1 for b in boxes if _box_splits_ltr_band(lines, b, page_width)
    )

    if coverage < COVERAGE_HYBRID_MIN:
        meta["reason"] = "low_coverage"
        meta["overflow_policy"] = "hv_combined_full"
        return [], meta

    box_sections = _sections_from_box_groups(groups)
    overflow_secs: list[Section] = []
    if overflow:
        ov_fw = _overflow_full_width_lines(overflow, full_width_lines)
        overflow_secs, _, _ = lines_to_sections_hv_combined(
            overflow,
            page_width=page_width,
            full_width_lines=ov_fw,
            min_gap_px=min_gap_px,
        )

    if coverage >= COVERAGE_BOX_ONLY:
        meta["overflow_policy"] = "box_only"
    else:
        meta["overflow_policy"] = "hybrid"

    meta["use_opencv"] = True
    merged = box_sections + overflow_secs
    return _reindex_sections(merged), meta
