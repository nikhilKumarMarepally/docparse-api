"""Detect printed rectangular boxes on form scans using OpenCV line morphology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FormBox:
    """Axis-aligned rectangle in image pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    area: float
    contour_area: float
    depth: int  # nesting level from findContours hierarchy
    confidence: float = 0.0

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "area": round(self.area, 1),
            "depth": self.depth,
        }


@dataclass(frozen=True)
class FormLine:
    """Axis-aligned printed rule (solid or dotted/dashed)."""

    x: int
    y: int
    w: int
    h: int
    orientation: str  # "h" or "v"
    style: str  # "solid" or "dotted"
    confidence: float

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def as_form_box(self) -> FormBox:
        area = float(max(1, self.w) * max(1, self.h))
        return FormBox(
            x=self.x,
            y=self.y,
            w=self.w,
            h=self.h,
            area=area,
            contour_area=area,
            depth=0,
            confidence=self.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "orientation": self.orientation,
            "style": self.style,
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class LayoutDet:
    """Unified OpenCV layout hit: closed box or H/V rule."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    cls: str  # box, solid, dotted
    confidence: float
    orientation: str = ""  # h, v, or empty for boxes

    def as_form_box(self) -> FormBox:
        w = max(1, int(self.max_x - self.min_x))
        h = max(1, int(self.max_y - self.min_y))
        area = float(w * h)
        return FormBox(
            x=int(self.min_x),
            y=int(self.min_y),
            w=w,
            h=h,
            area=area,
            contour_area=area,
            depth=0,
            confidence=self.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_x": round(self.min_x, 1),
            "min_y": round(self.min_y, 1),
            "max_x": round(self.max_x, 1),
            "max_y": round(self.max_y, 1),
            "cls": self.cls,
            "confidence": round(self.confidence, 4),
            "orientation": self.orientation,
        }


def _quad_rectangularity(cnt: np.ndarray) -> float:
    area = cv2.contourArea(cnt)
    if area <= 0:
        return 0.0
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        return 0.0
    x, y, w, h = cv2.boundingRect(approx)
    box_area = float(w * h)
    if box_area <= 0:
        return 0.0
    return float(area / box_area)


def _ink_binary(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _connect_dashed_gaps(binary: np.ndarray, w_img: int, h_img: int) -> np.ndarray:
    """Close short gaps so dotted/dashed rules join into extractable strokes."""
    gap = max(7, int(min(w_img, h_img) * 0.008))
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_k)
    return cv2.morphologyEx(closed, cv2.MORPH_CLOSE, v_k)


def detect_printed_boxes(
    image_bgr: np.ndarray,
    *,
    min_area_frac: float = 0.0015,
    max_area_frac: float = 0.42,
    min_rectangularity: float = 0.72,
    line_scale: float = 1.0,
) -> list[FormBox]:
    """Find closed rectangular frames (tables, option boxes, panels)."""
    h_img, w_img = image_bgr.shape[:2]
    page_area = float(h_img * w_img)
    min_area = page_area * min_area_frac
    max_area = page_area * max_area_frac

    binary = _ink_binary(image_bgr)
    binary = _connect_dashed_gaps(binary, w_img, h_img)

    h_len = max(25, int(w_img * 0.025 * line_scale))
    v_len = max(25, int(h_img * 0.025 * line_scale))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    horiz = cv2.erode(binary, h_kernel, iterations=1)
    horiz = cv2.dilate(horiz, h_kernel, iterations=1)
    vert = cv2.erode(binary, v_kernel, iterations=1)
    vert = cv2.dilate(vert, v_kernel, iterations=1)
    grid = cv2.bitwise_or(horiz, vert)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(
        grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []

    boxes: list[FormBox] = []
    hier = hierarchy[0]
    for i, cnt in enumerate(contours):
        rect_score = _quad_rectangularity(cnt)
        if rect_score < min_rectangularity:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        area = float(w * h)
        if area < min_area or area > max_area:
            continue
        if w < 30 or h < 30:
            continue
        depth = 0
        parent = hier[i][3]
        while parent != -1:
            depth += 1
            parent = hier[parent][3]
        boxes.append(
            FormBox(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=area,
                contour_area=float(cv2.contourArea(cnt)),
                depth=depth,
                confidence=round(float(rect_score), 4),
            )
        )

    boxes.sort(key=lambda b: b.area, reverse=True)
    return _suppress_nested_duplicates(boxes)


def _suppress_nested_duplicates(boxes: list[FormBox], *, iou_thresh: float = 0.92) -> list[FormBox]:
    """Drop boxes that are almost identical to a larger kept box."""
    kept: list[FormBox] = []
    for b in boxes:
        dup = False
        for k in kept:
            if _iou(b, k) >= iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(b)
    kept.sort(key=lambda b: (b.y, b.x))
    return kept


def _iou(a: FormBox, b: FormBox) -> float:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x2, b.x2)
    y1 = min(a.y2, b.y2)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def draw_form_boxes(
    image_bgr: np.ndarray,
    boxes: list[FormBox],
    *,
    title: str = "",
) -> np.ndarray:
    """Overlay detected boxes with labels."""
    out = image_bgr.copy()
    palette = [
        (0, 180, 255),
        (255, 120, 0),
        (0, 220, 120),
        (220, 80, 255),
        (255, 255, 0),
        (180, 180, 255),
    ]
    if title:
        cv2.rectangle(out, (0, 0), (out.shape[1], 32), (20, 20, 20), -1)
        cv2.putText(
            out,
            title[:120],
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    for i, box in enumerate(boxes):
        color = palette[i % len(palette)]
        cv2.rectangle(out, (box.x, box.y), (box.x2, box.y2), color, 3)
        label = f"B{i} d{box.depth}"
        cv2.rectangle(
            out,
            (box.x, max(32, box.y - 22)),
            (box.x + 72, max(32, box.y - 2)),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (box.x + 4, max(28, box.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def boxes_to_sections(boxes: list[FormBox], *, page_area: float | None = None) -> list[FormBox]:
    """Keep boxes that match printed form panels (drop tiny inner check cells)."""
    if not boxes:
        return []
    page = page_area or max(b.area for b in boxes)
    # Tall panels (program columns) or wide bands (header table rows).
    sections: list[FormBox] = []
    for b in boxes:
        frac = b.area / page
        aspect = b.h / max(1, b.w)
        if frac < 0.012:
            continue
        if aspect >= 1.15 and b.h >= 120 and frac < 0.25:
            sections.append(b)
            continue
        if aspect < 0.35 and b.w >= 200 and frac < 0.12:
            sections.append(b)
            continue
        if frac >= 0.04 and b.depth <= 2:
            sections.append(b)
    sections.sort(key=lambda b: (b.y, b.x))
    return _suppress_nested_duplicates(sections, iou_thresh=0.85)


def detect_column_panels(
    image_bgr: np.ndarray,
    *,
    y_frac: tuple[float, float] = (0.32, 0.72),
) -> list[FormBox]:
    """Three-across (or N) vertical panels: closed rectangles in a horizontal band."""
    h_img, w_img = image_bgr.shape[:2]
    y0 = int(h_img * y_frac[0])
    y1 = int(h_img * y_frac[1])
    crop = image_bgr[y0:y1, :]
    local = detect_printed_boxes(
        crop,
        min_area_frac=0.02,
        max_area_frac=0.22,
        min_rectangularity=0.68,
        line_scale=0.85,
    )
    panels: list[FormBox] = []
    for b in local:
        if b.h < 0.55 * (y1 - y0):
            continue
        if b.w < 0.12 * w_img or b.w > 0.45 * w_img:
            continue
        panels.append(
            FormBox(
                x=b.x,
                y=b.y + y0,
                w=b.w,
                h=b.h,
                area=b.area,
                contour_area=b.contour_area,
                depth=b.depth,
                confidence=b.confidence,
            )
        )
    panels.sort(key=lambda b: b.x)
    return _suppress_nested_duplicates(panels, iou_thresh=0.7)


def detect_form_sections(image_bgr: np.ndarray) -> list[FormBox]:
    """Printed boxes likely to be section boundaries."""
    h_img, w_img = image_bgr.shape[:2]
    page_area = float(h_img * w_img)
    all_boxes = detect_printed_boxes(image_bgr)
    sections = boxes_to_sections(all_boxes, page_area=page_area)
    panels = detect_column_panels(image_bgr)
    if len(panels) >= 2:
        # Prefer column panels for the program area; keep non-overlapping header boxes.
        merged = panels[:]
        for s in sections:
            if s.h < 100:
                merged.append(s)
                continue
            if any(_iou(s, p) > 0.2 for p in panels):
                continue
            merged.append(s)
        merged.sort(key=lambda b: (b.y, b.x))
        return _suppress_nested_duplicates(merged, iou_thresh=0.85)
    return sections


def detect_high_confidence_sections(
    image_bgr: np.ndarray,
    *,
    min_confidence: float = 0.9,
) -> list[FormBox]:
    """OpenCV printed frames with rectangularity >= min_confidence."""
    return [
        box
        for box in detect_form_sections(image_bgr)
        if box.confidence >= min_confidence
    ]


def _classify_line_style(
    binary: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    orientation: str,
) -> tuple[str, float]:
    """solid vs dotted/dashed from ink runs on the original (unclosed) binary."""
    h_img, w_img = binary.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    roi = binary[y0:y1, x0:x1]
    if roi.size == 0:
        return "solid", 0.0
    if orientation == "h":
        row_sums = roi.sum(axis=1)
        profile = roi[int(np.argmax(row_sums))].astype(np.uint8)
        length = w
    else:
        col_sums = roi.sum(axis=0)
        profile = roi[:, int(np.argmax(col_sums))].astype(np.uint8)
        length = h
    ink = profile > 127
    if ink.size < 8:
        return "solid", 0.5
    fill = float(ink.mean())
    padded = np.concatenate([[0], ink.astype(np.uint8), [0]])
    n_runs = int(np.sum(np.diff(padded) == 1))
    length_score = min(1.0, length / 80.0)
    if n_runs >= 4 and fill < 0.75:
        conf = min(0.99, 0.55 + 0.08 * min(n_runs, 6) + 0.2 * length_score)
        return "dotted", round(float(conf), 4)
    conf = min(0.99, 0.55 + 0.4 * fill + 0.1 * length_score)
    return "solid", round(float(conf), 4)


def _components_to_lines(
    mask: np.ndarray,
    original_binary: np.ndarray,
    orientation: str,
    w_img: int,
    h_img: int,
) -> list[FormLine]:
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_h = max(50, int(w_img * 0.06))
    min_v = max(40, int(h_img * 0.04))
    max_thick = max(12, int(min(w_img, h_img) * 0.012))
    lines: list[FormLine] = []
    for i in range(1, n_labels):
        x, y, w, h, _area = (int(v) for v in stats[i][:5])
        if orientation == "h":
            if w < min_h or h > max_thick:
                continue
        elif h < min_v or w > max_thick:
            continue
        style, conf = _classify_line_style(original_binary, x, y, w, h, orientation)
        lines.append(
            FormLine(
                x=x,
                y=y,
                w=w,
                h=h,
                orientation=orientation,
                style=style,
                confidence=conf,
            )
        )
    return lines


def _merge_collinear_lines(lines: list[FormLine], *, gap: int = 12) -> list[FormLine]:
    """Join collinear fragments of the same style that almost touch."""
    if not lines:
        return []
    kept: list[FormLine] = []
    remaining = sorted(lines, key=lambda ln: (ln.orientation, ln.style, ln.y, ln.x))
    while remaining:
        cur = remaining.pop(0)
        merged = True
        while merged:
            merged = False
            nxt: list[FormLine] = []
            for other in remaining:
                same = (
                    other.orientation == cur.orientation and other.style == cur.style
                )
                if not same:
                    nxt.append(other)
                    continue
                if cur.orientation == "h":
                    y_ok = abs((cur.y + cur.h / 2) - (other.y + other.h / 2)) <= gap
                    x_gap = max(0, max(cur.x, other.x) - min(cur.x2, other.x2))
                    if y_ok and x_gap <= gap:
                        x0 = min(cur.x, other.x)
                        y0 = min(cur.y, other.y)
                        x1 = max(cur.x2, other.x2)
                        y1 = max(cur.y2, other.y2)
                        cur = FormLine(
                            x=x0,
                            y=y0,
                            w=x1 - x0,
                            h=max(1, y1 - y0),
                            orientation=cur.orientation,
                            style=cur.style,
                            confidence=max(cur.confidence, other.confidence),
                        )
                        merged = True
                        continue
                else:
                    x_ok = abs((cur.x + cur.w / 2) - (other.x + other.w / 2)) <= gap
                    y_gap = max(0, max(cur.y, other.y) - min(cur.y2, other.y2))
                    if x_ok and y_gap <= gap:
                        x0 = min(cur.x, other.x)
                        y0 = min(cur.y, other.y)
                        x1 = max(cur.x2, other.x2)
                        y1 = max(cur.y2, other.y2)
                        cur = FormLine(
                            x=x0,
                            y=y0,
                            w=max(1, x1 - x0),
                            h=y1 - y0,
                            orientation=cur.orientation,
                            style=cur.style,
                            confidence=max(cur.confidence, other.confidence),
                        )
                        merged = True
                        continue
                nxt.append(other)
            remaining = nxt
        kept.append(cur)
    kept.sort(key=lambda ln: (ln.y, ln.x))
    return kept


def detect_rule_lines(
    image_bgr: np.ndarray,
    *,
    style_image_bgr: np.ndarray | None = None,
    orientations: tuple[str, ...] = ("h", "v"),
) -> list[FormLine]:
    """Find long solid and dotted/dashed horizontal and vertical rules."""
    h_img, w_img = image_bgr.shape[:2]
    binary = _ink_binary(image_bgr)
    style_binary = _ink_binary(style_image_bgr) if style_image_bgr is not None else binary
    connected = _connect_dashed_gaps(binary, w_img, h_img)

    h_len = max(40, int(w_img * 0.08))
    v_len = max(40, int(h_img * 0.08))
    lines: list[FormLine] = []
    if "h" in orientations:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        horiz = cv2.bitwise_or(
            cv2.morphologyEx(connected, cv2.MORPH_OPEN, h_kernel),
            cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel),
        )
        lines.extend(_components_to_lines(horiz, style_binary, "h", w_img, h_img))
    if "v" in orientations:
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        vert = cv2.bitwise_or(
            cv2.morphologyEx(connected, cv2.MORPH_OPEN, v_kernel),
            cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel),
        )
        lines.extend(_components_to_lines(vert, style_binary, "v", w_img, h_img))
    return _merge_collinear_lines(lines)


def detect_residual_figure_boxes(
    redacted_bgr: np.ndarray,
    *,
    min_area_frac: float = 0.004,
    max_area_frac: float = 0.55,
) -> list[FormBox]:
    """Cluster leftover ink/color after OCR white-out into figure-sized boxes.

    Closes OCR holes so a diagram stays one component. Does not emit title
    bands or filled text panels — those are OCR sections, not figures.
    """
    h_img, w_img = redacted_bgr.shape[:2]
    page_area = float(h_img * w_img)
    gray = cv2.cvtColor(redacted_bgr, cv2.COLOR_BGR2GRAY)
    chroma = np.max(redacted_bgr.astype(np.float32), axis=2) - np.min(
        redacted_bgr.astype(np.float32), axis=2
    )
    remain = ((gray < 248) | (chroma >= 8.0)).astype(np.uint8) * 255

    rules = detect_rule_lines(redacted_bgr)
    rule_mask = np.zeros_like(remain)
    for ln in rules:
        if ln.orientation == "h" and ln.w >= 0.4 * w_img:
            cv2.rectangle(
                rule_mask,
                (ln.x, max(0, ln.y - 3)),
                (ln.x2, min(h_img, ln.y2 + 3)),
                255,
                -1,
            )
        elif ln.orientation == "v" and ln.h >= 0.25 * h_img:
            cv2.rectangle(
                rule_mask,
                (max(0, ln.x - 3), ln.y),
                (min(w_img, ln.x2 + 3), ln.y2),
                255,
                -1,
            )
    remain = cv2.bitwise_and(remain, cv2.bitwise_not(rule_mask))

    kx = max(36, int(w_img * 0.022))
    ky = max(36, int(h_img * 0.018))
    closed = cv2.morphologyEx(
        remain, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    )
    _num, _labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    boxes: list[FormBox] = []
    min_area = page_area * min_area_frac
    max_area = page_area * max_area_frac
    for i in range(1, _num):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if min(int(w), int(h)) < 50:
            continue
        aspect = float(h) / max(1, int(w))
        if aspect > 10.0 or aspect < 0.06:
            continue
        boxes.append(
            FormBox(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=float(area),
                contour_area=float(area),
                depth=0,
                confidence=0.94,
            )
        )
    boxes.sort(key=lambda b: (b.y, b.x))
    return _suppress_nested_duplicates(boxes, iou_thresh=0.85)


def _line_to_layout_det(line: FormLine) -> LayoutDet:
    return LayoutDet(
        min_x=float(line.x),
        min_y=float(line.y),
        max_x=float(line.x2),
        max_y=float(line.y2),
        cls=line.style,
        confidence=line.confidence,
        orientation=line.orientation,
    )


def _box_to_layout_det(box: FormBox) -> LayoutDet:
    return LayoutDet(
        min_x=float(box.x),
        min_y=float(box.y),
        max_x=float(box.x2),
        max_y=float(box.y2),
        cls="box",
        confidence=box.confidence,
        orientation="",
    )


def keep_vertical_lines_only(dets: list[LayoutDet]) -> list[LayoutDet]:
    """Column/gutter rules only — drop boxes and horizontal strokes."""
    kept: list[LayoutDet] = []
    for det in dets:
        if det.cls == "box":
            continue
        ori = det.orientation or (
            "v" if (det.max_y - det.min_y) > (det.max_x - det.min_x) else "h"
        )
        if ori == "v":
            kept.append(det)
    kept.sort(key=lambda d: (d.min_x, d.min_y, d.cls))
    return kept


def detect_layout_elements(
    image_bgr: np.ndarray,
    *,
    extra_lines: list[FormLine] | None = None,
    style_image_bgr: np.ndarray | None = None,
    include_boxes: bool = True,
    line_orientations: tuple[str, ...] = ("h", "v"),
) -> list[LayoutDet]:
    """Closed printed boxes plus solid/dotted H/V rules."""
    dets: list[LayoutDet] = []
    if include_boxes:
        dets.extend(_box_to_layout_det(b) for b in detect_printed_boxes(image_bgr))
    lines = detect_rule_lines(
        image_bgr,
        style_image_bgr=style_image_bgr,
        orientations=line_orientations,
    )
    if extra_lines:
        lines = _merge_collinear_lines(lines + list(extra_lines))
    dets.extend(_line_to_layout_det(ln) for ln in lines)
    dets.sort(key=lambda d: (d.min_y, d.min_x, d.cls))
    return dets


def _word_rect(word: Any, pad: float) -> tuple[float, float, float, float]:
    box = word.box if hasattr(word, "box") else word
    return (
        float(box.min_x) - pad,
        float(box.min_y) - pad,
        float(box.max_x) + pad,
        float(box.max_y) + pad,
    )


def _rects_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _line_stroke_rect(det: LayoutDet) -> tuple[float, float, float, float]:
    """Thin ink stroke, not the full connected-component bbox."""
    ori = det.orientation
    if not ori:
        ori = "v" if (det.max_y - det.min_y) > (det.max_x - det.min_x) else "h"
    if ori == "v":
        cx = 0.5 * (det.min_x + det.max_x)
        half = max(3.0, 0.5 * (det.max_x - det.min_x))
        return (cx - half, det.min_y, cx + half, det.max_y)
    cy = 0.5 * (det.min_y + det.max_y)
    half = max(3.0, 0.5 * (det.max_y - det.min_y))
    return (det.min_x, cy - half, det.max_x, cy + half)


def _ocr_cuts_box_border(
    det: LayoutDet,
    word_rect: tuple[float, float, float, float],
    *,
    band: float,
) -> bool:
    """True when OCR sits on the frame stroke, not merely inside the panel."""
    outer = (det.min_x - 2.0, det.min_y - 2.0, det.max_x + 2.0, det.max_y + 2.0)
    if not _rects_overlap(word_rect, outer):
        return False
    inner = (
        det.min_x + band,
        det.min_y + band,
        det.max_x - band,
        det.max_y - band,
    )
    if inner[2] <= inner[0] or inner[3] <= inner[1]:
        return True
    fully_inside = (
        word_rect[0] >= inner[0]
        and word_rect[1] >= inner[1]
        and word_rect[2] <= inner[2]
        and word_rect[3] <= inner[3]
    )
    return not fully_inside


def _is_field_underline(
    det: LayoutDet,
    word_rects: list[tuple[float, float, float, float]],
    *,
    below_px: float = 12.0,
) -> bool:
    """Short rule sitting just under a glyph — a fill-in underline, not a border."""
    cy = 0.5 * (det.min_y + det.max_y)
    for wr in word_rects:
        x_overlap = min(det.max_x, wr[2]) - max(det.min_x, wr[0])
        if x_overlap <= 4:
            continue
        gap = cy - wr[3]
        if -2.0 <= gap <= below_px:
            return True
    return False


def drop_vertical_lines_crossing_ink(
    dets: list[LayoutDet],
    image_bgr: np.ndarray,
    *,
    max_stroke: int = 10,
    hit_frac: float = 0.30,
) -> list[LayoutDet]:
    """Drop vertical rules that pass through glyph-wide ink, even if OCR missed it."""
    binary = _ink_binary(image_bgr)
    h_img, w_img = binary.shape[:2]
    kept: list[LayoutDet] = []
    for det in dets:
        ori = det.orientation or (
            "v" if (det.max_y - det.min_y) > (det.max_x - det.min_x) else "h"
        )
        if ori != "v":
            kept.append(det)
            continue
        x = int(np.clip(0.5 * (det.min_x + det.max_x), 0, w_img - 1))
        y0 = max(0, int(det.min_y))
        y1 = min(h_img, int(det.max_y))
        hits = 0
        checked = 0
        x0 = max(0, x - 18)
        x1 = min(w_img, x + 19)
        for y in range(y0, y1, 2):
            row = binary[y, x0:x1]
            local = x - x0
            if local < 0 or local >= row.size or row[local] == 0:
                continue
            left = local
            while left > 0 and row[left - 1] > 0:
                left -= 1
            right = local
            while right + 1 < row.size and row[right + 1] > 0:
                right += 1
            checked += 1
            if (right - left + 1) > max_stroke:
                hits += 1
        if checked >= 8 and hits / checked >= hit_frac:
            continue
        kept.append(det)
    return kept


def drop_layout_overlapping_ocr(
    dets: list[LayoutDet],
    words: list[Any],
    *,
    pad: float = 3.0,
    box_border_band: float = 8.0,
    page_width: float | None = None,
    min_separator_frac: float = 0.35,
) -> list[LayoutDet]:
    """Drop OpenCV hits whose stroke intersects OCR, plus short field underlines.

    Keeps closed frames (text inside, not on the border) and long H/V rules
    that sit in gutters — borders and separators, not lines through glyphs.
    """
    if not dets:
        return []
    if not words:
        return list(dets)
    word_rects = [_word_rect(w, pad) for w in words]
    kept: list[LayoutDet] = []
    for det in dets:
        if det.cls in ("solid", "dotted"):
            stroke = _line_stroke_rect(det)
            if any(_rects_overlap(stroke, wr) for wr in word_rects):
                continue
            ori = det.orientation or (
                "v" if (det.max_y - det.min_y) > (det.max_x - det.min_x) else "h"
            )
            if ori == "h" and _is_field_underline(det, word_rects):
                continue
            kept.append(det)
            continue
        if any(
            _ocr_cuts_box_border(det, wr, band=box_border_band) for wr in word_rects
        ):
            continue
        kept.append(det)
    kept.sort(key=lambda d: (d.min_y, d.min_x, d.cls))
    return kept


def detect_layout_on_redacted(
    original_bgr: np.ndarray,
    redacted_bgr: np.ndarray,
    words: list[Any] | None = None,
) -> list[LayoutDet]:
    """Vertical separators from redacted ink; drop strokes that cut OCR text."""
    dets = detect_layout_elements(
        redacted_bgr,
        style_image_bgr=original_bgr,
        include_boxes=False,
        line_orientations=("v",),
    )
    if words:
        dets = drop_layout_overlapping_ocr(
            dets,
            words,
            page_width=float(redacted_bgr.shape[1]),
        )
    dets = keep_vertical_lines_only(dets)
    dets = drop_vertical_lines_crossing_ink(dets, original_bgr)
    return dets
