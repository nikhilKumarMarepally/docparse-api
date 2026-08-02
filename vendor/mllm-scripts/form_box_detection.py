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

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

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
