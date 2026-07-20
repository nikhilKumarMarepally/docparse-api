"""Detect bold vs regular text from B&W stroke thickness and intensity."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

import numpy as np
from PIL import Image

from app.paths import ensure_script_path

ensure_script_path()

from ocr_word_to_line_boxes import Box, Word  # noqa: E402


@dataclass(frozen=True)
class WordMetrics:
    text: str
    ink_ratio: float
    avg_intensity: float
    stroke_thickness: float
    bold: bool
    avg_ink_intensity: float = 255.0


WordStyle = WordMetrics


def _clamp_box(box: Box, width: int, height: int) -> tuple[int, int, int, int] | None:
    x0 = max(0, int(box.min_x))
    y0 = max(0, int(box.min_y))
    x1 = min(width, int(box.max_x))
    y1 = min(height, int(box.max_y))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = gray.size
    if total == 0:
        return 180
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 180
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return int(threshold)


def _similar_height(h: float, ref: float) -> bool:
    return abs(h - ref) <= max(3.0, ref * 0.25)


def measure_word(
    image: Image.Image,
    box: Box,
) -> tuple[float, float, float, float, float] | None:
    """
    Return (ink_ratio, avg_intensity, avg_ink_intensity, stroke_px, word_height).
    Per-word grayscale → Otsu B&W; stroke = median vertical ink depth per column.
    """
    region = _clamp_box(box, image.width, image.height)
    if region is None:
        return None
    gray = np.asarray(image.crop(region).convert("L"), dtype=np.uint8)
    h, w = gray.shape
    if h < 2 or w < 2:
        return None

    thr = _otsu_threshold(gray)
    ink = gray < thr
    ink_ratio = float(ink.sum()) / float(ink.size)
    if ink_ratio < 0.02:
        return None

    avg_intensity = float(gray.mean())
    ink_vals = gray[ink]
    avg_ink_intensity = float(ink_vals.mean()) if ink_vals.size else 255.0

    col_depths = [int(ink[:, col].sum()) for col in range(w) if ink[:, col].any()]
    stroke_px = float(median(col_depths)) if col_depths else 0.0

    return ink_ratio, avg_intensity, avg_ink_intensity, stroke_px, float(h)


def annotate_word_styles(
    image: Image.Image,
    words: Sequence[Word],
) -> list[WordMetrics]:
    """Compare B&W ink fill, stroke thickness, and ink darkness vs same-height peers."""
    if not words:
        return []

    measured: list[tuple[Word, float, float, float, float, float]] = []
    for word in words:
        m = measure_word(image, word.box)
        if m is None:
            continue
        measured.append((word, *m))

    if not measured:
        return [
            WordMetrics(text=w.text, ink_ratio=0.0, avg_intensity=255.0, stroke_thickness=0.0, bold=False)
            for w in words
        ]

    metrics_by_word: dict[int, WordMetrics] = {}
    for word, ink, avg_int, avg_ink_int, stroke_px, height in measured:
        peers = [row for row in measured if _similar_height(row[5], height)]
        peer_inks = [p[1] for p in peers]
        peer_ink_ints = [p[3] for p in peers]
        peer_strokes = [p[4] / p[5] if p[5] > 0 else 0.0 for p in peers]

        med_ink = median(peer_inks)
        med_ink_int = median(peer_ink_ints)
        med_stroke = median(peer_strokes)
        mad_ink = median([abs(x - med_ink) for x in peer_inks]) or 0.02
        mad_ink_int = median([abs(x - med_ink_int) for x in peer_ink_ints]) or 4.0
        mad_stroke = median([abs(x - med_stroke) for x in peer_strokes]) or 0.02

        stroke_norm = stroke_px / height if height > 0 else 0.0
        stroke_score = (stroke_norm - med_stroke) / (mad_stroke + 0.012)
        ink_score = (ink - med_ink) / (mad_ink + 0.015)
        dark_score = (med_ink_int - avg_ink_int) / (mad_ink_int + 3.0)

        bold = (
            stroke_score >= 0.85
            or dark_score >= 1.0
            or (stroke_score >= 0.45 and dark_score >= 0.55)
            or (ink_score >= 0.75 and dark_score >= 0.75)
        )

        metrics_by_word[id(word)] = WordMetrics(
            text=word.text,
            ink_ratio=round(ink, 4),
            avg_intensity=round(avg_int, 1),
            avg_ink_intensity=round(avg_ink_int, 1),
            stroke_thickness=round(stroke_norm, 4),
            bold=bold,
        )

    return [
        metrics_by_word.get(
            id(word),
            WordMetrics(text=word.text, ink_ratio=0.0, avg_intensity=255.0, stroke_thickness=0.0, bold=False),
        )
        for word in words
    ]


def ink_density_ratio(image: Image.Image, box: Box, *, dark_threshold: int = 200) -> float:
    m = measure_word(image, box)
    return m[0] if m else 0.0


def section_has_bold(styles: Sequence[WordMetrics]) -> bool:
    return any(s.bold for s in styles)


def bold_text_in_section(styles: Sequence[WordMetrics]) -> list[str]:
    seen: set[str] = set()
    bold: list[str] = []
    for s in styles:
        if s.bold and s.text not in seen:
            seen.add(s.text)
            bold.append(s.text)
    return bold


def value_is_bold(value: str, styles: Sequence[WordMetrics]) -> bool | None:
    norm = (value or "").strip().lower()
    if not norm:
        return None
    for s in styles:
        if s.bold and s.text.strip().lower() in norm:
            return True
    return False


def styles_to_dicts(styles: Sequence[WordMetrics]) -> list[dict[str, Any]]:
    return [
        {
            "text": s.text,
            "ink_ratio": s.ink_ratio,
            "avg_intensity": s.avg_intensity,
            "avg_ink_intensity": s.avg_ink_intensity,
            "stroke_thickness": s.stroke_thickness,
            "bold": s.bold,
        }
        for s in styles
    ]
