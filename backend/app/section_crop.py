"""Section crop + enhancement for Gemini (standalone, no gallery deps)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

MIN_SECTION_CROP_WIDTH = 1200


def enhance_section_crop(img: Image.Image) -> Image.Image:
    out = img.convert("RGB")
    w, h = out.size
    if w < MIN_SECTION_CROP_WIDTH:
        ratio = MIN_SECTION_CROP_WIDTH / w
        out = out.resize(
            (MIN_SECTION_CROP_WIDTH, max(1, int(h * ratio))),
            Image.Resampling.LANCZOS,
        )
    out = ImageEnhance.Contrast(out).enhance(1.12)
    out = ImageEnhance.Sharpness(out).enhance(1.15)
    return out


def crop_section_image(full_png: Path, bounds: dict[str, Any], *, padding: int = 12) -> Image.Image:
    with Image.open(full_png) as img:
        rgb = img.convert("RGB")
        x0 = max(0, int(bounds.get("min_x", 0)) - padding)
        y0 = max(0, int(bounds.get("min_y", 0)) - padding)
        x1 = min(rgb.width, int(bounds.get("max_x", 0)) + padding)
        y1 = min(rgb.height, int(bounds.get("max_y", 0)) + padding)
        return rgb.crop((x0, y0, x1, y1))
