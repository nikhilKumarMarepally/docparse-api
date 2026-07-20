from __future__ import annotations

from typing import Any, Protocol, Sequence

from PIL import Image


class VisionExtractor(Protocol):
    """Protocol for section-level value extraction (Gemini now, Qwen later)."""

    def extract_section(
        self,
        crop_img: Image.Image,
        ocr_text: str,
        *,
        section_words: Sequence[Any] | None = None,
        bounds: dict[str, Any] | None = None,
        table_band: bool = False,
    ) -> dict[str, Any]: ...


EXTRACT_PROVIDER = "gemini"  # future: qwen
