"""
Sectioning copied verbatim from deployed Render commit b2ad4c9.

Uses frozen vendor code in ``vendor/mllm-scripts-b2ad4c9`` so step 1 matches
``git show b2ad4c9`` output even when live ``vendor/mllm-scripts`` has evolved.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from app.b2ad4c9_vendor_loader import get_b2ad4c9_vendor_modules
from app.paths import ensure_script_path

ensure_script_path()

from section_layout_breaks import lines_to_sections_human  # noqa: E402
from section_table_layout import classify_section_layout  # noqa: E402


def run_b2ad4c9_process_page_sectioning(
    vision: dict[str, Any],
    words: list[Any],
    *,
    page_width: float,
    page_rgb: Image.Image,
) -> tuple[list[Any], list[Any], list[Any], Any, dict[str, Any]]:
    """
    Exact ``process_page`` sectioning block from origin/main @ b2ad4c9.

    ``split_lines`` primary → ``lines_to_sections_hv_combined`` with frozen vendor.
    """
    ocr_line_to_sections, ocr_word_to_line_boxes = get_b2ad4c9_vendor_modules()
    lines_to_sections_hv_combined = ocr_line_to_sections.lines_to_sections_hv_combined
    _make_section = ocr_line_to_sections._make_section
    words_to_lines = ocr_word_to_line_boxes.words_to_lines

    use_human_layout = bool((vision.get("image_blocks") or {}).get("with_polygons"))
    fw_lines = words_to_lines(words, page_width=page_width, full_width=True)
    split_lines = words_to_lines(
        words,
        page_width=page_width,
        full_width=False,
        split_columns=True,
    )

    if use_human_layout:
        human_chunks, _section_meta = lines_to_sections_human(
            split_lines,
            vision=vision,
            page_width=float(page_width),
            image_rgb=page_rgb,
            min_gap_px=8.0,
        )
        sections_obj = [
            _make_section(i, line_group, None, 6.0)
            for i, (line_group, _, _) in enumerate(human_chunks)
        ]
        gap_stats_obj = None
        column_meta: dict[str, Any] = {"mode": "human_layout"}
    else:
        sections_obj, gap_stats_obj, column_meta = lines_to_sections_hv_combined(
            split_lines,
            page_width=float(page_width),
            full_width_lines=fw_lines,
            min_gap_px=18.0,
        )

    return sections_obj, fw_lines, split_lines, gap_stats_obj, column_meta


def sections_to_dicts(sections_obj: list[Any]) -> list[dict[str, Any]]:
    """Same as b2ad4c9 raw_sections write to sections.json."""
    return [
        s.to_dict(layout=classify_section_layout(s.lines).to_dict())
        for s in sections_obj
    ]
