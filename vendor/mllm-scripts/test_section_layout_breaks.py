"""Tests for human-like section break detection."""

from __future__ import annotations

import unittest

from ocr_word_to_line_boxes import Box, Line, Word
from section_layout_breaks import (
    adaptive_gap_threshold,
    cluster_prose_lines,
    context_header_break,
    layout_regions_from_vision,
    lines_to_sections_human,
    should_break_section,
)


def _line(idx: int, text: str, y0: float, y1: float, x0: float = 60, x1: float = 500) -> Line:
    box = Box(x0, y0, x1, y1)
    w = Word(text=text, box=box, index=0)
    return Line(index=idx, text=text, box=box, words=[w])


class TestSectionLayoutBreaks(unittest.TestCase):
    def test_context_header_option(self) -> None:
        self.assertEqual(
            context_header_break("5 Amount Financed", "OPTION : You pay no finance charge"),
            "context_header",
        )

    def test_adaptive_gap_lower_than_fixed_18(self) -> None:
        gaps = [4.0] * 20 + [9.0, 4.0]
        thresh = adaptive_gap_threshold(gaps, floor=8.0)
        self.assertLess(thresh, 18.0)
        self.assertGreaterEqual(thresh, 8.0)

    def test_cluster_prose_keeps_every_line(self) -> None:
        lines = [
            _line(0, "para one", 100, 118),
            _line(1, "para one continued", 119, 136),
            _line(2, "Late Charge. If you pay late", 140, 158),
            _line(3, "more late charge text", 159, 176),
        ]
        chunks, _ = cluster_prose_lines(
            lines, page_width=1200, min_gap_px=8, pad=4
        )
        seen = {ln.index for group, _r, _b in chunks for ln in group}
        self.assertEqual(seen, {0, 1, 2, 3})

    def test_layout_band_break(self) -> None:
        prev = _line(0, "item", 100, 120)
        curr = _line(1, "OPTION :", 130, 150)
        d = should_break_section(
            prev,
            curr,
            10.0,
            gap_threshold=18.0,
            prev_key="grid:left",
            curr_key="fw:819",
            page_width=1200,
        )
        self.assertTrue(d.break_before)
        self.assertTrue(any(r.startswith("layout_band") for r in d.reasons))

    def test_column_flip_no_break(self) -> None:
        prev = _line(0, "left col", 100, 120, x0=60, x1=400)
        curr = _line(1, "right col", 125, 145, x0=1200, x1=1600)
        d = should_break_section(
            prev,
            curr,
            5.0,
            gap_threshold=18.0,
            prev_key="grid:left",
            curr_key="grid:right",
            page_width=1700,
        )
        self.assertFalse(d.break_before)

    def test_regions_from_fixture_shape(self) -> None:
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[5]
        vision_path = root / "risc_sample_5/c15433a5/layout/c15433a5_p1_vision.json"
        if not vision_path.exists():
            self.skipTest("risc p1 vision fixture not present")
        vision = json.loads(vision_path.read_text())
        regions = layout_regions_from_vision(vision, page_width=1700)
        kinds = {r.kind for r in regions}
        self.assertIn("full_width", kinds)
        self.assertGreaterEqual(len(regions), 4)

    def test_p1_human_more_than_two_sections(self) -> None:
        import json
        from pathlib import Path

        from ocr_word_to_line_boxes import load_vision, load_words, words_to_lines
        from PIL import Image

        root = Path(__file__).resolve().parents[5]
        vision_path = root / "risc_sample_5/c15433a5/layout/c15433a5_p1_vision.json"
        image_path = root / "risc_sample_5/c15433a5/c15433a5_p1.png"
        if not vision_path.exists() or not image_path.exists():
            self.skipTest("risc p1 fixtures not present")

        vision = load_vision(vision_path)
        with Image.open(image_path) as img:
            pw = img.width
            rgb = img.convert("RGB")
        lines = words_to_lines(load_words(vision), page_width=pw, full_width=True)
        chunks, meta = lines_to_sections_human(
            lines, vision=vision, page_width=float(pw), image_rgb=rgb
        )
        self.assertGreater(len(chunks), 2, meta.get("breaks"))
        reasons = {r for _lines, rs, _b in chunks for r in rs}
        self.assertTrue(
            any(r.startswith("layout_region:") for r in reasons),
            f"expected vision region sections, got {reasons}",
        )


if __name__ == "__main__":
    unittest.main()
