"""Tests for row-coupling vertical partition (unified grid vs parallel regions)."""

from __future__ import annotations

import unittest
from pathlib import Path

from ocr_word_to_line_boxes import load_vision, load_words, words_to_lines
from section_table_layout import analyze_vertical_partition

GALLERY = Path("/Users/nikhilmarepally/Desktop/docparse_layout_gallery/work")


@unittest.skipUnless(GALLERY.is_dir(), "layout gallery work dir not present")
class TestVerticalPartitionFixtures(unittest.TestCase):
    def _fw(self, stem: str):
        from PIL import Image

        work = GALLERY / stem
        vision = load_vision(work / "vision.json")
        pw = float(Image.open(work / "source.png").width)
        return words_to_lines(load_words(vision), page_width=pw), pw

    def test_amount_column_grid_unified(self) -> None:
        lines, pw = self._fw("risc_risc_sample_5_759ad153_759ad153_p1")
        vp = analyze_vertical_partition(lines, pw)
        self.assertTrue(vp.is_unified_grid)
        self.assertIn("amount_column_grid", vp.reasons)

    def test_prose_sidebar_parallel(self) -> None:
        lines, pw = self._fw("risc_risc_sample_5_16cbb1db_16cbb1db_p1")
        vp = analyze_vertical_partition(lines, pw)
        self.assertFalse(vp.is_unified_grid)
        self.assertTrue(vp.allows_column_split())

    def test_inline_bank_table_unified(self) -> None:
        lines, pw = self._fw("looker_cash_sale_receipt_777867dc_777867dc_p0")
        vp = analyze_vertical_partition(lines, pw)
        self.assertTrue(vp.is_unified_grid)


if __name__ == "__main__":
    unittest.main()
