"""Tests for vision word bounds parsing."""

from __future__ import annotations

import unittest

from ocr_word_to_line_boxes import parse_bounds


class TestParseBounds(unittest.TestCase):
    def test_polygon_missing_y_on_some_vertices(self) -> None:
        raw = [{"x": 583}, {"x": 584, "y": 10}, {"x": 553, "y": 11}, {"x": 552}]
        box = parse_bounds(raw)
        self.assertIsNotNone(box)
        assert box is not None
        self.assertEqual(box.min_x, 552.0)
        self.assertEqual(box.min_y, 10.0)

    def test_polygon_only_x_returns_none(self) -> None:
        self.assertIsNone(parse_bounds([{"x": 1}, {"x": 2}]))


if __name__ == "__main__":
    unittest.main()
