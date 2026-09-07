"""Tests for YOLO-format section bounds export."""

from __future__ import annotations

import unittest

from app.yolo_bounds import build_page_yolo_bounds, section_to_yolo_entry


class TestYoloBounds(unittest.TestCase):
    def test_section_to_yolo_entry(self) -> None:
        section = {
            "index": 2,
            "label": "Header",
            "bounds": {"min_x": 100, "min_y": 50, "max_x": 300, "max_y": 150},
        }
        entry = section_to_yolo_entry(section, 1000, 2000)
        self.assertEqual(entry["class"], "section")
        self.assertEqual(entry["bbox"], [100.0, 50.0, 300.0, 150.0])
        self.assertEqual(entry["index"], 2)
        self.assertEqual(entry["label"], "Header")
        self.assertEqual(entry["confidence"], 1.0)
        self.assertAlmostEqual(entry["bbox_yolo"][0], 0.2)
        self.assertAlmostEqual(entry["bbox_yolo"][1], 0.05)
        self.assertAlmostEqual(entry["bbox_yolo"][2], 0.2)
        self.assertAlmostEqual(entry["bbox_yolo"][3], 0.05)

    def test_build_page_yolo_bounds_skips_empty_bounds(self) -> None:
        payload = build_page_yolo_bounds(
            [
                {"index": 1, "bounds": {"min_x": 0, "min_y": 0, "max_x": 10, "max_y": 10}},
                {"index": 0, "bounds": {}},
            ],
            job_id="abc123",
            page_index=0,
            image_width=100,
            image_height=100,
        )
        self.assertEqual(payload["job_id"], "abc123")
        self.assertEqual(payload["source"], "production_pipeline")
        self.assertEqual(len(payload["sections"]), 1)
        self.assertEqual(payload["sections"][0]["index"], 1)
        self.assertEqual(len(payload["detections"]), 1)


if __name__ == "__main__":
    unittest.main()
