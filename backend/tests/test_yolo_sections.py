"""Tests for YOLO section detection helpers."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.yolo_bounds import build_page_yolo_bounds, entries_to_detections, section_to_yolo_entry
from app.yolo_sections import detect_opencv_sections, detect_sections_from_image_bytes


class TestYoloSections(unittest.TestCase):
    def test_section_to_yolo_entry_includes_bounds(self) -> None:
        section = {
            "index": 0,
            "bounds": {"min_x": 10, "min_y": 20, "max_x": 110, "max_y": 120},
        }
        entry = section_to_yolo_entry(section, 200, 200)
        self.assertEqual(entry["bounds"]["max_x"], 110.0)

    def test_entries_to_detections(self) -> None:
        entries = [
            {
                "index": 1,
                "class": "section",
                "confidence": 0.95,
                "bounds": {"min_x": 0, "min_y": 0, "max_x": 10, "max_y": 10},
                "label": "section_1",
            }
        ]
        detections = entries_to_detections(entries)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["index"], 1)
        self.assertEqual(detections[0]["bounds"]["max_x"], 10)

    def test_build_page_yolo_bounds_includes_detections(self) -> None:
        payload = build_page_yolo_bounds(
            [{"index": 0, "bounds": {"min_x": 0, "min_y": 0, "max_x": 5, "max_y": 5}}],
            job_id="job1",
            page_index=0,
            image_width=100,
            image_height=100,
        )
        self.assertEqual(len(payload["detections"]), 1)

    def test_detect_opencv_sections_on_blank_image(self) -> None:
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        detections = detect_opencv_sections(image)
        self.assertIsInstance(detections, list)

    def test_detect_sections_from_image_bytes(self) -> None:
        img = Image.new("RGB", (80, 60), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        payload = detect_sections_from_image_bytes(buf.getvalue())
        self.assertEqual(payload["image_width"], 80)
        self.assertEqual(payload["image_height"], 60)
        self.assertIn(payload["source"], {"opencv", "opencv_fallback", "yolo"})
        self.assertIn("detections", payload)

    @patch("app.yolo_sections.yolo_model_path", return_value=None)
    def test_detect_sections_from_image_bytes_source_opencv(self, _mock: object) -> None:
        img = Image.new("RGB", (40, 40), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        payload = detect_sections_from_image_bytes(buf.getvalue())
        self.assertEqual(payload["source"], "opencv")


if __name__ == "__main__":
    unittest.main()
