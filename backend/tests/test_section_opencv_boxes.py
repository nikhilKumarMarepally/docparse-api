import json
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from form_box_detection import FormBox
from ocr_word_to_line_boxes import Box, Word
from section_opencv_boxes import detect_image_boxes_on_page, residual_box_is_image

ROOT = Path(__file__).resolve().parents[2]
HK_WORDS = (
    ROOT
    / "demo_output/opencv_risc_layout/occupancy_separator/downloads/ocr_cache/dl_1_sySXf5oSctvUO9XI5XsdVQ_words.json"
)
HK_IMAGE = Path("/Users/nikhilmarepally/Downloads/1_sySXf5oSctvUO9XI5XsdVQ.jpg")
PNG_W = 1800


def _scale_to_pipeline_width(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    if rgb.width == PNG_W:
        return rgb
    scale = PNG_W / float(rgb.width)
    h = max(1, int(round(rgb.height * scale)))
    return rgb.resize((PNG_W, h), Image.Resampling.LANCZOS)


def _load_hk_fixture() -> tuple[np.ndarray, list[Word]]:
    if not HK_IMAGE.is_file() or not HK_WORDS.is_file():
        raise unittest.SkipTest("HK invoice fixture image or OCR cache missing")
    img = _scale_to_pipeline_width(Image.open(HK_IMAGE))
    raw = json.loads(HK_WORDS.read_text())
    words = [
        Word(text=r["text"], box=Box(*r["box"]), index=r.get("index", i))
        for i, r in enumerate(raw)
    ]
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return bgr, words


class ResidualBoxIsImageTests(unittest.TestCase):
    def test_hk_invoice_totals_column_is_not_figure(self) -> None:
        bgr, words = _load_hk_fixture()
        totals_box = FormBox(
            x=1056,
            y=1687,
            w=567,
            h=548,
            area=567 * 548,
            contour_area=567 * 548,
            depth=0,
            confidence=0.94,
        )
        self.assertFalse(residual_box_is_image(bgr, totals_box, words))

    def test_colored_logo_without_ocr_stays_figure(self) -> None:
        bgr, words = _load_hk_fixture()
        logo_box = FormBox(
            x=66,
            y=64,
            w=186,
            h=211,
            area=186 * 211,
            contour_area=186 * 211,
            depth=0,
            confidence=0.94,
        )
        self.assertTrue(residual_box_is_image(bgr, logo_box, words))

    def test_detect_image_boxes_skips_hk_totals_column(self) -> None:
        bgr, words = _load_hk_fixture()
        boxes = detect_image_boxes_on_page(bgr, words)
        self.assertFalse(
            any(
                b.x >= 1000
                and b.y >= 1600
                and b.y2 >= 2100
                for b in boxes
            )
        )


if __name__ == "__main__":
    unittest.main()
