import json
import os
import unittest
from pathlib import Path

from ocr_word_to_line_boxes import (
    Box,
    Word,
    is_vertical_angle,
    split_row_by_horizontal_gaps,
    words_to_lines,
    load_words,
)
from ocr_line_to_sections import (
    Section,
    _attach_margin_vertical_sections,
    _is_margin_vertical_line,
    _make_section,
    _partition_margin_vertical_lines,
    _strip_margin_words_from_lines,
    combine_flowing_sections,
    lines_to_sections_hv_combined,
    split_prose_sections_by_lanes,
)
from ocr_word_to_line_boxes import Line

PAGE_000_VISION_CANDIDATES = [
    Path(os.environ.get("PAGE_000_VISION", "")),
    Path(
        "/Users/nikhilmarepally/Desktop/classification/nikhil_desktop/"
        "doc-extract-render-prod/output/arxiv_vibevoice_page_000/opencv_bounds/page_000/vision.json"
    ),
    Path(
        "/Users/nikhilmarepally/Downloads/opencv_demo_user_confirm/"
        "page_000-e7938c9d-097a-47c4-9e85-9f497b2abe39.png/vision.json"
    ),
    Path("/Users/nikhilmarepally/Downloads/opencv_demo/page_000.png/vision.json"),
]


def _page_000_vision_path() -> Path | None:
    for path in PAGE_000_VISION_CANDIDATES:
        if path.is_file():
            return path
    return None


def _median_positive_word_gap(line: Line) -> float:
    gaps = [
        line.words[i + 1].box.min_x - line.words[i].box.max_x
        for i in range(len(line.words) - 1)
    ]
    positive = [g for g in gaps if g > 0]
    if not positive:
        return 0.0
    return float(sorted(positive)[len(positive) // 2])


def _w(text: str, x0: float, x1: float, y: float = 100.0, idx: int = 0) -> Word:
    return Word(text=text, box=Box(x0, y, x1, y + 14.0), index=idx)


def _line(text: str, x0: float, x1: float, y: float, idx: int) -> Line:
    w = _w(text, x0, x1, y, idx)
    return Line(index=idx, text=text, box=w.box, words=[w])


class HorizontalGapSplitTests(unittest.TestCase):
    def test_split_row_at_wide_gaps(self) -> None:
        row = [
            _w("GitHub", 40, 90, idx=0),
            _w("Code", 95, 130, idx=1),
            _w("Demo", 260, 300, idx=2),
            _w("HuggingFace", 420, 520, idx=3),
        ]
        chunks = split_row_by_horizontal_gaps(row, prose=True, min_gap_px=20.0)
        self.assertGreaterEqual(len(chunks), 3)

    def test_words_to_lines_splits_link_row(self) -> None:
        words = [
            _w("GitHub", 40, 90, idx=0),
            _w("Code", 95, 130, idx=1),
            _w("Demo", 260, 300, idx=2),
            _w("Models", 420, 480, idx=3),
        ]
        lines = words_to_lines(words, split_columns=True, full_width=False)
        self.assertGreaterEqual(len(lines), 2)

    def test_split_prose_section_side_by_side_lanes(self) -> None:
        left = _line("DER chart", 40, 280, 200, 0)
        right = _line("tcpWER chart", 420, 680, 200, 1)
        sec = _make_section(0, [left, right], None, 6.0)
        parts = split_prose_sections_by_lanes([sec], pad=6.0)
        self.assertEqual(len(parts), 2)

    def test_margin_vertical_line_detection(self) -> None:
        # Vertical margin tokens are narrow (high aspect), not one wide horizontal word box.
        margin = _line("arXiv", 50, 62, 1200, 0)
        body = _line("Abstract paragraph text", 350, 1400, 820, 1)
        self.assertTrue(_is_margin_vertical_line(margin, 1800.0))
        self.assertFalse(_is_margin_vertical_line(body, 1800.0))
        peeled_body, peeled_margin = _partition_margin_vertical_lines([margin, body], 1800.0)
        self.assertEqual(len(peeled_body), 1)
        self.assertEqual(len(peeled_margin), 1)
        self.assertEqual(peeled_margin[0].text, "arXiv")

    def test_margin_vertical_splits_from_horizontal_body_by_column_gap(self) -> None:
        """Wide horizontal gutter between margin stamp and prose must not merge sections."""
        margin_lines = [
            _line("arXiv", 48, 60, 650, 0),
            _line("2601", 48, 60, 820, 1),
            _line("[", 48, 55, 1050, 2),
        ]
        abstract_lines = [
            _line(
                "This report presents VIBE VOICE-ASR, a general-purpose speech",
                370,
                1400,
                720,
                3,
            ),
            _line(
                "understanding framework built upon VIBEVOICE.",
                370,
                1400,
                752,
                4,
            ),
        ]
        interleaved = [abstract_lines[0], margin_lines[0], abstract_lines[1], margin_lines[1]]
        page_width = 1800.0
        secs, _, _ = lines_to_sections_hv_combined(
            interleaved,
            page_width=page_width,
            full_width_lines=interleaved,
        )
        margin_sec = next(s for s in secs if "arXiv" in s.text or "2601" in s.text)
        body_sec = next(s for s in secs if "This report presents" in s.text)
        self.assertIsNot(margin_sec.index, body_sec.index)
        column_gap = body_sec.box.min_x - margin_sec.box.max_x
        word_gap = _median_positive_word_gap(body_sec.lines[0])
        self.assertGreater(column_gap, word_gap * 3.0)
        self.assertGreater(column_gap, 72.0)

    def test_same_y_margin_and_body_lines_stay_separate(self) -> None:
        margin = _line("arXiv", 50, 62, 800, 0)
        body = _line("This report presents VIBE VOICE-ASR", 380, 1400, 800, 1)
        secs, _, _ = lines_to_sections_hv_combined(
            [margin, body],
            page_width=1800.0,
            full_width_lines=[margin, body],
        )
        self.assertGreaterEqual(len(secs), 2)
        texts = [s.text for s in secs]
        self.assertTrue(any("arXiv" in t for t in texts))
        self.assertTrue(any("This report" in t for t in texts))

    @unittest.skipUnless(_page_000_vision_path(), "page_000 vision.json not available")
    def test_page_000_margin_vertical_and_abstract_are_split(self) -> None:
        """Regression: arXiv sidebar vs abstract — peeled margin + attach, column gap >> word spacing."""
        vision_path = _page_000_vision_path()
        vision = json.loads(vision_path.read_text())
        words = load_words(vision)
        page_width = 1800.0
        fw = words_to_lines(words, page_width=page_width, full_width=True)
        split = words_to_lines(
            words, page_width=page_width, full_width=False, split_columns=True
        )
        secs, _, meta = lines_to_sections_hv_combined(
            split, page_width=page_width, full_width_lines=fw
        )
        margin = next(s for s in secs if "arXiv" in s.text or "2601" in s.text)
        abstract = next(
            s for s in secs
            if "presents VIBE VOICE" in s.text or "This report presents" in s.text
        )
        self.assertIsNot(margin.index, abstract.index)
        self.assertLess(margin.box.max_x, page_width * 0.08)
        column_gap = abstract.box.min_x - margin.box.max_x
        word_gap = _median_positive_word_gap(abstract.lines[0])
        self.assertGreater(column_gap, 72.0)
        self.assertGreater(word_gap, 0.0)
        self.assertGreater(column_gap, word_gap * 3.0)

    @unittest.skipUnless(_page_000_vision_path(), "page_000 vision.json not available")
    def test_page_000_vertical_margin_words_have_orientation(self) -> None:
        vision_path = _page_000_vision_path()
        vision = json.loads(vision_path.read_text())
        words = load_words(vision)
        vertical = [
            w for w in words if w.angle_deg is not None and is_vertical_angle(w.angle_deg)
        ]
        self.assertGreaterEqual(len(vertical), 5)
        margin_vertical = [w for w in vertical if w.box.max_x <= 1800.0 * 0.12]
        self.assertGreaterEqual(len(margin_vertical), 3)

    def test_combine_flowing_sections_merges_prose_shards(self) -> None:
        heading = _line("2.3.1 Pre-training", 316, 500, 1265, 0)
        p1 = _line("First paragraph of body text here.", 316, 900, 1325, 1)
        p2 = _line("Second paragraph continues below.", 316, 900, 1360, 2)
        secs = [
            _make_section(0, [heading], None, 6.0),
            _make_section(1, [p1], 29.0, 6.0),
            _make_section(2, [p2], 21.0, 6.0),
        ]
        pool = [heading, p1, p2]
        merged, n = combine_flowing_sections(secs, pad=6.0, line_pool=pool)
        self.assertGreaterEqual(n, 2)
        self.assertEqual(len(merged), 1)
        self.assertIn("Pre-training", merged[0].text)
        self.assertIn("Second paragraph", merged[0].text)


if __name__ == "__main__":
    unittest.main()
