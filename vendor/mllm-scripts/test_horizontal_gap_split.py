import unittest

from ocr_word_to_line_boxes import Box, Word, split_row_by_horizontal_gaps, words_to_lines
from ocr_line_to_sections import Section, _make_section, split_prose_sections_by_lanes
from ocr_word_to_line_boxes import Line


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


if __name__ == "__main__":
    unittest.main()
