import unittest

from app.section_merge import merge_table_fragments, should_merge_table_fragments
from ocr_word_to_line_boxes import Box, Line, Word


def _word(text: str, x0: float, x1: float, y: float, idx: int) -> Word:
    return Word(text=text, box=Box(x0, y, x1, y + 12.0), index=idx)


def _line(idx: int, y: float, cells: list[tuple[str, float, float]]) -> Line:
    words = [_word(t, x0, x1, y, idx * 10 + i) for i, (t, x0, x1) in enumerate(cells)]
    box = words[0].box
    for w in words[1:]:
        box = box.union(w.box)
    return Line(index=idx, text=" ".join(w.text for w in words), box=box, words=words)


def _section(index: int, lines: list[Line], kind: str) -> dict:
    box = lines[0].box
    for ln in lines[1:]:
        box = box.union(ln.box)
    return {
        "index": index,
        "line_indices": [ln.index for ln in lines],
        "line_count": len(lines),
        "text": "\n".join(ln.text for ln in lines),
        "bounds": box.to_dict(),
        "layout_kind": kind,
        "layout_detect": {
            "layout_kind": kind,
            "aligned_column_count": 3 if kind != "prose" else 0,
            "row_count": len(lines),
        },
    }


class TableFragmentMergeTests(unittest.TestCase):
    def test_stacked_aligned_table_shards_merge(self) -> None:
        def row(idx: int, y: float, a: str, b: str, c: str) -> Line:
            return _line(
                idx,
                y,
                [
                    (a, 80, 100),
                    ("x", 104, 118),
                    (b, 280, 300),
                    ("y", 304, 318),
                    (c, 480, 500),
                    ("z", 504, 518),
                ],
            )

        upper = [row(0, 100, "A", "B", "C"), row(1, 130, "1", "2", "3"), row(2, 160, "4", "5", "6")]
        lower = [row(3, 200, "7", "8", "9"), row(4, 230, "0", "1", "2"), row(5, 260, "3", "4", "5")]
        lines = upper + lower
        left = _section(0, upper, "section_table")
        right = _section(1, lower, "section_table")
        self.assertTrue(
            should_merge_table_fragments(
                left, right, lines, gap_threshold=40.0, page_width=600.0
            )
        )
        merged = merge_table_fragments(
            [left, right], lines, gap_threshold=40.0, page_width=600.0
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["bounds"]["min_y"], 100.0)
        self.assertEqual(merged[0]["bounds"]["max_y"], 272.0)

    def test_stacked_tables_with_different_column_x_do_not_merge(self) -> None:
        upper = [
            _line(0, 100, [("A", 80, 110), ("B", 480, 510)]),
            _line(1, 130, [("1", 80, 110), ("2", 480, 510)]),
            _line(2, 160, [("3", 80, 110), ("4", 480, 510)]),
        ]
        lower = [
            _line(3, 200, [("X", 220, 250), ("Y", 340, 370)]),
            _line(4, 230, [("5", 220, 250), ("6", 340, 370)]),
            _line(5, 260, [("7", 220, 250), ("8", 340, 370)]),
        ]
        lines = upper + lower
        left = _section(0, upper, "section_table")
        right = _section(1, lower, "section_table")
        self.assertFalse(
            should_merge_table_fragments(
                left, right, lines, gap_threshold=40.0, page_width=600.0
            )
        )
        merged = merge_table_fragments(
            [left, right], lines, gap_threshold=40.0, page_width=600.0
        )
        self.assertEqual(len(merged), 2)

    def test_table_above_prose_paragraph_does_not_merge(self) -> None:
        table = [
            _line(0, 100, [("A", 80, 110), ("B", 280, 310), ("C", 480, 510)]),
            _line(1, 130, [("1", 80, 110), ("2", 280, 310), ("3", 480, 510)]),
            _line(2, 160, [("4", 80, 110), ("5", 280, 310), ("6", 480, 510)]),
        ]
        prose = [
            _line(3, 210, [("The", 80, 115)]),
            _line(4, 232, [("quick", 80, 130)]),
            _line(5, 254, [("brown", 80, 128)]),
            _line(6, 276, [("fox", 80, 112)]),
        ]
        lines = table + prose
        left = _section(0, table, "section_table")
        right = _section(1, prose, "prose")
        right["layout_detect"]["aligned_column_count"] = 0
        self.assertFalse(
            should_merge_table_fragments(
                left, right, lines, gap_threshold=40.0, page_width=600.0
            )
        )


if __name__ == "__main__":
    unittest.main()
