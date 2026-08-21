"""Ensure pipeline outputs are JSON-serializable (job result.json / API payloads)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from PIL import Image

from app.paths import ensure_script_path

ensure_script_path()

from ocr_line_to_sections import gap_stats  # noqa: E402
from ocr_word_to_line_boxes import load_words  # noqa: E402

from app.pipeline import run_pipeline
from app.section_pipeline import SectionPipelineHandler, run_page_section_pipeline
from app.section_pipeline_context import PageSectionContext


def _word(text: str, x0: float, y0: float, x1: float, y1: float) -> dict[str, Any]:
    return {
        "text": text,
        "bounds": [
            {"x": x0, "y": y0},
            {"x": x1, "y": y0},
            {"x": x1, "y": y1},
            {"x": x0, "y": y1},
        ],
    }


def table_like_vision(*, width: int = 489, height: int = 408) -> dict[str, Any]:
    """Synthetic page similar to a stats table + footnote (triggers gap_stats)."""
    words: list[dict[str, Any]] = []

    header_y = 12
    for i, label in enumerate(("Results", "from", "the", "1998", "Survey")):
        x = 10 + i * 72
        words.append(_word(label, x, header_y, x + 60, header_y + 18))

    table_y = 55
    for row in range(8):
        y = table_y + row * 28
        for col, cell in enumerate(("Income", f"{row * 10}", f"{48.8 - row}")):
            x = 10 + col * 140
            words.append(_word(cell, x, y, x + 90, y + 20))

    body_y = 330
    for i, chunk in enumerate(
        (
            "Declines",
            "were",
            "spread",
            "across",
            "most",
            "groups",
            "except",
            "families",
            "with",
            "incomes",
            "of",
            "$100,000",
            "or",
            "more.",
        )
    ):
        x = 10 + (i % 7) * 62
        y = body_y + (i // 7) * 22
        words.append(_word(chunk, x, y, x + 55, y + 18))

    return {
        "dimensions": {"width": width, "height": height},
        "ocr_data": {
            "document_text": {
                "words": words,
            }
        },
    }


def assert_json_serializable(obj: Any, *, path: str = "$") -> None:
    """Fail with a clear path if obj cannot be encoded as JSON."""
    try:
        json.dumps(obj)
    except TypeError as exc:
        raise AssertionError(f"{path} is not JSON-serializable: {exc}") from exc


def walk_json_serializable(obj: Any, *, path: str = "$") -> None:
    """Recursively verify JSON encoding (catches nested dataclass leaks)."""
    assert_json_serializable(obj, path=path)
    if isinstance(obj, dict):
        for key, value in obj.items():
            walk_json_serializable(value, path=f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            walk_json_serializable(item, path=f"{path}[{i}]")


class TestPipelineJsonSerialization(unittest.TestCase):
    def test_finalize_mode_stores_gap_stats_as_dict(self) -> None:
        """Regression: GapStats in section_meta broke result.json on Render."""
        vision = table_like_vision()
        words = load_words(vision)
        page_rgb = Image.new("RGB", (489, 408), "white")
        ctx = PageSectionContext(
            vision=vision,
            words=words,
            page_width=float(vision["dimensions"]["width"]),
            page_rgb=page_rgb,
            column_meta={"mode": "gap"},
            gap_stats_obj=gap_stats([]),
        )

        SectionPipelineHandler(ctx, steps=())._finalize_mode(ctx)

        gap_stats_value = ctx.section_meta.get("gap_stats")
        self.assertIsInstance(gap_stats_value, dict)
        walk_json_serializable(ctx.section_meta, path="section_meta")

    def test_run_page_section_pipeline_section_meta_serializable(self) -> None:
        vision = table_like_vision()
        words = load_words(vision)
        page_rgb = Image.new("RGB", (489, 408), "white")

        ctx = run_page_section_pipeline(
            vision,
            words,
            page_width=float(vision["dimensions"]["width"]),
            page_rgb=page_rgb,
        )

        self.assertGreater(len(ctx.sections), 0)
        walk_json_serializable(ctx.section_meta, path="section_meta")

    def test_run_pipeline_result_serializable_with_skip_llm(self) -> None:
        vision = table_like_vision()
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            page_dir = job_dir / "page_000"
            page_dir.mkdir(parents=True)
            page_png = page_dir / "page.png"
            Image.new("RGB", (489, 408), "white").save(page_png)
            (page_dir / "vision.json").write_text(json.dumps(vision))

            class _SkipExtractor:
                model = "test-skip"

            with mock.patch("app.pipeline.get_extractor", return_value=_SkipExtractor()):
                with mock.patch("app.pipeline.run_ocr", return_value=vision):
                    result = run_pipeline(
                        "testjob",
                        job_dir,
                        [page_png],
                        filename="table.jpeg",
                        skip_llm=True,
                    )

            walk_json_serializable(result, path="result")
            self.assertEqual(result["status"], "completed")
            page = result["pages"][0]
            self.assertIn("section_meta", page)
            gap_stats_value = page["section_meta"].get("gap_stats")
            self.assertIsInstance(gap_stats_value, dict)


if __name__ == "__main__":
    unittest.main()
