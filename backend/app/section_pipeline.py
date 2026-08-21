"""Section pipeline handler — runs object-oriented geometry steps in order."""

from __future__ import annotations

from typing import Any, Sequence

from PIL import Image

from app.paths import ensure_script_path
from app.section_pipeline_context import MIN_GAP_PX, bounds_dicts_match, PageSectionContext
from app.section_pipeline_steps import pipeline_steps, SectionPipelineStep

ensure_script_path()

from ocr_line_to_sections import _make_section  # noqa: E402
from ocr_word_to_line_boxes import words_to_lines  # noqa: E402


class SectionPipelineHandler:
    """
    Runs section geometry step-by-step via ``SectionPipelineStep`` objects.

    | Step | Class | Snapshot key |
    |------|-------|--------------|
    | 1 | B2ad4c9SectioningStep | step1_b2ad4c9 |
    | 2 | VerticalTableMergeStep | step2_table_merge |
    | 3 | HorizontalXGapSplitStep | step3_x_gap_split |
    """

    def __init__(
        self,
        ctx: PageSectionContext,
        steps: Sequence[SectionPipelineStep] | None = None,
    ) -> None:
        self.ctx = ctx
        self._custom_steps = steps

    def run(self) -> PageSectionContext:
        ctx = self.ctx
        steps = self._custom_steps or pipeline_steps()
        if not ctx.section_meta:
            ctx.section_meta = {
                "unified_pipeline": True,
                "pipeline": "oop_three_step",
            }

        for step in steps:
            step.run(ctx)

        self._finalize_mode(ctx)
        return ctx

    def _finalize_mode(self, ctx: PageSectionContext) -> None:
        mode = ctx.column_meta.get("mode", "horizontal_gap")
        step3 = ctx.section_meta.get("step3_x_gap_split") or {}
        if int(step3.get("x_gap_split_section_count", 0)) > 0:
            mode = f"{mode}_x_gap_split"
        if ctx.merge_count:
            mode = f"{mode}_table_merge"
        ctx.section_meta["mode"] = mode
        if ctx.column_meta:
            ctx.section_meta["horizontal"] = ctx.column_meta
        if ctx.gap_stats_obj is not None:
            gs = ctx.gap_stats_obj
            ctx.section_meta["gap_stats"] = (
                gs.to_dict() if hasattr(gs, "to_dict") else gs
            )


def run_page_section_pipeline(
    vision: dict[str, Any],
    words: list[Any],
    *,
    page_width: float,
    page_rgb: Image.Image,
    record_snapshots: bool = False,
) -> PageSectionContext:
    ctx = PageSectionContext(
        vision=vision,
        words=words,
        page_width=page_width,
        page_rgb=page_rgb,
        record_snapshots=record_snapshots,
    )
    return SectionPipelineHandler(ctx).run()


def render_production_sections(
    lines: list[Any],
    *,
    page_width: float,
    full_width_lines: list[Any] | None = None,
    min_gap_px: float = MIN_GAP_PX,
) -> tuple[list[Any], Any, dict[str, Any]]:
    """
    Render step-1 only: ``lines_to_sections_hv_combined``.

    See ``B2ad4c9SectioningStep`` for origin/main vs live API line inputs.
    """
    from ocr_line_to_sections import lines_to_sections_hv_combined

    return lines_to_sections_hv_combined(
        lines,
        page_width=float(page_width),
        full_width_lines=full_width_lines,
        min_gap_px=min_gap_px,
    )
