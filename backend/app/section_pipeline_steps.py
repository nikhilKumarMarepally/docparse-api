"""
Object-oriented section pipeline steps.

| Step | Class | Role |
|------|-------|------|
| 1 | B2ad4c9SectioningStep | b2ad4c9 process_page sectioning + horizontal-gap vendor code |
| 2 | VerticalTableMergeStep | Glue adjacent table row/header/footer shards vertically |
| 3 | HorizontalXGapSplitStep | Split non-table sections at L/R gutter when gap >> row word spacing |
| 4 | OpenCVImageBoxStep | Merge residual-ink OpenCV **image** boxes only (drop text panels) |
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from app.cloud_section_rules import sections_from_dicts
from app.paths import ensure_script_path
from app.section_pipeline_context import (
    merge_raw_table_sections,
    PageSectionContext,
    reindex_sections_globally,
    sections_from_merged_dicts,
    sections_objs_to_dicts,
)

ensure_script_path()

from section_opencv_boxes import (  # noqa: E402
    detect_image_boxes_on_page,
    merge_image_boxes_into_sections,
    split_sections_by_horizontal_x_gaps,
)


class SectionPipelineStep(ABC):
    """One geometry stage in the page section pipeline."""

    name: str
    snapshot_key: str

    @abstractmethod
    def run(self, ctx: PageSectionContext) -> None:
        ...


class B2ad4c9SectioningStep(SectionPipelineStep):
    """Step 1 — origin/main @ b2ad4c9 deploy (split_lines primary hv_combined)."""

    name = "b2ad4c9_sectioning"
    snapshot_key = "step1_b2ad4c9"

    def run(self, ctx: PageSectionContext) -> None:
        from app.b2ad4c9_sectioning import run_b2ad4c9_process_page_sectioning

        sections_obj, fw_lines, split_lines, gap_stats_obj, column_meta = (
            run_b2ad4c9_process_page_sectioning(
                ctx.vision,
                ctx.words,
                page_width=ctx.page_width,
                page_rgb=ctx.page_rgb,
            )
        )
        ctx.sections = sections_obj
        ctx.fw_lines = fw_lines
        ctx.split_lines = split_lines
        ctx.gap_stats_obj = gap_stats_obj
        ctx.column_meta = column_meta
        ctx.section_meta["step1_b2ad4c9"] = {
            "section_count": len(sections_obj),
            "fw_line_count": len(fw_lines),
            "split_line_count": len(split_lines),
            "column_mode": column_meta.get("mode"),
        }
        ctx.snapshot(self.snapshot_key)


class VerticalTableMergeStep(SectionPipelineStep):
    """Step 2 — Merge vertically adjacent table fragments."""

    name = "vertical_table_merge"
    snapshot_key = "step2_table_merge"

    def run(self, ctx: PageSectionContext) -> None:
        ctx.sections, ctx.line_pool = reindex_sections_globally(ctx.sections)
        raw_sections = sections_objs_to_dicts(ctx.sections)
        merged_sections, merge_count = merge_raw_table_sections(
            raw_sections,
            ctx.line_pool,
            page_width=float(ctx.page_width),
        )
        ctx.merge_count = merge_count
        if merge_count:
            ctx.section_meta[self.snapshot_key] = {"fragments_merged": merge_count}
        ctx.sections = sections_from_dicts(merged_sections, ctx.line_pool)
        ctx.snapshot(self.snapshot_key)


class HorizontalXGapSplitStep(SectionPipelineStep):
    """
    Step 3 — Split at the section left/right column gutter when horizontal whitespace
    between word clusters is far larger than typical inter-word gaps in OCR rows.
    """

    name = "horizontal_x_gap_split"
    snapshot_key = "step3_x_gap_split"

    def run(self, ctx: PageSectionContext) -> None:
        sections, split_meta = split_sections_by_horizontal_x_gaps(
            ctx.sections,
            ctx.fw_lines,
            [],
            page_width=float(ctx.page_width),
            lines=ctx.fw_lines,
        )
        ctx.sections = sections
        ctx.section_meta[self.snapshot_key] = split_meta
        ctx.snapshot(self.snapshot_key)


class OpenCVImageBoxStep(SectionPipelineStep):
    """Step 4 — residual-ink OpenCV figures only; ignore text boxes from that detector."""

    name = "opencv_image_boxes"
    snapshot_key = "step4_opencv_image_boxes"

    def run(self, ctx: PageSectionContext) -> None:
        import cv2
        import numpy as np

        rgb = np.asarray(ctx.page_rgb.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image_boxes = detect_image_boxes_on_page(bgr, ctx.words)
        ctx.sections, meta = merge_image_boxes_into_sections(
            ctx.sections,
            image_boxes,
            page_width=float(ctx.page_width),
            page_height=float(ctx.page_rgb.height),
        )
        fig_idx = list(meta.get("figure_section_indices") or [])
        orig_fig_boxes = [
            ctx.sections[i].box for i in fig_idx if 0 <= i < len(ctx.sections)
        ]
        raw_sections = sections_objs_to_dicts(ctx.sections)
        line_pool: dict[int, object] = {}
        for sec in ctx.sections:
            for ln in sec.lines:
                line_pool[int(ln.index)] = ln
        merged_sections, extra_merges = merge_raw_table_sections(
            raw_sections,
            line_pool or ctx.line_pool,
            page_width=float(ctx.page_width),
        )
        if extra_merges:
            ctx.merge_count += extra_merges
            ctx.sections = sections_from_merged_dicts(merged_sections, line_pool)
            new_fig: list[int] = []
            for i, sec in enumerate(ctx.sections):
                for fb in orig_fig_boxes:
                    if (
                        abs(sec.box.min_x - fb.min_x) < 1.0
                        and abs(sec.box.min_y - fb.min_y) < 1.0
                        and abs(sec.box.max_x - fb.max_x) < 1.0
                        and abs(sec.box.max_y - fb.max_y) < 1.0
                    ):
                        new_fig.append(i)
                        break
            meta["figure_section_indices"] = new_fig
            meta["table_fragments_merged"] = extra_merges
            meta["section_count"] = len(ctx.sections)
        for idx in meta.get("figure_section_indices") or []:
            ctx.section_meta[f"S{idx}"] = {"layout_kind": "figure"}
        ctx.section_meta[self.snapshot_key] = meta
        ctx.snapshot(self.snapshot_key)


DEFAULT_PIPELINE_STEPS: tuple[SectionPipelineStep, ...] = (
    B2ad4c9SectioningStep(),
    VerticalTableMergeStep(),
    HorizontalXGapSplitStep(),
)


def pipeline_steps() -> Sequence[SectionPipelineStep]:
    """Return active production geometry steps (OCR gaps only — no OpenCV)."""
    return DEFAULT_PIPELINE_STEPS
