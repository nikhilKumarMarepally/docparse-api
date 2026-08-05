"""
Object-oriented section pipeline steps.

| Step | Class | Role |
|------|-------|------|
| 1 | B2ad4c9SectioningStep | b2ad4c9 process_page sectioning + horizontal-gap vendor code |
| 2 | VerticalTableMergeStep | Glue adjacent table row/header/footer shards vertically |
| 3 | HorizontalXGapSplitStep | Split non-table sections at L/R gutter when gap >> row word spacing |
| 4 | OpenCvHorizontalBoxSplitStep | Section bounds = OpenCV printed frames (confidence >= 0.9) |
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import cv2
import numpy as np

from app.cloud_section_rules import sections_from_dicts
from app.paths import ensure_script_path
from app.section_pipeline_context import (
    merge_raw_table_sections,
    PageSectionContext,
    reindex_sections_globally,
    sections_objs_to_dicts,
)

ensure_script_path()

from section_opencv_boxes import (  # noqa: E402
    detect_merged_high_confidence_sections,
    opencv_box_min_confidence,
    redact_ocr_words,
    sections_from_opencv_boxes_direct,
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

        # Detect OpenCV boxes here so step-2 review PDF can overlay them before step 3 runs.
        if not ctx.opencv_boxes:
            img_bgr = cv2.cvtColor(np.array(ctx.page_rgb), cv2.COLOR_RGB2BGR)
            min_conf = opencv_box_min_confidence()
            redacted_bgr = redact_ocr_words(img_bgr, ctx.words)
            ctx.opencv_boxes = detect_merged_high_confidence_sections(
                img_bgr,
                ctx.words,
                min_confidence=min_conf,
            )
            ctx.section_meta["opencv_detect"] = {
                "box_count": len(ctx.opencv_boxes),
                "min_confidence": min_conf,
                "high_confidence_boxes": sum(
                    1 for b in ctx.opencv_boxes if b.confidence >= 0.9
                ),
            }
            ctx.section_meta["opencv_redacted_image_shape"] = list(redacted_bgr.shape)

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
            ctx.opencv_boxes,
            page_width=float(ctx.page_width),
            lines=ctx.fw_lines,
        )
        ctx.sections = sections
        ctx.section_meta[self.snapshot_key] = split_meta
        ctx.snapshot(self.snapshot_key)


class OpenCvHorizontalBoxSplitStep(SectionPipelineStep):
    """
    Step 4 — Section bounds = OpenCV printed frames (confidence >= 0.9).
    Lines assigned to the best-matching box; bounds taken directly from OpenCV.
    """

    name = "opencv_horizontal_box_split"
    snapshot_key = "step4_opencv_split"

    def run(self, ctx: PageSectionContext) -> None:
        if not ctx.opencv_boxes:
            img_bgr = cv2.cvtColor(np.array(ctx.page_rgb), cv2.COLOR_RGB2BGR)
            min_conf = opencv_box_min_confidence()
            ctx.opencv_boxes = detect_merged_high_confidence_sections(
                img_bgr,
                ctx.words,
                min_confidence=min_conf,
            )

        min_conf = opencv_box_min_confidence()
        sections, split_meta = sections_from_opencv_boxes_direct(
            ctx.sections,
            ctx.opencv_boxes,
            page_width=float(ctx.page_width),
            page_height=float(ctx.page_rgb.height),
            min_confidence=min_conf,
        )
        ctx.sections = sections
        ctx.section_meta[self.snapshot_key] = split_meta
        ctx.snapshot(self.snapshot_key)


# Active pipeline order — comment out any line to skip that step.
# Example: only b2ad4c9 → use (B2ad4c9SectioningStep(),)
# Example: skip OpenCV split → omit OpenCvHorizontalBoxSplitStep()
# Runtime shortcut: include_optional=False runs only steps 1–3 (no step 4 OpenCV).
DEFAULT_PIPELINE_STEPS: tuple[SectionPipelineStep, ...] = (
    B2ad4c9SectioningStep(),           # step 1 — frozen b2ad4c9 deploy sectioning
    VerticalTableMergeStep(),          # step 2 — vertical table fragment merge
    HorizontalXGapSplitStep(),         # step 3 — L/R gutter split when gap >> row word spacing
    OpenCvHorizontalBoxSplitStep(),    # step 4 — OpenCV box bounds >= 0.9
)


def pipeline_steps(include_optional: bool = True) -> Sequence[SectionPipelineStep]:
    """Return active steps; step 4 (OpenCV split) is optional."""
    if include_optional:
        return DEFAULT_PIPELINE_STEPS
    return DEFAULT_PIPELINE_STEPS[:3]
