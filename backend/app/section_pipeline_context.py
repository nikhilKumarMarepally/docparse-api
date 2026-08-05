"""Shared context and helpers for the section pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from app.paths import ensure_script_path

ensure_script_path()

from section_table_layout import classify_section_layout  # noqa: E402

from ocr_line_to_sections import (  # noqa: E402
    _make_section,
    _partition_margin_vertical_lines,
    _strip_margin_words_from_lines,
    gap_stats,
    Section,
)
from ocr_word_to_line_boxes import Line  # noqa: E402
from app.section_merge import merge_table_fragments  # noqa: E402

MIN_GAP_PX = 18.0


def reindex_sections_globally(
    sections: list[Section],
    *,
    pad: float = 6.0,
) -> tuple[list[Section], dict[int, Line]]:
    """
    Assign unique global line indices across sections.

    b2ad4c9 (and hv_combined) sections reuse per-section indices (0..n); dict round-trips
    must not share one pool keyed by line.index.
    """
    pool: dict[int, Line] = {}
    out: list[Section] = []
    next_idx = 0
    for sec in sections:
        new_lines: list[Line] = []
        for ln in sec.lines:
            new_line = Line(index=next_idx, text=ln.text, box=ln.box, words=ln.words)
            pool[next_idx] = new_line
            new_lines.append(new_line)
            next_idx += 1
        out.append(_make_section(len(out), new_lines, sec.gap_above, pad))
    return out, pool


def sections_objs_to_dicts(sections_obj: list[Section]) -> list[dict[str, Any]]:
    dicts: list[dict[str, Any]] = []
    for s in sections_obj:
        d = s.to_dict(layout=classify_section_layout(s.lines).to_dict())
        d["line_indices"] = [ln.index for ln in s.lines]
        dicts.append(d)
    return dicts


def section_line_index_pool(
    fw_lines: list[Any],
    body_sl: list[Any],
    sections_obj: list[Section],
) -> dict[int, Any]:
    pool: dict[int, Any] = {}
    for sec in sections_obj:
        for ln in sec.lines:
            pool[ln.index] = ln
    for ln in fw_lines:
        pool.setdefault(ln.index, ln)
    for ln in body_sl:
        pool.setdefault(ln.index, ln)
    return pool


def merge_raw_table_sections(
    raw_sections: list[dict[str, Any]],
    lines: list[Any] | dict[int, Any],
    *,
    page_width: float,
) -> tuple[list[dict[str, Any]], int]:
    if len(raw_sections) < 2:
        return raw_sections, 0
    line_list = list(lines.values()) if isinstance(lines, dict) else lines
    ordered = sorted(line_list, key=lambda ln: ln.content_box.min_y)
    stats = gap_stats(ordered, multiplier=2.0, min_gap_px=MIN_GAP_PX)
    before = len(raw_sections)
    merged = merge_table_fragments(
        raw_sections,
        line_list,
        gap_threshold=stats.threshold,
        page_width=page_width,
    )
    return merged, before - len(merged)


@dataclass
class PageSectionContext:
    vision: dict[str, Any]
    words: list[Any]
    page_width: float
    page_rgb: Image.Image
    record_snapshots: bool = False
    include_optional: bool = True
    fw_lines: list[Any] = field(default_factory=list)
    split_lines: list[Any] = field(default_factory=list)
    body_sl: list[Any] = field(default_factory=list)
    margin_sl: list[Any] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    section_meta: dict[str, Any] = field(default_factory=dict)
    line_pool: dict[int, Any] = field(default_factory=dict)
    opencv_boxes: list[Any] = field(default_factory=list)
    gap_stats_obj: Any = None
    column_meta: dict[str, Any] = field(default_factory=dict)
    skip_vertical_for_table: bool = False
    snapshots: list[tuple[str, list[dict[str, Any]]]] = field(default_factory=list)
    merge_count: int = 0

    def snapshot(self, name: str) -> None:
        if self.record_snapshots:
            self.snapshots.append((name, sections_objs_to_dicts(self.sections)))


def prepare_margin_lines(ctx: PageSectionContext) -> None:
    if ctx.body_sl:
        return
    body_sl, margin_sl = _partition_margin_vertical_lines(
        ctx.split_lines, float(ctx.page_width)
    )
    ctx.body_sl = _strip_margin_words_from_lines(body_sl, float(ctx.page_width))
    ctx.margin_sl = margin_sl
    ctx.section_meta["margin_line_count"] = len(margin_sl)


def bounds_dicts_match(
    local: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    tolerance_px: float = 2.0,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if len(local) != len(reference):
        issues.append(f"count {len(local)} != {len(reference)}")
    for i, (a, b) in enumerate(zip(local, reference)):
        ba, bb = a.get("bounds") or {}, b.get("bounds") or {}
        for key in ("min_x", "min_y", "max_x", "max_y"):
            av, bv = float(ba.get(key, 0)), float(bb.get(key, 0))
            if abs(av - bv) > tolerance_px:
                issues.append(
                    f"S{i} {key}: local={av:.0f} api={bv:.0f} (Δ{abs(av - bv):.0f}px)"
                )
    return not issues, issues
