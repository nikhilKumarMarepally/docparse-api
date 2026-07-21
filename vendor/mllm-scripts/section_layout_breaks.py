"""Human-like section breaks: layout regions, rules, spacing, and context cues.

Combines signals a reviewer uses on dealer forms:
  1. Printed boxes / bands from vision ``image_blocks`` (full-width sections)
  2. Dark horizontal rules in the raster gap between OCR lines
  3. Vertical whitespace larger than a local adaptive threshold
  4. Strong context headers (OPTION, ITEMIZATION, signature blocks, etc.)

When vision layout blocks exist, lines map to layout regions, then **co-planar cells**
merge into table bands (vehicle row, TIL grid, payment schedule, etc.).
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, replace
from typing import Any

from ocr_word_to_line_boxes import Box, Line

# Strong printed section headers (start of line or after barcode junk).
_CONTEXT_HEADER = re.compile(
    r"(?i)^(?:"
    r"itemization of amount financed|"
    r"option\s*:|"
    r"optional\s+gap\s+contract|"
    r"returned check charge|"
    r"notice to the consumer|"
    r"agreement to arbitrate|"
    r"you agree to the terms|"
    r"federal truth-?in-?lending|"
    r"retail installment sale contract"
    r")"
)

_CONTEXT_COLUMN_RIGHT = re.compile(
    r"(?i)^(insurance\.|check the insurance|optional credit insurance|"
    r"credit life|credit disability|this insurance does not include)"
)


@dataclass(frozen=True)
class LayoutRegion:
    region_id: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    word_count: int
    preview: str
    kind: str  # full_width | column_left | column_right | other

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


@dataclass
class BreakDecision:
    break_before: bool
    reasons: list[str]


def _polygon_bbox(points: list[list[float]]) -> Box | None:
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return Box(min(xs), min(ys), max(xs), max(ys))


def _vertical_overlap_ratio(a: LayoutRegion, b: LayoutRegion) -> float:
    top = max(a.min_y, b.min_y)
    bot = min(a.max_y, b.max_y)
    overlap = max(0.0, bot - top)
    shorter = min(a.height, b.height)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _relabel_side_by_side_columns(regions: list[LayoutRegion], page_width: float) -> list[LayoutRegion]:
    """Parallel columns: a wide left block beside a right block is not full-width."""
    staged: list[LayoutRegion] = []
    for r in regions:
        if r.kind != "full_width" or r.width < 0.45 * page_width:
            staged.append(r)
            continue
        has_right_neighbor = any(
            o.region_id != r.region_id
            and o.min_x >= r.max_x - 50
            and abs(o.min_y - r.min_y) < 80
            and _vertical_overlap_ratio(r, o) >= 0.25
            for o in regions
        )
        if has_right_neighbor:
            staged.append(replace(r, kind="column_left"))
        else:
            staged.append(r)

    left_ids = {r.region_id for r in staged if r.kind == "column_left"}
    finalized: list[LayoutRegion] = []
    for r in staged:
        if r.kind != "other":
            finalized.append(r)
            continue
        beside_left = any(
            o.region_id in left_ids
            and o.max_x <= r.min_x + 50
            and abs(o.min_y - r.min_y) < 80
            and _vertical_overlap_ratio(o, r) >= 0.25
            for o in staged
        )
        finalized.append(replace(r, kind="column_right") if beside_left else r)
    return finalized


def layout_regions_from_vision(
    vision: dict[str, Any],
    *,
    page_width: float,
    min_words: int = 1,
    min_height: float = 35.0,
    min_width: float = 100.0,
) -> list[LayoutRegion]:
    """Parse Document AI layout blocks from prod vision JSON."""
    ib = vision.get("image_blocks") or {}
    wp = ib.get("with_polygons") or {}
    blocks = wp.get("blocks") or {}
    if not blocks:
        return []

    poly_by_id = {str(p["id"]): p for p in (vision.get("polygons") or [])}
    regions: list[LayoutRegion] = []
    for bid, blk in blocks.items():
        poly = poly_by_id.get(str(bid))
        if not poly:
            continue
        box = _polygon_bbox(poly.get("points") or [])
        if box is None:
            continue
        words = blk.get("words") or []
        if len(words) < min_words:
            continue
        if box.height < min_height or box.width < min_width:
            continue
        preview = (words[0].get("text") or "").strip()[:60]
        fw = box.width >= 0.55 * page_width
        kind = "full_width" if fw else "other"
        regions.append(
            LayoutRegion(
                region_id=str(bid),
                min_x=box.min_x,
                min_y=box.min_y,
                max_x=box.max_x,
                max_y=box.max_y,
                word_count=len(words),
                preview=preview,
                kind=kind,
            )
        )

    regions.sort(key=lambda r: (r.min_y, r.min_x))
    if not regions:
        return regions
    return _relabel_side_by_side_columns(regions, page_width)


def _overlap_y(a: Box, b: LayoutRegion) -> float:
    top = max(a.min_y, b.min_y)
    bot = min(a.max_y, b.max_y)
    return max(0.0, bot - top)


def _overlap_x(a: Box, b: LayoutRegion) -> float:
    left = max(a.min_x, b.min_x)
    right = min(a.max_x, b.max_x)
    return max(0.0, right - left)


def assign_line_stack_key(
    line: Line,
    regions: list[LayoutRegion],
    *,
    page_width: float,
) -> str:
    """Map a line to a vertical stack band (ignores left/right column flips)."""
    box = line.content_box
    cy, cx = box.centroid_y, box.centroid_x

    full_width = [r for r in regions if r.kind == "full_width"]
    full_width.sort(key=lambda r: r.min_y)
    for r in full_width:
        if cy >= r.min_y - 8 and cy <= r.max_y + 8:
            return f"fw:{r.region_id}"

    for r in regions:
        if r.kind not in ("column_left", "column_right"):
            continue
        if cy < r.min_y - 10 or cy > r.max_y + 10:
            continue
        if cx < r.min_x - 30 or cx > r.max_x + 30:
            continue
        side = "left" if r.kind == "column_left" else "right"
        return f"grid:{side}"

    if full_width:
        first_fw_y = min(r.min_y for r in full_width)
        if cy < first_fw_y - 5:
            return "grid:body"

    # Fallback: best region by overlap area.
    best_id = "unknown"
    best_score = 0.0
    for r in regions:
        oy = _overlap_y(box, r)
        ox = _overlap_x(box, r)
        score = oy * ox
        if score > best_score:
            best_score = score
            best_id = r.region_id
    return f"region:{best_id}"


def _normalize_header_text(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    # RouteOne / barcode header only — do not strip arbitrary leading words like "the".
    if re.match(r"^T\d{6,}", t, re.I):
        t = re.sub(
            r"^T[\w\-]+(?:\s+THIS IS A CUSTOMER COMPLETED COPY.*)?",
            "",
            t,
            flags=re.I,
        ).strip()
    return t


def context_header_break(prev_text: str, curr_text: str) -> str | None:
    curr = _normalize_header_text(curr_text)
    if _CONTEXT_HEADER.match(curr):
        prev = _normalize_header_text(prev_text)
        if prev[:40] != curr[:40]:
            return "context_header"
    return None


_WATERMARK_PREFIX = re.compile(
    r"^(?:(?:ACCURATE|TRUE|AND|COPY|NON|UCC|COMPLETED|AUTHORITATIVE)\s+)+",
    re.I,
)

_PROSE_SECTION_START = re.compile(
    r"(?i)^(?:"
    r"late charge|"
    r"for married|"
    r"vendor'?s single interest|"
    r"notice\s*:|"
    r"agreement to arbitrate"
    r")"
)


def _strip_watermark_prefix(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    while True:
        n = _WATERMARK_PREFIX.sub("", t, count=1).strip()
        if n == t:
            return t
        t = n


_MARGIN_SLOT = re.compile(r"^(?:N/A|YES|NO|X|\*|-{1,2})$", re.I)


def _margin_slot_line(text: str) -> bool:
    t = _strip_watermark_prefix(text).strip()
    return bool(_MARGIN_SLOT.match(t)) or len(t) <= 3


def _right_column_continuation(prev: Line, curr: Line, page_width: float) -> bool:
    """Wide left/full line with a short right-gutter finish (e.g. 'apply to this contract.')."""
    if prev.content_box.width < page_width * 0.45:
        return False
    text = _strip_watermark_prefix(curr.text).strip()
    if len(text) > 50:
        return False
    return bool(text) and (text[0].islower() or text.endswith("."))


def _line_y_overlap(a: Line, b: Line) -> float:
    top = max(a.content_box.min_y, b.content_box.min_y)
    bot = min(a.content_box.max_y, b.content_box.max_y)
    return max(0.0, bot - top)


def prose_section_start_break(prev_text: str, curr_text: str) -> str | None:
    curr = _strip_watermark_prefix(curr_text)
    if not _PROSE_SECTION_START.match(curr):
        return None
    prev = _strip_watermark_prefix(prev_text)
    if prev[:40] == curr[:40]:
        return None
    return "prose_section_start"


def _column_side(line: Line, page_width: float) -> str:
    cx = line.content_box.centroid_x
    width = line.content_box.width
    if width >= page_width * 0.52:
        return "full"
    if cx < page_width * 0.58:
        return "left"
    if cx > page_width * 0.62:
        return "right"
    return "center"


def should_break_prose_section(
    prev: Line,
    curr: Line,
    gap: float,
    *,
    gap_threshold: float,
    image_rgb=None,
    page_width: int,
) -> BreakDecision:
    """Gap + context breaks; same visual row (y overlap) stays one section."""
    section_hit = prose_section_start_break(prev.text, curr.text)
    if section_hit:
        return BreakDecision(True, [section_hit])

    if _line_y_overlap(prev, curr) >= 4.0:
        prev_side = _column_side(prev, float(page_width))
        curr_side = _column_side(curr, float(page_width))
        if (
            prev_side != curr_side
            and "full" not in (prev_side, curr_side)
        ):
            if _margin_slot_line(prev.text) or _margin_slot_line(curr.text):
                return BreakDecision(False, [])
            if _right_column_continuation(prev, curr, float(page_width)):
                return BreakDecision(False, [])
            return BreakDecision(True, ["column_change:same_row"])
        return BreakDecision(False, [])

    reasons: list[str] = []
    prev_side = _column_side(prev, float(page_width))
    curr_side = _column_side(curr, float(page_width))
    if (
        prev_side != curr_side
        and "full" not in (prev_side, curr_side)
    ):
        if not _right_column_continuation(prev, curr, float(page_width)):
            reasons.append("column_change")

    if gap > gap_threshold:
        reasons.append(f"whitespace:{gap:.0f}px>{gap_threshold:.0f}px")

    hit = context_header_break(prev.text, curr.text)
    if hit:
        reasons.append(hit)

    if image_rgb is not None and gap >= 8:
        if horizontal_rule_in_gap(
            image_rgb,
            prev.content_box.max_y,
            curr.content_box.min_y,
            page_width=page_width,
        ):
            reasons.append("horizontal_rule")

    return BreakDecision(break_before=bool(reasons), reasons=reasons)


def cluster_prose_lines(
    lines: list[Line],
    *,
    image_rgb=None,
    page_width: float,
    min_gap_px: float,
    pad: float,
) -> list[tuple[list[Line], list[str], Box]]:
    if not lines:
        return []
    ordered = sorted(lines, key=lambda ln: ln.index)
    gaps = [
        max(0.0, ordered[i + 1].content_box.min_y - ordered[i].content_box.max_y)
        for i in range(len(ordered) - 1)
    ]
    gap_threshold = adaptive_gap_threshold(gaps, floor=min_gap_px)

    out: list[tuple[list[Line], list[str], Box]] = []
    current: list[Line] = [ordered[0]]

    for i in range(1, len(ordered)):
        prev, curr = ordered[i - 1], ordered[i]
        gap = gaps[i - 1]
        decision = should_break_prose_section(
            prev,
            curr,
            gap,
            gap_threshold=gap_threshold,
            image_rgb=image_rgb,
            page_width=int(page_width),
        )
        if decision.break_before:
            out.append(
                (
                    current,
                    decision.reasons,
                    _bounds_from_lines(current, pad=pad),
                )
            )
            current = [curr]
        else:
            current.append(curr)

    out.append((current, ["prose_cluster"], _bounds_from_lines(current, pad=pad)))
    return out, gap_threshold


def _bounds_from_lines(lines: list[Line], *, pad: float) -> Box:
    box = lines[0].content_box
    for ln in lines[1:]:
        box = box.union(ln.content_box)
    return Box(box.min_x - pad, box.min_y - pad, box.max_x + pad, box.max_y + pad)


def adaptive_gap_threshold(gaps: list[float], *, floor: float = 8.0) -> float:
    if not gaps:
        return floor
    positive = [g for g in gaps if g > 0] or gaps
    if not positive:
        return floor
    median = statistics.median(positive)
    p90 = sorted(positive)[int(0.9 * (len(positive) - 1))]
    return max(floor, median * 1.6, p90 * 0.75, statistics.fmean(positive) * 1.1)


def horizontal_rule_in_gap(
    image_rgb,
    y_top: float,
    y_bot: float,
    *,
    page_width: int,
    min_span_frac: float = 0.45,
    dark_threshold: int = 120,
    min_dark_frac: float = 0.35,
) -> bool:
    """True if a dark horizontal rule sits in the whitespace between two lines."""
    if image_rgb is None:
        return False
    gap = y_bot - y_top
    if gap < 4:
        return False
    y0 = int(max(0, y_top))
    y1 = int(min(image_rgb.height - 1, y_bot))
    if y1 <= y0:
        return False

    x_margin = int(page_width * 0.08)
    x0, x1 = x_margin, page_width - x_margin
    span_need = (x1 - x0) * min_span_frac

    for y in range(y0, y1 + 1):
        dark_run = 0
        max_run = 0
        for x in range(x0, x1):
            r, g, b = image_rgb.getpixel((x, y))[:3]
            if (r + g + b) / 3 < dark_threshold:
                dark_run += 1
                max_run = max(max_run, dark_run)
            else:
                dark_run = 0
        if max_run >= span_need:
            return True
    return False


def should_break_section(
    prev: Line,
    curr: Line,
    gap: float,
    *,
    gap_threshold: float,
    prev_key: str,
    curr_key: str,
    image_rgb=None,
    page_width: int,
) -> BreakDecision:
    reasons: list[str] = []

    if prev_key != curr_key:
        if prev_key.startswith("grid:") and curr_key.startswith("grid:"):
            pass
        elif prev_key == "grid:body" and curr_key == "grid:body":
            pass
        else:
            reasons.append(f"layout_band:{prev_key}->{curr_key}")

    if gap > gap_threshold:
        reasons.append(f"whitespace:{gap:.0f}px>{gap_threshold:.0f}px")

    hdr = context_header_break(prev.text, curr.text)
    if hdr:
        reasons.append(hdr)

    if image_rgb is not None and gap >= 8:
        if horizontal_rule_in_gap(
            image_rgb,
            prev.content_box.max_y,
            curr.content_box.min_y,
            page_width=page_width,
        ):
            if prev_key != curr_key or gap >= 10:
                reasons.append("horizontal_rule")

    return BreakDecision(break_before=bool(reasons), reasons=reasons)


def _box_from_region(r: LayoutRegion, pad: float = 6.0) -> Box:
    return Box(r.min_x - pad, r.min_y - pad, r.max_x + pad, r.max_y + pad)


def _regions_by_id(regions: list[LayoutRegion]) -> dict[str, LayoutRegion]:
    return {r.region_id: r for r in regions}


def _first_full_width_y(regions: list[LayoutRegion]) -> float | None:
    fw = [r for r in regions if r.kind == "full_width"]
    return min(r.min_y for r in fw) if fw else None


def _full_width_regions(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    return sorted((r for r in regions if r.kind == "full_width"), key=lambda r: r.min_y)


def _grid_regions(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    return [r for r in regions if r.kind in ("column_left", "column_right")]


def _column_split_x(regions: list[LayoutRegion]) -> float | None:
    rights = [r for r in regions if r.kind == "column_right"]
    return min(r.min_x for r in rights) if rights else None


def y_band_group(
    cy: float,
    cx: float,
    regions: list[LayoutRegion],
    *,
    first_fw_y: float | None,
    header_y_max: float,
) -> str:
    """Horizontal band id from printed full-width boxes (OPTION, GAP, etc.)."""
    if cy <= header_y_max:
        return "header"
    if first_fw_y is not None and cy < first_fw_y:
        return "grid"

    split_x = _column_split_x(regions)

    fw = _full_width_regions(regions)
    for r in fw:
        if r.min_y <= cy <= r.max_y:
            if split_x is not None and cx >= split_x - 30:
                continue
            return f"box:{r.region_id}"

    if split_x is not None and cx >= split_x - 30 and first_fw_y is not None and cy >= first_fw_y:
        return "rc"

    for prev, nxt in zip(fw, fw[1:]):
        if prev.max_y + 4 < cy < nxt.min_y - 4:
            return "tail"

    if fw and cy > fw[-1].max_y + 4:
        return "tail"

    return "tail"


def lateral_group(
    line: Line,
    regions: list[LayoutRegion],
    *,
    first_fw_y: float | None,
    header_y_max: float,
) -> str:
    """Left printed box vs right insurance column — never merge across the gutter."""
    base = y_band_group(
        line.content_box.centroid_y,
        line.content_box.centroid_x,
        regions,
        first_fw_y=first_fw_y,
        header_y_max=header_y_max,
    )
    split_x = _column_split_x(regions)
    if base == "grid":
        return "grid"
    if split_x is None:
        return base
    cx = line.content_box.centroid_x
    if cx < split_x - 25:
        return f"L:{base}"
    if first_fw_y is not None and line.content_box.centroid_y >= first_fw_y:
        return "R:rc"
    return base


def context_column_break(prev_text: str, curr_text: str) -> bool:
    """Right-column insurance prose must not sit in the same section as OPTION/GAP boxes."""
    curr = re.sub(r"\s+", " ", (curr_text or "").strip())
    prev = re.sub(r"\s+", " ", (prev_text or "").strip())
    if _CONTEXT_COLUMN_RIGHT.match(curr):
        if re.search(r"(?i)option\s*:|optional gap|itemization", prev):
            return True
    if re.search(r"(?i)other optional insurance", curr) and re.search(r"(?i)option\s*:", prev):
        return True
    if re.search(r"(?i)decision to buy or not buy", curr) and re.search(r"(?i)seller.?s initials|option\s*:", prev):
        return True
    return False


def _base_group(lateral: str) -> str:
    if lateral.startswith("L:"):
        return lateral[2:]
    if lateral.startswith("R:"):
        return lateral[2:]
    return lateral


def bounds_for_section_group(
    group: str,
    lines: list[Line],
    regions: list[LayoutRegion],
    *,
    pad: float,
) -> Box:
    """Prefer vision layout polygon bounds so overlays match printed boxes."""
    by_id = _regions_by_id(regions)
    if group == "header" or group == "tail":
        box = lines[0].content_box
        for line in lines[1:]:
            box = box.union(line.content_box)
        return Box(box.min_x - pad, box.min_y - pad, box.max_x + pad, box.max_y + pad)

    if group.startswith("L:grid") or group == "grid":
        cols = _grid_regions(regions)
        ymin = min(line.content_box.min_y for line in lines)
        ymax = max(line.content_box.max_y for line in lines)
        if cols:
            box = _box_from_region(cols[0], pad=0)
            for r in cols[1:]:
                box = box.union(_box_from_region(r, pad=0))
            return Box(box.min_x - pad, ymin - pad, box.max_x + pad, ymax + pad)

    if group.startswith("L:box:") or group.startswith("box:"):
        rid = group.split(":")[-1]
        r = by_id.get(rid)
        if r:
            return _box_from_region(r, pad=pad)

    if group.startswith("R:rc") or group == "rc":
        cols = [r for r in regions if r.kind == "column_right"]
        ymin = min(line.content_box.min_y for line in lines)
        ymax = max(line.content_box.max_y for line in lines)
        if cols:
            r = cols[0]
            return Box(r.min_x - pad, ymin - pad, r.max_x + pad, ymax + pad)

    if group.startswith("box:"):
        rid = group.split(":", 1)[1]
        r = by_id.get(rid)
        if r:
            return _box_from_region(r, pad=pad)

    box = lines[0].content_box
    for line in lines[1:]:
        box = box.union(line.content_box)
    return Box(box.min_x - pad, box.min_y - pad, box.max_x + pad, box.max_y + pad)


def _region_row_mate(a: LayoutRegion, b: LayoutRegion, *, y_tol: float = 32.0) -> bool:
    if _vertical_overlap_ratio(a, b) >= 0.2:
        return True
    return abs(a.min_y - b.min_y) <= y_tol and abs(a.max_y - b.max_y) <= y_tol * 1.5


def _region_splittable_for_tables(
    r: LayoutRegion,
    *,
    page_height: float,
    page_width: float,
) -> bool:
    """Small layout cells that should merge into rows/tables, not stand alone."""
    if r.kind in ("column_left", "column_right"):
        return False
    if r.kind == "full_width":
        # Thin in-table band titles only — not page-wide prose clauses.
        return r.height <= 55 and r.width < page_width * 0.75
    if r.kind == "other":
        if r.height < 120:
            return True
        return r.min_x < page_width * 0.66
    return r.height <= min(220, page_height * 0.12)


def _row_bbox(
    members: list[LayoutRegion],
) -> tuple[float, float, float, float]:
    return (
        min(m.min_y for m in members),
        max(m.max_y for m in members),
        min(m.min_x for m in members),
        max(m.max_x for m in members),
    )


def _rows_should_stack(
    prev: tuple[float, float, float, float, list[LayoutRegion]],
    nxt: tuple[float, float, float, float, list[LayoutRegion]],
    *,
    page_width: float,
    inter_row_gap_px: float,
) -> bool:
    prev_ymin, prev_ymax, prev_xmin, prev_xmax, _prev_m = prev
    next_ymin, _next_ymax, next_xmin, next_xmax, _next_m = nxt
    gap = next_ymin - prev_ymax
    if gap > inter_row_gap_px:
        return False
    overlap = min(prev_xmax, next_xmax) - max(prev_xmin, next_xmin)
    min_w = min(prev_xmax - prev_xmin, next_xmax - next_xmin)
    if min_w <= 0 or overlap / min_w < 0.35:
        return False
    return True


def _is_margin_watermark_line(line: Line, page_width: float) -> bool:
    """Drop edge OCR tokens (diagonal stamps), not body copy."""
    cx = line.content_box.centroid_x
    text = re.sub(r"\s+", " ", (line.text or "").strip())
    if not text:
        return True
    words = text.split()
    if len(words) > 4:
        return False
    in_margin = cx < page_width * 0.04 or cx > page_width * 0.91
    if not in_margin:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    upper = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper >= 0.85


def cluster_table_region_groups(
    regions: list[LayoutRegion],
    *,
    page_height: float,
    page_width: float,
    inter_row_gap_px: float = 55.0,
    header_attach_px: float = 90.0,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Map each region_id to a section group id; return table band metadata for orphans."""
    main_x = page_width * 0.68
    splittable = [
        r
        for r in regions
        if _region_splittable_for_tables(r, page_height=page_height, page_width=page_width)
    ]
    splittable_ids = {r.region_id for r in splittable}

    main_cells = [
        r
        for r in splittable
        if r.min_x < main_x - 10 and ((r.min_x + r.max_x) / 2 < main_x or r.height < 150)
    ]

    parent = {r.region_id: r.region_id for r in main_cells}

    def find(i: str) -> str:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(main_cells):
        for b in main_cells[i + 1 :]:
            if _region_row_mate(a, b):
                union(a.region_id, b.region_id)

    from collections import defaultdict

    row_buckets: dict[str, list[LayoutRegion]] = defaultdict(list)
    for r in main_cells:
        row_buckets[find(r.region_id)].append(r)

    row_list: list[tuple[float, float, float, float, list[LayoutRegion]]] = []
    for members in row_buckets.values():
        row_list.append(
            (
                min(m.min_y for m in members),
                max(m.max_y for m in members),
                min(m.min_x for m in members),
                max(m.max_x for m in members),
                members,
            )
        )
    row_list.sort(key=lambda t: t[0])

    table_members: list[list[LayoutRegion]] = []
    current: list[LayoutRegion] = []
    prev_row: tuple[float, float, float, float, list[LayoutRegion]] | None = None
    for row in row_list:
        if prev_row is not None and not _rows_should_stack(
            prev_row,
            row,
            page_width=page_width,
            inter_row_gap_px=inter_row_gap_px,
        ):
            if current:
                table_members.append(current)
            current = []
        current.extend(row[4])
        prev_row = row
    if current:
        table_members.append(current)

    region_to_group: dict[str, str] = {}
    table_meta: list[dict[str, Any]] = []

    for r in regions:
        if r.region_id not in splittable_ids:
            region_to_group[r.region_id] = f"solo:{r.region_id}"

    for spl in splittable:
        if spl.region_id not in region_to_group:
            region_to_group[spl.region_id] = f"solo:{spl.region_id}"

    for ti, members in enumerate(table_members):
        gid = f"table:{ti}"
        ids = sorted({m.region_id for m in members})
        ymin = min(m.min_y for m in members)
        ymax = max(m.max_y for m in members)
        xmin = min(m.min_x for m in members)
        xmax = max(m.max_x for m in members)
        table_meta.append(
            {
                "group_id": gid,
                "region_ids": ids,
                "min_y": ymin,
                "max_y": ymax,
                "min_x": xmin,
                "max_x": xmax,
                "attach_y_min": ymin - header_attach_px,
            }
        )
        for m in members:
            region_to_group[m.region_id] = gid

    return region_to_group, table_meta


def _orphan_table_group(
    line: Line,
    table_meta: list[dict[str, Any]],
    *,
    main_x: float,
) -> str | None:
    cy = line.content_box.centroid_y
    cx = line.content_box.centroid_x
    if cx > main_x + 40:
        return None
    for band in table_meta:
        if band["attach_y_min"] <= cy <= band["max_y"] + 12:
            if cx < band["min_x"] - 50 or cx > band["max_x"] + 50:
                continue
            return band["group_id"]
    return None


def _group_section_bounds(
    lines: list[Line],
    members: list[LayoutRegion],
    *,
    pad: float,
) -> Box:
    ymin = min(ln.content_box.min_y for ln in lines)
    ymax = max(ln.content_box.max_y for ln in lines)
    xmin = min(ln.content_box.min_x for ln in lines)
    xmax = max(ln.content_box.max_x for ln in lines)
    return Box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)


def _region_section_bounds(
    lines: list[Line],
    region: LayoutRegion,
    *,
    pad: float,
) -> Box:
    ymin = min(ln.content_box.min_y for ln in lines)
    ymax = max(ln.content_box.max_y for ln in lines)
    lxmin = min(ln.content_box.min_x for ln in lines)
    lxmax = max(ln.content_box.max_x for ln in lines)
    return Box(
        max(region.min_x, lxmin) - pad,
        ymin - pad,
        min(region.max_x, lxmax) + pad,
        ymax + pad,
    )


def primary_layout_region_for_line(
    line: Line,
    regions: list[LayoutRegion],
    *,
    pad: float = 8.0,
) -> LayoutRegion | None:
    """Pick one vision layout block for this line (geometry only)."""
    box = line.content_box
    cx, cy = box.centroid_x, box.centroid_y
    hits = [
        r
        for r in regions
        if r.min_x - pad <= cx <= r.max_x + pad and r.min_y - pad <= cy <= r.max_y + pad
    ]
    if hits:

        def _rank(r: LayoutRegion) -> tuple[int, float]:
            # Column blocks are tighter than page-spanning bands when both match.
            kind_rank = 0 if r.kind in ("column_left", "column_right") else 1
            return (kind_rank, r.width * r.height)

        return min(hits, key=_rank)

    best: LayoutRegion | None = None
    best_score = 0.0
    for r in regions:
        score = _overlap_y(box, r) * _overlap_x(box, r)
        if score > best_score:
            best_score = score
            best = r
    return best if best_score > 0 else None


def _is_tall_column_region(region: LayoutRegion, *, page_height: float) -> bool:
    return region.kind in ("column_left", "column_right") and region.height > page_height * 0.35


def _is_narrow_table_sliver_group(
    members: list[LayoutRegion],
    *,
    page_width: float,
    page_height: float,
) -> bool:
    """Right-gutter labels (e.g. APPLICABLE LAW) are prose, not TIL-style table bands."""
    if len(members) > 4:
        return False
    if max(m.height for m in members) > page_height * 0.12:
        return False
    return all(
        m.kind in ("other", "column_right")
        and ((m.min_x + m.max_x) / 2) > page_width * 0.55
        for m in members
    )


def lines_to_sections_by_layout_regions(
    lines: list[Line],
    regions: list[LayoutRegion],
    *,
    pad: float = 6.0,
    page_width: float | None = None,
    image_rgb=None,
    min_gap_px: float = 8.0,
) -> tuple[list[tuple[list[Line], list[str], Box]], dict[str, Any]]:
    """Tables/columns from vision; prose blocks from gap + context + same-row alignment."""
    if not lines or not regions:
        return [], {"mode": "layout_regions", "breaks": []}

    from collections import defaultdict

    if page_width is None:
        page_width = max(ln.content_box.max_x for ln in lines) + 20
    page_height = max(ln.content_box.max_y for ln in lines) + 20
    main_x = page_width * 0.68

    region_to_group, table_meta = cluster_table_region_groups(
        regions,
        page_height=page_height,
        page_width=page_width,
    )
    group_members: dict[str, list[LayoutRegion]] = defaultdict(list)
    for r in regions:
        group_members[region_to_group[r.region_id]].append(r)

    by_group: dict[str, list[Line]] = defaultdict(list)
    prose_pool: list[Line] = []
    skipped_watermark: list[Line] = []

    for line in sorted(lines, key=lambda ln: ln.index):
        if _is_margin_watermark_line(line, page_width):
            skipped_watermark.append(line)
            continue
        reg = primary_layout_region_for_line(line, regions, pad=pad)
        if reg is None:
            gid = _orphan_table_group(line, table_meta, main_x=main_x)
            if gid:
                by_group[gid].append(line)
            else:
                prose_pool.append(line)
            continue

        gid = region_to_group[reg.region_id]
        if gid.startswith("table:"):
            by_group[gid].append(line)
        elif _is_tall_column_region(reg, page_height=page_height):
            by_group[gid].append(line)
        else:
            prose_pool.append(line)

    for gid in list(by_group.keys()):
        if not gid.startswith("table:"):
            continue
        members = group_members[gid]
        if _is_narrow_table_sliver_group(
            members, page_width=page_width, page_height=page_height
        ):
            prose_pool.extend(by_group.pop(gid))

    regions_by_id = _regions_by_id(regions)
    sections: list[tuple[list[Line], list[str], Box]] = []
    break_log: list[dict[str, Any]] = []
    orphan_lines: list[Line] = []
    prose_gap_threshold: float | None = None

    region_top = min(r.min_y for r in regions)
    header_from_prose: list[Line] = []
    body_prose: list[Line] = []
    for ln in prose_pool:
        if ln.content_box.centroid_y < region_top - 8:
            header_from_prose.append(ln)
        else:
            body_prose.append(ln)

    def group_sort_key(gid: str) -> tuple[float, float]:
        members = group_members[gid]
        return (min(m.min_y for m in members), min(m.min_x for m in members))

    for gid in sorted(by_group.keys(), key=group_sort_key):
        group_lines = by_group[gid]
        members = group_members[gid]
        if gid.startswith("table:"):
            bounds = _group_section_bounds(group_lines, members, pad=pad)
            rids = sorted({m.region_id for m in members})
            reasons = ["table_band", f"regions:{','.join(rids)}"]
        else:
            reg = members[0]
            bounds = _region_section_bounds(group_lines, reg, pad=pad)
            reasons = [f"layout_region:{reg.region_id}", f"kind:{reg.kind}"]
        sections.append((group_lines, reasons, bounds))
        break_log.append(
            {
                "reasons": reasons,
                "line_count": len(group_lines),
                "preview": group_lines[0].text[:80],
            }
        )

    prose_sections, prose_gap_threshold = cluster_prose_lines(
        body_prose,
        image_rgb=image_rgb,
        page_width=page_width,
        min_gap_px=min_gap_px,
        pad=pad,
    )
    for group_lines, reasons, bounds in prose_sections:
        sections.append((group_lines, reasons, bounds))
        break_log.append(
            {
                "reasons": reasons,
                "line_count": len(group_lines),
                "preview": group_lines[0].text[:80],
            }
        )

    sections.sort(
        key=lambda item: (
            min(ln.content_box.min_y for ln in item[0]),
            min(ln.content_box.min_x for ln in item[0]),
        )
    )

    if header_from_prose:
        sections.insert(
            0,
            (
                header_from_prose,
                ["unassigned:above_layout"],
                bounds_for_section_group("header", header_from_prose, regions, pad=pad),
            ),
        )

    meta = {
        "mode": "layout_regions",
        "prose_gap_threshold": prose_gap_threshold,
        "table_bands": table_meta,
        "orphan_line_count": len(orphan_lines),
        "orphan_line_indices": [ln.index for ln in orphan_lines],
        "watermark_line_count": len(skipped_watermark),
        "watermark_line_indices": [ln.index for ln in skipped_watermark],
        "layout_regions": [
            {
                "id": r.region_id,
                "kind": r.kind,
                "bounds": {
                    "min_x": r.min_x,
                    "min_y": r.min_y,
                    "max_x": r.max_x,
                    "max_y": r.max_y,
                },
                "preview": r.preview,
            }
            for r in regions
        ],
        "breaks": break_log,
    }
    return sections, meta


def lines_to_sections_human(
    lines: list[Line],
    *,
    vision: dict[str, Any] | None = None,
    page_width: float,
    image_rgb=None,
    pad: float = 6.0,
    min_gap_px: float = 8.0,
) -> tuple[list[tuple[list[Line], list[str], Box | None]], dict[str, Any]]:
    """Cluster lines into sections with explainable break reasons per section."""
    if not lines:
        return [], {"mode": "human", "breaks": []}

    regions = layout_regions_from_vision(vision, page_width=page_width) if vision else []
    if regions:
        region_sections, meta = lines_to_sections_by_layout_regions(
            lines,
            regions,
            pad=pad,
            page_width=page_width,
            image_rgb=image_rgb,
            min_gap_px=min_gap_px,
        )
        return [(a, b, c) for a, b, c in region_sections], meta

    gaps = [
        max(0.0, lines[i + 1].content_box.min_y - lines[i].content_box.max_y)
        for i in range(len(lines) - 1)
    ]
    gap_threshold = adaptive_gap_threshold(gaps, floor=min_gap_px)

    keys = [
        assign_line_stack_key(line, regions, page_width=page_width) if regions else "gap_only"
        for line in lines
    ]

    sections: list[tuple[list[Line], list[str], Box | None]] = []
    break_log: list[dict[str, Any]] = []
    current: list[Line] = [lines[0]]

    for i in range(1, len(lines)):
        prev, curr = lines[i - 1], lines[i]
        gap = gaps[i - 1]
        decision = should_break_section(
            prev,
            curr,
            gap,
            gap_threshold=gap_threshold,
            prev_key=keys[i - 1],
            curr_key=keys[i],
            image_rgb=image_rgb,
            page_width=int(page_width),
        )
        if decision.break_before:
            sections.append((current, decision.reasons, None))
            break_log.append(
                {
                    "after_line": i - 1,
                    "before_line": i,
                    "gap_px": round(gap, 1),
                    "reasons": decision.reasons,
                    "preview_after": prev.text[:80],
                    "preview_before": curr.text[:80],
                }
            )
            current = [curr]
        else:
            current.append(curr)

    sections.append((current, [], None))
    meta = {
        "mode": "human",
        "gap_threshold": round(gap_threshold, 2),
        "layout_regions": [
            {
                "id": r.region_id,
                "kind": r.kind,
                "bounds": {
                    "min_x": r.min_x,
                    "min_y": r.min_y,
                    "max_x": r.max_x,
                    "max_y": r.max_y,
                },
                "preview": r.preview,
            }
            for r in regions
        ],
        "breaks": break_log,
    }
    return sections, meta
