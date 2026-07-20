#!/usr/bin/env python3
"""Cluster line-level OCR into vertical sections using inter-line gap analysis.

Uses average/median line spacing to detect larger gaps (section breaks), then
consolidates all lines in each section into one bounding box (leftmost→rightmost
of contained OCR) and a merged text block.

Examples:
  python ocr_line_to_sections.py \\
    --vision page.json --image page.png --out /tmp/ocr_sections

  # Reuse prior line JSON instead of re-parsing vision:
  python ocr_line_to_sections.py \\
    --lines-json wa577_gallery/vin_ticket/87c842be_p0_lines/87c842be_p0_lines.json \\
    --image wa577_gallery/vin_ticket/87c842be_p0.png \\
    --out wa577_gallery/vin_ticket/87c842be_p0_sections
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ocr_word_to_line_boxes import (  # noqa: E402
    Box,
    Line,
    Word,
    load_font,
    load_vision,
    load_words,
    resolve_paths,
    words_to_lines,
)


@dataclass
class GapStats:
    gaps: list[float]
    mean: float
    median: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gaps": [round(g, 2) for g in self.gaps],
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "threshold": round(self.threshold, 2),
        }


@dataclass
class Section:
    index: int
    lines: list[Line]
    text: str
    box: Box
    gap_above: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "line_indices": [line.index for line in self.lines],
            "line_count": len(self.lines),
            "gap_above": None if self.gap_above is None else round(self.gap_above, 2),
            "text": self.text,
            "bounds": self.box.to_dict(),
        }


def line_gaps(lines: list[Line]) -> list[float]:
    gaps: list[float] = []
    for prev, curr in zip(lines, lines[1:]):
        gap = curr.content_box.min_y - prev.content_box.max_y
        gaps.append(max(0.0, gap))
    return gaps


def gap_stats(
    lines: list[Line],
    *,
    multiplier: float = 2.0,
    min_gap_px: float = 18.0,
) -> GapStats:
    gaps = line_gaps(lines)
    if not gaps:
        return GapStats(gaps=[], mean=0.0, median=0.0, threshold=min_gap_px)

    positive = [g for g in gaps if g > 0]
    if not positive:
        positive = gaps

    mean = statistics.fmean(positive)
    median = statistics.median(positive)
    threshold = max(min_gap_px, median * multiplier, mean * 1.25)
    return GapStats(gaps=gaps, mean=mean, median=median, threshold=threshold)


def lines_to_sections(
    lines: list[Line],
    *,
    multiplier: float = 2.0,
    min_gap_px: float = 18.0,
    pad: float = 6.0,
) -> tuple[list[Section], GapStats]:
    if not lines:
        return [], GapStats([], 0.0, 0.0, min_gap_px)

    stats = gap_stats(lines, multiplier=multiplier, min_gap_px=min_gap_px)
    sections: list[Section] = []
    current: list[Line] = [lines[0]]
    gap_above: float | None = None

    for i, line in enumerate(lines[1:], start=1):
        gap = stats.gaps[i - 1]
        if gap > stats.threshold:
            sections.append(_make_section(len(sections), current, gap_above, pad))
            current = [line]
            gap_above = gap
        else:
            current.append(line)

    sections.append(_make_section(len(sections), current, gap_above, pad))
    return sections, stats


def _make_section(
    index: int,
    lines: list[Line],
    gap_above: float | None,
    pad: float,
) -> Section:
    box = lines[0].content_box
    for line in lines[1:]:
        box = box.union(line.content_box)
    box = Box(
        box.min_x - pad,
        box.min_y - pad,
        box.max_x + pad,
        box.max_y + pad,
    )
    text = "\n".join(line.text for line in lines if line.text.strip())
    return Section(index=index, lines=lines, text=text, box=box, gap_above=gap_above)


def lines_from_json(path: Path) -> list[Line]:
    data = json.loads(path.read_text())
    lines: list[Line] = []
    for row in data["lines"]:
        words = [
            Word(
                text=w["text"],
                box=Box(**w["bounds"]),
                index=i,
            )
            for i, w in enumerate(row.get("words") or [])
        ]
        if not words:
            bounds = row["bounds"]
            box = Box(**bounds)
            content = Box(**row.get("content_bounds", bounds))
            words = [Word(text=row["text"], box=content, index=0)]
        else:
            box = Box(**row["bounds"])
        lines.append(
            Line(
                index=row["index"],
                text=row["text"],
                box=box,
                words=words,
            )
        )
    return lines


SECTION_PALETTE: list[tuple[int, int, int]] = [
    (255, 99, 71),
    (50, 205, 50),
    (30, 144, 255),
    (255, 165, 0),
    (148, 0, 211),
    (0, 206, 209),
    (255, 20, 147),
    (128, 128, 0),
    (220, 40, 40),
    (100, 149, 237),
]


def _section_bounds(section: Section | dict[str, Any]) -> tuple[int, float, float, float, float]:
    if isinstance(section, dict):
        idx = int(section.get("index", 0))
        b = section["bounds"]
        return idx, b["min_x"], b["min_y"], b["max_x"], b["max_y"]
    return section.index, section.box.min_x, section.box.min_y, section.box.max_x, section.box.max_y


def draw_sections_overlay(
    full_img: Image.Image,
    sections: list[Section] | list[dict[str, Any]],
    picked_section_idx: int | None = None,
    crop_bounds: dict[str, float] | None = None,
) -> Image.Image:
    """Draw each OCR section in a unique color; highlight the picked VIN section.

    When ``crop_bounds`` is set, draw an inner green rectangle for the VIN crop
    sent to image enhancement (typically ``tight_vin_bounds`` + pad).
    """
    base = full_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(overlay)
    font = load_font(14)
    label_h = 18

    for section in sections:
        idx, min_x, min_y, max_x, max_y = _section_bounds(section)
        color = SECTION_PALETTE[idx % len(SECTION_PALETTE)]
        is_picked = picked_section_idx is not None and idx == picked_section_idx
        fill_alpha = int(255 * 0.30) if is_picked else int(255 * 0.20)
        fill_draw.rectangle(
            [(min_x, min_y), (max_x, max_y)],
            fill=(*color, fill_alpha),
        )
        label = f"S{idx}"
        fill_draw.rectangle(
            [(min_x, min_y - label_h), (min_x + 34, min_y)],
            fill=(*color, 230),
        )
        fill_draw.text(
            (min_x + 3, min_y - label_h + 2),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    result = Image.alpha_composite(base, overlay)
    border_draw = ImageDraw.Draw(result)
    for section in sections:
        idx, min_x, min_y, max_x, max_y = _section_bounds(section)
        color = SECTION_PALETTE[idx % len(SECTION_PALETTE)]
        is_picked = picked_section_idx is not None and idx == picked_section_idx
        width = 5 if is_picked else 3
        border_draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=color, width=width)
        if is_picked:
            inset = 4
            border_draw.rectangle(
                [
                    (min_x + inset, min_y + inset),
                    (max_x - inset, max_y - inset),
                ],
                outline=(255, 255, 255),
                width=2,
            )

    if crop_bounds:
        cx0 = int(crop_bounds["min_x"])
        cy0 = int(crop_bounds["min_y"])
        cx1 = int(crop_bounds["max_x"])
        cy1 = int(crop_bounds["max_y"])
        crop_color = (20, 180, 80)
        border_draw.rectangle([(cx0, cy0), (cx1, cy1)], outline=crop_color, width=4)
        label = "crop"
        lw = border_draw.textlength(label, font=font)
        border_draw.rectangle(
            [(cx0, cy0 - label_h), (cx0 + lw + 8, cy0)],
            fill=(*crop_color, 255),
        )
        border_draw.text((cx0 + 4, cy0 - label_h + 2), label, fill=(255, 255, 255), font=font)

    return result.convert("RGB")


def draw_sections(
    image_path: Path,
    sections: list[Section],
    *,
    show_lines: bool = False,
    lines: list[Line] | None = None,
    section_color: tuple[int, int, int] = (20, 150, 60),
    line_color: tuple[int, int, int] = (220, 40, 40),
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = load_font(12)

    if show_lines and lines:
        for line in lines:
            b = line.content_box
            draw.rectangle([(b.min_x, b.min_y), (b.max_x, b.max_y)], outline=line_color, width=1)

    for section in sections:
        b = section.box
        draw.rectangle([(b.min_x, b.min_y), (b.max_x, b.max_y)], outline=section_color, width=4)
        label = f"S{section.index}"
        draw.rectangle([(b.min_x, b.min_y - 16), (b.min_x + 34, b.min_y)], fill=section_color)
        draw.text((b.min_x + 3, b.min_y - 15), label, fill=(255, 255, 255), font=font)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", help="Path to vision page JSON")
    parser.add_argument("--lines-json", help="Reuse output from ocr_word_to_line_boxes.py")
    parser.add_argument("--image", help="Path to page PNG/JPG")
    parser.add_argument("--payload", help="mllm_test payload JSON (optional, for S3 vision key)")
    parser.add_argument("--out", required=True, help="Output directory or .png path")
    parser.add_argument("--gap-multiplier", type=float, default=2.0, help="Section break when gap > median * multiplier")
    parser.add_argument("--min-gap", type=float, default=18.0, help="Minimum px gap treated as a section break")
    parser.add_argument("--show-lines", action="store_true", help="Overlay faint line boxes")
    parser.add_argument("--download-image", action="store_true")
    parser.add_argument("--aws-profile", default="prod")
    args = parser.parse_args()

    if args.lines_json:
        if not args.image:
            raise SystemExit("--image required with --lines-json")
        lines = lines_from_json(Path(args.lines_json))
        image_path = Path(args.image)
        vision_path = None
        out = Path(args.out)
    else:
        vision_path, image_path, out = resolve_paths(args)
        vision = load_vision(vision_path)
        words = load_words(vision)
        with Image.open(image_path) as img:
            page_width = img.width
        lines = words_to_lines(words, page_width=page_width, full_width=True)

    sections, stats = lines_to_sections(
        lines,
        multiplier=args.gap_multiplier,
        min_gap_px=args.min_gap,
    )

    if str(out).endswith((".png", ".jpg", ".jpeg")):
        annotated = out
        json_out = out.with_suffix(".json")
        txt_out = out.with_suffix(".txt")
    else:
        out.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        annotated = out / f"{stem}_sections.png"
        json_out = out / f"{stem}_sections.json"
        txt_out = out / f"{stem}_ocr_sections.txt"

    annotated_img = draw_sections(
        image_path,
        sections,
        show_lines=args.show_lines,
        lines=lines,
    )
    annotated_img.save(annotated)

    payload = {
        "image": str(image_path),
        "vision": str(vision_path) if vision_path else None,
        "lines_json": args.lines_json,
        "line_count": len(lines),
        "section_count": len(sections),
        "gap_stats": stats.to_dict(),
        "sections": [section.to_dict() for section in sections],
    }
    json_out.write_text(json.dumps(payload, indent=2))

    txt_blocks = []
    for section in sections:
        txt_blocks.append(f"=== Section {section.index} (lines {section.lines[0].index}-{section.lines[-1].index}) ===")
        txt_blocks.append(section.text)
    txt_out.write_text("\n\n".join(txt_blocks) + "\n")

    print(f"lines={len(lines)} sections={len(sections)}")
    print(f"gap mean={stats.mean:.1f}px median={stats.median:.1f}px threshold={stats.threshold:.1f}px")
    print(f"annotated: {annotated}")
    print(f"json:      {json_out}")
    print(f"text:      {txt_out}")


if __name__ == "__main__":
    main()
