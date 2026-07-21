#!/usr/bin/env python3
"""Group word-level vision OCR into lines and draw bounding boxes on a page image.

Accepts prod vision page JSON (`document_text.words` or legacy `bounds` quads).
Outputs:
  - annotated PNG with line boxes (and optional word boxes)
  - lines.json with per-line text + bounds
  - ocr_lines.txt (one line per row)

Examples:
  python ocr_word_to_line_boxes.py \\
    --vision page.json --image page.png --out /tmp/ocr_viz

  # Derive vision S3 key from mllm_test payload (uses image_uri page number):
  python ocr_word_to_line_boxes.py \\
    --payload payloads/wa513_577_v1/87c842be-*_p0.json \\
    --image wa577_gallery/vin_ticket/87c842be_p0.png \\
    --out wa577_gallery/vin_ticket/87c842be_p0_lines
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

BUCKET = "informed-techno-core-prod-exchange"


@dataclass
class Box:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def centroid_y(self) -> float:
        return (self.min_y + self.max_y) / 2

    @property
    def centroid_x(self) -> float:
        return (self.min_x + self.max_x) / 2

    def union(self, other: Box) -> Box:
        return Box(
            min_x=min(self.min_x, other.min_x),
            min_y=min(self.min_y, other.min_y),
            max_x=max(self.max_x, other.max_x),
            max_y=max(self.max_y, other.max_y),
        )

    def intersects_horizontally(self, other: Box, slack: float = 0.0) -> bool:
        return not (
            self.max_y + slack < other.min_y or other.max_y + slack < self.min_y
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class Word:
    text: str
    box: Box
    index: int


@dataclass
class Line:
    index: int
    text: str
    box: Box
    words: list[Word]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "bounds": self.box.to_dict(),
            "content_bounds": self.content_box.to_dict(),
            "word_count": len(self.words),
            "words": [{"text": w.text, "bounds": w.box.to_dict()} for w in self.words],
        }

    @property
    def content_box(self) -> Box:
        """Tight bounds around OCR tokens (ignores full-page line width)."""
        box = self.words[0].box
        for word in self.words[1:]:
            box = box.union(word.box)
        return box


def parse_bounds(raw: Any) -> Box | None:
    if not raw:
        return None
    if isinstance(raw, dict):
        if all(k in raw for k in ("min_x", "min_y", "max_x", "max_y")):
            return Box(
                float(raw["min_x"]),
                float(raw["min_y"]),
                float(raw["max_x"]),
                float(raw["max_y"]),
            )
        pts = raw.get("points")
        if pts:
            xs = [float(p["x"]) for p in pts]
            ys = [float(p["y"]) for p in pts]
            return Box(min(xs), min(ys), max(xs), max(ys))
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "x" in raw[0]:
        xs = [float(p["x"]) for p in raw]
        ys = [float(p["y"]) for p in raw]
        return Box(min(xs), min(ys), max(xs), max(ys))
    return None


def load_vision_doc_text(vision: dict[str, Any]) -> dict[str, Any]:
    if isinstance(vision.get("ocr_data"), dict):
        dt = vision["ocr_data"].get("document_text")
        if isinstance(dt, dict):
            return dt
    if isinstance(vision.get("document_text"), dict):
        return vision["document_text"]
    if isinstance(vision.get("text"), dict):
        return vision["text"]
    return vision if isinstance(vision, dict) else {}


def load_words(vision: dict[str, Any]) -> list[Word]:
    doc_text = load_vision_doc_text(vision)
    words_raw = doc_text.get("words") or []
    words: list[Word] = []
    for i, w in enumerate(words_raw):
        text = (w.get("corrected_text") or w.get("text") or "").strip()
        if not text:
            continue
        box = parse_bounds(w.get("bounds"))
        if box is None:
            continue
        words.append(Word(text=text, box=box, index=i))
    return words


def combine_phrase(parts: list[str]) -> str:
    """Light port of Augmenters::Grouper#construct_phrase."""
    result: list[str] = []
    for part in parts:
        if not result:
            result.append(part)
            continue
        prev = result[-1]
        if (
            (prev[-1].isalnum() or prev.endswith(","))
            and part
            and (part[0].isalnum() or part[0] == "(")
        ) or prev.endswith(":") or part.startswith(":"):
            result.append(part)
        elif prev.endswith(".") and not (
            re.fullmatch(r"[\d,]+\.", prev) and re.fullmatch(r"\d+", part)
        ):
            result.append(part)
        else:
            result[-1] = prev + part
    return " ".join(result)


def split_row_by_column_gutter(
    row: list[Word],
    *,
    min_gutter_px: float = 72.0,
) -> list[list[Word]]:
    if len(row) < 2:
        return [row]
    gaps: list[tuple[float, int]] = []
    for i in range(len(row) - 1):
        gap = row[i + 1].box.min_x - row[i].box.max_x
        gaps.append((gap, i))
    best_gap, split_at = max(gaps, key=lambda t: t[0])
    if best_gap < min_gutter_px:
        return [row]
    left, right = row[: split_at + 1], row[split_at + 1 :]
    parts: list[list[Word]] = []
    parts.extend(split_row_by_column_gutter(left, min_gutter_px=min_gutter_px))
    parts.extend(split_row_by_column_gutter(right, min_gutter_px=min_gutter_px))
    return parts


def group_into_rows(words: list[Word]) -> list[list[Word]]:
    """Cluster OCR tokens into horizontal rows by vertical position."""
    if not words:
        return []

    heights = sorted(w.box.height for w in words)
    median_h = heights[len(heights) // 2] if heights else 12.0
    y_tol = max(8.0, median_h * 0.55)

    rows: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (w.box.centroid_y, w.box.min_x)):
        placed = False
        for row in rows:
            row_y = sum(w.box.centroid_y for w in row) / len(row)
            if abs(word.box.centroid_y - row_y) <= y_tol:
                row.append(word)
                placed = True
                break
        if not placed:
            rows.append([word])

    rows.sort(key=lambda row: sum(w.box.centroid_y for w in row) / len(row))
    return rows


def words_to_lines(
    words: list[Word],
    *,
    page_width: int | None = None,
    full_width: bool = True,
    split_columns: bool = False,
) -> list[Line]:
    lines: list[Line] = []
    for row in group_into_rows(words):
        chunks = split_row_by_column_gutter(row) if split_columns else [row]
        for chunk in chunks:
            chunk.sort(key=lambda w: w.box.min_x)
            text = combine_phrase([w.text for w in chunk])
            box = chunk[0].box
            for w in chunk[1:]:
                box = box.union(w.box)
            if full_width and page_width and not split_columns:
                box = Box(0.0, box.min_y, float(page_width), box.max_y)
            lines.append(Line(index=len(lines), text=text, box=box, words=chunk))
    return lines


@dataclass
class AlignedTextColumn:
    """Vertical strip where line left edges (x) line up on the left or right page margin."""

    side: str
    anchor_min_x: float
    bounds: Box
    line_indices: list[int]
    split_x: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "anchor_min_x": round(self.anchor_min_x, 1),
            "split_x": round(self.split_x, 1),
            "bounds": self.bounds.to_dict(),
            "line_count": len(self.line_indices),
            "line_indices": self.line_indices,
        }


# Generic aligned-column rule: infer the page gutter from wide-row whitespace, cluster
# outer-margin word left-edges into an anchor x, keep only lines whose column-side OCR
# mass dominates, reject strips that would bisect opposing text, and scope column y to the
# aligned cluster (not the full page).


def estimate_page_gutter_x(lines: list[Line], page_width: float) -> float:
    """Infer the main vertical gutter from wide OCR rows (largest inter-word gaps near page center)."""
    import statistics

    gap_candidates: list[float] = []
    for ln in lines:
        words = sorted(ln.words, key=lambda w: w.box.min_x)
        if len(words) < 2 or ln.content_box.width < page_width * 0.28:
            continue
        for left_w, right_w in zip(words, words[1:]):
            gap = right_w.box.min_x - left_w.box.max_x
            if gap < 44.0:
                continue
            midpoint = (left_w.box.max_x + right_w.box.min_x) * 0.5
            if page_width * 0.22 <= midpoint <= page_width * 0.82:
                gap_candidates.append(midpoint)
    if not gap_candidates:
        return page_width * 0.55
    return float(statistics.median(gap_candidates))


def _words_on_side(
    line: Line,
    *,
    side: str,
    gutter_x: float,
    anchor_min_x: float,
    anchor_slack_px: float = 28.0,
) -> list[Word]:
    if side == "right":
        edge_floor = anchor_min_x - anchor_slack_px
        return [
            w
            for w in line.words
            if w.box.centroid_x >= gutter_x - 4.0 and w.box.min_x >= edge_floor
        ]
    edge_ceil = anchor_min_x + anchor_slack_px
    return [
        w
        for w in line.words
        if w.box.centroid_x <= gutter_x + 4.0 and w.box.max_x <= edge_ceil
    ]


def _line_side_mass(line: Line, split_x: float) -> tuple[float, float]:
    left = 0.0
    right = 0.0
    for w in line.words:
        wmass = max(1.0, w.box.width)
        if w.box.centroid_x < split_x:
            left += wmass
        else:
            right += wmass
    return left, right


def vertical_column_bbox_invalid(
    lines: list[Line],
    *,
    col_bounds: Box,
    split_x: float,
    gutter_slack_px: float = 4.0,
) -> bool:
    """True when the vertical strip bisects OCR (straddle or body text inside the strip)."""
    for ln in lines:
        cb = ln.content_box
        if cb.max_y < col_bounds.min_y - 2.0 or cb.min_y > col_bounds.max_y + 2.0:
            continue
        if cb.max_x <= col_bounds.min_x + 2.0 or cb.min_x >= col_bounds.max_x - 2.0:
            continue
        for w in ln.words:
            if (
                w.box.min_x < split_x - gutter_slack_px
                and w.box.max_x > split_x + gutter_slack_px
            ):
                return True
        for w in ln.words:
            wb = w.box
            if wb.max_y < col_bounds.min_y or wb.min_y > col_bounds.max_y:
                continue
            if wb.max_x <= col_bounds.min_x or wb.min_x >= col_bounds.max_x:
                continue
            if wb.centroid_x < split_x - gutter_slack_px and wb.max_x > col_bounds.min_x + 2.0:
                return True
    return False


def _line_assignable_to_column(
    line: Line,
    *,
    side: str,
    split_x: float,
    gutter_x: float,
    anchor_min_x: float,
    gutter_slack_px: float = 4.0,
) -> bool:
    """True when aligned column-side words exist and no token straddles the gutter."""
    for w in line.words:
        if (
            w.box.min_x < split_x - gutter_slack_px
            and w.box.max_x > split_x + gutter_slack_px
        ):
            return False
    side_words = _words_on_side(
        line, side=side, gutter_x=gutter_x, anchor_min_x=anchor_min_x
    )
    if not side_words:
        return False
    for w in line.words:
        if w in side_words:
            continue
        if side == "right" and w.box.max_x > split_x - gutter_slack_px:
            return False
        if side == "left" and w.box.min_x < split_x + gutter_slack_px:
            return False
    return True


def aligned_column_valid_for_vertical(
    lines: list[Line],
    column: AlignedTextColumn,
) -> bool:
    """Validity pass: no OCR row in the strip y-band is bisected by the column rectangle."""
    return not vertical_column_bbox_invalid(
        lines,
        col_bounds=column.bounds,
        split_x=column.split_x,
    )


def _detect_aligned_text_column_one_side(
    lines: list[Line],
    *,
    page_width: float,
    side: str,
    gutter_x: float,
    min_lines: int,
    anchor_tolerance_px: float,
    bin_width_px: float,
) -> AlignedTextColumn | None:
    from collections import Counter

    if side == "right":
        stripe_lo = max(gutter_x + 8.0, page_width * 0.62)
        stripe_hi = page_width
    else:
        stripe_lo = 0.0
        stripe_hi = min(gutter_x - 8.0, page_width * 0.38)

    word_edges: list[float] = []
    for ln in lines:
        for w in ln.words:
            cx = w.box.centroid_x
            mx = w.box.min_x
            if side == "right":
                if mx >= stripe_lo and cx >= gutter_x - 4.0:
                    word_edges.append(mx)
            elif cx <= stripe_hi and w.box.max_x <= gutter_x + 4.0:
                word_edges.append(mx)

    if len(word_edges) < max(8, min_lines * 2):
        return None

    hist = Counter(int(round(x / bin_width_px)) * int(bin_width_px) for x in word_edges)
    anchor = float(hist.most_common(1)[0][0])

    split_guess = max(gutter_x + 4.0, anchor - 18.0) if side == "right" else min(gutter_x - 4.0, anchor + 24.0)

    kept: list[Line] = []
    for ln in lines:
        side_words = _words_on_side(
            ln, side=side, gutter_x=gutter_x, anchor_min_x=anchor
        )
        if not side_words:
            continue
        edge = min(w.box.min_x for w in side_words)
        if abs(edge - anchor) > anchor_tolerance_px:
            continue
        if not _line_assignable_to_column(
            ln,
            side=side,
            split_x=split_guess,
            gutter_x=gutter_x,
            anchor_min_x=anchor,
        ):
            continue
        kept.append(ln)

    if len(kept) < min_lines:
        return None

    kept.sort(key=lambda ln: ln.content_box.min_y)
    y0 = min(ln.content_box.min_y for ln in kept)
    y1 = max(ln.content_box.max_y for ln in kept)

    if side == "right":
        split_x = max(gutter_x + 4.0, anchor - 18.0)
        column_left = split_x
    else:
        split_x = min(gutter_x - 4.0, anchor + max(24.0, bin_width_px))
        column_left = min(w.box.min_x for ln in kept for w in ln.words)

    bounds: Box | None = None
    for ln in kept:
        side_words = _words_on_side(
            ln, side=side, gutter_x=gutter_x, anchor_min_x=anchor
        )
        for w in side_words:
            bounds = w.box if bounds is None else bounds.union(w.box)
    if bounds is None:
        return None

    if side == "right":
        bounds = Box(column_left, y0, bounds.max_x, y1)
    else:
        bounds = Box(bounds.min_x, y0, split_x, y1)

    if vertical_column_bbox_invalid(lines, col_bounds=bounds, split_x=split_x):
        return None

    return AlignedTextColumn(
        side=side,
        anchor_min_x=anchor,
        bounds=bounds,
        line_indices=[ln.index for ln in kept],
        split_x=split_x,
    )


def detect_aligned_text_column(
    lines: list[Line],
    *,
    page_width: float,
    side: str | None = None,
    min_lines: int = 6,
    gutter_x: float | None = None,
    anchor_tolerance_px: float = 80.0,
    bin_width_px: float = 20.0,
    y_min: float | None = None,
    y_max: float | None = None,
    min_anchor_frac: float = 0.0,
) -> AlignedTextColumn | None:
    """Detect a vertically aligned text column on the left or right.

    Generic rule: cluster word left-edges past the page gutter; keep lines whose
    column-side words share an anchor x; reject candidates whose bounding box
    intersects bisected OCR rows (significant mass on both sides of the split).
    Column y-extent is only the aligned cluster, not the full page.
    """
    scoped = lines
    if y_min is not None or y_max is not None:
        lo = y_min if y_min is not None else float("-inf")
        hi = y_max if y_max is not None else float("inf")
        scoped = [
            ln
            for ln in lines
            if ln.content_box.max_y >= lo and ln.content_box.min_y <= hi
        ]
    if not scoped:
        return None

    gx = gutter_x if gutter_x is not None else estimate_page_gutter_x(scoped, page_width)
    band_min_lines = max(3, min(min_lines, len(scoped) // 5))

    candidates: list[AlignedTextColumn] = []
    sides = (side,) if side in ("left", "right") else ("left", "right")
    for s in sides:
        col = _detect_aligned_text_column_one_side(
            scoped,
            page_width=page_width,
            side=s,
            gutter_x=gx,
            min_lines=band_min_lines,
            anchor_tolerance_px=anchor_tolerance_px,
            bin_width_px=bin_width_px,
        )
        if col is None:
            continue
        if s == "right" and min_anchor_frac > 0 and col.anchor_min_x < page_width * min_anchor_frac:
            continue
        candidates.append(col)

    if not candidates:
        return None

    def _score(col: AlignedTextColumn) -> tuple[float, float, float]:
        outer = col.anchor_min_x if col.side == "right" else page_width - col.anchor_min_x
        height = col.bounds.max_y - col.bounds.min_y
        return (float(len(col.line_indices)), outer, height)

    return max(candidates, key=_score)


def line_from_words(words: list[Word], *, index: int) -> Line | None:
    if not words:
        return None
    box = words[0].box
    for w in words[1:]:
        box = box.union(w.box)
    text = " ".join(w.text for w in words if w.text.strip())
    if not text.strip():
        return None
    return Line(index=index, text=text, box=box, words=words)


def partition_lines_at_column(
    lines: list[Line],
    column: AlignedTextColumn,
    *,
    page_width: float,
    split_x: float | None = None,
    gutter_x: float | None = None,
) -> tuple[list[Line], list[Line]]:
    """Word-split lines at the detected column gutter; emit separate left/right line fragments."""
    gx = gutter_x if gutter_x is not None else estimate_page_gutter_x(lines, page_width)
    anchor = column.anchor_min_x
    column_split = split_x if split_x is not None else column.split_x
    left_lines: list[Line] = []
    right_lines: list[Line] = []
    for ln in lines:
        side = column.side
        if side == "right":
            col_words = _words_on_side(
                ln, side="right", gutter_x=gx, anchor_min_x=anchor
            )
        else:
            col_words = _words_on_side(
                ln, side="left", gutter_x=gx, anchor_min_x=anchor
            )
        col_ids = {id(w) for w in col_words}
        left_words = [
            w
            for w in ln.words
            if id(w) not in col_ids and w.box.centroid_x < column_split - 4.0
        ]
        right_words = [
            w
            for w in ln.words
            if id(w) not in col_ids and w.box.centroid_x >= column_split + 4.0
        ]
        if side == "right":
            right_words = col_words + right_words
        else:
            left_words = col_words + left_words

        left_ln = line_from_words(left_words, index=len(left_lines))
        if left_ln is not None:
            left_lines.append(left_ln)
        right_ln = line_from_words(right_words, index=len(right_lines))
        if right_ln is not None:
            right_lines.append(right_ln)
    return left_lines, right_lines


def load_font(size: int = 12) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_boxes(
    image_path: Path,
    lines: list[Line],
    *,
    show_words: bool = False,
    line_color: tuple[int, int, int] = (220, 40, 40),
    word_color: tuple[int, int, int] = (80, 120, 220),
    label_lines: bool = True,
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = load_font(11)

    if show_words:
        for line in lines:
            for word in line.words:
                b = word.box
                draw.rectangle(
                    [(b.min_x, b.min_y), (b.max_x, b.max_y)],
                    outline=word_color,
                    width=1,
                )

    for line in lines:
        b = line.box
        pad = 2
        draw.rectangle(
            [(b.min_x - pad, b.min_y - pad), (b.max_x + pad, b.max_y + pad)],
            outline=line_color,
            width=2,
        )
        if label_lines:
            label = f"L{line.index}"
            label_x = 2 if b.min_x <= 1 else b.min_x
            draw.rectangle(
                [(label_x, b.min_y - 14), (label_x + 28, b.min_y)],
                fill=line_color,
            )
            draw.text((label_x + 2, b.min_y - 13), label, fill=(255, 255, 255), font=font)
    return img


def parse_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 URI: {uri}")
    rest = uri.removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def vision_key_from_image_uri(image_uri: str) -> str:
    # .../file/{fid}/img/{fid}-{page}.png -> .../file/{fid}/vision/{fid}-{page}.json
    bucket, key = parse_s3(image_uri)
    m = re.search(r"/file/([^/]+)/img/\1-(\d+)\.png$", key)
    if not m:
        raise ValueError(f"Cannot derive vision key from image_uri: {image_uri}")
    fid, page = m.group(1), m.group(2)
    prefix = key.rsplit("/file/", 1)[0]
    return f"{prefix}/file/{fid}/vision/{fid}-{page}.json"


def s3_download(uri: str, dest: Path, profile: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "cp", uri, str(dest), "--profile", profile],
        check=True,
        capture_output=True,
        text=True,
    )


def load_vision(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    image_path = Path(args.image) if args.image else None
    vision_path = Path(args.vision) if args.vision else None

    if args.payload:
        payload = json.loads(Path(args.payload).read_text())
        data = payload["detail"]["data"]
        if image_path is None and args.download_image:
            image_uri = data["image_uri"]
            image_path = Path(args.out).with_suffix(".png") if args.out else Path("page.png")
            if str(args.out).endswith((".png", ".jpg")):
                image_path = Path(args.out)
            else:
                image_path = Path(args.out) / "page.png"
            s3_download(image_uri, image_path, args.aws_profile)
        elif image_path is None:
            raise SystemExit("--image required unless --download-image is set with --payload")

        if vision_path is None:
            vision_uri = f"s3://{BUCKET}/{vision_key_from_image_uri(data['image_uri'])}"
            if str(args.out).endswith((".png", ".jpg")):
                vision_path = Path(args.out).with_suffix(".json")
            else:
                vision_path = Path(args.out) / "vision.json"
            s3_download(vision_uri, vision_path, args.aws_profile)

    if vision_path is None or image_path is None:
        raise SystemExit("Provide --vision and --image, or --payload with --image")
    return vision_path, image_path, Path(args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", help="Path to vision page JSON")
    parser.add_argument("--image", help="Path to page PNG/JPG")
    parser.add_argument("--payload", help="mllm_test payload JSON (optional, for S3 vision key)")
    parser.add_argument("--out", required=True, help="Output path prefix or .png path")
    parser.add_argument("--show-words", action="store_true", help="Draw word-level boxes")
    parser.add_argument(
        "--content-width",
        action="store_true",
        help="Tight line boxes around OCR content only (default: span full page width)",
    )
    parser.add_argument("--download-image", action="store_true", help="With --payload, fetch image from S3")
    parser.add_argument("--aws-profile", default="prod")
    args = parser.parse_args()

    vision_path, image_path, out = resolve_paths(args)
    vision = load_vision(vision_path)
    words = load_words(vision)
    with Image.open(image_path) as img:
        page_width = img.width
    lines = words_to_lines(words, page_width=page_width, full_width=not args.content_width)

    if str(out).endswith((".png", ".jpg", ".jpeg")):
        out_dir = out.parent
        annotated = out
        json_out = out.with_suffix(".json")
        txt_out = out.with_suffix(".txt")
    else:
        out_dir = out
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        annotated = out_dir / f"{stem}_lines.png"
        json_out = out_dir / f"{stem}_lines.json"
        txt_out = out_dir / f"{stem}_ocr_lines.txt"

    annotated_img = draw_boxes(image_path, lines, show_words=args.show_words)
    annotated_img.save(annotated)

    payload = {
        "image": str(image_path),
        "vision": str(vision_path),
        "word_count": len(words),
        "line_count": len(lines),
        "lines": [line.to_dict() for line in lines],
    }
    json_out.write_text(json.dumps(payload, indent=2))
    txt_out.write_text("\n".join(line.text for line in lines) + "\n")

    print(f"words={len(words)} lines={len(lines)}")
    print(f"annotated: {annotated}")
    print(f"json:      {json_out}")
    print(f"text:      {txt_out}")


if __name__ == "__main__":
    main()
