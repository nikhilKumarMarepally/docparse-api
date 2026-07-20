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
) -> list[Line]:
    lines: list[Line] = []
    for row in group_into_rows(words):
        row.sort(key=lambda w: w.box.min_x)
        text = combine_phrase([w.text for w in row])
        box = row[0].box
        for w in row[1:]:
            box = box.union(w.box)
        if full_width and page_width:
            box = Box(0.0, box.min_y, float(page_width), box.max_y)
        lines.append(Line(index=len(lines), text=text, box=box, words=row))
    return lines


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
