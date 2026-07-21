#!/usr/bin/env python3
"""Detect a vertically aligned left/right text column and draw a guide box."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ocr_word_to_line_boxes import (  # noqa: E402
    detect_aligned_text_column,
    load_vision,
    load_words,
    resolve_paths,
    words_to_lines,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", help="Path to vision page JSON")
    parser.add_argument("--image", help="Path to page PNG/JPG")
    parser.add_argument("--payload", help="mllm_test payload JSON (optional)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="right",
        help="Which aligned column to detect (default: right insurance strip)",
    )
    args = parser.parse_args()

    vision_path, image_path, out = resolve_paths(args)
    vision = load_vision(vision_path)
    words = load_words(vision)
    with Image.open(image_path) as im:
        page_width = im.width
    lines = words_to_lines(words, page_width=page_width, full_width=True)

    col = detect_aligned_text_column(
        lines,
        page_width=float(page_width),
        side=args.side,
        min_anchor_frac=0.0 if args.side == "left" else 0.60,
    )

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    json_path = out / f"{stem}_aligned_{args.side}_column.json"
    png_path = out / f"{stem}_aligned_{args.side}_column.png"

    payload = {
        "image": str(image_path),
        "vision": str(vision_path),
        "page_width": page_width,
        "side": args.side,
        "column": col.to_dict() if col else None,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    img = Image.open(image_path).convert("RGB")
    if col is not None:
        draw = ImageDraw.Draw(img)
        b = col.bounds
        draw.rectangle(
            [(b.min_x, b.min_y), (b.max_x, b.max_y)],
            outline=(20, 180, 80),
            width=4,
        )
        draw.text(
            (b.min_x, max(0, b.min_y - 18)),
            f"{args.side} anchor x~{col.anchor_min_x:.0f}",
            fill=(20, 180, 80),
        )
    img.save(png_path)

    if col:
        print(
            f"detected {args.side} column anchor={col.anchor_min_x:.0f} "
            f"lines={len(col.line_indices)} bounds=({col.bounds.min_x:.0f},"
            f"{col.bounds.min_y:.0f})-({col.bounds.max_x:.0f},{col.bounds.max_y:.0f})"
        )
    else:
        print(f"no {args.side} aligned column detected")
    print(f"json: {json_path}")
    print(f"png:  {png_path}")


if __name__ == "__main__":
    main()
