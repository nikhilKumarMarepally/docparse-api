#!/usr/bin/env python3
"""Apply content classification to existing section gallery JSONs (no OCR rerun)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from section_content_classifier import classify_sections_payload, load_default_model  # noqa: E402

ROOT = SCRIPT_DIR.parents[4]
GALLERY_ROOTS = [
    ROOT / "wa577_gallery" / "credit_app_sections",
]


def apply_to_gallery(
    gallery_root: Path,
    *,
    document_type: str = "credit_application",
) -> int:
    model = load_default_model()
    count = 0
    for sections_json in sorted(gallery_root.rglob("*_sections.json")):
        try:
            payload = json.loads(sections_json.read_text())
        except Exception:
            continue
        sections = payload.get("sections") or []
        if not sections:
            continue
        classified = classify_sections_payload(
            sections,
            document_type=document_type,
            model=model,
        )
        payload["document_type"] = document_type
        payload["sections"] = classified
        sections_json.write_text(json.dumps(payload, indent=2))
        classified_path = sections_json.with_name(
            sections_json.name.replace("_sections.json", "_sections_classified.json")
        )
        classified_path.write_text(
            json.dumps(
                {
                    "document_type": document_type,
                    "sections": classified,
                    "source": str(sections_json),
                },
                indent=2,
            )
        )
        count += 1
    return count


def main() -> None:
    total = 0
    for root in GALLERY_ROOTS:
        if root.exists():
            n = apply_to_gallery(root)
            print(f"{root}: classified {n} section files")
            total += n
    print(f"Done: {total} files")


if __name__ == "__main__":
    main()
