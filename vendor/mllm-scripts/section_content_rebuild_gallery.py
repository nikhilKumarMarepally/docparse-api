#!/usr/bin/env python3
"""Rebuild credit_app_sections manifest + HTML from classified section JSONs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_credit_app_sections import OUT_ROOT, _summarize_content_types, build_html  # noqa: E402


def main() -> None:
    manifest_path = OUT_ROOT / "manifest.json"
    if not manifest_path.exists():
        print(f"Missing {manifest_path}")
        return
    data = json.loads(manifest_path.read_text())
    results: list[dict[str, Any]] = data.get("results") or []
    for row in results:
        sections_json = Path(row.get("sections_json") or "")
        if sections_json.exists():
            payload = json.loads(sections_json.read_text())
            row["content_types_summary"] = _summarize_content_types(payload.get("sections") or [])
            classified = sections_json.with_name(
                sections_json.name.replace("_sections.json", "_sections_classified.json")
            )
            row["sections_classified_json"] = str(classified.resolve()) if classified.exists() else ""
    out = {
        **data,
        "results": results,
        "classifier": "section_content_classifier",
    }
    manifest_path.write_text(json.dumps(out, indent=2))
    gallery = Path.home() / "Desktop" / "credit_app_sections_gallery.html"
    build_html(results, gallery)
    print(f"Updated manifest + gallery ({len(results)} pages)")


if __name__ == "__main__":
    main()
