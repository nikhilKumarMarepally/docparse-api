"""OCR router — Document AI when configured, else Google Cloud Vision."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _docai_configured() -> bool:
    project = (
        os.environ.get("GCP_PROJECT_NUMBER")
        or os.environ.get("GCP_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    processor = os.environ.get("DOCUMENT_AI_PROCESSOR_ID") or os.environ.get("PROCESSOR_ID")
    return bool(project and processor)


def run_ocr(image_path: Path, vision_out: Path) -> dict[str, Any]:
    if _docai_configured():
        from app.ocr.docai import run_ocr as run_docai

        return run_docai(image_path, vision_out)

    from app.ocr.vision_api import run_ocr as run_vision

    return run_vision(image_path, vision_out)
