"""Google Cloud Vision OCR fallback (service account or API key)."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account


def _quota_project() -> str | None:
    return (
        os.environ.get("GCP_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT_NUMBER")
    )


def _bearer_headers(creds) -> dict[str, str]:
    if not creds.valid:
        creds.refresh(Request())
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }
    project = _quota_project()
    if project:
        headers["X-Goog-User-Project"] = project
    return headers


def _auth_headers() -> dict[str, str]:
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY") or os.environ.get("GOOGLE_VISION_API_KEY")
    if api_key:
        return {"Content-Type": "application/json"}

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    keyfile_json = os.environ.get("GOOGLE_CLOUD_KEYFILE_JSON")
    if keyfile_json and not creds_path:
        tmp = Path(tempfile.gettempdir()) / "doc_extract_vision_creds.json"
        tmp.write_text(keyfile_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(tmp)
        creds_path = str(tmp)

    if creds_path and Path(creds_path).is_file():
        creds = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return _bearer_headers(creds)

    # No JSON key file — use gcloud ADC (works when org blocks SA key download).
    try:
        import google.auth

        creds, _project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return _bearer_headers(creds)
    except Exception as exc:
        raise RuntimeError(
            "OCR not configured. Your org blocks service-account JSON keys — use one of:\n"
            "  1) GOOGLE_CLOUD_API_KEY with billing enabled on the GCP project, or\n"
            "  2) gcloud auth application-default login --project=YOUR_PROJECT_ID\n"
            f"ADC error: {exc}"
        ) from exc


def _vision_url() -> str:
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY") or os.environ.get("GOOGLE_VISION_API_KEY")
    if api_key:
        return f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    return "https://vision.googleapis.com/v1/images:annotate"


def _call_vision(image_path: Path) -> dict[str, Any]:
    content = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "requests": [
            {
                "image": {"content": content},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            }
        ]
    }
    resp = requests.post(_vision_url(), headers=_auth_headers(), json=payload, timeout=120)
    if resp.status_code == 403:
        raise RuntimeError(
            "Google Vision API returned 403 Forbidden. Enable Cloud Vision API and billing "
            "on your GCP project, or verify GOOGLE_CLOUD_API_KEY in tools/doc-extract-web/.env.local"
        )
    resp.raise_for_status()
    responses = resp.json().get("responses") or []
    if not responses:
        raise RuntimeError("Google Vision returned no responses")
    if "error" in responses[0]:
        raise RuntimeError(f"Google Vision error: {responses[0]['error']}")
    return responses[0]


def _word_text(word: dict[str, Any]) -> str:
    symbols = word.get("symbols") or []
    return "".join(s.get("text", "") for s in symbols).strip()


def _vertices(box: dict[str, Any] | None) -> list[dict[str, float]]:
    if not box:
        return []
    verts = box.get("vertices") or []
    out: list[dict[str, float]] = []
    for v in verts:
        out.append({"x": float(v.get("x", 0)), "y": float(v.get("y", 0))})
    return out


def _vision_to_standard(vision_resp: dict[str, Any], image_path: Path) -> dict[str, Any]:
    from PIL import Image

    annotation = vision_resp.get("fullTextAnnotation") or {}
    full_text = annotation.get("text") or ""
    pages = annotation.get("pages") or []
    words: list[dict[str, Any]] = []

    for page in pages:
        for block in page.get("blocks") or []:
            for paragraph in block.get("paragraphs") or []:
                for word in paragraph.get("words") or []:
                    text = _word_text(word)
                    if not text:
                        continue
                    bounds = _vertices(word.get("boundingBox"))
                    if bounds:
                        words.append({"text": text, "bounds": bounds})

    with Image.open(image_path) as img:
        dimensions = {"width": img.width, "height": img.height}

    page_bounds = []
    if pages:
        page_bounds = _vertices((pages[0].get("property") or {}).get("detectedBreak"))
        if not page_bounds and words:
            xs = [v["x"] for w in words for v in w["bounds"]]
            ys = [v["y"] for w in words for v in w["bounds"]]
            if xs and ys:
                page_bounds = [
                    {"x": min(xs), "y": min(ys)},
                    {"x": max(xs), "y": min(ys)},
                    {"x": max(xs), "y": max(ys)},
                    {"x": min(xs), "y": max(ys)},
                ]

    return {
        "dimensions": dimensions,
        "ocr_data": {
            "text": full_text,
            "document_text": {
                "locale": "en",
                "text": full_text,
                "bounds": page_bounds,
                "words": words,
            },
        },
    }


def run_ocr(image_path: Path, vision_out: Path) -> dict[str, Any]:
    raw = _call_vision(image_path)
    vision = _vision_to_standard(raw, image_path)
    if not vision["ocr_data"]["document_text"]["words"]:
        raise RuntimeError("Google Vision returned no word bounds")
    vision_out.parent.mkdir(parents=True, exist_ok=True)
    vision_out.write_text(json.dumps(vision, indent=2))
    return vision
