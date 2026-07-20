"""Google Document AI OCR adapter."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

OCR_DATA = "ocr_data"
DOCUMENT_TEXT = "document_text"
TOKENS = "tokens"
LAYOUT = "layout"
TEXT_ANCHOR = "text_anchor"
TEXT_SEGMENTS = "text_segments"
BOUNDING_POLY = "bounding_poly"
START_INDEX = "start_index"
END_INDEX = "end_index"
VERTICES = "vertices"
CONFIDENCE = "confidence"
LANGUAGE_CODE = "language_code"
LOCATION = "us"


def _project_id() -> str:
    return (
        os.environ.get("GCP_PROJECT_NUMBER")
        or os.environ.get("GCP_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or ""
    )


def _processor_id() -> str:
    return os.environ.get("DOCUMENT_AI_PROCESSOR_ID") or os.environ.get("PROCESSOR_ID") or ""


def _auth_headers() -> dict[str, str]:
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    keyfile_json = os.environ.get("GOOGLE_CLOUD_KEYFILE_JSON")
    if keyfile_json and not creds_path:
        tmp = Path("/tmp/doc_extract_gcp_creds.json")
        tmp.write_text(keyfile_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(tmp)
        creds_path = str(tmp)
    if not creds_path:
        raise RuntimeError(
            "Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_KEYFILE_JSON for Document AI OCR"
        )
    creds = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    return {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }


def _docai_url() -> str:
    project = _project_id()
    processor = _processor_id()
    if not project or not processor:
        raise RuntimeError("Set GCP_PROJECT_NUMBER (or GCP_PROJECT_ID) and DOCUMENT_AI_PROCESSOR_ID")
    return (
        f"https://{LOCATION}-documentai.googleapis.com/v1/projects/{project}"
        f"/locations/{LOCATION}/processors/{processor}:process"
    )


def _call_docai(image_path: Path, *, mime_type: str = "image/png") -> dict[str, Any]:
    content = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "rawDocument": {"content": content, "mimeType": mime_type},
        "processOptions": {"ocrConfig": {"enableSymbol": False}},
    }
    headers = _auth_headers()
    url = _docai_url()
    for attempt in range(3):
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        if resp.status_code in {429, 500, 502, 503, 504} and attempt < 2:
            time.sleep(5)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Document AI request failed after retries")


def _process_language(node: list[dict[str, Any]]) -> str:
    best = ""
    best_conf = -1.0
    for entry in node:
        conf = float(entry.get(CONFIDENCE, 0))
        if conf > best_conf:
            best_conf = conf
            best = str(entry.get(LANGUAGE_CODE, ""))
    return best


def _token_layout(token: dict[str, Any]) -> dict[str, Any]:
    return token.get("layout") or token.get(LAYOUT) or {}


def _token_segments(layout: dict[str, Any]) -> list[dict[str, Any]]:
    anchor = layout.get("textAnchor") or layout.get(TEXT_ANCHOR) or {}
    return anchor.get("textSegments") or anchor.get(TEXT_SEGMENTS) or []


def _token_vertices(layout: dict[str, Any]) -> list[dict[str, Any]]:
    poly = layout.get("boundingPoly") or layout.get(BOUNDING_POLY) or {}
    return poly.get("vertices") or poly.get(VERTICES) or []


def _process_tokens(tokens: list[dict[str, Any]], page_text: str) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for token in tokens:
        layout = _token_layout(token)
        segments = _token_segments(layout)
        if not segments:
            continue
        start = int(segments[0].get("startIndex", segments[0].get(START_INDEX, 0)))
        end = int(segments[0].get("endIndex", segments[0].get(END_INDEX, -1)))
        text = page_text[start:end].strip()
        if not text:
            continue
        words.append({"text": text, "bounds": _token_vertices(layout)})
    return words


def _standardize_docai_response(raw: dict[str, Any], image_path: Path) -> dict[str, Any]:
    """Convert Document AI response to prod vision JSON shape."""
    from PIL import Image

    document = raw.get("document") or {}
    pages = document.get("pages") or []
    if not pages:
        raise RuntimeError("Document AI returned no pages")

    page = pages[0]
    tokens = page.get("tokens") or []
    page_text = document.get("text") or ""
    page_layout = page.get("layout") or {}
    bounds_poly = page_layout.get("boundingPoly") or page_layout.get("bounding_poly") or {}
    bounds = bounds_poly.get("vertices") or bounds_poly.get(VERTICES) or []
    for coord in bounds:
        coord["x"] = coord.get("x", 0)
        coord["y"] = coord.get("y", 0)

    language = ""
    langs = page.get("detectedLanguages") or page.get("detected_languages") or []
    if langs:
        language = _process_language(langs)

    with Image.open(image_path) as img:
        dimensions = {"width": img.width, "height": img.height}

    words = _process_tokens(tokens, page_text)
    return {
        "dimensions": dimensions,
        "ocr_data": {
            "text": page_text,
            "pages": [{"tokens": tokens}],
            "document_text": {
                "locale": language,
                "text": page_text,
                "bounds": bounds,
                "words": words,
            },
        },
    }


def run_ocr(image_path: Path, vision_out: Path) -> dict[str, Any]:
    raw = _call_docai(image_path)
    vision = _standardize_docai_response(raw, image_path)
    vision_out.parent.mkdir(parents=True, exist_ok=True)
    vision_out.write_text(json.dumps(vision, indent=2))
    return vision
