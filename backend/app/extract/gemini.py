from __future__ import annotations

import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from app.extract.base import VisionExtractor
from app.paths import is_cloud_deploy
from app.table_detect import looks_like_table

DEFAULT_MODEL = os.environ.get("DOC_EXTRACT_GEMINI_MODEL", "gemini-3.1-flash-lite")
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)

SCHEMA_FREE_PROMPT = """Extract every filled label→value pair visible in this form section.
Return JSON only: {{"fields": {{"<snake_case_label>": "<value>"}}}}.
Derive labels from printed field labels. Omit blanks and boilerplate.
Dates ISO 8601; numbers as strings or numbers.

OCR text for this section:
{ocr_text}
"""

TABLE_CLASSIFY_PROMPT = """Is this a multi-row line-item table with repeating item rows, not a form block or summary/totals band?
Return JSON only: {{"is_line_item_table": true|false}}

OCR text for this section:
{ocr_text}
"""

TABLE_PROMPT = """This section is a data table (column headers plus one or more data rows).
Return JSON only:
{{"table": {{"columns": ["<snake_case_header>", ...], "rows": [{{<header>: <value>, ...}}, ...]}}}}
Use printed column headers as snake_case keys. Include one object per data row, not the header row.
Preserve cell text as printed. Include sub-rows (e.g. country of origin) inside the row object.

OCR text for this section:
{ocr_text}
"""

def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def resolve_location(model: str) -> str:
    if os.environ.get("GOOGLE_CLOUD_LOCATION"):
        return os.environ["GOOGLE_CLOUD_LOCATION"]
    if model.startswith("gemini-3.1"):
        return "global"
    return "us-central1"


def setup_gemini_client(*, model: str):
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    if is_cloud_deploy():
        raise RuntimeError(
            "Set GEMINI_API_KEY on Render (personal API key only; company Vertex creds are disabled)"
        )

    project = (
        os.environ.get("DOC_EXTRACT_GEMINI_PROJECT_ID")
        or os.environ.get("GCP_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    if not project:
        raise RuntimeError("Set GCP_PROJECT_ID (Vertex) or GEMINI_API_KEY / GOOGLE_API_KEY")

    keyfile_json = os.environ.get("DOC_EXTRACT_GEMINI_KEYFILE_JSON") or os.environ.get(
        "GOOGLE_CLOUD_KEYFILE_JSON"
    )
    if keyfile_json and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        creds_path = Path(tempfile.gettempdir()) / "doc_extract_gcp_creds.json"
        creds_path.write_text(keyfile_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

    location = resolve_location(model)
    return genai.Client(vertexai=True, project=project, location=location)


def parse_json_response(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    fence = JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def call_gemini(client: Any, model: str, crop_img: Image.Image, prompt: str) -> dict[str, Any]:
    from google.genai import types

    buf = io.BytesIO()
    crop_img.save(buf, format="PNG")
    image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=model,
        contents=[image_part, prompt],
        config=config,
    )
    text = getattr(response, "text", None) or ""
    if not text and getattr(response, "candidates", None):
        parts = response.candidates[0].content.parts
        text = "".join(getattr(p, "text", "") or "" for p in parts)
    return parse_json_response(text)


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "yes", "1"}:
            return True
        if norm in {"false", "no", "0"}:
            return False
    return None


def gemini_confirms_line_item_table(
    client: Any,
    model: str,
    crop_img: Image.Image,
    ocr_text: str,
) -> bool:
    """Second opinion when geometry looks grid-like — filters summary/footer false positives."""
    prompt = TABLE_CLASSIFY_PROMPT.format(ocr_text=ocr_text or "(empty)")
    parsed = call_gemini(client, model, crop_img, prompt)
    verdict = _parse_bool(parsed.get("is_line_item_table"))
    if verdict is None:
        verdict = _parse_bool(parsed.get("is_table"))
    return verdict is True


def _extract_table_fields(
    client: Any,
    model: str,
    crop_img: Image.Image,
    ocr_text: str,
) -> dict[str, Any] | None:
    prompt = TABLE_PROMPT.format(ocr_text=ocr_text or "(empty)")
    parsed = call_gemini(client, model, crop_img, prompt)
    table = parsed.get("table")
    if not isinstance(table, dict):
        return None
    rows = table.get("rows")
    columns = table.get("columns")
    if not isinstance(rows, list) or not rows:
        return None
    # A single wide row is a horizontal form block, not a line-item table.
    if len(rows) == 1 and isinstance(rows[0], dict):
        return {
            str(k): v
            for k, v in rows[0].items()
            if v is not None and str(v).strip()
        }
    out: dict[str, Any] = {"line_items": rows}
    if isinstance(columns, list) and columns:
        out["table_columns"] = columns
    return out


def _extract_schema_free_fields(
    client: Any,
    model: str,
    crop_img: Image.Image,
    ocr_text: str,
) -> dict[str, Any]:
    prompt = SCHEMA_FREE_PROMPT.format(ocr_text=ocr_text or "(empty)")
    parsed = call_gemini(client, model, crop_img, prompt)
    fields = parsed.get("fields")
    if isinstance(fields, dict):
        return {str(k): v for k, v in fields.items() if v is not None and str(v).strip()}
    return parsed if isinstance(parsed, dict) else {}


class GeminiExtractor(VisionExtractor):
    def __init__(self, *, model: str = DEFAULT_MODEL, client: Any | None = None) -> None:
        self.model = model
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = setup_gemini_client(model=self.model)
        return self._client

    def extract_section(
        self,
        crop_img: Image.Image,
        ocr_text: str,
        *,
        section_words: Sequence[Any] | None = None,
        bounds: dict[str, Any] | None = None,
        table_band: bool = False,
    ) -> dict[str, Any]:
        use_table_prompt = table_band
        if not use_table_prompt and looks_like_table(section_words, bounds=bounds):
            use_table_prompt = gemini_confirms_line_item_table(
                self.client,
                self.model,
                crop_img,
                ocr_text,
            )

        if use_table_prompt:
            table_fields = _extract_table_fields(self.client, self.model, crop_img, ocr_text)
            if table_fields is not None:
                return table_fields

        return _extract_schema_free_fields(self.client, self.model, crop_img, ocr_text)


def get_extractor() -> VisionExtractor:
    provider = (os.environ.get("EXTRACT_PROVIDER") or "gemini").lower()
    if provider == "qwen":
        raise NotImplementedError("Qwen 2.5 VL provider not yet implemented; set EXTRACT_PROVIDER=gemini")
    return GeminiExtractor()
