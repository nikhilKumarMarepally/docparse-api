#!/usr/bin/env python3
"""Prompt-based field detection + value extraction demo for title_application patches.

Uses prod Google Vision OCR (word bounds) to populate section text, crops each
section from the full page, runs Gemini field detection per patch, then a second
Gemini call to extract values for detected fields (prod schema shape).

Usage:
  python title_app_prompt_field_demo.py
  python title_app_prompt_field_demo.py --doc 0e66117c 899d7cbd d4eb2430
  python title_app_prompt_field_demo.py --skip-llm
  python title_app_prompt_field_demo.py --force
  python title_app_prompt_field_demo.py --vision wa577_gallery/.../vision.json
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ocr_word_to_line_boxes import (  # noqa: E402
    Box,
    Word,
    load_vision,
    load_words,
    parse_bounds,
    s3_download,
    vision_key_from_image_uri,
    words_to_lines,
)
from section_content_gemini_gallery import (  # noqa: E402
    DEFAULT_MODEL,
    ENV_LOCAL,
    build_section_prompt,
    call_gemini_section,
    crop_section_image,
    enhance_section_crop,
    flatten_paths,
    load_dotenv,
    nested_skeleton_from_paths,
    parse_json_response,
    setup_gemini_client,
)
from section_content_taxonomy import load_document_field_paths  # noqa: E402

TITLE_APP_YAML = (
    ROOT.parent
    / "techno-configs/techno_configs/envs/qa/document_fields/extractions/llm_configs/title_application.yml"
)
DOC_TYPE = "title_application"
S3_BUCKET = "informed-techno-core-prod-exchange"
VIN_SECTIONS_INDEX = ROOT / "wa577_gallery/vin_sections_batch/index.json"

CACHE_ROOT = (
    ROOT / "wa577_gallery" / "section_classifier" / "prompt_field_experiment_cache" / DOC_TYPE
)
CROP_ROOT = CACHE_ROOT / "crops"
DESKTOP_HTML = Path.home() / "Desktop" / "title_app_prompt_field_demo.html"
RESULTS_MD = (
    ROOT / "wa577_gallery" / "section_classifier" / "prompt_field_experiment.md"
)

DEFAULT_SHORT_IDS = ("0e66117c", "899d7cbd", "d4eb2430")

# After Gemini field detection, sections with no extractable fields are treated as
# garbage (boilerplate, section headers, barcodes, blank margins) and skip extraction.
GARBAGE_FIELD_THRESHOLD = 0

# title_application schema has top-level `year`, but QA llm_config has no field_prompt for it.
_CATALOG_SUPPLEMENTS: dict[str, str] = {
    "year": (
        "extract the 4-digit model year of the primary vehicle from the "
        "Year/Make/Model/Body Style section; return integer only"
    ),
}

# Demo detection groups use pseudo-prefixes; prod schema uses top-level keys.
_DETECTION_GROUP_PREFIXES = ("vehicle.", "financial_dealer.", "document.")
_SIGNATURE_LEAVES = frozenset(
    {"section_present", "signature_present", "signature_date", "e_signed"}
)


@dataclass
class DocDemoConfig:
    short_id: str
    doc_id: str
    sections: list[dict[str, Any]] = field(default_factory=list)
    page: str = "p0"

    @property
    def batch_dir(self) -> Path:
        return ROOT / "wa577_gallery/vin_sections_batch" / self.short_id

    @property
    def crop_wins_dir(self) -> Path:
        return ROOT / "wa577_gallery/vin_crop_wins" / self.short_id


# Per-doc section picks — indices differ by form layout.
DEMO_DOCS: dict[str, DocDemoConfig] = {
    "0e66117c": DocDemoConfig(
        short_id="0e66117c",
        doc_id="0e66117c-50f4-4963-ae5a-311162caca17",
        sections=[
            {"index": 0, "label": "owner_applicant", "note": "Primary owner/applicant identity + address"},
            {"index": 1, "label": "vin_vehicle", "note": "VIN, odometer, year/make/model"},
            {"index": 4, "label": "signature", "note": "Applicant signature block"},
        ],
    ),
    "899d7cbd": DocDemoConfig(
        short_id="899d7cbd",
        doc_id="899d7cbd-ba36-46f2-88d3-8ef79282365f",
        sections=[
            {"index": 0, "label": "vin_vehicle", "note": "Colorado DMV header + VIN/YMM/odometer block"},
            {"index": 2, "label": "owner_applicant", "note": "Registrant legal name + address"},
            {"index": 5, "label": "lien_holder", "note": "First lienholder name + ELT amount"},
        ],
    ),
    "d4eb2430": DocDemoConfig(
        short_id="d4eb2430",
        doc_id="d4eb2430-6dff-401a-8783-e7ddb37e5155",
        sections=[
            {"index": 1, "label": "privacy_boilerplate", "note": "SC DMV privacy disclosure + mailing instructions — no extractable fields"},
            {"index": 3, "label": "vehicle_owner", "note": "SC DMV: VIN + primary owner/address"},
            {"index": 4, "label": "leasing_contact", "note": "Voter registration + leasing company block"},
            {"index": 6, "label": "odometer_lien", "note": "Odometer certification + first lien"},
        ],
    ),
}


def lookup_doc_id(short_id: str) -> str | None:
    if short_id in DEMO_DOCS:
        return DEMO_DOCS[short_id].doc_id
    if VIN_SECTIONS_INDEX.exists():
        payload = json.loads(VIN_SECTIONS_INDEX.read_text())
        for row in payload.get("results") or []:
            if row.get("short_id") == short_id and row.get("doc_id"):
                return str(row["doc_id"])
    return None


def resolve_asset_paths(
    config: DocDemoConfig,
    *,
    vision_override: Path | None = None,
) -> tuple[Path, Path, Path, str]:
    """Return (full_png, sections_json, vision_json, ocr_source_note)."""
    short_id = config.short_id
    page = config.page
    batch_dir = config.batch_dir
    crop_wins_dir = config.crop_wins_dir
    payload_json = crop_wins_dir / f"payload_{page}_full.json"

    full_png = batch_dir / f"{short_id}_{page}.png"
    sections_json = batch_dir / f"{short_id}_{page}_sections/{short_id}_{page}_sections.json"
    vision_json = vision_override or (batch_dir / f"{short_id}_{page}_sections/vision.json")

    if not full_png.exists():
        full_png = crop_wins_dir / f"{short_id}_{page}.png"
    if not sections_json.exists():
        sections_json = crop_wins_dir / f"{short_id}_{page}_sections/{short_id}_{page}_sections.json"
    if not vision_json.exists():
        vision_json = crop_wins_dir / f"{short_id}_{page}_sections/vision.json"

    if vision_json.exists():
        return full_png, sections_json, vision_json, f"local Google Vision JSON ({vision_json})"

    if payload_json.exists():
        try:
            payload = json.loads(payload_json.read_text())
            image_uri = payload["detail"]["data"]["image_uri"]
            vision_key = vision_key_from_image_uri(image_uri)
            dest_dir = batch_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            if not full_png.exists():
                s3_download(image_uri, dest_dir / f"{short_id}_{page}.png", "prod")
                full_png = dest_dir / f"{short_id}_{page}.png"
            vision_dest = dest_dir / f"{short_id}_{page}_sections/vision.json"
            vision_dest.parent.mkdir(parents=True, exist_ok=True)
            s3_download(f"s3://{S3_BUCKET}/{vision_key}", vision_dest, "prod")
            return (
                full_png,
                sections_json,
                vision_dest,
                f"prod S3 vision ({vision_key})",
            )
        except Exception as exc:
            raise SystemExit(
                f"No local vision JSON and S3 fetch failed ({exc}). "
                f"Expected {batch_dir}/.../vision.json or {crop_wins_dir}/.../vision.json"
            ) from exc

    raise SystemExit(
        f"Missing Google Vision OCR JSON for {short_id}. Tried:\n"
        f"  {batch_dir / f'{short_id}_{page}_sections/vision.json'}\n"
        f"  {crop_wins_dir / f'{short_id}_{page}_sections/vision.json'}"
    )


def word_in_bounds(word: Word, box: Box) -> bool:
    """Keep words whose centroid lies inside the section box."""
    cx, cy = word.box.centroid_x, word.box.centroid_y
    return box.min_x <= cx <= box.max_x and box.min_y <= cy <= box.max_y


def text_from_bounds(words: list[Word], bounds: dict[str, Any]) -> str:
    """Extract Google OCR text for words inside a section bounding box."""
    box = parse_bounds(bounds)
    if box is None:
        return ""
    filtered = [w for w in words if word_in_bounds(w, box)]
    if not filtered:
        return ""
    lines = words_to_lines(filtered, full_width=False)
    return "\n".join(line.text for line in lines)


def format_bounds(bounds: dict[str, Any]) -> str:
    if not bounds:
        return "(none)"
    return (
        f"min_x={bounds.get('min_x')}, min_y={bounds.get('min_y')}, "
        f"max_x={bounds.get('max_x')}, max_y={bounds.get('max_y')}"
    )


def parse_field_prompts(yml_path: Path) -> dict[str, str]:
    data = yaml.safe_load(yml_path.read_text()) or {}
    payload = (data.get("model_info") or {}).get("payload_config") or {}
    field_prompts = (payload.get("prompt_config") or {}).get("field_prompts") or {}
    out: dict[str, str] = {}
    for key, entry in field_prompts.items():
        field_key = str(key).strip()
        if not field_key:
            continue
        if isinstance(entry, dict):
            desc = entry.get("default") or entry.get("prompt") or ""
            if isinstance(desc, list):
                desc = " ".join(str(x) for x in desc)
        else:
            desc = str(entry)
        desc = str(desc).strip()
        if desc:
            out[field_key] = desc
    return out


def supplement_field_prompts(field_prompts: dict[str, str]) -> dict[str, str]:
    """Merge schema fields that prod injects via $$_SCHEMA but lack field_prompt keys."""
    out = dict(field_prompts)
    for path, desc in _CATALOG_SUPPLEMENTS.items():
        out.setdefault(path, desc)
    return out


def field_catalog(field_prompts: dict[str, str]) -> dict[str, str]:
    paths = set(load_document_field_paths(DOC_TYPE))
    paths.update(_CATALOG_SUPPLEMENTS)
    catalog: dict[str, str] = {}
    for path in sorted(paths):
        desc = field_prompts.get(path, "")
        if not desc:
            parts = path.split(".")
            for end in range(len(parts) - 1, 0, -1):
                parent = ".".join(parts[:end])
                if parent in field_prompts:
                    desc = field_prompts[parent]
                    break
        catalog[path] = desc or "(no description)"
    return catalog


def group_field_catalog(catalog: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, desc in sorted(catalog.items()):
        if path.startswith("applicants."):
            groups["applicants"].append((path, desc))
        elif path.startswith("signatures."):
            groups["signatures"].append((path, desc))
        elif path.startswith("owner."):
            groups["owner"].append((path, desc))
        elif any(
            path.startswith(p)
            for p in ("vin", "make", "model", "year", "odometer", "vehicle", "trade_in", "collateral")
        ) or path in {"vin", "make", "model", "odometer", "collateral_type", "additional_collaterals"}:
            groups["vehicle"].append((path, desc))
        elif any(
            k in path
            for k in ("lien", "dealer", "sale_price", "purchase_date", "taxes", "fi_manager", "elt_code")
        ):
            groups["financial_dealer"].append((path, desc))
        else:
            groups["document"].append((path, desc))
    return dict(groups)


def build_detection_prompt(catalog: dict[str, str], ocr_text: str, *, ocr_label: str = "OCR") -> str:
    groups = group_field_catalog(catalog)
    lines = [
        f"You are analyzing a cropped region from a {DOC_TYPE} page image.",
        f"Below is the catalog of possible extractable field paths for {DOC_TYPE}.",
        "Return ONLY field paths that have extractable data IN THIS PATCH — filled form values",
        "or clearly completed fields visible in the image. Do not list blank label rows alone.",
        "Do not list fields mentioned only in legal/disclosure boilerplate without a dedicated form field.",
        'If none apply, return {"fields": []}.',
        "",
        "Field catalog (grouped):",
    ]
    for group_name, entries in sorted(groups.items()):
        lines.append(f"\n[{group_name}]")
        for path, desc in entries:
            short_desc = desc.split(".")[0][:120] if len(desc) > 120 else desc
            lines.append(f"- {path}: {short_desc}")
    lines.extend(
        [
            "",
            'Return JSON only: {"fields": ["dotted.path", ...]}',
            "",
            f"{ocr_label} text for this patch:",
            ocr_text or "(empty)",
        ]
    )
    return "\n".join(lines)


def parse_fields_response(parsed: dict[str, Any]) -> list[str]:
    if not parsed:
        return []
    if "fields" in parsed and isinstance(parsed["fields"], list):
        return sorted(str(f) for f in parsed["fields"] if f)
    fields = [k for k, v in parsed.items() if v and k != "fields"]
    return sorted(fields)


def is_garbage_section(prompt_fields: list[str]) -> bool:
    """True when detection found no extractable fields (len <= GARBAGE_FIELD_THRESHOLD)."""
    return len(prompt_fields) <= GARBAGE_FIELD_THRESHOLD


def map_detection_path_to_schema(path: str) -> str:
    """Map demo detection paths (vehicle.vin) to prod schema keys (vin)."""
    for prefix in _DETECTION_GROUP_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :]
    parts = path.split(".")
    if (
        len(parts) >= 4
        and parts[0] == "signatures"
        and parts[2] in _SIGNATURE_LEAVES
        and parts[2] == parts[3]
    ):
        return ".".join(parts[:3])
    return path


def map_detection_paths_to_schema(paths: list[str]) -> list[str]:
    mapped = {map_detection_path_to_schema(p) for p in paths if p}
    return sorted(mapped)


def format_value_display(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def flattened_nonempty_values(extracted: dict[str, Any]) -> list[tuple[str, Any]]:
    if not extracted:
        return []
    flat = flatten_paths(extracted)
    return sorted(
        (path, val)
        for path, val in flat.items()
        if val is not None and val != "" and val != [] and val != {}
    )


def cache_path(short_id: str, section_index: int) -> Path:
    return CACHE_ROOT / f"{short_id}_s{section_index}.json"


def load_cached(short_id: str, section_index: int) -> dict[str, Any] | None:
    path = cache_path(short_id, section_index)
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_cached(short_id: str, section_index: int, payload: dict[str, Any]) -> None:
    path = cache_path(short_id, section_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def call_gemini_detection(
    client: Any,
    model: str,
    crop_img: Image.Image,
    prompt: str,
) -> tuple[list[str], str, dict[str, Any]]:
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
    parsed = parse_json_response(text)
    fields = parse_fields_response(parsed)
    return fields, text, parsed


def process_section(
    config: DocDemoConfig,
    section: dict[str, Any],
    meta: dict[str, Any],
    catalog: dict[str, str],
    field_prompts: dict[str, str],
    *,
    words: list[Word],
    full_png: Path,
    sections_json: Path,
    vision_json: Path,
    ocr_source: str,
    client: Any | None,
    model: str,
    skip_llm: bool,
    force: bool,
) -> dict[str, Any]:
    short_id = config.short_id
    idx = int(section["index"])
    bounds = section.get("bounds") or {}
    google_ocr_text = text_from_bounds(words, bounds)

    cached = None if force else load_cached(short_id, idx)
    crop_rel = f"crops/{short_id}_s{idx}.png"
    crop_path = CACHE_ROOT / crop_rel

    prompt_str = build_detection_prompt(
        catalog,
        google_ocr_text,
        ocr_label="Google OCR",
    )
    prompt_fields: list[str] = []
    prompt_text = ""
    prompt_json: dict[str, Any] = {}
    schema_fields: list[str] = []
    extracted_values: dict[str, Any] = {}
    extraction_prompt = ""
    raw_extraction_response = ""

    if cached is not None:
        prompt_fields = cached.get("prompt_fields") or []
        prompt_text = cached.get("raw_response") or ""
        prompt_json = cached.get("parsed_response") or {}
        schema_fields = cached.get("schema_fields") or map_detection_paths_to_schema(prompt_fields)
        extracted_values = cached.get("extracted_values") or {}
        extraction_prompt = cached.get("extraction_prompt") or ""
        raw_extraction_response = cached.get("raw_extraction_response") or ""
        if cached.get("prompt") and cached.get("google_ocr_text") == google_ocr_text:
            prompt_str = cached.get("prompt") or prompt_str

    need_detection = (
        not skip_llm
        and client is not None
        and bounds
        and full_png.exists()
        and (force or cached is None or not prompt_text)
    )
    need_extraction = (
        not skip_llm
        and client is not None
        and bounds
        and full_png.exists()
        and (force or cached is None or not raw_extraction_response)
    )

    if bounds and full_png.exists():
        crop_img = enhance_section_crop(crop_section_image(full_png, bounds))
        CROP_ROOT.mkdir(parents=True, exist_ok=True)
        crop_img.save(crop_path)

        if need_detection:
            prompt_fields, prompt_text, prompt_json = call_gemini_detection(
                client, model, crop_img, prompt_str
            )
            schema_fields = map_detection_paths_to_schema(prompt_fields)

        if need_extraction and schema_fields:
            skeleton = nested_skeleton_from_paths(schema_fields)
            extraction_prompt = build_section_prompt(
                schema_fields,
                field_prompts,
                google_ocr_text,
                skeleton,
            )
            extracted_values = call_gemini_section(
                client, model, crop_img, extraction_prompt
            )
            raw_extraction_response = json.dumps(extracted_values, indent=2)

        if need_detection or need_extraction:
            save_cached(
                short_id,
                idx,
                {
                    "doc_id": config.doc_id,
                    "short_id": short_id,
                    "document_type": DOC_TYPE,
                    "section_index": idx,
                    "label": meta["label"],
                    "note": meta["note"],
                    "crop_path": str(crop_path),
                    "full_png": str(full_png),
                    "sections_json": str(sections_json),
                    "vision_json": str(vision_json),
                    "ocr_source": ocr_source,
                    "bounds": bounds,
                    "google_ocr_text": google_ocr_text,
                    "prompt": prompt_str,
                    "raw_response": prompt_text,
                    "parsed_response": prompt_json,
                    "prompt_fields": prompt_fields,
                    "schema_fields": schema_fields,
                    "extraction_prompt": extraction_prompt,
                    "raw_extraction_response": raw_extraction_response,
                    "extracted_values": extracted_values,
                    "model": model,
                },
            )
        elif cached is not None and cached.get("google_ocr_text") != google_ocr_text:
            cached = dict(cached)
            cached.update(
                {
                    "vision_json": str(vision_json),
                    "ocr_source": ocr_source,
                    "bounds": bounds,
                    "google_ocr_text": google_ocr_text,
                    "prompt": prompt_str,
                }
            )
            save_cached(short_id, idx, cached)

    garbage = is_garbage_section(prompt_fields)

    return {
        "section_index": idx,
        "label": meta["label"],
        "note": meta["note"],
        "is_garbage": garbage,
        "google_ocr_text": google_ocr_text,
        "google_ocr_preview": google_ocr_text[:400].replace("\n", " "),
        "bounds": bounds,
        "bounds_str": format_bounds(bounds),
        "crop_path": str(crop_path),
        "prompt_fields": prompt_fields,
        "schema_fields": schema_fields,
        "prompt": prompt_str,
        "raw_response": prompt_text,
        "parsed_response": prompt_json,
        "extracted_values": extracted_values,
        "extraction_prompt": extraction_prompt,
        "raw_extraction_response": raw_extraction_response,
        "value_pairs": flattened_nonempty_values(extracted_values),
    }


def img_data_uri(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_html(
    doc_results: list[dict[str, Any]],
    model: str,
) -> str:
    doc_blocks: list[str] = []
    total_sections = 0
    for doc in doc_results:
        config: DocDemoConfig = doc["config"]
        results: list[dict[str, Any]] = doc["results"]
        total_sections += len(results)
        rows = []
        for row in results:
            crop_path = Path(row["crop_path"])
            img_src = img_data_uri(crop_path) if crop_path.exists() else ""
            ocr_full = html.escape(row["google_ocr_text"] or "(empty)")
            resp = html.escape(row["raw_response"] or "{}")
            fields = html.escape(json.dumps(row["prompt_fields"], indent=2))
            schema = html.escape(json.dumps(row.get("schema_fields") or [], indent=2))
            is_garbage = bool(row.get("is_garbage"))
            patch_class = "patch patch-garbage" if is_garbage else "patch"
            if is_garbage:
                banner_html = (
                    '<div class="garbage-banner">'
                    "<strong>Useless information — filtered out</strong>"
                    f"<p>Detection returned {len(row['prompt_fields'])} fields "
                    f"(threshold ≤ {GARBAGE_FIELD_THRESHOLD}); extraction skipped.</p>"
                    "</div>"
                )
                values_html = '<p class="muted">(extraction skipped — no fields detected)</p>'
                extraction_html = (
                    '<p class="muted">Skipped — section classified as garbage after field detection.</p>'
                )
            else:
                banner_html = ""
                value_pairs = row.get("value_pairs") or []
                if value_pairs:
                    value_lines = "\n".join(
                        f'<div class="value-line"><span class="value-key">{html.escape(path)}:</span> '
                        f'<span class="value-val">{html.escape(format_value_display(val))}</span></div>'
                        for path, val in value_pairs
                    )
                    values_html = f'<div class="extracted-values">{value_lines}</div>'
                else:
                    values_html = '<p class="muted">(no values extracted)</p>'
                extraction_json = html.escape(row.get("raw_extraction_response") or "{}")
                extraction_html = f'<pre class="json">{extraction_json}</pre>'
            rows.append(
                f"""<section class="{patch_class}">
  <h3>s{row['section_index']} — {html.escape(row['label'])}</h3>
  <p class="note">{html.escape(row['note'])}</p>
  {banner_html}
  <div class="values-banner{' values-banner-garbage' if is_garbage else ''}">
    <h4>Extracted values</h4>
    {values_html}
  </div>
  <div class="grid">
    <div class="col"><h4>Crop image</h4><img src="{img_src}" alt="s{row['section_index']} crop"/></div>
    <div class="col"><h4>Google OCR (bounds region)</h4><pre class="ocr">{ocr_full}</pre></div>
    <div class="col"><h4>Detection output</h4><pre class="json">{resp}</pre>
    <p class="fields"><strong>detected ({len(row['prompt_fields'])}):</strong></p>
    <pre class="json">{fields}</pre>
    <p class="fields"><strong>schema fields ({len(row.get('schema_fields') or [])}):</strong></p>
    <pre class="json">{schema}</pre></div>
    <div class="col"><h4>Extraction JSON</h4>{extraction_html}</div>
  </div>
</section>"""
            )
        doc_blocks.append(
            f"""<section class="doc">
  <h2>{html.escape(config.short_id)} — {html.escape(config.doc_id)} ({config.page})</h2>
  <p class="meta">OCR: <code>{html.escape(doc['ocr_source'])}</code><br/>
  Vision JSON: <code>{html.escape(str(doc['vision_json']))}</code><br/>
  Full page: <code>{html.escape(str(doc['full_png']))}</code></p>
  {''.join(rows)}
</section>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>title_application field detection + extraction (Google OCR)</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f0f2f8; color: #1a1a2e; }}
h1 {{ margin-bottom: 4px; }}
.meta {{ color: #555; margin-bottom: 16px; }}
.doc {{ background: #e8ebf5; border-radius: 10px; padding: 16px 20px; margin-bottom: 32px; }}
.doc > h2 {{ margin-top: 0; }}
.patch {{ background: #fff; border-radius: 8px; padding: 16px 20px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
.patch-garbage {{ background: #fff5f5; border: 2px solid #e53935; box-shadow: 0 2px 8px rgba(229,57,53,.15); }}
.garbage-banner {{ background: #ffcdd2; border: 2px solid #e53935; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px; color: #b71c1c; }}
.garbage-banner strong {{ font-size: 15px; text-transform: uppercase; letter-spacing: 0.03em; }}
.garbage-banner p {{ margin: 6px 0 0; font-size: 13px; }}
.note {{ color: #666; font-style: italic; margin-top: 0; }}
.values-banner {{ background: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px; }}
.values-banner-garbage {{ background: #ffebee; border-color: #ef9a9a; }}
.values-banner-garbage h4 {{ color: #c62828; }}
.values-banner h4 {{ margin: 0 0 8px; font-size: 13px; text-transform: uppercase; color: #2e7d32; }}
.extracted-values {{ font-family: ui-monospace, monospace; font-size: 14px; line-height: 1.6; }}
.value-key {{ color: #1565c0; font-weight: 600; }}
.value-val {{ color: #1a1a2e; }}
.muted {{ color: #888; font-style: italic; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px; }}
.col h3, .col h4 {{ margin: 0 0 8px; font-size: 13px; text-transform: uppercase; color: #446; }}
img {{ max-width: 100%; border: 1px solid #ccd; border-radius: 4px; }}
pre {{ background: #f8f9fc; border: 1px solid #dde; border-radius: 4px; padding: 10px; font-size: 11px;
       white-space: pre-wrap; word-break: break-word; max-height: 420px; overflow: auto; margin: 0; }}
pre.ocr {{ font-family: inherit; font-size: 12px; }}
pre.json {{ font-family: ui-monospace, monospace; }}
@media (max-width: 1400px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
@media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<h1>title_application — field detection + value extraction (Google OCR)</h1>
<p class="meta">Docs: {len(doc_results)} · Sections: {total_sections} · Model: <code>{html.escape(model)}</code> ·
Cache: <code>{html.escape(str(CACHE_ROOT))}</code></p>
{''.join(doc_blocks)}
</body></html>"""


def append_markdown_section(
    doc_results: list[dict[str, Any]],
    model: str,
) -> None:
    lines = [
        "",
        "---",
        "",
        "## title_application prompt-field demo",
        "",
        f"Model: `{model}` · Cache: `{CACHE_ROOT}`",
        "",
    ]
    for doc in doc_results:
        config: DocDemoConfig = doc["config"]
        results: list[dict[str, Any]] = doc["results"]
        lines.extend(
            [
                f"### `{config.short_id}` ({config.page})",
                "",
                f"Doc ID: `{config.doc_id}` · OCR: {doc['ocr_source']}",
                "",
                "| Section | Label | Fields detected | Status |",
                "|---------|-------|-----------------|--------|",
            ]
        )
        for row in results:
            fields = ", ".join(row["prompt_fields"]) if row["prompt_fields"] else "∅"
            status = "garbage (filtered)" if row.get("is_garbage") else "kept"
            lines.append(
                f"| s{row['section_index']} | {row['label']} | {len(row['prompt_fields'])} — `{fields[:80]}{'…' if len(fields)>80 else ''}` | {status} |"
            )
        lines.extend(["", "#### Prompt responses", ""])
        for row in results:
            lines.extend(
                [
                    f"##### s{row['section_index']} — {row['label']}",
                    "",
                    f"Crop: `{row['crop_path']}`",
                    f"Bounds: `{row['bounds_str']}`",
                    "",
                    "Google OCR (bounds region):",
                    "",
                    "```",
                    row["google_ocr_text"] or "(empty)",
                    "```",
                    "",
                    "```json",
                    row["raw_response"] or "{}",
                    "```",
                    "",
                ]
            )

    existing = RESULTS_MD.read_text() if RESULTS_MD.exists() else ""
    marker = "## title_application prompt-field demo"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip()
    elif "## title_application demo" in existing:
        existing = existing.split("## title_application demo")[0].rstrip()
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(existing + "\n".join(lines))


def process_doc(
    config: DocDemoConfig,
    catalog: dict[str, str],
    field_prompts: dict[str, str],
    *,
    client: Any | None,
    model: str,
    skip_llm: bool,
    force: bool,
    vision_override: Path | None,
) -> dict[str, Any]:
    full_png, sections_json, vision_json, ocr_source = resolve_asset_paths(
        config,
        vision_override=vision_override,
    )
    if not sections_json.exists():
        raise SystemExit(f"Missing {sections_json}")
    if not full_png.exists():
        raise SystemExit(f"Missing page image: {full_png}")

    vision = load_vision(vision_json)
    words = load_words(vision)
    print(f"\n=== {config.short_id} ({config.doc_id}) ===", flush=True)
    print(f"Google OCR: {len(words)} words from {vision_json}", flush=True)
    print(f"OCR source: {ocr_source}", flush=True)

    payload = json.loads(sections_json.read_text())
    sections_by_idx = {int(s["index"]): s for s in payload.get("sections") or []}

    results: list[dict[str, Any]] = []
    for meta in config.sections:
        section_idx = int(meta["index"])
        if section_idx not in sections_by_idx:
            raise SystemExit(
                f"{config.short_id}: section s{section_idx} not in {sections_json}"
            )
        section = sections_by_idx[section_idx]
        google_preview = text_from_bounds(words, section.get("bounds") or {})
        print(
            f"Section s{section_idx} ({meta['label']}) — "
            f"{len(google_preview)} chars Google OCR …",
            flush=True,
        )
        results.append(
            process_section(
                config,
                section,
                meta,
                catalog,
                field_prompts,
                words=words,
                full_png=full_png,
                sections_json=sections_json,
                vision_json=vision_json,
                ocr_source=ocr_source,
                client=client,
                model=model,
                skip_llm=skip_llm,
                force=force,
            )
        )

    return {
        "config": config,
        "results": results,
        "full_png": full_png,
        "sections_json": sections_json,
        "vision_json": vision_json,
        "ocr_source": ocr_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--doc",
        nargs="*",
        metavar="SHORT_ID",
        help="8-char doc prefix(es); default: all demo docs",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore Gemini cache")
    parser.add_argument("--vision", type=Path, help="Override path to Google Vision page JSON")
    args = parser.parse_args()

    load_dotenv(ENV_LOCAL)

    if not TITLE_APP_YAML.exists():
        print(f"Missing {TITLE_APP_YAML}")
        sys.exit(1)

    short_ids = args.doc or list(DEFAULT_SHORT_IDS)
    configs: list[DocDemoConfig] = []
    for short_id in short_ids:
        short_id = short_id[:8]
        if short_id in DEMO_DOCS:
            configs.append(DEMO_DOCS[short_id])
            continue
        doc_id = lookup_doc_id(short_id)
        if not doc_id:
            print(f"Unknown doc {short_id}; add to DEMO_DOCS or vin_sections_batch/index.json")
            sys.exit(1)
        configs.append(
            DocDemoConfig(
                short_id=short_id,
                doc_id=doc_id,
                sections=DEMO_DOCS["0e66117c"].sections,
            )
        )

    field_prompts = supplement_field_prompts(parse_field_prompts(TITLE_APP_YAML))
    catalog = field_catalog(field_prompts)
    print(f"Field catalog: {len(catalog)} paths from {len(field_prompts)} field_prompts", flush=True)

    client = None
    if not args.skip_llm:
        client = setup_gemini_client(model=args.model)

    doc_results: list[dict[str, Any]] = []
    for config in configs:
        doc_results.append(
            process_doc(
                config,
                catalog,
                field_prompts,
                client=client,
                model=args.model,
                skip_llm=args.skip_llm,
                force=args.force,
                vision_override=args.vision,
            )
        )

    html_out = build_html(doc_results, args.model)
    DESKTOP_HTML.write_text(html_out)
    append_markdown_section(doc_results, args.model)

    for doc in doc_results:
        config = doc["config"]
        print(f"\nDoc: {config.doc_id} ({config.short_id})")
        print(f"Sections JSON: {doc['sections_json']}")
        print(f"Vision JSON: {doc['vision_json']}")
        print(f"Full page: {doc['full_png']}")
        for row in doc["results"]:
            print(f"  s{row['section_index']} ({row['label']})")
            print(f"    bounds: {row['bounds_str']}")
            print(f"    Google OCR chars: {len(row['google_ocr_text'])}")
            print(f"    crop: {row['crop_path']}")
            print(f"    fields ({len(row['prompt_fields'])}): {row['prompt_fields']}")
            if row.get("is_garbage"):
                print("    status: GARBAGE — filtered out (no extraction)")
            value_pairs = row.get("value_pairs") or []
            if value_pairs:
                for path, val in value_pairs[:8]:
                    print(f"    {path}: {format_value_display(val)}")
                if len(value_pairs) > 8:
                    print(f"    … +{len(value_pairs) - 8} more")
    print(f"\nHTML: {DESKTOP_HTML}")
    print(f"Cache: {CACHE_ROOT}")


if __name__ == "__main__":
    main()
