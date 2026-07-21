#!/usr/bin/env python3
"""Section classifier gallery with per-section Gemini extraction.

Four columns per document:
  1. Full page image
  2. Highlight overlay (kept sections only)
  3. Filtered sections list + crop thumbnails
  4. Merged Gemini extraction JSON (cached per doc)

Requires Vertex credentials (GCP_PROJECT_ID + GOOGLE_CLOUD_KEYFILE_JSON or
GOOGLE_APPLICATION_CREDENTIALS) or GEMINI_API_KEY / GOOGLE_API_KEY.

Usage:
  python section_content_gemini_gallery.py
  python section_content_gemini_gallery.py --limit 3
  python section_content_gemini_gallery.py --skip-llm   # HTML only, no API calls
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageEnhance

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]
MLLM_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_credit_app_sections import CONTENT_TYPE_COLORS  # noqa: E402
from ocr_line_to_sections import draw_sections_overlay  # noqa: E402
from section_content_gallery_html import (  # noqa: E402
    CROP_ROOT,
    MANIFEST_PATH,
    PICK_COUNT,
    crop_section_thumb,
    file_url,
    pick_diverse_docs,
    section_content_types,
    section_fields,
)
from section_preprocess import section_passes_preprocess  # noqa: E402

CREDIT_APP_YAML = (
    ROOT.parent
    / "techno-configs/techno_configs/envs/qa/document_fields/extractions/llm_configs/credit_application.yml"
)
ENV_LOCAL = MLLM_DIR / ".env.local"
CACHE_ROOT = ROOT / "wa577_gallery" / "section_classifier" / "gemini_extractions"
HIGHLIGHT_ROOT = ROOT / "wa577_gallery" / "section_classifier" / "highlight_overlays"
DESKTOP_HTML = Path.home() / "Desktop" / "section_classifier_gemini_gallery.html"
REPO_HTML = ROOT / "wa577_gallery" / "section_classifier" / "gemini_gallery.html"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
MIN_SECTION_CROP_WIDTH = 1200
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)


def enhance_section_crop(img: Image.Image) -> Image.Image:
    """Upscale small section crops and apply mild contrast/sharpness before Gemini."""
    out = img.convert("RGB")
    w, h = out.size
    if w < MIN_SECTION_CROP_WIDTH:
        ratio = MIN_SECTION_CROP_WIDTH / w
        out = out.resize(
            (MIN_SECTION_CROP_WIDTH, max(1, int(h * ratio))),
            Image.Resampling.LANCZOS,
        )
    out = ImageEnhance.Contrast(out).enhance(1.12)
    out = ImageEnhance.Sharpness(out).enhance(1.15)
    return out


def load_sections(row: dict[str, Any]) -> list[dict[str, Any]]:
    classified = Path(row.get("sections_classified_json") or "")
    sections_json = Path(row.get("sections_json") or "")
    path = classified if classified.exists() else sections_json
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("sections") or []


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
        else:
            desc = str(entry)
        desc = str(desc).strip()
        if desc:
            out[field_key] = desc
    return out


def match_prompt_key(field_path: str, prompt_keys: set[str]) -> str | None:
    if field_path in prompt_keys:
        return field_path
    parts = field_path.split(".")
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in prompt_keys:
            return candidate
    return None


def field_descriptions_for_section(
    fields: list[str],
    field_prompts: dict[str, str],
) -> dict[str, str]:
    prompt_keys = set(field_prompts)
    descriptions: dict[str, str] = {}
    for field in fields:
        key = match_prompt_key(field, prompt_keys)
        if key and key not in descriptions:
            descriptions[key] = field_prompts[key]
    return descriptions


def nested_skeleton_from_paths(paths: list[str]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path in paths:
        node = root
        parts = path.split(".")
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                node.setdefault(part, None)
            else:
                child = node.get(part)
                if not isinstance(child, dict):
                    child = {}
                    node[part] = child
                node = child
    return root


def is_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def deep_merge_preserve(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, new_val in update.items():
        if key not in base:
            base[key] = new_val
            continue
        old_val = base[key]
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            deep_merge_preserve(old_val, new_val)
        elif not is_populated(old_val) and is_populated(new_val):
            base[key] = new_val
    return base


def flatten_paths(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_paths(value, path))
        else:
            out[path] = value
    return out


def unflatten_paths(flat: dict[str, Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path, value in flat.items():
        node = root
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return root


def passes_field_filter(section: dict[str, Any]) -> bool:
    """Kept by preprocessor and has detected field paths for Gemini extraction."""
    return section_passes_preprocess(section) and bool(section_fields(section))


def draw_kept_sections_overlay(
    full_png: Path,
    kept_sections: list[dict[str, Any]],
    out_path: Path,
) -> None:
    with Image.open(full_png) as img:
        overlay = draw_sections_overlay(img, kept_sections, picked_section_idx=None)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(out_path)


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

    project = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "Set GCP_PROJECT_ID (Vertex) or GEMINI_API_KEY / GOOGLE_API_KEY"
        )

    keyfile_json = os.environ.get("GOOGLE_CLOUD_KEYFILE_JSON")
    if keyfile_json and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        creds_path = Path(tempfile.gettempdir()) / "section_classifier_gcp_creds.json"
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


def build_section_prompt(
    fields: list[str],
    field_prompts: dict[str, str],
    ocr_text: str,
    skeleton: dict[str, Any],
) -> str:
    descriptions = field_descriptions_for_section(fields, field_prompts)
    lines = [
        "Extract only the listed fields from the attached credit application section image.",
        "Use the OCR text as a hint; prefer what is visible in the image.",
        "Return a single JSON object matching the schema shape below.",
        "Use null for fields not present in this section.",
        "",
        "Fields to extract:",
    ]
    for key, desc in sorted(descriptions.items()):
        lines.append(f"- {key}: {desc}")
    lines.extend(
        [
            "",
            "Expected JSON shape:",
            json.dumps(skeleton, indent=2),
            "",
            "OCR text for this section:",
            ocr_text or "(empty)",
        ]
    )
    return "\n".join(lines)


def call_gemini_section(
    client: Any,
    model: str,
    crop_img: Image.Image,
    prompt: str,
) -> dict[str, Any]:
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


def crop_section_image(full_png: Path, bounds: dict[str, Any], *, padding: int = 12) -> Image.Image:
    with Image.open(full_png) as img:
        rgb = img.convert("RGB")
        x0 = max(0, int(bounds.get("min_x", 0)) - padding)
        y0 = max(0, int(bounds.get("min_y", 0)) - padding)
        x1 = min(rgb.width, int(bounds.get("max_x", 0)) + padding)
        y1 = min(rgb.height, int(bounds.get("max_y", 0)) + padding)
        return rgb.crop((x0, y0, x1, y1))


def cache_path(short_id: str) -> Path:
    return CACHE_ROOT / f"{short_id}.json"


def load_cache(short_id: str) -> dict[str, Any] | None:
    path = cache_path(short_id)
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_cache(short_id: str, payload: dict[str, Any]) -> None:
    path = cache_path(short_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def extract_doc(
    row: dict[str, Any],
    sections: list[dict[str, Any]],
    field_prompts: dict[str, str],
    *,
    client: Any | None,
    model: str,
    skip_llm: bool,
    force: bool,
) -> dict[str, Any]:
    short_id = row["short_id"]
    if not force:
        cached = load_cache(short_id)
        if cached is not None:
            return cached

    full_png = Path(row["full_png"])
    kept = [s for s in sections if passes_field_filter(s)]
    merged: dict[str, Any] = {}
    section_results: list[dict[str, Any]] = []
    errors: list[str] = []

    for section in kept:
        idx = section.get("index", "?")
        fields = section_fields(section)
        bounds = section.get("bounds") or {}
        ocr_text = section.get("text") or ""
        section_entry: dict[str, Any] = {
            "index": idx,
            "fields": fields,
            "content_types": section_content_types(section),
            "extraction": None,
            "error": None,
        }

        if skip_llm or client is None:
            section_results.append(section_entry)
            continue

        try:
            crop_img = enhance_section_crop(crop_section_image(full_png, bounds))
            skeleton = nested_skeleton_from_paths(fields)
            prompt = build_section_prompt(fields, field_prompts, ocr_text, skeleton)
            extraction = call_gemini_section(client, model, crop_img, prompt)
            section_entry["extraction"] = extraction
            if extraction:
                deep_merge_preserve(merged, extraction)
        except Exception as exc:  # noqa: BLE001
            section_entry["error"] = str(exc)
            errors.append(f"S{idx}: {exc}")

        section_results.append(section_entry)

    payload = {
        "short_id": short_id,
        "doc_id": row.get("doc_id"),
        "partner": row.get("partner"),
        "page": row.get("page"),
        "model": model if not skip_llm else None,
        "kept_section_count": len(kept),
        "merged_extraction": merged,
        "section_results": section_results,
        "errors": errors,
        "skipped_llm": skip_llm,
    }
    if not skip_llm:
        save_cache(short_id, payload)
    return payload


def build_sections_column(
    row: dict[str, Any],
    kept: list[dict[str, Any]],
) -> str:
    short_id = row["short_id"]
    full_png = Path(row["full_png"])
    crop_dir = CROP_ROOT / short_id
    blocks: list[str] = []

    for section in kept:
        idx = section.get("index", "?")
        fields = section_fields(section)
        ctypes = section_content_types(section)
        bounds = section.get("bounds")
        ctype_badges = "".join(
            f'<span class="ctype" style="border-color:{CONTENT_TYPE_COLORS.get(ct, "#999")}">{html.escape(ct)}</span>'
            for ct in ctypes
        ) or '<span class="muted">—</span>'
        fields_txt = html.escape(", ".join(fields)) if fields else '<span class="muted">—</span>'

        thumb_html = '<span class="muted">—</span>'
        thumb_path = crop_dir / f"s{idx}.png"
        if crop_section_thumb(full_png, bounds or {}, thumb_path):
            thumb_html = (
                f'<img class="thumb" src="{file_url(thumb_path)}" alt="S{idx}" title="S{idx}" />'
            )

        blocks.append(
            f"""<div class="section-block">
  <div class="section-head"><span class="mono">S{idx}</span> {ctype_badges}</div>
  <div class="fields mono">{fields_txt}</div>
  <div class="thumb-cell">{thumb_html}</div>
</div>"""
        )

    if not blocks:
        return '<p class="muted">No sections with non-empty fields</p>'
    return "\n".join(blocks)


def build_json_column(extraction_payload: dict[str, Any]) -> str:
    merged = extraction_payload.get("merged_extraction") or {}
    errors = extraction_payload.get("errors") or []
    skipped = extraction_payload.get("skipped_llm")
    parts = [
        f'<pre class="json-block">{html.escape(json.dumps(merged, indent=2))}</pre>',
    ]
    if skipped:
        parts.append('<p class="muted">LLM skipped — run without --skip-llm to populate.</p>')
    if errors:
        parts.append(
            "<p class=\"error\">"
            + html.escape("; ".join(errors[:5]))
            + (" …" if len(errors) > 5 else "")
            + "</p>"
        )
    section_rows = []
    for sec in extraction_payload.get("section_results") or []:
        if sec.get("error"):
            section_rows.append(
                f'<li class="mono">S{sec["index"]}: error — {html.escape(sec["error"])}</li>'
            )
        elif sec.get("extraction"):
            section_rows.append(
                f'<li class="mono">S{sec["index"]}: '
                f'{html.escape(json.dumps(sec["extraction"], separators=(",", ":")))}</li>'
            )
    if section_rows:
        parts.append("<details><summary>Per-section extractions</summary><ul>")
        parts.extend(section_rows)
        parts.append("</ul></details>")
    return "\n".join(parts)


def build_card(
    row: dict[str, Any],
    sections: list[dict[str, Any]],
    extraction_payload: dict[str, Any],
) -> str:
    short_id = row["short_id"]
    partner = row["partner"]
    page = row.get("page", "p1")
    full_png = Path(row["full_png"])
    kept = [s for s in sections if passes_field_filter(s)]
    highlight_png = HIGHLIGHT_ROOT / f"{short_id}_{page}_kept.png"
    draw_kept_sections_overlay(full_png, kept, highlight_png)

    return f"""
<article class="card" id="{html.escape(short_id)}">
  <header>
    <div class="title-row">
      <span class="doc-id">{html.escape(short_id)}</span>
      <span class="badge partner">{html.escape(partner)}</span>
      <span class="badge">{html.escape(page)}</span>
      <span class="meta">{len(sections)} sections · {len(kept)} kept (preprocess+fields)</span>
    </div>
  </header>
  <div class="quad">
    <figure>
      <img src="{file_url(full_png)}" alt="full {html.escape(short_id)}" loading="lazy" />
      <figcaption>1. Full page</figcaption>
    </figure>
    <figure>
      <img src="{file_url(highlight_png)}" alt="highlight {html.escape(short_id)}" loading="lazy" />
      <figcaption>2. Kept sections highlighted</figcaption>
    </figure>
    <div class="col-sections">
      <div class="col-label">3. Filtered sections</div>
      {build_sections_column(row, kept)}
    </div>
    <div class="col-json">
      <div class="col-label">4. Merged Gemini JSON</div>
      {build_json_column(extraction_payload)}
    </div>
  </div>
</article>"""


def build_html(
    picked: list[dict[str, Any]],
    extractions: dict[str, dict[str, Any]],
    *,
    model: str,
    skip_llm: bool,
    out_path: Path,
) -> None:
    cards = []
    for row in picked:
        sections = load_sections(row)
        payload = extractions.get(row["short_id"]) or {
            "merged_extraction": {},
            "section_results": [],
            "errors": [],
            "skipped_llm": skip_llm,
        }
        cards.append(build_card(row, sections, payload))

    doc_list = ", ".join(f"{r['partner']}/{r['short_id']} {r.get('page','p1')}" for r in picked)
    mode = "skip-llm" if skip_llm else f"model={model}"
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Section Classifier + Gemini Gallery</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 24px;
      background: #f0f2f8;
      color: #1a1a2e;
      line-height: 1.45;
    }}
    h1 {{ margin-bottom: 4px; }}
    .sub {{ color: #555; margin-bottom: 16px; max-width: 1200px; }}
    .doc-list {{ font-family: ui-monospace, monospace; font-size: 12px; color: #444; margin-bottom: 20px; }}
    .grid {{ display: flex; flex-direction: column; gap: 32px; }}
    .card {{
      background: #fff; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,.08);
      padding: 18px;
    }}
    .title-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 10px; }}
    .doc-id {{ font-family: ui-monospace, monospace; font-weight: 700; font-size: 16px; }}
    .badge {{ font-size: 11px; padding: 2px 8px; border-radius: 999px; background: #e8ecf4; color: #334; }}
    .badge.partner {{ background: #dbeafe; color: #1e3a8a; }}
    .meta {{ color: #666; font-size: 13px; }}
    .quad {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr 1fr;
      gap: 14px;
      align-items: start;
    }}
    figure {{ margin: 0; }}
    img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
    figcaption, .col-label {{
      text-align: center; font-size: 12px; color: #777; margin: 4px 0 8px; font-weight: 600;
    }}
    .col-sections, .col-json {{
      border: 1px solid #e2e6ef; border-radius: 6px; padding: 8px;
      background: #fafbfd; max-height: 720px; overflow: auto;
    }}
    .section-block {{
      border-bottom: 1px solid #e8ebf2; padding: 8px 0;
    }}
    .section-block:last-child {{ border-bottom: none; }}
    .section-head {{ margin-bottom: 4px; }}
    .fields {{ font-size: 11px; word-break: break-word; margin-bottom: 6px; }}
    .ctype {{
      display: inline-block; font-size: 10px; padding: 1px 5px; margin: 0 2px;
      border-left: 3px solid #999; background: #fff; border-radius: 2px;
    }}
    .thumb {{ max-width: 100%; max-height: 90px; border: 1px solid #ccc; border-radius: 3px; }}
    .mono {{ font-family: ui-monospace, monospace; }}
    .muted {{ color: #999; }}
    .error {{ color: #b42318; font-size: 12px; }}
    .json-block {{
      font-size: 11px; white-space: pre-wrap; word-break: break-word;
      margin: 0; background: #fff; border: 1px solid #e2e6ef; border-radius: 4px; padding: 8px;
    }}
    details {{ margin-top: 8px; font-size: 11px; }}
    @media (max-width: 1400px) {{ .quad {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 800px) {{ .quad {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Section Classifier + Gemini Extraction</h1>
  <p class="sub">
    Credit application pages — full page, kept-section highlights, filtered sections
    (preprocessor-kept with non-empty <code>content_classification.fields</code>), and
    per-section Gemini extraction
    merged into one JSON. Mode: <code>{html.escape(mode)}</code>.
    Cache: <code>wa577_gallery/section_classifier/gemini_extractions/</code>.
  </p>
  <p class="doc-list">Selected ({len(picked)}): {html.escape(doc_list)}</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=PICK_COUNT, help="Number of docs (default 10)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument(
        "--skip-llm",
        "--dry-run",
        action="store_true",
        dest="skip_llm",
        help="Build HTML only; do not call Gemini (use cached extractions if present)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run Gemini even if cache exists")
    parser.add_argument("--output", type=Path, default=DESKTOP_HTML)
    args = parser.parse_args()

    load_dotenv(ENV_LOCAL)

    if not MANIFEST_PATH.exists():
        print(f"Missing manifest: {MANIFEST_PATH}")
        sys.exit(1)
    if not CREDIT_APP_YAML.exists():
        print(f"Missing credit_application.yml: {CREDIT_APP_YAML}")
        sys.exit(1)

    field_prompts = parse_field_prompts(CREDIT_APP_YAML)
    manifest_data = json.loads(MANIFEST_PATH.read_text())
    manifest: list[dict[str, Any]] = manifest_data.get("results") or []
    picked = pick_diverse_docs(manifest, args.limit)

    client = None
    if not args.skip_llm:
        try:
            client = setup_gemini_client(model=args.model)
        except Exception as exc:  # noqa: BLE001
            print(f"Gemini setup failed: {exc}")
            print("Re-run with --skip-llm to build HTML without API calls.")
            sys.exit(1)

    extractions: dict[str, dict[str, Any]] = {}
    for row in picked:
        short_id = row["short_id"]
        sections = load_sections(row)
        print(f"Processing {row['partner']}/{short_id} {row.get('page', 'p1')} …", flush=True)
        payload = extract_doc(
            row,
            sections,
            field_prompts,
            client=client,
            model=args.model,
            skip_llm=args.skip_llm,
            force=args.force,
        )
        extractions[short_id] = payload
        kept = sum(1 for s in sections if passes_field_filter(s))
        merged_keys = len(flatten_paths(payload.get("merged_extraction") or {}))
        err_count = len(payload.get("errors") or [])
        print(
            f"  kept={kept} merged_fields={merged_keys} errors={err_count}",
            flush=True,
        )

    build_html(
        picked,
        extractions,
        model=args.model,
        skip_llm=args.skip_llm,
        out_path=args.output,
    )
    REPO_HTML.parent.mkdir(parents=True, exist_ok=True)
    REPO_HTML.write_text(args.output.read_text())

    print(f"\nHTML: {args.output}")
    print(f"Repo: {REPO_HTML}")
    print(f"Cache: {CACHE_ROOT}")
    if args.skip_llm:
        print("\nEnv for Gemini calls:")
        print("  Vertex: GCP_PROJECT_ID + GOOGLE_CLOUD_KEYFILE_JSON (or GOOGLE_APPLICATION_CREDENTIALS)")
        print("  API key: GEMINI_API_KEY or GOOGLE_API_KEY")
        print(f"  Model: {args.model}")


if __name__ == "__main__":
    main()
