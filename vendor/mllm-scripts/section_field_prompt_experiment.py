#!/usr/bin/env python3
"""Prompt-based field detection experiment for credit-app section patches.

Compares heuristic, FastText, and Gemini prompt field detection on representative
section crops. Caches Gemini responses per (doc, section).

Usage:
  python section_field_prompt_experiment.py
  python section_field_prompt_experiment.py --skip-llm
  python section_field_prompt_experiment.py --force
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]
MLLM_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_credit_app_sections import OUT_ROOT  # noqa: E402
from section_content_gemini_gallery import (  # noqa: E402
    CREDIT_APP_YAML,
    DEFAULT_MODEL,
    ENV_LOCAL,
    crop_section_image,
    enhance_section_crop,
    load_dotenv,
    parse_json_response,
    setup_gemini_client,
)
from section_content_gallery_html import MANIFEST_PATH, load_sections  # noqa: E402
from section_content_taxonomy import load_document_field_paths  # noqa: E402
from section_field_classifier import (  # noqa: E402
    DEFAULT_FIELD_MODEL_PATH,
    DEFAULT_FIELD_THRESHOLD,
    FastTextFieldClassifier,
    _classify_section_fields_heuristic,
    _filter_fields_by_context,
)
from section_preprocess import annotate_preprocess, section_passes_preprocess  # noqa: E402

OUT_DIR = ROOT / "wa577_gallery" / "section_classifier"
CACHE_ROOT = OUT_DIR / "prompt_field_experiment_cache"
RESULTS_MD = OUT_DIR / "prompt_field_experiment.md"
RESULTS_HTML = OUT_DIR / "prompt_field_experiment.html"

DOC_TYPE = "credit_application"

# Representative patch types for side-by-side comparison.
REPRESENTATIVE_PATCHES: list[dict[str, Any]] = [
    {
        "label": "legal_consent",
        "short_id": "6f99d76d",
        "section_index": 5,
        "note": "AGREEMENT boilerplate — expect no extractable fields",
    },
    {
        "label": "co_applicant",
        "short_id": "6f99d76d",
        "section_index": 0,
        "note": "Co-applicant identity/address block",
    },
    {
        "label": "employment",
        "short_id": "6f99d76d",
        "section_index": 2,
        "note": "Applicant1 employment/income block",
    },
]

FULL_DOC_SHORT_ID = "6f99d76d"


def load_manifest_row(short_id: str) -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    for row in manifest.get("results") or []:
        if row.get("short_id") == short_id:
            return row
    raise KeyError(f"short_id {short_id!r} not in manifest")


def parse_credit_app_field_prompts(yml_path: Path) -> dict[str, str]:
    """Load field_prompts from section_aware or custom credit_application YAML."""
    data = yaml.safe_load(yml_path.read_text()) or {}
    payload = (data.get("model_info") or {}).get("payload_config") or {}
    field_prompts = payload.get("field_prompts") or {}
    if not field_prompts:
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


def field_catalog(field_prompts: dict[str, str]) -> dict[str, str]:
    """All credit_application field paths with descriptions."""
    paths = load_document_field_paths(DOC_TYPE)
    catalog: dict[str, str] = {}
    for path in paths:
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
        if path.startswith("applicants.applicant1"):
            groups["applicant1"].append((path, desc))
        elif path.startswith("applicants.applicant2"):
            groups["applicant2"].append((path, desc))
        elif path.startswith("signatures."):
            groups["signatures"].append((path, desc))
        elif path.startswith("vehicle.") or path.startswith("trade_in."):
            groups["vehicle"].append((path, desc))
        else:
            groups["document"].append((path, desc))
    return dict(groups)


def build_detection_prompt(
    catalog: dict[str, str],
    ocr_text: str,
) -> str:
    groups = group_field_catalog(catalog)
    lines = [
        "You are analyzing a cropped region from a credit application page image.",
        "Below is the catalog of possible extractable field paths for credit_application.",
        "Return ONLY field paths that have extractable data IN THIS PATCH — filled form values",
        "or clearly completed fields visible in the image. Do not list blank label rows alone.",
        "Do not list fields mentioned only in legal/disclosure boilerplate without a dedicated form field.",
        "If none apply, return {\"fields\": []}.",
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
            "Return JSON only: {\"fields\": [\"dotted.path\", ...]}",
            "",
            "OCR text for this patch:",
            ocr_text or "(empty)",
        ]
    )
    return "\n".join(lines)


def parse_fields_response(parsed: dict[str, Any]) -> list[str]:
    if not parsed:
        return []
    if "fields" in parsed and isinstance(parsed["fields"], list):
        return sorted(str(f) for f in parsed["fields"] if f)
    # Flat object with boolean values?
    fields = [k for k, v in parsed.items() if v and k != "fields"]
    return sorted(fields)


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


def classify_fasttext_only(
    ft: FastTextFieldClassifier | None,
    text: str,
) -> list[str]:
    if ft is None:
        return []
    raw_fields, _scores, is_empty = ft.predict(text, threshold=DEFAULT_FIELD_THRESHOLD)
    if is_empty and not raw_fields:
        return []
    filtered = _filter_fields_by_context(raw_fields, text, DOC_TYPE)
    return filtered if filtered else raw_fields


def process_section(
    row: dict[str, Any],
    section: dict[str, Any],
    catalog: dict[str, str],
    *,
    client: Any | None,
    model: str,
    ft: FastTextFieldClassifier | None,
    skip_llm: bool,
    force: bool,
) -> dict[str, Any]:
    short_id = row["short_id"]
    idx = int(section.get("index", 0))
    section = annotate_preprocess(section, document_type=DOC_TYPE)
    text = section.get("text") or ""
    bounds = section.get("bounds") or {}
    full_png = Path(row["full_png"])
    preprocess = section.get("preprocess") or {}

    if not section_passes_preprocess(section):
        return {
            "short_id": short_id,
            "section_index": idx,
            "text_preview": text[:120].replace("\n", " "),
            "preprocess": preprocess,
            "heuristic_fields": [],
            "fasttext_fields": [],
            "prompt_fields": [],
            "prompt": "",
            "raw_response": "",
            "parsed_response": {},
        }

    heuristic = _classify_section_fields_heuristic(text, DOC_TYPE)
    fasttext = classify_fasttext_only(ft, text)

    cached = None if force else load_cached(short_id, idx)
    if cached is not None:
        prompt_fields = cached.get("prompt_fields") or []
        prompt_text = cached.get("raw_response") or ""
        prompt_json = cached.get("parsed_response") or {}
        prompt_str = cached.get("prompt") or ""
    else:
        prompt_str = build_detection_prompt(catalog, text)
        prompt_fields: list[str] = []
        prompt_text = ""
        prompt_json: dict[str, Any] = {}

        if not skip_llm and client is not None and bounds:
            crop_img = enhance_section_crop(crop_section_image(full_png, bounds))
            prompt_fields, prompt_text, prompt_json = call_gemini_detection(
                client, model, crop_img, prompt_str
            )
            save_cached(
                short_id,
                idx,
                {
                    "short_id": short_id,
                    "section_index": idx,
                    "prompt": prompt_str,
                    "raw_response": prompt_text,
                    "parsed_response": prompt_json,
                    "prompt_fields": prompt_fields,
                    "model": model,
                },
            )

    return {
        "short_id": short_id,
        "section_index": idx,
        "text_preview": text[:120].replace("\n", " "),
        "preprocess": preprocess,
        "heuristic_fields": heuristic,
        "fasttext_fields": fasttext,
        "prompt_fields": prompt_fields,
        "prompt": prompt_str,
        "raw_response": prompt_text,
        "parsed_response": prompt_json,
    }


def format_field_list(fields: list[str]) -> str:
    if not fields:
        return "∅"
    return ", ".join(fields)


def agreement_row(heuristic: list[str], fasttext: list[str], prompt: list[str]) -> str:
    h, f, p = set(heuristic), set(fasttext), set(prompt)
    if h == f == p:
        return "all agree"
    parts = []
    if h == p and h != f:
        parts.append("heuristic≈prompt ≠ fasttext")
    elif f == p and f != h:
        parts.append("fasttext≈prompt ≠ heuristic")
    elif h == f and h != p:
        parts.append("heuristic≈fasttext ≠ prompt")
    else:
        parts.append("all differ")
    return "; ".join(parts) if parts else "partial overlap"


def build_markdown(
    representative: list[dict[str, Any]],
    full_doc_sections: list[dict[str, Any]],
    merged_prompt_fields: list[str],
    model: str,
    skip_llm: bool,
) -> str:
    lines = [
        "# Prompt-based field detection experiment",
        "",
        f"Model: `{model}` (Vertex global) · Document: `credit_application`",
        f"Cache: `{CACHE_ROOT}`",
        "",
        "## Approach",
        "",
        "Pass section patch (enhanced crop + OCR text) to Gemini with the full",
        "credit_application field catalog. Model returns only field paths with",
        "extractable data in the patch. Compare against heuristic and FastText-only.",
        "",
        "## Representative patches (3 types)",
        "",
        "| Patch type | Doc | S# | Heuristic | FastText | Prompt | Agreement |",
        "|------------|-----|----|-----------|----------|--------|-----------|",
    ]
    for row in representative:
        meta = next(
            p for p in REPRESENTATIVE_PATCHES if p["section_index"] == row["section_index"]
        )
        lines.append(
            f"| {meta['label']} | {row['short_id']} | s{row['section_index']} "
            f"| {len(row['heuristic_fields'])} | {len(row['fasttext_fields'])} "
            f"| {len(row['prompt_fields'])} | {agreement_row(row['heuristic_fields'], row['fasttext_fields'], row['prompt_fields'])} |"
        )

    lines.extend(["", "### Side-by-side field lists", ""])
    for row in representative:
        meta = next(
            p for p in REPRESENTATIVE_PATCHES if p["section_index"] == row["section_index"]
        )
        lines.extend(
            [
                f"#### {meta['label']} — {row['short_id']} s{row['section_index']}",
                f"_{meta['note']}_",
                "",
                f"- OCR: _{row['text_preview']}_",
                f"- **Heuristic** ({len(row['heuristic_fields'])}): `{format_field_list(row['heuristic_fields'])}`",
                f"- **FastText** ({len(row['fasttext_fields'])}): `{format_field_list(row['fasttext_fields'])}`",
                f"- **Prompt** ({len(row['prompt_fields'])}): `{format_field_list(row['prompt_fields'])}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Full document — all sections (`6f99d76d` p2)",
            "",
            "| S# | Heuristic | FastText | Prompt | H | F | P |",
            "|----|-----------|----------|--------|---|---|---|",
        ]
    )
    for row in full_doc_sections:
        lines.append(
            f"| s{row['section_index']} | {len(row['heuristic_fields'])} "
            f"| {len(row['fasttext_fields'])} | {len(row['prompt_fields'])} "
            f"| {len(row['heuristic_fields'])} | {len(row['fasttext_fields'])} "
            f"| {len(row['prompt_fields'])} |"
        )

    lines.extend(
        [
            "",
            f"**Merged prompt fields** ({len(merged_prompt_fields)}):",
            "",
            "```",
            json.dumps(merged_prompt_fields, indent=2),
            "```",
            "",
            "### Per-section detail (full doc)",
            "",
        ]
    )
    for row in full_doc_sections:
        lines.extend(
            [
                f"**s{row['section_index']}** — _{row['text_preview']}_",
                f"- heuristic: `{format_field_list(row['heuristic_fields'])}`",
                f"- fasttext: `{format_field_list(row['fasttext_fields'])}`",
                f"- prompt: `{format_field_list(row['prompt_fields'])}`",
                "",
            ]
        )

  # Sample prompts/responses for representative patches
    lines.extend(["## Sample prompts & responses", ""])
    for row in representative:
        meta = next(
            p for p in REPRESENTATIVE_PATCHES if p["section_index"] == row["section_index"]
        )
        lines.extend(
            [
                f"### {meta['label']}",
                "",
                "<details><summary>Prompt (truncated)</summary>",
                "",
                "```",
                (row["prompt"][:2000] + ("…" if len(row["prompt"]) > 2000 else "")),
                "```",
                "",
                "</details>",
                "",
                "**Response:**",
                "",
                "```json",
                row["raw_response"] or "(skipped)" if skip_llm else row["raw_response"] or "{}",
                "```",
                "",
            ]
        )

    if skip_llm:
        lines.append("_LLM calls skipped — using cache only._")
    return "\n".join(lines)


def build_html(
    representative: list[dict[str, Any]],
    full_doc_sections: list[dict[str, Any]],
    merged_prompt_fields: list[str],
    model: str,
) -> str:
    rep_rows = []
    for row in representative:
        meta = next(
            p for p in REPRESENTATIVE_PATCHES if p["section_index"] == row["section_index"]
        )
        rep_rows.append(
            f"<tr><td>{html.escape(meta['label'])}</td>"
            f"<td class='mono'>{row['short_id']} s{row['section_index']}</td>"
            f"<td class='mono small'>{html.escape(format_field_list(row['heuristic_fields']))}</td>"
            f"<td class='mono small'>{html.escape(format_field_list(row['fasttext_fields']))}</td>"
            f"<td class='mono small'>{html.escape(format_field_list(row['prompt_fields']))}</td></tr>"
        )

    full_rows = []
    for row in full_doc_sections:
        full_rows.append(
            f"<tr><td>s{row['section_index']}</td>"
            f"<td>{len(row['heuristic_fields'])}</td>"
            f"<td>{len(row['fasttext_fields'])}</td>"
            f"<td>{len(row['prompt_fields'])}</td>"
            f"<td class='mono small'>{html.escape(format_field_list(row['prompt_fields']))}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Prompt field detection experiment</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 24px; background: #f4f6fb; }}
table {{ border-collapse: collapse; background: #fff; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #dde; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #eef2fa; }}
.mono {{ font-family: ui-monospace, monospace; font-size: 11px; }}
.small {{ max-width: 320px; word-break: break-word; }}
pre {{ background: #fff; border: 1px solid #ddd; padding: 10px; overflow: auto; font-size: 11px; }}
</style></head><body>
<h1>Prompt-based field detection</h1>
<p>Model: <code>{html.escape(model)}</code> · Merged fields: {len(merged_prompt_fields)}</p>
<h2>Representative patches</h2>
<table><tr><th>Type</th><th>Section</th><th>Heuristic</th><th>FastText</th><th>Prompt</th></tr>
{''.join(rep_rows)}</table>
<h2>Full doc 6f99d76d p2</h2>
<table><tr><th>S#</th><th>H cnt</th><th>FT cnt</th><th>P cnt</th><th>Prompt fields</th></tr>
{''.join(full_rows)}</table>
<h3>Merged prompt fields</h3>
<pre>{html.escape(json.dumps(merged_prompt_fields, indent=2))}</pre>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore Gemini cache")
    args = parser.parse_args()

    load_dotenv(ENV_LOCAL)

    if not CREDIT_APP_YAML.exists():
        print(f"Missing {CREDIT_APP_YAML}")
        sys.exit(1)

    field_prompts = parse_credit_app_field_prompts(CREDIT_APP_YAML)
    catalog = field_catalog(field_prompts)
    print(f"Field catalog: {len(catalog)} paths", flush=True)

    ft: FastTextFieldClassifier | None = None
    if DEFAULT_FIELD_MODEL_PATH.exists():
        ft = FastTextFieldClassifier(DEFAULT_FIELD_MODEL_PATH)
    else:
        print("Warning: FastText model not found", flush=True)

    client = None
    if not args.skip_llm:
        try:
            client = setup_gemini_client(model=args.model)
        except Exception as exc:  # noqa: BLE001
            print(f"Gemini setup failed: {exc}")
            print("Use --skip-llm to report from cache only.")
            sys.exit(1)

    # Representative patches
    rep_row = load_manifest_row(REPRESENTATIVE_PATCHES[0]["short_id"])
    rep_sections = {int(s.get("index", i)): s for i, s in enumerate(load_sections(rep_row))}
    representative_results: list[dict[str, Any]] = []
    for patch in REPRESENTATIVE_PATCHES:
        section = rep_sections[patch["section_index"]]
        print(f"Rep patch {patch['label']} s{patch['section_index']} …", flush=True)
        representative_results.append(
            process_section(
                rep_row,
                section,
                catalog,
                client=client,
                model=args.model,
                ft=ft,
                skip_llm=args.skip_llm,
                force=args.force,
            )
        )

    # Full document
    full_row = load_manifest_row(FULL_DOC_SHORT_ID)
    all_sections = load_sections(full_row)
    full_doc_results: list[dict[str, Any]] = []
    merged_prompt: set[str] = set()
    for section in all_sections:
        idx = int(section.get("index", 0))
        print(f"Full doc section s{idx} …", flush=True)
        result = process_section(
            full_row,
            section,
            catalog,
            client=client,
            model=args.model,
            ft=ft,
            skip_llm=args.skip_llm,
            force=args.force,
        )
        full_doc_results.append(result)
        merged_prompt.update(result["prompt_fields"])

    merged_list = sorted(merged_prompt)
    md = build_markdown(
        representative_results,
        full_doc_results,
        merged_list,
        args.model,
        args.skip_llm,
    )
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(md)
    RESULTS_HTML.write_text(
        build_html(representative_results, full_doc_results, merged_list, args.model)
    )

    print(f"\nMarkdown: {RESULTS_MD}")
    print(f"HTML: {RESULTS_HTML}")
    print(f"Cache: {CACHE_ROOT}")
    print(f"Merged prompt fields: {len(merged_list)}")


if __name__ == "__main__":
    main()
