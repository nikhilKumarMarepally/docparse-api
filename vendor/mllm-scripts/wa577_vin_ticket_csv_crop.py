#!/usr/bin/env python3
"""Crop-only VIN extraction on Linear ticket CSV docs (triage / Looker attachments).

Uses prod S3 baseline for full_page_vin — never reruns full-page MLLM.
Crop path: pick_vin_section + tight_vin_bounds + enhance_vin_crop + MLLM.
"""

from __future__ import annotations

import argparse
import copy
import csv
import html
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import boto3
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import wa577_vin_full_ne_crop_disagreements as disag
import wa577_vin_linear_partner_closed as lpc
import wa577_vin_ticket_crop_ab as tab
from wa577_vin_crop_helpers import best_vin_section_score, classify_from_gt_audit, gt_in_ocr_text, gt_visible_on_page, infer_printed_vin, norm_vin, pick_vin_section, score_vin_section, summarize_three_class, tight_vin_bounds, to_three_class, vin_matches_printed
from wa577_vin_ticket_discover import get_doc_meta

OUT_DIR = ROOT / "wa577_gallery/vin_ticket_csv_crop"
CSV_DIR = OUT_DIR / "csvs"
SEED_JSON = OUT_DIR / "csv_rows_seed.json"
DESKTOP_HTML = Path.home() / "Desktop/vin_ticket_csv_crop.html"
DESKTOP_ASSETS = Path.home() / "Desktop/vin_ticket_csv_crop_assets"
ASSETS_PREFIX = "vin_ticket_csv_crop_assets"
BUNDLE_DIR = Path.home() / "Desktop/wa577_vin_crop_galleries"
BUNDLE_ZIP = Path.home() / "Desktop/wa577_vin_crop_galleries.zip"
BUNDLE_README = """WA-577 VIN Crop Galleries
=========================

Contents:
  - vin_ticket_csv_crop.html          Ticket CSV crop A/B gallery
  - vin_null_both_wrong_audit.html    Null / both-wrong audit gallery

How to view:
  1. Unzip this folder anywhere on your machine.
  2. Open the HTML file in Google Chrome (double-click or File > Open).
  3. In the summary table, click a View link or thumbnail to scroll to that doc's card.
  4. Keep each *_assets/ folder next to its HTML file — image paths are relative.

Do not move HTML files without their matching assets folder.
"""
TICKET_AB_DIR = ROOT / "wa577_gallery/vin_ticket_crop_ab"
BUCKET = "informed-techno-core-prod-exchange"
QA_ROOT = ROOT.parent / "techno-configs/techno_configs/envs/qa/document_fields"

PRIORITY_TICKETS = frozenset({"WA-489", "WA-411", "WA-640"})
MATCHES_CONTRACT_VIN_TICKETS = PRIORITY_TICKETS

PARTNER_BY_UUID: dict[str, str] = {
    "14983b26-babc-42b6-b004-a86869c5e357": "consumerscu",
    "596dfc48-fcba-49de-8001-70b52aad72de": "autonationfinance",
    "4cc97cd4-3d53-45e7-9ac0-23a554925d54": "suncoast",
}

TICKET_META: list[dict[str, Any]] = [
    {"ticket": "WA-640", "url": "https://linear.app/informediq/issue/WA-640", "partner": "cuofco", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "Multi-page merge picks wrong page VIN"},
    {"ticket": "WA-553", "url": "https://linear.app/informediq/issue/WA-553", "partner": "consumerscu", "document_type": "verification_of_coverage", "question_code": "vin", "notes": "Insurance VIN extraction failures"},
    {"ticket": "WA-609", "url": "https://linear.app/informediq/issue/WA-609", "partner": "consumerscu", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "VIN ticket cluster"},
    {"ticket": "WA-610", "url": "https://linear.app/informediq/issue/WA-610", "partner": "consumerscu", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "VIN ticket cluster"},
    {"ticket": "WA-524", "url": "https://linear.app/informediq/issue/WA-524", "partner": "consumerscu", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "VIN ticket cluster"},
    {"ticket": "WA-538", "url": "https://linear.app/informediq/issue/WA-538", "partner": "consumerscu", "document_type": "odometer_statement", "verification_doc_type": "odometer_statement", "question_code": "vin", "notes": "Odometer VIN verify — extract title_application prompt"},
    {"ticket": "WA-540", "url": "https://linear.app/informediq/issue/WA-540", "partner": "consumerscu", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "VIN ticket cluster"},
    {"ticket": "WA-477", "url": "https://linear.app/informediq/issue/WA-477", "partner": "autonationfinance", "document_type": "gap_binder", "verification_doc_type": "gap_waiver_contract", "question_code": "vin", "notes": "GAP model-name concatenated into VIN"},
    {"ticket": "WA-539", "url": "https://linear.app/informediq/issue/WA-539", "partner": "consumerscu", "document_type": "retail_installment_sales_contract", "question_code": "vin", "notes": "RISC wrong VIN / multi-page selection"},
    {"ticket": "WA-411", "url": "https://linear.app/informediq/issue/WA-411", "partner": "autonationfinance", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "Null-heavy title_application VIN failures"},
    {"ticket": "WA-530", "url": "https://linear.app/informediq/issue/WA-530", "partner": "consumerscu", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "VIN ticket cluster"},
    {"ticket": "WA-489", "url": "https://linear.app/informediq/issue/WA-489", "partner": "consumerscu", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "92% wrong_value — 1–2 char OCR diffs"},
    {"ticket": "WA-494", "url": "https://linear.app/informediq/issue/WA-494", "partner": "consumerscu", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "VIN ticket cluster"},
    {"ticket": "WA-439", "url": "https://linear.app/informediq/issue/WA-439", "partner": "consumerscu", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "VIN ticket cluster"},
    {"ticket": "WA-528", "url": "https://linear.app/informediq/issue/WA-528", "partner": "consumerscu", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "VIN ticket cluster"},
    {"ticket": "WA-577", "url": "https://linear.app/informediq/issue/WA-577", "partner": "penair", "document_type": "title_application", "question_code": "matches_contract_vin", "notes": "Crop investigation parent ticket"},
    {"ticket": "WA-153", "url": "https://linear.app/informediq/issue/WA-153", "partner": "stellantis", "document_type": "buyers_order", "verification_doc_type": "order_form", "question_code": "matches_contract_vin", "notes": "Partner-closed unsolvable (<1% fix rate)"},
    {"ticket": "WA-244", "url": "https://linear.app/informediq/issue/WA-244", "partner": "suncoast", "document_type": "documentary_draft", "question_code": "vin", "notes": "HITL OCR confusion — re-score with hitl_value"},
    {"ticket": "WA-231", "url": "https://linear.app/informediq/issue/WA-231", "partner": "desertfinancial", "document_type": "title_application", "question_code": "vin", "notes": "RouteOne VIN garbling — HITL CSV"},
    {"ticket": "WA-221", "url": "https://linear.app/informediq/issue/WA-221", "partner": "stellantis", "document_type": "buyers_order", "question_code": "vin", "notes": "Trade-in vs primary VIN — section targeting"},
]

INLINE_DOCS: list[dict[str, Any]] = [
    {"ticket": "WA-577", "doc_id": "87c842be-0578-452a-a9ba-99aefa0e0a4d", "document_type": "title_application", "partner": "penair", "ground_truth": "1FMEE8BH0TLA94758", "prod_vin": "1FMEE8BH0TLA97458", "status": "fail", "bucket": "B1", "csv_source": "inline"},
    {"ticket": "WA-577", "doc_id": "4aeb9d2c-0328-4699-acc1-5993630c1258", "document_type": "title_application", "partner": "desertfinancial", "ground_truth": "1ft8w2bt5tee77678", "prod_vin": None, "status": "review", "bucket": "B1", "csv_source": "inline"},
    {"ticket": "WA-577", "doc_id": "899d7cbd-ba36-46f2-88d3-8ef79282365f", "document_type": "title_application", "partner": "cuofco", "ground_truth": "jf2sjaec6hh410265", "prod_vin": None, "status": "fail", "bucket": "B1", "csv_source": "inline"},
    {"ticket": "WA-640", "doc_id": "899d7cbd-ba36-46f2-88d3-8ef79282365f", "document_type": "title_application", "partner": "cuofco", "ground_truth": "jf2sjaec6hh410265", "prod_vin": None, "status": "fail", "bucket": "B1", "csv_source": "inline"},
]
for _spec in lpc.WA153_DOCS:
    INLINE_DOCS.append(
        {
            "ticket": "WA-153",
            "doc_id": _spec["doc_id"],
            "document_type": "buyers_order",
            "partner": "stellantis",
            "ground_truth": norm_vin(_spec["ground_truth"]),
            "prod_vin": norm_vin(_spec["prod_mllm"]) if _spec.get("prod_mllm") else None,
            "status": "fail",
            "bucket": "B1",
            "csv_source": "inline_wa153",
        }
    )

TICKET_FROM_CSV_NAME: dict[str, str] = {
    "wa489_triage.csv": "WA-489",
    "wa489_looker.csv": "WA-489",
    "wa411_triage.csv": "WA-411",
    "wa411_looker.csv": "WA-411",
    "wa538_looker.csv": "WA-538",
    "wa477_triage.csv": "WA-477",
    "wa539_triage.csv": "WA-539",
    "wa553_triage.csv": "WA-553",
    "wa244_ticket.csv": "WA-244",
    "wa231_hitl.csv": "WA-231",
    "wa221_hitl.csv": "WA-221",
    "wa494_triage.csv": "WA-494",
    "wa528_triage.csv": "WA-528",
    "wa524_triage.csv": "WA-524",
    "wa609_triage.csv": "WA-609",
    "wa610_triage.csv": "WA-610",
    "wa610_looker.csv": "WA-610",
    "wa530_triage.csv": "WA-530",
    "wa540_triage.csv": "WA-540",
    "wa439_triage.csv": "WA-439",
}

_s3 = None


def load_config(doc_type: str) -> dict[str, Any]:
    return lpc.load_config(doc_type)


def init_s3() -> None:
    global _s3
    if _s3 is None:
        kwargs: dict[str, Any] = {"region_name": "us-west-2"}
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            kwargs["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
            kwargs["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
            kwargs["aws_session_token"] = os.environ.get("AWS_SESSION_TOKEN")
        else:
            kwargs["profile_name"] = os.environ.get("AWS_PROFILE", "prod")
        _s3 = boto3.Session(**kwargs).client("s3")


def fetch_prod_vin(doc_id: str, partner_id: str, app_id: str, doc_type: str) -> str | None:
    init_s3()
    key = f"{partner_id}/{app_id}/raw_extracted_data_mllm/{doc_id}.json"
    try:
        raw = json.loads(_s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except Exception:
        return None
    if isinstance(raw, list):
        merged: dict[str, Any] | None = None
        for page in raw:
            if not isinstance(page, dict):
                continue
            clean = {k: v for k, v in page.items() if not str(k).startswith("_") and v is not None}
            if merged is None:
                merged = copy.deepcopy(clean)
            else:
                for k, v in clean.items():
                    if k not in merged or merged[k] is None:
                        merged[k] = v
        raw = merged or {}
    vin = tab.extract_vin(raw if isinstance(raw, dict) else {})
    return norm_vin(vin) if vin else None


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        val = row.get(key)
        if val is not None and str(val).strip() not in ("", "None", "null"):
            return val
    return None


def extraction_document_type(row: dict[str, Any]) -> str:
    doc_type = str(row.get("document_type") or row.get("doc_type") or "title_application")
    doc_name = str(row.get("document_name") or "")
    verify = str(row.get("verification_doc_type") or row.get("verification_document_type") or "")

    if doc_name == "title_application" and doc_type != "title_application":
        return "title_application"
    if doc_type in ("gap_waiver", "gap_waiver_contract") or verify in ("gap_waiver", "gap_waiver_contract"):
        return "gap_binder"
    if doc_type == "gap_binder":
        return "gap_binder"
    return doc_type


def bucket_from_classification(classification: str | None, explicit: str | None = None) -> str:
    if explicit and str(explicit).upper().startswith("B"):
        return str(explicit).upper()[:2]
    c = (classification or "").lower()
    if c == "fixable":
        return "B1"
    if "value_differs" in c or "source" in c:
        return "B2"
    if "field_not_present" in c or "not_present" in c:
        return "B3"
    if "bad_ocr" in c:
        return "B3"
    if c == "uncertain":
        return "B3"
    return "B1"


def partner_slug(row: dict[str, Any], ticket: str) -> str:
    pid = _first(row, "partner_id", "partner")
    if pid and pid in PARTNER_BY_UUID:
        return PARTNER_BY_UUID[pid]
    meta = {m["ticket"]: m for m in TICKET_META}.get(ticket, {})
    return str(row.get("partner") or meta.get("partner") or "unknown")


def parse_csv_file(path: Path, ticket: str | None = None) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.lstrip().startswith("{"):
        return []
    ticket = ticket or TICKET_FROM_CSV_NAME.get(path.name, "UNKNOWN")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []
    for raw in reader:
        doc_id = _first(raw, "doc_id", "document_id")
        if not doc_id:
            continue
        doc_type = _first(raw, "document_type", "doc_type") or "title_application"
        doc_name = _first(raw, "document_name")
        ground_truth = _first(raw, "expected", "hitl_value", "ground_truth")
        prod_vin = _first(raw, "answer", "extracted_value", "prod", "prod_vin")
        status = _first(raw, "status") or ""
        classification = _first(raw, "classification", "bucket")
        bucket = bucket_from_classification(
            str(classification) if classification else None,
            str(classification) if classification and str(classification).upper().startswith("B") else None,
        )
        row = {
            "ticket": ticket,
            "doc_id": str(doc_id).strip(),
            "document_type": doc_type,
            "document_name": doc_name,
            "extraction_document_type": extraction_document_type(
                {"document_type": doc_type, "document_name": doc_name}
            ),
            "partner": partner_slug(raw, ticket),
            "ground_truth": norm_vin(ground_truth) if ground_truth else None,
            "prod_vin": norm_vin(prod_vin) if prod_vin else None,
            "status": status,
            "bucket": bucket,
            "classification": classification,
            "csv_source": path.name,
        }
        rows.append(row)
    return rows


def load_csv_rows(*, use_seed: bool = False) -> list[dict[str, Any]]:
    if use_seed and SEED_JSON.exists():
        return json.loads(SEED_JSON.read_text())
    rows: list[dict[str, Any]] = []
    has_csv = CSV_DIR.exists() and any(CSV_DIR.glob("*.csv"))
    if has_csv:
        for path in sorted(CSV_DIR.glob("*.csv")):
            rows.extend(parse_csv_file(path))
    if rows:
        return rows
    if SEED_JSON.exists():
        return json.loads(SEED_JSON.read_text())
    return []


def row_priority(row: dict[str, Any]) -> tuple[int, int, int]:
    ticket = row.get("ticket", "")
    tier = 0
    if ticket in PRIORITY_TICKETS:
        tier = 3
    elif row.get("document_type") == "title_application":
        tier = 2
    elif row.get("extraction_document_type") == "title_application":
        tier = 2
    bucket = str(row.get("bucket") or "")
    b_score = 2 if bucket == "B1" else (1 if bucket == "B2" else 0)
    has_gt = 1 if row.get("ground_truth") else 0
    return (tier, b_score, has_gt)


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        doc_id = row["doc_id"]
        prev = best.get(doc_id)
        if prev is None or row_priority(row) > row_priority(prev):
            best[doc_id] = row
    return list(best.values())


def merge_all_rows(csv_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined = list(csv_rows) + list(INLINE_DOCS)
    for row in combined:
        if "extraction_document_type" not in row:
            row["extraction_document_type"] = extraction_document_type(row)
    return dedupe_rows(combined)


def select_docs(rows: list[dict[str, Any]], max_docs: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=row_priority, reverse=True)
    return ranked[:max_docs]


def build_ticket_docs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    meta_by_ticket = {m["ticket"]: m for m in TICKET_META}
    docs: list[dict[str, Any]] = []
    for row in rows:
        doc_id = row["doc_id"]
        short = doc_id.split("-")[0]
        ticket = row["ticket"]
        meta = meta_by_ticket.get(ticket, {})
        ext_type = row.get("extraction_document_type") or row.get("document_type") or "title_application"
        docs.append(
            {
                "ticket": ticket,
                "ticket_url": meta.get("url"),
                "partner": row.get("partner") or meta.get("partner"),
                "partner_id": None,
                "short": short,
                "doc_id": doc_id,
                "document_type": ext_type,
                "verification_doc_type": meta.get("verification_doc_type"),
                "ground_truth": row.get("ground_truth"),
                "prod_mllm_hint": row.get("prod_vin"),
                "status": row.get("status"),
                "bucket": row.get("bucket"),
                "csv_source": row.get("csv_source"),
                "classification": row.get("classification"),
            }
        )
    return docs


def resolve_full_vin(
    ticket: dict[str, Any],
    meta: dict[str, Any],
    doc_dir: Path,
    short: str,
    page: str,
) -> tuple[str | None, str]:
    cached = disag.load_cached_full_page_vin(doc_dir, short, page, ticket)
    if cached is not None:
        vin, src = cached
        return (norm_vin(vin) if vin else None, src)

    prod = fetch_prod_vin(ticket["doc_id"], meta["partner_id"], meta["app_id"], ticket["document_type"])
    if prod:
        disag.backfill_full_extraction_stub(doc_dir, short, page, prod)
        return prod, "s3_prod_baseline"

    hint = ticket.get("prod_mllm_hint")
    if hint:
        hv = norm_vin(hint)
        disag.backfill_full_extraction_stub(doc_dir, short, page, hv)
        return hv, "csv_prod_hint"

    return None, "missing"


def classify_result(row: dict[str, Any]) -> str:
    """Raw extraction-vs-GT label (used at run time).

  3-class scoring (crop / null / both_wrong) is via ``to_three_class``:
  - crop: crop matches GT *or* ``extraction_correct_vs_doc`` (printed doc)
  - null: crop returned null
  - both_wrong: extraction wrong vs what is printed on the document
    """
    if row.get("error"):
        return "error"
    full_ok = row.get("full_matches_truth")
    crop_ok = row.get("crop_matches_truth")
    crop_vin = row.get("crop_vin_enhanced")
    if crop_vin is None:
        return "crop_null"
    if row.get("extraction_correct_vs_doc"):
        return "crop_wins" if not crop_ok else "both_ok"
    if crop_ok and not full_ok:
        return "crop_fixes_full"
    if crop_ok:
        return "both_ok" if full_ok else "crop_wins"
    if full_ok:
        return "both_ok"
    return "both_wrong"


USER_CORRECTIONS: dict[str, dict[str, str]] = {
    "31f54996": {
        "notes": "GT wrong; full==crop match printed VIN",
    },
}

# Visual-audit overrides for null crop rows where GT is app-context only (B3).
VIN_ABSENT_CORRECTIONS: dict[str, dict[str, Any]] = {}

_TITLE_APP_MARKERS = re.compile(
    r"APPLICATION\s+FOR\s+VEHICLE\s+TITLE|MVD-\d+|TITLE\s+APPLICATION|HSMV\s+\d+",
    re.I,
)

_B2_SOURCE_NOTE = (
    "GT wrong; full==crop match printed VIN"
)

# Docs that entered results as classification=both_wrong (ticket CSV batch).
BOTH_WRONG_AUDIT_IDS = frozenset({
    "04ecf650", "073d17f5", "08a39be0", "1d35eae6", "31f54996", "484a17ee",
    "5a92718b", "62f8eeb2", "6e115533", "7340b78f", "9c6d96e3", "b5cff309",
    "b9bff01f", "cc727d66", "ce485f6c", "d5bc85ba", "e19851c3",
})

_VIN17 = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
_BOTH_WRONG_AUDIT: dict[str, dict[str, Any]] | None = None


def _load_both_wrong_audit() -> dict[str, dict[str, Any]]:
    global _BOTH_WRONG_AUDIT
    if _BOTH_WRONG_AUDIT is None:
        audit_path = OUT_DIR / "both_wrong_audit.json"
        if audit_path.exists():
            rows = json.loads(audit_path.read_text())
            _BOTH_WRONG_AUDIT = {r["short_id"]: r for r in rows}
        else:
            _BOTH_WRONG_AUDIT = {}
    return _BOTH_WRONG_AUDIT


def is_b2_completely_different_vin(row: dict[str, Any]) -> bool:
    """True when full==crop, valid 17-char VINs, GT is a wholly different vehicle (B2)."""
    if row.get("error"):
        return False
    full = norm_vin(row.get("full_page_vin") or "")
    crop = norm_vin(row.get("crop_vin_enhanced") or "")
    if not full or full != crop:
        return False
    if not _VIN17.match(full) or not _VIN17.match(crop):
        return False
    gt = norm_vin(row.get("ground_truth") or "")
    if not gt or gt == crop:
        return False
    audit = _load_both_wrong_audit().get(row.get("short_id") or (row.get("doc_id") or "")[:8], {})
    return (
        audit.get("pattern") == "completely_different_vin"
        and audit.get("root_cause") == "b2_source_disagree"
    )


def _find_sections_json_for_row(row: dict[str, Any]) -> Path | None:
    short = row.get("short_id") or (row.get("doc_id") or "")[:8]
    page = row.get("page", "p0")
    image_path = Path(row.get("image_path") or OUT_DIR / short / f"{short}_{page}.png")
    if not image_path.exists():
        return None
    stem = image_path.stem
    parent = image_path.parent
    candidates = [
        parent / f"{stem}_sections" / f"{stem}_sections.json",
        parent / f"{stem}_sections.json",
        OUT_DIR / short / f"{short}_{page}_sections" / f"{short}_{page}_sections.json",
        TICKET_AB_DIR / short / f"{short}_{page}_sections" / f"{short}_{page}_sections.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_ocr_blob(sections_json: Path | None) -> tuple[str, list[dict[str, Any]] | None, str]:
    if not sections_json:
        return "", None, ""
    sections_data = json.loads(sections_json.read_text())
    sections = sections_data.get("sections") or []
    section_text = "\n".join(s.get("text", "") for s in sections)
    ocr_path = sections_json.parent / sections_json.name.replace("_sections.json", "_ocr_sections.txt")
    full_ocr = ocr_path.read_text(encoding="utf-8", errors="replace") if ocr_path.exists() else ""
    return section_text + "\n" + full_ocr, sections, full_ocr


def audit_row_gt_ocr(row: dict[str, Any]) -> dict[str, Any]:
    """GT/OCR audit for a both_wrong row; does not mutate row."""
    short = row.get("short_id") or (row.get("doc_id") or "")[:8]
    sections_json = _find_sections_json_for_row(row)
    ocr_blob, sections, _ = _load_ocr_blob(sections_json)
    gt = row.get("ground_truth")
    full = row.get("full_page_vin")
    crop = row.get("crop_vin_enhanced")
    picked = row.get("picked_section")

    gt_in_ocr = gt_in_ocr_text(gt, ocr_blob)
    printed_vin = infer_printed_vin(full, crop, sections, picked, ocr_blob)
    gt_visible = gt_visible_on_page(gt, ocr_blob)
    if not gt_visible and gt and printed_vin:
        gt_visible = vin_matches_printed(gt, printed_vin, max_edits=0)

    manual = USER_CORRECTIONS.get(short, {})
    if manual:
        gt_visible = False

    new_class, bucket, extraction_ok, notes = classify_from_gt_audit(
        row,
        gt_in_ocr=gt_in_ocr,
        gt_visible_on_page=gt_visible,
        printed_vin=printed_vin,
    )
    if manual:
        new_class, bucket, extraction_ok = "crop", "B2", True
        notes = manual.get("notes") or notes

    return {
        "doc": short,
        "full": full,
        "crop": crop,
        "GT": gt,
        "gt_in_ocr": gt_in_ocr,
        "gt_visible": gt_visible,
        "printed_vin": printed_vin,
        "new_class": new_class,
        "bucket": bucket,
        "extraction_correct_vs_doc": extraction_ok,
        "notes": notes,
    }


def apply_gt_ocr_reclassify(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reclassify both_wrong rows using GT/OCR audit; returns (results, audit_rows)."""
    audits: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    for row in results:
        r = dict(row)
        short = r.get("short_id") or (r.get("doc_id") or "")[:8]
        crop_n = norm_vin(r.get("crop_vin_enhanced") or "")
        full_null_crop_valid = (
            r.get("full_page_vin") is None
            and crop_n
            and _VIN17.match(crop_n)
            and (r.get("class") == "both_wrong" or r.get("classification") == "both_wrong")
        )
        if r.get("extraction_correct_vs_doc") and r.get("class") == "crop":
            out.append(r)
            continue
        is_both_wrong = (
            r.get("classification") == "both_wrong"
            or short in BOTH_WRONG_AUDIT_IDS
            or full_null_crop_valid
        )
        if not is_both_wrong or r.get("error"):
            out.append(r)
            continue
        audit = audit_row_gt_ocr(r)
        audits.append(audit)
        r["gt_in_ocr"] = audit["gt_in_ocr"]
        r["gt_visible_on_page"] = audit["gt_visible"]
        r["printed_vin"] = audit["printed_vin"]
        r["class"] = audit["new_class"]
        r["bucket"] = audit["bucket"]
        r["result_bucket"] = audit["bucket"]
        r["extraction_correct_vs_doc"] = audit["extraction_correct_vs_doc"]
        if audit.get("notes"):
            r["notes"] = audit["notes"]
        if audit["new_class"] == "crop":
            r["classification"] = "crop_wins" if not r.get("crop_matches_truth") else "both_ok"
        elif audit["new_class"] == "both_wrong":
            r["classification"] = "both_wrong"
        out.append(r)
    return out, audits


def _gt_on_title_app_pages(short: str, gt: str | None, *, strict: bool = True) -> bool:
    """True when GT appears on any page that looks like a title application form."""
    if not gt:
        return False
    doc_dir = OUT_DIR / short
    if not doc_dir.is_dir():
        return False
    checker = gt_in_ocr_text if strict else gt_visible_on_page
    for sections_json in sorted(doc_dir.glob(f"{short}_p*_sections/{short}_p*_sections.json")):
        page = sections_json.parent.name.split("_", 1)[1]  # p0, p1, …
        ocr_blob, _ = _page_ocr_blob(short, page, doc_dir)
        if not _TITLE_APP_MARKERS.search(ocr_blob):
            continue
        if checker(gt, ocr_blob):
            return True
    return False


def apply_null_vin_absent_reclassify(
    results: list[dict[str, Any]],
    *,
    manual_audit: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Reclassify null crop rows to B3 when VIN is absent from the document."""
    manual_audit = manual_audit or {}
    out: list[dict[str, Any]] = []
    for row in results:
        r = dict(row)
        if to_three_class(r) != "null":
            out.append(r)
            continue

        short = r.get("short_id") or (r.get("doc_id") or "")[:8]
        corr = VIN_ABSENT_CORRECTIONS.get(short)
        if corr:
            r["bucket"] = corr.get("bucket", "B3")
            r["result_bucket"] = r["bucket"]
            if corr.get("notes"):
                r["notes"] = corr["notes"]
            if "gt_in_ocr" in corr:
                r["gt_in_ocr"] = corr["gt_in_ocr"]
            if "gt_visible_on_page" in corr:
                r["gt_visible_on_page"] = corr["gt_visible_on_page"]
            out.append(r)
            continue

        manual = manual_audit.get(short, {})
        root = manual.get("root_cause", "")
        if root in ("vin_absent", "vin_absent_on_page") and not manual.get("fixable", True):
            r["bucket"] = "B3"
            r["result_bucket"] = "B3"
            r["gt_visible_on_page"] = False
            if manual.get("notes"):
                r["notes"] = manual["notes"]
            out.append(r)
            continue

        bucket = r.get("result_bucket") or r.get("bucket")
        gt = r.get("ground_truth")
        if bucket == "B1" and gt and not _gt_on_title_app_pages(short, gt, strict=True):
            r["bucket"] = "B3"
            r["result_bucket"] = "B3"
            r["gt_visible_on_page"] = False
            if not r.get("notes"):
                r["notes"] = "no VIN on title application page(s); GT app-context only"
        out.append(r)
    return out


def apply_user_corrections(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply explicit user overrides and legacy both_wrong_audit B2 flags."""
    out: list[dict[str, Any]] = []
    audits = _load_both_wrong_audit()
    for row in results:
        r = dict(row)
        short = r.get("short_id") or (r.get("doc_id") or "")[:8]
        manual = USER_CORRECTIONS.get(short)
        audit = audits.get(short, {})
        if manual or is_b2_completely_different_vin(r):
            r["extraction_correct_vs_doc"] = True
            r["class"] = "crop"
            r["bucket"] = "B2"
            r["result_bucket"] = "B2"
            r["notes"] = (manual or {}).get("notes") or audit.get("notes") or _B2_SOURCE_NOTE
        cls = to_three_class(r)
        if cls:
            r["class"] = cls
        out.append(r)
    return out


def bucket_for_result(row: dict[str, Any]) -> str:
    explicit = row.get("bucket")
    if explicit:
        return str(explicit)
    classification = str(row.get("classification") or "")
    return bucket_from_classification(classification)


def _page_ocr_blob(short: str, page: str, doc_dir: Path) -> tuple[str, dict[str, Any] | None]:
    sections_json = doc_dir / f"{short}_{page}_sections" / f"{short}_{page}_sections.json"
    if not sections_json.exists():
        return "", None
    sections_data = json.loads(sections_json.read_text())
    sections = sections_data.get("sections") or []
    section_text = "\n".join(s.get("text", "") for s in sections)
    ocr_path = sections_json.parent / sections_json.name.replace("_sections.json", "_ocr_sections.txt")
    full_ocr = ocr_path.read_text(encoding="utf-8", errors="replace") if ocr_path.exists() else ""
    return section_text + "\n" + full_ocr, sections_data


def _find_gt_section(sections: list[dict[str, Any]], target: str) -> int | None:
    norm = norm_vin(target)
    if not norm:
        return None
    best_idx: int | None = None
    best_score = 0
    for s in sections:
        text = s.get("text", "")
        compact = re.sub(r"[^A-Z0-9]", "", text.upper())
        score = 0
        if norm in compact:
            score += 30
        else:
            for n in range(17, 10, -1):
                for i in range(len(norm) - n + 1):
                    if norm[i : i + n] in compact:
                        score += n
                        break
        if score > best_score:
            best_score = score
            best_idx = s["index"]
    return best_idx if best_score >= 12 else None


def _doc_assigned_page_label(meta: dict[str, Any]) -> str:
    start = int(meta.get("start") or 1)
    return f"p{start - 1}"


def scan_file_pages_for_gt(
    *,
    short: str,
    truth: str | None,
    full_hint: str | None,
    meta: dict[str, Any],
    page_payloads: list[tuple[str, Path, dict[str, Any]]],
    doc_dir: Path,
    profile: str,
) -> list[dict[str, Any]]:
    """OCR-scan every file page; return GT hits sorted best-first for crop rerun."""
    target = norm_vin(truth) or norm_vin(full_hint)
    if not target:
        return []
    assigned = _doc_assigned_page_label(meta)
    hits: list[dict[str, Any]] = []
    for page, payload_path, data in page_payloads:
        image_path = doc_dir / f"{short}_{page}.png"
        try:
            tab.download_image(data["image_uri"], image_path, profile)
            sections_json = tab.ensure_sections(short, page, payload_path, image_path, profile)
        except Exception as exc:
            print(f"  {short} {page}: skip page ({exc})", flush=True)
            continue
        ocr_blob, sections_data = _page_ocr_blob(short, page, doc_dir)
        if not sections_data:
            continue
        visible = gt_visible_on_page(target, ocr_blob)
        strict = gt_in_ocr_text(target, ocr_blob)
        if not visible and not strict:
            continue
        vin_section = pick_vin_section(sections_data)
        gt_sec = _find_gt_section(sections_data.get("sections") or [], target)
        if gt_sec is not None:
            vin_section = next(
                (s for s in sections_data["sections"] if s["index"] == gt_sec),
                vin_section,
            )
        sec_score = score_vin_section(vin_section, sections_data["sections"]) if vin_section else 0.0
        page_num = int(page[1:]) + 1
        rank = sec_score + (40 if strict else 20) + (10 if page != assigned else 0)
        hits.append(
            {
                "page": page,
                "file_page": page_num,
                "doc_assigned_page": int(meta.get("start") or page_num),
                "gt_strict": strict,
                "gt_visible": visible,
                "vin_section": vin_section,
                "sections_data": sections_data,
                "payload_path": payload_path,
                "data": data,
                "rank": rank,
                "wrong_page": page != assigned,
            }
        )
    hits.sort(key=lambda h: h["rank"], reverse=True)
    return hits


def _run_crop_row(
    base: dict[str, Any],
    *,
    ticket: dict[str, Any],
    meta: dict[str, Any],
    doc_dir: Path,
    short: str,
    truth: str | None,
    page: str,
    payload_path: Path,
    sections_data: dict[str, Any],
    vin_section: dict[str, Any],
    sec_score: float,
    env: dict[str, str],
) -> dict[str, Any]:
    image_path = doc_dir / f"{short}_{page}.png"
    full_vin, full_src = resolve_full_vin(ticket, meta, doc_dir, short, page)
    section_bounds = vin_section["bounds"]
    section_crop_img = tab.crop_bounds(image_path, section_bounds)
    section_crop_path = doc_dir / f"{short}_{page}_section_crop.png"
    section_crop_img.save(section_crop_path)
    crop_vin, enhanced_path, _ = disag._run_enhanced_arm(
        payload_path.resolve(), env, section_crop_img, doc_dir, short, page
    )
    crop_vin_norm = norm_vin(crop_vin) if crop_vin else None
    truth_norm = norm_vin(truth) if truth else None
    full_ok = norm_vin(full_vin) == truth_norm if full_vin and truth_norm else False
    crop_ok = crop_vin_norm == truth_norm if crop_vin_norm and truth_norm else False
    row = dict(base)
    row.update(
        {
            "page": page,
            "file_page": int(page[1:]) + 1,
            "doc_assigned_page": int(meta.get("start") or 0),
            "full_page_vin": full_vin,
            "full_page_source": full_src,
            "crop_vin_enhanced": crop_vin,
            "full_matches_truth": full_ok,
            "crop_matches_truth": crop_ok,
            "crop_fixes": crop_ok and not full_ok,
            "enhanced_crop_path": enhanced_path,
            "section_crop_path": str(section_crop_path),
            "image_path": str(image_path),
            "picked_section": vin_section["index"],
            "section_score": sec_score,
        }
    )
    row["classification"] = classify_result(row)
    row["result_bucket"] = bucket_for_result(row)
    row["class"] = to_three_class(row) or row.get("class")
    return row


def try_multi_page_gt_scan(
    best: dict[str, Any],
    *,
    ticket: dict[str, Any],
    meta: dict[str, Any],
    page_payloads: list[tuple[str, Path, dict[str, Any]]],
    doc_dir: Path,
    short: str,
    truth: str | None,
    env: dict[str, str],
    profile: str,
    ran_page: str,
) -> dict[str, Any]:
    """When crop is null on a multi-page file, OCR-scan all pages for GT and crop-rerun."""
    if len(page_payloads) <= 1:
        return best
    if norm_vin(best.get("crop_vin_enhanced")):
        return best
    hits = scan_file_pages_for_gt(
        short=short,
        truth=truth,
        full_hint=best.get("full_page_vin") or ticket.get("prod_mllm_hint"),
        meta=meta,
        page_payloads=page_payloads,
        doc_dir=doc_dir,
        profile=profile,
    )
    if not hits:
        return best
    base = {k: v for k, v in best.items() if k not in ("wrong_page_rerun", "page_note", "gt_scan")}
    gt_pages = [h["page"] for h in hits]
    wrong_page_hits = [h for h in hits if h["wrong_page"]]
    print(
        f"  {short} multi-page GT scan: {len(hits)} hit(s) on {gt_pages}; "
        f"ran={ran_page} assigned={_doc_assigned_page_label(meta)}",
        flush=True,
    )
    candidates = wrong_page_hits or hits
    improved = best
    for hit in candidates:
        vin_section = hit.get("vin_section")
        if not vin_section:
            continue
        sec_score = score_vin_section(vin_section, hit["sections_data"]["sections"])
        row = _run_crop_row(
            base,
            ticket=ticket,
            meta=meta,
            doc_dir=doc_dir,
            short=short,
            truth=truth,
            page=hit["page"],
            payload_path=hit["payload_path"],
            sections_data=hit["sections_data"],
            vin_section=vin_section,
            sec_score=sec_score,
            env=env,
        )
        print(
            f"  {short} GT-scan {hit['page']} sec{vin_section['index']}: "
            f"crop={row.get('crop_vin_enhanced')!r} gt={truth!r} class={row.get('classification')}",
            flush=True,
        )
        if row.get("crop_matches_truth"):
            row["wrong_page_rerun"] = True
            row["gt_scan"] = {"gt_pages": gt_pages, "ran_page": ran_page}
            row["page_note"] = (
                f"DynamoDB doc maps to file pg{meta['start']} ({_doc_assigned_page_label(meta)}); "
                f"GT VIN on file pg{hit['file_page']} ({hit['page']})"
            )
            row["bucket"] = "B1"
            row["result_bucket"] = "B1"
            row["class"] = "crop"
            return row
        if norm_vin(row.get("crop_vin_enhanced")) and not norm_vin(improved.get("crop_vin_enhanced")):
            improved = row
    if improved is not best and norm_vin(improved.get("crop_vin_enhanced")):
        improved["wrong_page_rerun"] = True
        improved["gt_scan"] = {"gt_pages": gt_pages, "ran_page": ran_page}
        improved["class"] = to_three_class(improved) or improved.get("class")
    return improved


def process_doc(ticket: dict[str, Any], env: dict[str, str], profile: str) -> dict[str, Any]:
    short = ticket["short"]
    truth = ticket.get("ground_truth")
    doc_type = ticket["document_type"]
    doc_dir = OUT_DIR / short
    doc_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "ticket": ticket["ticket"],
        "ticket_url": ticket.get("ticket_url"),
        "partner": ticket["partner"],
        "short_id": short,
        "doc_id": ticket["doc_id"],
        "document_type": doc_type,
        "verification_doc_type": ticket.get("verification_doc_type"),
        "ground_truth": truth,
        "bucket": ticket.get("bucket"),
        "csv_source": ticket.get("csv_source"),
        "status": ticket.get("status"),
        "full_page_vin": None,
        "crop_vin_enhanced": None,
        "full_page_source": None,
        "classification": None,
        "error": None,
    }

    try:
        config = load_config(doc_type)
    except FileNotFoundError as exc:
        result["error"] = str(exc)
        result["classification"] = "error"
        return result

    try:
        meta = get_doc_meta(ticket["doc_id"])
        ticket["partner_id"] = meta["partner_id"]
        page_payloads = tab.payloads_for_ticket(
            {
                "doc_id": ticket["doc_id"],
                "short": short,
                "document_type": doc_type,
                "partner": ticket["partner"],
            },
            config,
        )
        page_candidates: list[tuple[float, str, Path, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for page, payload_path, data in page_payloads:
            image_path = doc_dir / f"{short}_{page}.png"
            tab.download_image(data["image_uri"], image_path, profile)
            sections_json = tab.ensure_sections(short, page, payload_path, image_path, profile)
            sections_data = json.loads(sections_json.read_text())
            vin_section = pick_vin_section(sections_data)
            if not vin_section:
                continue
            sec_score = score_vin_section(vin_section, sections_data["sections"])
            page_candidates.append(
                (sec_score, page, payload_path, data, sections_data, vin_section)
            )

        page_candidates.sort(key=lambda item: item[0], reverse=True)
        best: dict[str, Any] | None = None
        ran_page = "p0"
        for sec_score, page, payload_path, data, sections_data, vin_section in page_candidates:
            ran_page = page
            row = _run_crop_row(
                result,
                ticket=ticket,
                meta=meta,
                doc_dir=doc_dir,
                short=short,
                truth=truth,
                page=page,
                payload_path=payload_path,
                sections_data=sections_data,
                vin_section=vin_section,
                sec_score=sec_score,
                env=env,
            )
            crop_vin_norm = norm_vin(row.get("crop_vin_enhanced"))
            crop_ok = row.get("crop_matches_truth")

            if best is None:
                best = row
            elif row.get("crop_fixes") and not best.get("crop_fixes"):
                best = row
            elif crop_ok and not best.get("crop_matches_truth"):
                best = row
            elif (
                not best.get("crop_matches_truth")
                and not row.get("crop_matches_truth")
                and sec_score > float(best.get("section_score") or -999)
            ):
                best = row

            print(
                f"  {short} {page} sec{vin_section['index']} score={sec_score:.1f}: "
                f"full={row.get('full_page_vin')!r} crop={row.get('crop_vin_enhanced')!r} "
                f"gt={truth!r} class={row.get('classification')}",
                flush=True,
            )

            if crop_ok or crop_vin_norm:
                break

        if best is None:
            out = {**result, "error": "no VIN section", "classification": "error"}
            return out
        best = try_multi_page_gt_scan(
            best,
            ticket=ticket,
            meta=meta,
            page_payloads=page_payloads,
            doc_dir=doc_dir,
            short=short,
            truth=truth,
            env=env,
            profile=profile,
            ran_page=best.get("page") or ran_page,
        )
        return best
    except Exception as exc:
        result["error"] = str(exc)
        result["classification"] = "error"
        return result


def vin_display(v: Any) -> str:
    return "null" if v is None else str(v)


def _first_existing_path(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path.resolve()
    return None


def resolve_result_paths(row: dict[str, Any]) -> dict[str, Any]:
    """Resolve gallery image paths from results.json, honoring non-p0 pages."""
    r = dict(row)
    short = r.get("short_id") or (r.get("doc_id") or "")[:8]
    page = r.get("page", "p0")
    doc_dir = OUT_DIR / short
    stem = f"{short}_{page}"

    image_path = _first_existing_path(
        r.get("image_path"),
        doc_dir / f"{stem}.png",
        doc_dir / f"{short}_p0.png",
    )
    if image_path:
        r["image_path"] = str(image_path)
        page_from_image = image_path.stem.rsplit("_", 1)[-1]
        if page_from_image.startswith("p"):
            page = page_from_image
            stem = f"{short}_{page}"

    sections_overlay = _first_existing_path(
        r.get("sections_overlay_path"),
        doc_dir / f"{stem}_sections.png",
        doc_dir / f"{stem}_sections" / f"{stem}_sections.png",
    )
    if sections_overlay:
        r["sections_overlay_path"] = str(sections_overlay)
    elif image_path:
        found = disag.find_sections_png(image_path)
        if found:
            r["sections_overlay_path"] = str(found.resolve())

    enhanced_crop = _first_existing_path(
        r.get("enhanced_crop_path"),
        doc_dir / f"{stem}_vin_crop_enhanced.png",
        doc_dir / f"{short}_p0_vin_crop_enhanced.png",
        r.get("section_crop_path"),
        doc_dir / f"{stem}_section_crop.png",
    )
    if enhanced_crop:
        r["enhanced_crop_path"] = str(enhanced_crop)

    for key in ("image_path", "enhanced_crop_path", "section_crop_path", "sections_overlay_path"):
        p = r.get(key)
        if p and Path(p).exists():
            r[key] = str(Path(p).resolve())
    return r


def ensure_sections_overlay_local(row: dict[str, Any]) -> tuple[str | None, bool]:
    """Resolve sections overlay from disk or draw from existing sections JSON (no OCR/AWS).

    Returns (path, generated) where generated is True only when a new PNG was written.
    """
    r = resolve_result_paths(row)
    overlay = r.get("sections_overlay_path")
    if overlay and Path(overlay).exists():
        return overlay, False
    image_path = r.get("image_path")
    if not image_path or not Path(image_path).exists():
        return None, False
    img = Path(image_path)
    found = disag.find_sections_png(img)
    if found:
        return str(found.resolve()), False
    sections_json = disag.find_sections_json(img)
    if not sections_json:
        return None, False
    overlay_path = img.parent / f"{img.stem}_sections.png"
    if overlay_path.exists():
        return str(overlay_path.resolve()), False
    sections_data = json.loads(sections_json.read_text())
    picked = r.get("picked_section")
    from ocr_line_to_sections import draw_sections_overlay

    full_img = Image.open(img).convert("RGB")
    overlay_img = draw_sections_overlay(
        full_img,
        sections_data.get("sections", []),
        picked if isinstance(picked, int) else None,
    )
    overlay_img.save(overlay_path)
    return str(overlay_path.resolve()), True


def fast_copy_desktop_assets(
    rows: list[dict[str, Any]],
    *,
    log_sections_skips: bool = True,
) -> dict[str, Any]:
    """Copy gallery PNGs from results.json paths without wiping assets or OCR overlay regen."""
    DESKTOP_ASSETS.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "docs": len(rows),
        "expected": 0,
        "copied": 0,
        "missing": [],
        "sections_generated": 0,
        "sections_skipped": [],
    }
    for row in rows:
        r = resolve_result_paths(row)
        short = r["short_id"]
        sections_src, generated = ensure_sections_overlay_local(r)
        if sections_src:
            r["sections_overlay_path"] = sections_src
            if generated:
                stats["sections_generated"] += 1
        elif log_sections_skips and r.get("image_path"):
            stats["sections_skipped"].append(short)
        copies: list[tuple[str, str | None, str, int]] = [
            ("full", r.get("image_path"), f"{short}_full.png", 560),
            ("sections", r.get("sections_overlay_path"), f"{short}_sections.png", 560),
            ("enhanced_crop", r.get("enhanced_crop_path"), f"{short}_enhanced_crop.png", 520),
        ]
        for kind, src, dest_name, max_w in copies:
            if not src:
                continue
            stats["expected"] += 1
            src_path = Path(src)
            if src_path.exists():
                disag.thumb_copy(src_path, DESKTOP_ASSETS / dest_name, max_w=max_w)
                stats["copied"] += 1
            else:
                stats["missing"].append(f"{short}/{kind}: {src}")
    return stats


def prepare_gallery_rows(results: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        rr = dict(r)
        short = rr.get("short_id") or (rr.get("doc_id") or "")[:8]
        page = rr.get("page", "p0")
        doc_dir = OUT_DIR / short
        for key in ("image_path", "enhanced_crop_path", "section_crop_path", "sections_overlay_path"):
            p = rr.get(key)
            if p and Path(p).exists():
                rr[key] = str(Path(p).resolve())
        if not rr.get("image_path"):
            for candidate in (doc_dir / f"{short}_{page}.png", doc_dir / f"{short}_p0.png"):
                if candidate.exists():
                    rr["image_path"] = str(candidate.resolve())
                    break
        rows.append(rr)

    overlay_rows = [r for r in rows if not r.get("error") and r.get("image_path")]
    for r in overlay_rows:
        image_path = Path(r["image_path"])
        short, page = r["short_id"], r.get("page", "p0")
        sections_json = disag.find_sections_json(image_path)
        if not sections_json:
            payload = disag.find_payload(short, page, image_path)
            if payload:
                try:
                    sections_json = disag.ensure_sections_json(short, page, payload, image_path, profile)
                except Exception:
                    continue
        if not sections_json:
            continue
        sections_data = json.loads(sections_json.read_text())
        picked = r.get("picked_section")
        if picked is None:
            vin_section = pick_vin_section(sections_data)
            if vin_section:
                picked = vin_section["index"]
        overlay_path = disag.ensure_sections_overlay(
            short, page, image_path, sections_data, picked if isinstance(picked, int) else None
        )
        dest = OUT_DIR / short / f"{short}_{page}_sections.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if overlay_path != dest:
            shutil.copy2(overlay_path, dest)
        r["sections_overlay_path"] = str(dest.resolve())
        r["picked_section"] = picked
    return rows


def write_desktop_assets(
    rows: list[dict[str, Any]],
    *,
    clean: bool = True,
    fast: bool = False,
) -> dict[str, Any]:
    if fast:
        return fast_copy_desktop_assets(rows)
    if clean and DESKTOP_ASSETS.exists():
        shutil.rmtree(DESKTOP_ASSETS)
    DESKTOP_ASSETS.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"expected": 0, "copied": 0, "missing": []}
    for row in rows:
        r = resolve_result_paths(row)
        short = r["short_id"]
        copies: list[tuple[str, str | None, str, int]] = [
            ("full", r.get("image_path"), f"{short}_full.png", 560),
            ("sections", r.get("sections_overlay_path"), f"{short}_sections.png", 560),
            ("enhanced_crop", r.get("enhanced_crop_path"), f"{short}_enhanced_crop.png", 520),
        ]
        for kind, src, dest_name, max_w in copies:
            if not src:
                continue
            stats["expected"] += 1
            src_path = Path(src)
            if src_path.exists():
                disag.thumb_copy(src_path, DESKTOP_ASSETS / dest_name, max_w=max_w)
                stats["copied"] += 1
            else:
                stats["missing"].append(f"{short}/{kind}: {src}")
    return stats


def verify_html_assets(html_path: Path, assets_dir: Path, assets_prefix: str) -> dict[str, Any]:
    if not html_path.exists():
        return {"html_refs": 0, "img_tags": 0, "asset_files": 0, "missing": [], "orphan_assets": []}
    html_text = html_path.read_text()
    img_tags = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html_text, re.I)
    refs = set(re.findall(rf'src="{re.escape(assets_prefix)}/([^"]+)"', html_text))
    on_disk = {p.name for p in assets_dir.glob("*.png")} if assets_dir.exists() else set()
    missing_srcs = []
    for src in img_tags:
        if src.startswith("http"):
            continue
        if not (html_path.parent / src).resolve().exists():
            missing_srcs.append(src)
    return {
        "html_refs": len(refs),
        "img_tags": len(img_tags),
        "asset_files": len(on_disk),
        "missing": sorted(refs - on_disk),
        "missing_srcs": sorted(set(missing_srcs)),
        "orphan_assets": sorted(on_disk - refs),
    }


def verify_all_gallery_html() -> dict[str, Any]:
    """Verify every <img src> in both Desktop gallery HTML files exists on disk."""
    galleries = [
        (DESKTOP_HTML, DESKTOP_ASSETS, ASSETS_PREFIX),
        (
            Path.home() / "Desktop/vin_null_both_wrong_audit.html",
            Path.home() / "Desktop/vin_null_both_wrong_audit_assets",
            "vin_null_both_wrong_audit_assets",
        ),
    ]
    reports: dict[str, Any] = {"galleries": {}, "total_missing": 0}
    for html_path, assets_dir, prefix in galleries:
        report = verify_html_assets(html_path, assets_dir, prefix)
        reports["galleries"][html_path.name] = report
        reports["total_missing"] += len(report.get("missing_srcs") or report["missing"])
    return reports


def build_shareable_zip() -> Path:
    """Package Desktop HTML galleries + assets into ~/Desktop/wa577_vin_crop_galleries.zip."""
    audit_html = Path.home() / "Desktop/vin_null_both_wrong_audit.html"
    audit_assets = Path.home() / "Desktop/vin_null_both_wrong_audit_assets"
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)
    (BUNDLE_DIR / "README.txt").write_text(BUNDLE_README)
    shutil.copy2(DESKTOP_HTML, BUNDLE_DIR / "vin_ticket_csv_crop.html")
    shutil.copytree(DESKTOP_ASSETS, BUNDLE_DIR / "vin_ticket_csv_crop_assets")
    if audit_html.exists():
        shutil.copy2(audit_html, BUNDLE_DIR / "vin_null_both_wrong_audit.html")
    if audit_assets.exists():
        shutil.copytree(audit_assets, BUNDLE_DIR / "vin_null_both_wrong_audit_assets")
    if BUNDLE_ZIP.exists():
        BUNDLE_ZIP.unlink()
    shutil.make_archive(str(BUNDLE_ZIP.with_suffix("")), "zip", root_dir=BUNDLE_DIR.parent, base_dir=BUNDLE_DIR.name)
    return BUNDLE_ZIP


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "—"


def ok_cell(ok: bool | None) -> str:
    if ok is True:
        return '<span class="winner-crop">✓</span>'
    if ok is False:
        return '<span class="winner-both">✗</span>'
    return '<span class="muted">—</span>'


def row_full_ok_gt(row: dict[str, Any]) -> bool:
    return bool(row.get("full_matches_truth"))


def row_crop_ok_gt(row: dict[str, Any]) -> bool:
    return bool(row.get("crop_matches_truth"))


def row_full_ok_printed(row: dict[str, Any]) -> bool | None:
    if row.get("full_extraction_correct_vs_doc") is not None:
        return bool(row["full_extraction_correct_vs_doc"])
    printed = row.get("printed_vin")
    if printed:
        return vin_matches_printed(row.get("full_page_vin"), printed)
    if row.get("extraction_correct_vs_doc"):
        return True
    return None


def row_crop_ok_printed(row: dict[str, Any]) -> bool | None:
    if row.get("extraction_correct_vs_doc") is not None:
        return bool(row["extraction_correct_vs_doc"])
    printed = row.get("printed_vin")
    if printed:
        return vin_matches_printed(row.get("crop_vin_enhanced"), printed)
    return None


def render_html(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    ticket_meta: list[dict[str, Any]],
) -> str:
    from collections import Counter, defaultdict

    meta_by_ticket = {m["ticket"]: m for m in ticket_meta}
    table_rows: list[str] = []
    cards: list[str] = []

    class_style = {
        "crop": "winner-crop",
        "null": "muted",
        "both_wrong": "winner-both",
        "error": "winner-both",
    }

    scored = sorted(
        [r for r in results if to_three_class(r)],
        key=lambda x: (x.get("ticket", ""), x.get("short_id", "")),
    )
    overall = Counter(to_three_class(r) for r in scored)
    errors_n = len(results) - len(scored)

    by_ticket: dict[str, list] = defaultdict(list)
    for r in results:
        by_ticket[r.get("ticket", "?")].append(r)

    n_scored = len(scored)
    crop_n = overall.get("crop", 0)
    null_n = overall.get("null", 0)
    both_n = overall.get("both_wrong", 0)

    full_gt_n = sum(1 for r in scored if row_full_ok_gt(r))
    crop_gt_n = sum(1 for r in scored if row_crop_ok_gt(r))
    full_printed_scored = [r for r in scored if row_full_ok_printed(r) is not None]
    crop_printed_scored = [r for r in scored if row_crop_ok_printed(r) is not None]
    full_printed_n = sum(1 for r in full_printed_scored if row_full_ok_printed(r))
    crop_printed_n = sum(1 for r in crop_printed_scored if row_crop_ok_printed(r))
    full_printed_d = len(full_printed_scored)
    crop_printed_d = len(crop_printed_scored)

    def count_pct(n: int) -> str:
        return f"{n} ({pct(n, n_scored)})"

    accuracy_rows = f"""      <tr>
        <td>full correct vs GT (<code>full_matches_truth</code>)</td>
        <td>{full_gt_n}</td>
        <td>{pct(full_gt_n, n_scored)}</td>
      </tr>
      <tr>
        <td>crop correct vs GT (<code>crop_matches_truth</code>)</td>
        <td>{crop_gt_n}</td>
        <td>{pct(crop_gt_n, n_scored)}</td>
      </tr>
      <tr>
        <td>full correct vs printed</td>
        <td>{full_printed_n if full_printed_d else "—"}</td>
        <td>{pct(full_printed_n, full_printed_d) if full_printed_d else "—"}</td>
      </tr>
      <tr>
        <td>crop correct vs printed</td>
        <td>{crop_printed_n if crop_printed_d else "—"}</td>
        <td>{pct(crop_printed_n, crop_printed_d) if crop_printed_d else "—"}</td>
      </tr>
      <tr>
        <td>CLASS: crop</td>
        <td>{crop_n}</td>
        <td>{pct(crop_n, n_scored)}</td>
      </tr>
      <tr>
        <td>CLASS: null</td>
        <td>{null_n}</td>
        <td>{pct(null_n, n_scored)}</td>
      </tr>
      <tr>
        <td>CLASS: both_wrong</td>
        <td>{both_n}</td>
        <td>{pct(both_n, n_scored)}</td>
      </tr>"""

    overall_row = f"""      <tr>
        <td><strong>Overall</strong></td>
        <td>{count_pct(full_gt_n)}</td>
        <td>{count_pct(crop_gt_n)}</td>
        <td>{count_pct(crop_n)}</td>
        <td>{count_pct(null_n)}</td>
        <td>{count_pct(both_n)}</td>
      </tr>"""

    ticket_summary_rows: list[str] = []
    for ticket in sorted(by_ticket):
        rows = by_ticket[ticket]
        scored_rows = [r for r in rows if to_three_class(r)]
        c = Counter(to_three_class(r) for r in scored_rows)
        crop_t = c.get("crop", 0)
        full_t = sum(1 for r in scored_rows if row_full_ok_gt(r))
        crop_ok_t = sum(1 for r in scored_rows if row_crop_ok_gt(r))
        ticket_summary_rows.append(
            f"""      <tr>
        <td><a href="{html.escape(meta_by_ticket.get(ticket, {}).get('url') or '#')}">{html.escape(ticket)}</a></td>
        <td>{full_t}</td>
        <td>{crop_ok_t}</td>
        <td>{crop_t}</td>
        <td>{c.get('null', 0)}</td>
        <td>{c.get('both_wrong', 0)}</td>
        <td>{pct(crop_t, len(scored_rows))}</td>
      </tr>"""
        )

    for r in scored:
        short = html.escape(r["short_id"])
        ticket = html.escape(r.get("ticket") or "—")
        ticket_url = html.escape(r.get("ticket_url") or "#")
        partner = html.escape(str(r.get("partner") or "—"))
        full_v = html.escape(vin_display(r.get("full_page_vin")))
        crop_v = html.escape(vin_display(r.get("crop_vin_enhanced")))
        gt_v = html.escape(vin_display(r.get("ground_truth")))
        three_cls = to_three_class(r) or "—"
        cls = html.escape(three_cls)
        bucket = html.escape(r.get("result_bucket") or r.get("bucket") or "—")
        row_class = class_style.get(three_cls, "muted")

        anchor = f"doc-{r['short_id']}"
        has_thumb = r.get("image_path") and Path(r["image_path"]).exists()
        view_cell = (
            f'<a href="#{anchor}"><img src="{ASSETS_PREFIX}/{short}_full.png" alt="view" class="view-thumb"></a>'
            if has_thumb
            else f'<a href="#{anchor}">view</a>'
        )
        table_rows.append(
            f"""      <tr>
        <td><a href="{ticket_url}">{ticket}</a></td>
        <td><a href="#{anchor}"><code>{short}</code></a></td>
        <td>{partner}</td>
        <td class="vin-full"><code>{full_v}</code></td>
        <td class="vin-crop"><code>{crop_v}</code></td>
        <td class="vin-gt"><code>{gt_v}</code></td>
        <td>{ok_cell(row_full_ok_gt(r))}</td>
        <td>{ok_cell(row_crop_ok_gt(r))}</td>
        <td class="{row_class}"><code>{cls}</code></td>
        <td>{bucket}</td>
        <td>{view_cell}</td>
      </tr>"""
        )

    for r in results:
        short = html.escape(r["short_id"])
        ticket = html.escape(r.get("ticket") or "—")
        ticket_url = html.escape(r.get("ticket_url") or "#")
        dtype = html.escape(r.get("document_type") or "")
        partner = html.escape(str(r.get("partner") or "—"))
        full_v = html.escape(vin_display(r.get("full_page_vin")))
        crop_v = html.escape(vin_display(r.get("crop_vin_enhanced")))
        gt_v = html.escape(vin_display(r.get("ground_truth")))
        three_cls = to_three_class(r) or "—"
        cls_display = html.escape(three_cls)
        bucket = html.escape(r.get("result_bucket") or r.get("bucket") or "—")
        row_class = class_style.get(three_cls, "muted")

        anchor = f"doc-{r['short_id']}"
        if r.get("error"):
            cards.append(
                f"""  <div class="doc-card error-card" id="{anchor}">
    <h2><a href="{ticket_url}">{ticket}</a> · <code>{short}</code> · {dtype}</h2>
    <p class="muted">GT <code class="vin-gt">{gt_v}</code> · ERR: {html.escape(str(r['error'])[:120])}</p>
  </div>"""
            )
            continue

        sections_block = (
            f'<img src="{ASSETS_PREFIX}/{short}_sections.png" alt="sections">'
            if r.get("sections_overlay_path") and Path(r["sections_overlay_path"]).exists()
            else '<p class="muted">No sections overlay</p>'
        )
        full_block = (
            f'<img src="{ASSETS_PREFIX}/{short}_full.png" alt="full">'
            if r.get("image_path") and Path(r["image_path"]).exists()
            else '<p class="muted">No full page</p>'
        )
        enhanced_block = (
            f'<img src="{ASSETS_PREFIX}/{short}_enhanced_crop.png" alt="crop">'
            if r.get("enhanced_crop_path") and Path(r["enhanced_crop_path"]).exists()
            else '<p class="muted">No enhanced crop</p>'
        )
        cards.append(
            f"""  <div class="doc-card" id="{anchor}">
    <h2><a href="{ticket_url}">{ticket}</a> · <code>{short}</code> · {dtype} · {partner}</h2>
    <p>Full/prod <code class="vin-full">{full_v}</code> · Crop <code class="vin-crop">{crop_v}</code> · GT <code class="vin-gt">{gt_v}</code> · Full OK {ok_cell(row_full_ok_gt(r))} · Crop OK {ok_cell(row_crop_ok_gt(r))} · <span class="{row_class}">{cls_display}</span> · {bucket}</p>
    <div class="images images-3">
      <div class="img-block"><label>Full page</label>{full_block}</div>
      <div class="img-block"><label>Sections</label>{sections_block}</div>
      <div class="img-block"><label>Enhanced crop</label>{enhanced_block}</div>
    </div>
  </div>"""
        )

    ticket_blurbs = []
    for m in ticket_meta:
        t = html.escape(m["ticket"])
        url = html.escape(m.get("url") or "#")
        notes = html.escape(m.get("notes", ""))
        ticket_blurbs.append(
            f'<li><a href="{url}"><strong>{t}</strong></a> ({html.escape(m.get("partner", ""))}): {notes}</li>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Ticket CSV VIN crop gallery</title>
  <style>
    html {{ scroll-behavior:smooth; }}
    :root {{ --bg:#0f1419; --surface:#1a2332; --border:#2d3a4d; --text:#e8edf4; --muted:#8b9cb3; --full:#fbbf24; --crop:#60a5fa; --gt:#34d399; }}
    body {{ margin:0; padding:2rem; font-family:-apple-system,sans-serif; background:var(--bg); color:var(--text); }}
    h1 {{ margin:0 0 .5rem; }}
    .subtitle {{ color:var(--muted); margin-bottom:1.5rem; }}
    .ticket-meta ul {{ margin:.5rem 0 0; padding-left:1.25rem; color:var(--muted); font-size:.95rem; }}
    table {{ width:100%; border-collapse:collapse; margin-bottom:2rem; background:var(--surface); border-radius:8px; overflow:hidden; font-size:.92rem; }}
    th,td {{ padding:.65rem .85rem; text-align:left; border-bottom:1px solid var(--border); vertical-align:top; }}
    th {{ background:#243044; color:var(--muted); font-size:.78rem; text-transform:uppercase; }}
    code {{ font-family:monospace; background:#0d1117; padding:.1em .35em; border-radius:4px; word-break:break-all; }}
    a {{ color:#93c5fd; }}
    .vin-full {{ color:var(--full); }} .vin-crop {{ color:var(--crop); }} .vin-gt {{ color:var(--gt); }}
    .winner-crop {{ color:#34d399; font-weight:600; }} .winner-full {{ color:var(--full); }} .winner-both {{ color:#f87171; }}
    .muted {{ color:var(--muted); }}
    .doc-card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1.5rem; margin-bottom:2rem; }}
    .error-card {{ border-color:#7f1d1d; }}
    .images {{ display:grid; gap:1rem; }} .images-3 {{ grid-template-columns:1fr 1fr 1fr; }}
    @media (max-width:1100px) {{ .images-3 {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width:800px) {{ .images-3 {{ grid-template-columns:1fr; }} }}
    .img-block label {{ display:block; font-size:.78rem; font-weight:600; color:var(--muted); margin-bottom:.5rem; text-transform:uppercase; }}
    .img-block img {{ width:100%; border:1px solid var(--border); border-radius:6px; }}
    .view-thumb {{ width:72px; height:auto; border:1px solid var(--border); border-radius:4px; display:block; }}
    .doc-card:target {{ outline:2px solid #60a5fa; outline-offset:4px; }}
  </style>
</head>
<body>
  <h1>Ticket CSV VIN — crop-only extraction</h1>
  <p class="subtitle">{summary['n_docs']} docs · {summary['n_tickets']} tickets · {n_scored} scored · {errors_n} errors excluded</p>
  <div class="ticket-meta"><strong>Tickets</strong><ul>{chr(10).join(ticket_blurbs)}</ul></div>
  <h2>Overall extraction accuracy (scored docs, {n_scored})</h2>
  <table>
    <thead><tr><th>Metric</th><th>Count</th><th>%</th></tr></thead>
    <tbody>{accuracy_rows}</tbody>
  </table>
  <h2>Summary (full · crop · CLASS)</h2>
  <table>
    <thead><tr><th></th><th>full correct</th><th>crop correct</th><th>crop class</th><th>null</th><th>both_wrong</th></tr></thead>
    <tbody>{overall_row}</tbody>
  </table>
  <h2>By ticket</h2>
  <table>
    <thead><tr><th>Ticket</th><th>full OK</th><th>crop OK</th><th>crop</th><th>null</th><th>both_wrong</th><th>crop %</th></tr></thead>
    <tbody>{chr(10).join(ticket_summary_rows)}</tbody>
  </table>
  <h2>Per doc (all scored docs)</h2>
  <table>
    <thead><tr><th>Ticket</th><th>Doc</th><th>Partner</th><th>Prod/full</th><th>Crop</th><th>GT</th><th>Full OK</th><th>Crop OK</th><th>CLASS</th><th>Bucket</th><th>View</th></tr></thead>
    <tbody>{chr(10).join(table_rows)}</tbody>
  </table>
{chr(10).join(cards)}
</body>
</html>
"""


def render_md(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Ticket CSV VIN — crop-only extraction",
        "",
        "Full-page VIN from prod S3 baseline or CSV prod hint (no full-page MLLM rerun).",
        "Crop: section bounds + `enhance_vin_crop` + MLLM.",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Tickets | {summary['n_tickets']} |",
        f"| Docs run | {summary['n_docs']} |",
        f"| Errors | {summary['n_errors']} |",
        f"| crop_fixes_full | {summary['crop_fixes_full']} |",
        f"| crop_wins | {summary['crop_wins']} |",
        f"| both_ok | {summary['both_ok']} |",
        f"| both_wrong | {summary['both_wrong']} |",
        f"| crop_null | {summary['crop_null']} |",
        "",
        "| Ticket | Doc | Partner | Type | Prod/full | Crop | GT | Class | Bucket |",
        "|--------|-----|---------|------|-----------|------|-----|-------|--------|",
    ]
    for r in results:
        if r.get("error"):
            lines.append(
                f"| {r['ticket']} | `{r['short_id']}` | {r.get('partner','')} | {r.get('document_type','')} | "
                f"— | — | `{r.get('ground_truth','')}` | error | {r.get('bucket','')} |"
            )
            continue
        lines.append(
            f"| {r['ticket']} | `{r['short_id']}` | {r.get('partner','')} | {r.get('document_type','')} | "
            f"`{r.get('full_page_vin')}` | `{r.get('crop_vin_enhanced')}` | `{r.get('ground_truth')}` | "
            f"{r.get('classification')} | {r.get('result_bucket') or r.get('bucket','')} |"
        )
    return "\n".join(lines) + "\n"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if not r.get("error")]
    by_class: dict[str, int] = {}
    for r in results:
        c = r.get("classification") or "error"
        by_class[c] = by_class.get(c, 0) + 1
    return {
        "n_tickets": len({r["ticket"] for r in results}),
        "n_docs": len(results),
        "n_errors": sum(1 for r in results if r.get("error")),
        "crop_fixes_full": by_class.get("crop_fixes_full", 0),
        "crop_wins": by_class.get("crop_wins", 0) + by_class.get("crop_fixes_full", 0),
        "both_ok": by_class.get("both_ok", 0),
        "both_wrong": by_class.get("both_wrong", 0),
        "crop_null": by_class.get("crop_null", 0),
        "by_classification": by_class,
        "b1_crop_fixes": sum(
            1 for r in ok if r.get("classification") == "crop_fixes_full" and r.get("result_bucket") == "B1"
        ),
    }


def rescan_docs_in_results(short_ids: list[str], profile: str) -> tuple[list[str], list[dict[str, Any]]]:
    """GT-scan + crop-rerun for null multi-page docs; merge into results.json."""
    results_path = OUT_DIR / "results.json"
    data = json.loads(results_path.read_text())
    by_short = {r["short_id"]: r for r in data["results"]}
    tickets_path = OUT_DIR / "tickets.json"
    ticket_by_short = {t["short"]: t for t in json.loads(tickets_path.read_text())["tickets"]}

    env = tab.refresh_aws_env(copy.copy(os.environ))
    env["SKIP_LLM_CACHE"] = "1"
    env["AWS_PROFILE"] = profile
    env.pop("BUNDLE_PATH", None)

    flipped: list[str] = []
    scan_rows: list[dict[str, Any]] = []
    for short in short_ids:
        prev = by_short.get(short)
        ticket = ticket_by_short.get(short)
        if not prev:
            print(f"SKIP {short}: missing from results.json", flush=True)
            continue
        if not ticket:
            ticket = {
                "short": short,
                "doc_id": prev["doc_id"],
                "document_type": prev.get("document_type") or "title_application",
                "partner": prev.get("partner"),
                "ticket": prev.get("ticket"),
                "ground_truth": prev.get("ground_truth"),
                "prod_mllm_hint": prev.get("full_page_vin"),
            }
        old_class = prev.get("class")
        print(f"=== GT-scan {short} (was {old_class} on {prev.get('page')}) ===", flush=True)
        meta = get_doc_meta(ticket["doc_id"])
        doc_type = ticket["document_type"]
        config = load_config(doc_type)
        doc_dir = OUT_DIR / short
        page_payloads = tab.payloads_for_ticket(
            {
                "doc_id": ticket["doc_id"],
                "short": short,
                "document_type": doc_type,
                "partner": ticket["partner"],
            },
            config,
        )
        s3_pages = tab.list_s3_file_pages(meta, profile)
        truth = prev.get("ground_truth") or ticket.get("ground_truth")
        new_row = try_multi_page_gt_scan(
            prev,
            ticket=ticket,
            meta=meta,
            page_payloads=page_payloads,
            doc_dir=doc_dir,
            short=short,
            truth=truth,
            env=env,
            profile=profile,
            ran_page=prev.get("page") or "p0",
        )
        if not new_row.get("class"):
            new_row["class"] = to_three_class(new_row) or old_class
        by_short[short] = new_row
        new_class = new_row.get("class")
        if old_class == "null" and new_class == "crop":
            flipped.append(short)
        scan_rows.append(
            {
                "short": short,
                "ticket": ticket.get("ticket"),
                "bucket": new_row.get("bucket"),
                "ran_page": prev.get("page"),
                "new_page": new_row.get("page"),
                "doc_file_pages": f"{meta['start']}-{meta['end']}",
                "s3_page_count": len(s3_pages),
                "s3_pages": [pnum for _, pnum, _ in s3_pages],
                "gt_scan": new_row.get("gt_scan"),
                "wrong_page_rerun": new_row.get("wrong_page_rerun"),
                "old_class": old_class,
                "new_class": new_class,
                "crop_vin": new_row.get("crop_vin_enhanced"),
                "ground_truth": new_row.get("ground_truth"),
                "multi_page_file": len(s3_pages) > 1,
                "flag": "MULTI_PAGE_NULL_CROP",
            }
        )

    data["results"] = [by_short.get(r["short_id"], r) for r in data["results"]]
    data["summary"] = summarize(data["results"])
    data["summary"]["by_class"] = summarize_three_class(data["results"])
    data["summary"]["three_class"] = data["summary"]["by_class"]
    data["summary"]["wrong_page_rescan"] = {"flipped_null_to_crop": flipped, "scanned": short_ids}
    results_path.write_text(json.dumps(data, indent=2))

    scan_path = OUT_DIR / "wrong_page_scan.json"
    prior = json.loads(scan_path.read_text()) if scan_path.exists() else []
    merged = {r["short"]: r for r in prior if isinstance(r, dict) and r.get("short")}
    for row in scan_rows:
        merged[row["short"]] = row
    scan_path.write_text(json.dumps(list(merged.values()), indent=2))
    return flipped, scan_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-docs", type=int, default=80, help="Max docs to run (default 80)")
    parser.add_argument("--docs-only", action="store_true", help="Write tickets.json / docs.json only")
    parser.add_argument("--html-only", action="store_true", help="Regenerate desktop HTML from results.json")
    parser.add_argument(
        "--fast-html-only",
        action="store_true",
        help="Regenerate HTML + copy existing gallery PNGs from results.json (no rmtree, no overlay regen)",
    )
    parser.add_argument(
        "--fast-copy-assets",
        "--copy-assets-only",
        action="store_true",
        dest="fast_copy_assets",
        help="Copy gallery PNGs from results.json only (no rmtree, no HTML, no OCR overlay regen)",
    )
    parser.add_argument(
        "--verify-html",
        action="store_true",
        help="Verify every <img src> in both Desktop gallery HTML files exists on disk",
    )
    parser.add_argument(
        "--bundle-zip",
        action="store_true",
        help="Rebuild ~/Desktop/wa577_vin_crop_galleries.zip from Desktop HTML + assets",
    )
    parser.add_argument(
        "--rescan-docs",
        type=str,
        help="Comma-separated short_ids to re-run with multi-page GT scan; merges into results.json",
    )
    parser.add_argument("--seed-json", action="store_true", help="Load csv_rows_seed.json instead of csvs/")
    parser.add_argument("--tickets-file", type=Path, help="Use prebuilt tickets.json")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tab.OUT_DIR = OUT_DIR

    profile = os.environ.get("AWS_PROFILE", "prod")

    if args.rescan_docs:
        shorts = [s.strip() for s in args.rescan_docs.split(",") if s.strip()]
        flipped, _ = rescan_docs_in_results(shorts, profile)
        print(f"\nFlipped null→crop: {flipped}")
        return

    if args.verify_html:
        reports = verify_all_gallery_html()
        for name, report in reports["galleries"].items():
            missing = report.get("missing_srcs") or report["missing"]
            print(
                f"{name}: {report.get('img_tags', report['html_refs'])} img tags, "
                f"{report['asset_files']} asset files, {len(missing)} missing"
            )
            for item in missing[:20]:
                print(f"  {item}")
        print(f"Total missing: {reports['total_missing']}")
        if reports["total_missing"]:
            raise SystemExit(1)
        return

    if args.fast_copy_assets:
        results_path = OUT_DIR / "results.json"
        if not results_path.exists():
            raise SystemExit(f"Missing {results_path}")
        results = json.loads(results_path.read_text()).get("results", [])
        asset_stats = fast_copy_desktop_assets(results)
        verify = verify_html_assets(DESKTOP_HTML, DESKTOP_ASSETS, ASSETS_PREFIX)
        print(
            f"Fast-copied {asset_stats['copied']}/{asset_stats['expected']} assets "
            f"for {asset_stats['docs']} docs → {DESKTOP_ASSETS}"
        )
        if asset_stats.get("sections_generated"):
            print(f"Sections overlays generated locally: {asset_stats['sections_generated']}")
        if asset_stats["missing"]:
            print(f"Missing source images ({len(asset_stats['missing'])}):")
            for line in asset_stats["missing"][:40]:
                print(f"  {line}")
            if len(asset_stats["missing"]) > 40:
                print(f"  ... and {len(asset_stats['missing']) - 40} more")
        if asset_stats.get("sections_skipped"):
            print(f"Sections skipped (no JSON): {', '.join(asset_stats['sections_skipped'][:20])}")
        if verify["missing"]:
            print(f"HTML refs still missing ({len(verify['missing'])}):")
            for name in verify["missing"][:30]:
                print(f"  {name}")
        else:
            print(f"HTML verify OK: {verify['html_refs']} img refs, {verify['asset_files']} asset files")
        if args.bundle_zip:
            zip_path = build_shareable_zip()
            print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1_048_576:.1f} MiB)")
        return

    if args.html_only or args.fast_html_only:
        results_path = OUT_DIR / "results.json"
        if not results_path.exists():
            raise SystemExit(f"Missing {results_path}")
        data = json.loads(results_path.read_text())
        results = data.get("results", [])
        summary = data.get("summary") or summarize(results)
        ticket_meta = data.get("ticket_meta") or TICKET_META
        gallery_rows = [resolve_result_paths(r) for r in results]
        asset_stats = write_desktop_assets(
            gallery_rows,
            clean=not args.fast_html_only,
            fast=args.fast_html_only,
        )
        DESKTOP_HTML.write_text(render_html(gallery_rows, summary, ticket_meta))
        verify = verify_html_assets(DESKTOP_HTML, DESKTOP_ASSETS, ASSETS_PREFIX)
        print(
            f"Wrote {len(gallery_rows)} cards to {DESKTOP_HTML} "
            f"({asset_stats['copied']}/{asset_stats['expected']} assets copied to {DESKTOP_ASSETS})"
        )
        if asset_stats.get("sections_generated"):
            print(f"Sections overlays generated locally: {asset_stats['sections_generated']}")
        if asset_stats["missing"]:
            print(f"Missing source images ({len(asset_stats['missing'])}):")
            for line in asset_stats["missing"][:40]:
                print(f"  {line}")
            if len(asset_stats["missing"]) > 40:
                print(f"  ... and {len(asset_stats['missing']) - 40} more")
        if asset_stats.get("sections_skipped"):
            print(f"Sections skipped (no JSON): {', '.join(asset_stats['sections_skipped'][:20])}")
        if verify["missing"]:
            print(
                f"HTML/asset gaps: {len(verify['missing'])} refs without files "
                f"(html_refs={verify['html_refs']}, asset_files={verify['asset_files']})"
            )
            for name in verify["missing"][:20]:
                print(f"  missing file: {name}")
        else:
            print(
                f"HTML verify OK: {verify['html_refs']} img refs, "
                f"{verify['asset_files']} asset files"
            )
        if args.bundle_zip:
            zip_path = build_shareable_zip()
            print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1_048_576:.1f} MiB)")
        return

    if args.bundle_zip:
        zip_path = build_shareable_zip()
        print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1_048_576:.1f} MiB)")
        return

    (OUT_DIR / "tickets_meta.json").write_text(json.dumps({"tickets": TICKET_META}, indent=2))

    if args.tickets_file:
        docs = json.loads(args.tickets_file.read_text()).get("tickets", [])
    else:
        has_csv = CSV_DIR.exists() and any(CSV_DIR.glob("*.csv"))
        csv_rows = load_csv_rows(use_seed=args.seed_json or not has_csv)
        merged = merge_all_rows(csv_rows)
        selected = select_docs(merged, args.max_docs)
        docs = build_ticket_docs(selected)
        (OUT_DIR / "docs.json").write_text(json.dumps({"rows": merged, "selected": selected}, indent=2))
        (OUT_DIR / "tickets.json").write_text(json.dumps({"tickets": docs}, indent=2))

    print(f"Resolved {len(docs)} docs from {len(TICKET_META)} tickets", flush=True)
    if args.docs_only:
        return

    profile = os.environ.get("AWS_PROFILE", "prod")
    env = tab.refresh_aws_env(copy.copy(os.environ))
    env["SKIP_LLM_CACHE"] = "1"
    env["AWS_PROFILE"] = profile
    env.pop("BUNDLE_PATH", None)

    results: list[dict[str, Any]] = []
    for i, doc in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] {doc['ticket']} / {doc['short']}", flush=True)
        results.append(process_doc(doc, env, profile))

    summary = summarize(results)
    out = {"summary": summary, "results": results, "ticket_meta": TICKET_META}
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "results.md").write_text(render_md(results, summary))
    gallery_rows = prepare_gallery_rows(results, profile)
    asset_stats = write_desktop_assets(gallery_rows)
    DESKTOP_HTML.write_text(render_html(gallery_rows, summary, TICKET_META))
    print("\n" + render_md(results, summary))
    print(f"Wrote {OUT_DIR / 'results.json'}")
    print(
        f"Wrote {DESKTOP_HTML} "
        f"({asset_stats['copied']}/{asset_stats['expected']} assets in {DESKTOP_ASSETS})"
    )


if __name__ == "__main__":
    main()
