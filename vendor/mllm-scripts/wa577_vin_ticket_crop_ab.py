#!/usr/bin/env python3
"""VIN crop A/B on ticket-sourced docs across document types (Linear + Jira lineage)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from wa577_vin_crop_helpers import (
    VIN_LABEL,
    VIN_TOKEN,
    pick_vin_section,
    score_vin_section,
    tight_vin_bounds,
)
from wa577_vin_ticket_discover import discover_and_resolve
MLLM = ROOT / "topologies/extraction/shared/mllm-invoker/bin/mllm_test"
MLLM_LOCAL = ROOT / "topologies/extraction/shared/mllm-invoker/bin/mllm_test_local_image.rb"
MLLM_DIR = MLLM.parent.parent
PAYLOAD_DIRS = [
    ROOT / "topologies/extraction/shared/mllm-invoker/payloads/wa577_v2",
    ROOT / "topologies/extraction/shared/mllm-invoker/payloads/wa513_577_v1",
    ROOT / "topologies/extraction/shared/mllm-invoker/payloads/wa625",
]
QA_ROOT = ROOT.parent / "techno-configs/techno_configs/envs/qa/document_fields"
DEFAULT_OUT_DIR = ROOT / "wa577_gallery/vin_ticket_crop_ab"
OUT_DIR = DEFAULT_OUT_DIR
BUCKET = "informed-techno-core-prod-exchange"

# 10 ticket docs — mixed document types (not title_application only).
TICKETS: list[dict[str, Any]] = [
    {
        "ticket": "WA-577",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-577",
        "partner": "penair",
        "short": "87c842be",
        "doc_id": "87c842be-0578-452a-a9ba-99aefa0e0a4d",
        "document_type": "title_application",
        "ground_truth": "1FMEE8BH0TLA94758",
        "issue": "digit transposition 94758→97458",
    },
    {
        "ticket": "WA-640",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-640",
        "partner": "cuofco",
        "short": "899d7cbd",
        "doc_id": "899d7cbd-ba36-46f2-88d3-8ef79282365f",
        "document_type": "title_application",
        "ground_truth": "jf2sjaec6hh410265",
        "issue": "multi-page merge picks wrong page VIN",
    },
    {
        "ticket": "WA-489",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-489",
        "partner": "consumerscu",
        "short": "8cebd2b2",
        "doc_id": "8cebd2b2-7477-4e1a-8179-0ae1bed16f5f",
        "document_type": "title_application",
        "ground_truth": "1ftfw3ld4rfa81006",
        "issue": "b→8 OCR char swap",
    },
    {
        "ticket": "WA-411",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-411",
        "partner": "autonationfinance",
        "short": "2565927a",
        "doc_id": "2565927a-ca67-4a8f-bb18-08ff5a45c9d9",
        "document_type": "title_application",
        "ground_truth": "JTND4MBE4P3206996",
        "issue": "leading 1 vs J misread",
    },
    {
        "ticket": "WA-538",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-538",
        "partner": "consumerscu",
        "short": "fcca8cfa",
        "doc_id": "fcca8cfa-4708-45ac-adb4-f6a5f7c56c7e",
        "document_type": "title_application",
        "verification_doc_type": "odometer_statement",
        "ground_truth": "jf1va1b61h9818909",
        "issue": "1→j OCR char swap (odometer VIN verify)",
    },
    {
        "ticket": "WA-477",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-477",
        "partner": "autonationfinance",
        "short": "00db9375",
        "doc_id": "00db9375-ce21-4123-b1af-21684f0d823d",
        "document_type": "gap_binder",
        "verification_doc_type": "gap_waiver_contract",
        "ground_truth": "5LMCJ1D98GUJ29964",
        "issue": "model name concatenated into VIN",
    },
    {
        "ticket": "WA-300",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-300",
        "partner": "chase",
        "short": "00024cb3",
        "doc_id": "00024cb3-f452-4b67-ba6e-1623c6f79784",
        "document_type": "vehicle_service_contract",
        "ground_truth": "4s4guhd68t3770803",
        "issue": "null VIN extraction",
    },
    {
        "ticket": "WA-539",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-539",
        "partner": "consumerscu",
        "short": "007c3eac",
        "doc_id": "007c3eac-26ca-4f63-8344-0d27d2a48ba2",
        "document_type": "retail_installment_sales_contract",
        "ground_truth": "1gykpdrs4sz142778",
        "issue": "wrong VIN on RISC",
    },
    {
        "ticket": "LIV-42",
        "source": "linear",
        "jira_key": "LIV-42",
        "url": "https://linear.app/informediq/issue/LIV-42",
        "partner": "glsauto",
        "short": "00044a39",
        "doc_id": "00044a39-12fe-4ed2-b138-dbd6a45baf24",
        "document_type": "factory_invoice",
        "ground_truth": "3n1ab8dv7sy319314",
        "issue": "factory invoice VIN mismatch",
    },
    {
        "ticket": "LIV-545",
        "source": "jira",
        "jira_key": "LIV-545",
        "url": "https://linear.app/informediq/issue/LIV-545",
        "jira_url": "https://informed-iq.atlassian.net/browse/LIV-545",
        "partner": "consumerscu",
        "short": "b1bc95a6",
        "doc_id": "b1bc95a6-fde4-4e29-9f73-2151b2c4004a",
        "document_type": "title_application",
        "ground_truth": "5lmpj8ja7tj049186",
        "issue": "char transposition (QC batch / WA-489 cluster)",
    },
]

_ddb = None
_s3 = None


def norm_vin(v: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(v or "")).upper()


@lru_cache(maxsize=16)
def load_config(doc_type: str) -> dict[str, Any]:
    yaml_path = QA_ROOT / f"extractions/llm_configs/{doc_type}.yml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"no QA llm_config for {doc_type}")
    config = yaml.safe_load(yaml_path.read_text())
    payload_type = config["model_info"]["payload_config"].get("type")
    if payload_type != "custom":
        return config
    schema_path = QA_ROOT / f"serialization/v1/{doc_type}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"no QA schema for {doc_type}")
    schema_body = json.loads(schema_path.read_text())["definitions"]["extracted_data"]
    intro = config["model_info"]["payload_config"]["prompt_config"]["intro"]
    config["model_info"]["payload_config"]["prompt_config"]["intro"] = intro.replace(
        "$$_SCHEMA", json.dumps(schema_body)
    )
    return config


def extract_vin(payload: dict[str, Any]) -> Any:
    if not isinstance(payload, dict):
        return None
    if payload.get("vin") is not None:
        return payload.get("vin")
    vehicle = payload.get("vehicle")
    if isinstance(vehicle, dict) and vehicle.get("vin") is not None:
        return vehicle.get("vin")
    vehicles = payload.get("vehicles")
    if isinstance(vehicles, list):
        for item in vehicles:
            if isinstance(item, dict) and item.get("vin") is not None:
                return item.get("vin")
    for value in payload.values():
        if isinstance(value, dict):
            found = extract_vin(value)
            if found is not None:
                return found
    return None


def refresh_aws_env(env: dict[str, str]) -> dict[str, str]:
    profile = env.get("AWS_PROFILE", "prod")
    proc = subprocess.run(
        ["aws", "configure", "export-credentials", "--profile", profile, "--format", "env"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = dict(env)
    for line in proc.stdout.splitlines():
        if line.startswith("export "):
            key, _, val = line.removeprefix("export ").partition("=")
            out[key] = val.strip().strip('"')
    return out


def boto_session():
    kwargs: dict[str, Any] = {"region_name": "us-west-2"}
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        kwargs["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
        kwargs["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
        kwargs["aws_session_token"] = os.environ.get("AWS_SESSION_TOKEN")
    else:
        kwargs["profile_name"] = os.environ.get("AWS_PROFILE", "prod")
    return boto3.Session(**kwargs)


def init_aws():
    global _ddb, _s3
    if _ddb is None:
        session = boto_session()
        _ddb = session.client("dynamodb")
        _s3 = session.client("s3")


def get_doc_meta(doc_id: str) -> dict[str, Any]:
    init_aws()
    item = _ddb.get_item(
        TableName="techno-core-prod-document-orchestrator",
        Key={"PK": {"S": doc_id}, "SK": {"S": "document"}},
        ProjectionExpression="partner_id, application_id, file_ids, parent_partition_params, document_type",
    )["Item"]
    pages = item["parent_partition_params"]["M"]["pages"]["L"][0]["M"]
    file_ids = item.get("file_ids", {}).get("L", [])
    if not file_ids:
        raise RuntimeError(f"document {doc_id} has no file_ids in DynamoDB")
    return {
        "partner_id": item["partner_id"]["S"],
        "app_id": item["application_id"]["S"],
        "file_id": file_ids[0]["S"],
        "document_type": item.get("document_type", {}).get("S"),
        "start": int(pages["start_page"]["N"]),
        "end": int(pages["end_page"]["N"]),
    }


def find_payloads(short: str) -> list[Path]:
    found: list[Path] = []
    for d in PAYLOAD_DIRS:
        if not d.exists():
            continue
        found.extend(sorted(d.glob(f"*{short}*_p*.json")))
        found.extend(sorted(d.glob(f"{short}*_p*.json")))
    seen: set[str] = set()
    unique: list[Path] = []
    for p in found:
        page = p.stem.rsplit("_", 1)[-1]
        key = f"{short}_{page}"
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return sorted(unique)


def build_payload(
    doc_id: str,
    meta: dict[str, Any],
    page_num: int,
    page_offset: int,
    config: dict,
    document_type: str,
) -> Path:
    short = meta["short"]
    payload_path = OUT_DIR / short / f"{short}_{page_offset}_payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    base = f"s3://{BUCKET}/{meta['partner_id']}/{meta['app_id']}"
    payload = {
        "detail": {
            "metadata": {},
            "data": {
                "partner_id": meta["partner_id"],
                "application_id": meta["app_id"],
                "application_data_uri": f"{base}/app_context/{meta['file_id']}.json",
                "document_id": doc_id,
                "document_type": document_type,
                "image_uri": f"{base}/file/{meta['file_id']}/img/{meta['file_id']}-{page_num}.png",
            },
        },
        "config": copy.deepcopy(config),
    }
    payload_path.write_text(json.dumps(payload))
    return payload_path


def download_image(uri: str, dest: Path, profile: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    key = uri.removeprefix(f"s3://{BUCKET}/")
    subprocess.run(
        ["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(dest), "--profile", profile],
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_sections(short: str, page: str, payload: Path, image: Path, profile: str) -> Path:
    sections_json = OUT_DIR / short / f"{short}_{page}_sections" / f"{short}_{page}_sections.json"
    if sections_json.exists():
        return sections_json
    out = OUT_DIR / short / f"{short}_{page}_sections"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "ocr_line_to_sections.py"),
            "--payload",
            str(payload),
            "--image",
            str(image),
            "--out",
            str(out),
            "--aws-profile",
            profile,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out / f"{short}_{page}_sections.json"


def crop_bounds(image_path: Path, bounds: dict[str, float], pad: int = 24) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    x0 = max(0, int(bounds["min_x"]) - pad)
    y0 = max(0, int(bounds["min_y"]) - pad)
    x1 = min(img.width, int(bounds["max_x"]) + pad)
    y1 = min(img.height, int(bounds["max_y"]) + pad)
    return img.crop((x0, y0, x1, y1))


def run_mllm(payload_path: Path, env: dict[str, str], *, local_image: Path | None = None) -> dict[str, Any]:
    run_env = dict(env)
    if local_image:
        cmd = ["ruby", str(MLLM_LOCAL), "--payload", str(payload_path)]
        run_env["LOCAL_IMAGE_PATH"] = str(local_image)
    else:
        cmd = [str(MLLM), "--payload", str(payload_path)]
    proc = subprocess.run(cmd, cwd=str(MLLM_DIR), capture_output=True, text=True, env=run_env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:])
    text = proc.stdout
    start = text.rfind("\n{")
    if start < 0:
        start = text.find("{")
    else:
        start += 1
    return json.loads(text[start:])


def list_s3_file_pages(meta: dict[str, Any], profile: str | None = None) -> list[tuple[int, int, str]]:
    """Return (page_offset, page_num, image_uri) for every PNG in the file's S3 img/ folder."""
    profile = profile or os.environ.get("AWS_PROFILE", "prod")
    prefix = f"{meta['partner_id']}/{meta['app_id']}/file/{meta['file_id']}/img/"
    session = boto3.Session(profile_name=profile, region_name="us-west-2")
    s3 = session.client("s3")
    keys: list[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".png"):
                keys.append(key)
    keys.sort()
    out: list[tuple[int, int, str]] = []
    file_id = meta["file_id"]
    standard_re = re.compile(rf"^{re.escape(file_id)}-(\d+)\.png$")
    for offset, key in enumerate(keys):
        fname = key.rsplit("/", 1)[-1]
        match = standard_re.match(fname)
        if not match:
            continue
        page_num = int(match.group(1))
        out.append((len(out), page_num, f"s3://{BUCKET}/{key}"))
    return out


def payloads_for_ticket(ticket: dict[str, Any], config: dict) -> list[tuple[str, Path, dict[str, Any]]]:
    short = ticket["short"]
    doc_id = ticket["doc_id"]
    doc_type = ticket["document_type"]
    meta = get_doc_meta(doc_id)
    meta["short"] = short
    profile = os.environ.get("AWS_PROFILE", "prod")
    s3_pages = list_s3_file_pages(meta, profile)
    if not s3_pages:
        s3_pages = [
            (po, meta["start"] + po, "")
            for po in range(meta["end"] - meta["start"] + 1)
        ]
    filled: list[tuple[str, Path, dict[str, Any]]] = []
    for page_offset, page_num, _uri in s3_pages:
        page = f"p{page_offset}"
        path = build_payload(doc_id, meta, page_num, page_offset, config, doc_type)
        data = json.loads(path.read_text())["detail"]["data"]
        filled.append((page, path, data))
    return filled


def process_ticket(ticket: dict[str, Any], env: dict[str, str], profile: str) -> dict[str, Any]:
    short = ticket["short"]
    truth = ticket["ground_truth"]
    doc_type = ticket["document_type"]
    print(f"=== {ticket['ticket']} / {short} ({ticket['partner']}, {doc_type}) ===", flush=True)
    doc_dir = OUT_DIR / short
    doc_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = load_config(doc_type)
    except FileNotFoundError as exc:
        return {
            "ticket": ticket["ticket"],
            "short_id": short,
            "partner": ticket["partner"],
            "doc_id": ticket["doc_id"],
            "document_type": doc_type,
            "ground_truth": truth,
            "error": str(exc),
        }

    best: dict[str, Any] | None = None
    error: str | None = None
    try:
        page_payloads = payloads_for_ticket(ticket, config)
        for page, payload_path, data in page_payloads:
            image_path = doc_dir / f"{short}_{page}.png"
            download_image(data["image_uri"], image_path, profile)
            sections_json = ensure_sections(short, page, payload_path, image_path, profile)
            sections_data = json.loads(sections_json.read_text())
            vin_section = pick_vin_section(sections_data)
            if not vin_section:
                continue
            full_img = Image.open(image_path).convert("RGB")
            bounds = tight_vin_bounds(vin_section, full_img, sections_data)
            crop_img = crop_bounds(image_path, bounds)
            crop_path = doc_dir / f"{short}_{page}_vin_crop.png"
            crop_img.save(crop_path)

            gallery = full_img.copy()
            draw = ImageDraw.Draw(gallery)
            b = bounds
            draw.rectangle([(b["min_x"], b["min_y"]), (b["max_x"], b["max_y"])], outline=(20, 150, 60), width=4)
            region_path = doc_dir / f"{short}_{page}_vin_region.png"
            gallery.save(region_path)
            full_copy = doc_dir / f"{short}_{page}_full.png"
            if not full_copy.exists():
                full_img.save(full_copy)

            full_ext = run_mllm(payload_path, env)
            crop_ext = run_mllm(payload_path, env, local_image=crop_path)
            full_vin = extract_vin(full_ext)
            crop_vin = extract_vin(crop_ext)
            full_ok = norm_vin(full_vin) == norm_vin(truth)
            crop_ok = norm_vin(crop_vin) == norm_vin(truth)
            row = {
                "ticket": ticket["ticket"],
                "ticket_url": ticket.get("url"),
                "jira_url": ticket.get("jira_url"),
                "source": ticket.get("source"),
                "partner": ticket["partner"],
                "document_type": doc_type,
                "verification_doc_type": ticket.get("verification_doc_type"),
                "short_id": short,
                "doc_id": ticket["doc_id"],
                "ground_truth": truth,
                "page": page,
                "full_page_vin": full_vin,
                "crop_vin": crop_vin,
                "full_matches_truth": full_ok,
                "crop_matches_truth": crop_ok,
                "crop_improved_vs_full": crop_ok and not full_ok,
                "crop_regressed_vs_full": full_ok and not crop_ok,
                "vin_section_index": vin_section["index"],
                "crop_size": list(crop_img.size),
                "image_path": str(image_path),
                "crop_path": str(crop_path),
                "region_path": str(region_path),
            }
            if best is None:
                best = row
            elif row["crop_improved_vs_full"] and not best.get("crop_improved_vs_full"):
                best = row
            elif crop_ok and not best.get("crop_matches_truth"):
                best = row
            elif not best.get("full_matches_truth") and (full_ok or crop_ok):
                best = row
            print(
                f"  {page}: full={full_vin!r} crop={crop_vin!r} truth=({full_ok},{crop_ok})",
                flush=True,
            )
    except Exception as exc:
        error = str(exc)
        print(f"  ERROR: {exc}", flush=True)

    if best is None:
        return {
            "ticket": ticket["ticket"],
            "short_id": short,
            "partner": ticket["partner"],
            "doc_id": ticket["doc_id"],
            "document_type": doc_type,
            "ground_truth": truth,
            "error": error or "no VIN section / no pages",
        }
    return best


def crop_outcome(row: dict[str, Any]) -> str:
    full_ok = row.get("full_matches_truth")
    crop_ok = row.get("crop_matches_truth")
    full_vin = row.get("full_page_vin")
    crop_vin = row.get("crop_vin")
    if full_ok:
        return "full_ok"
    if crop_ok:
        return "crop_win"
    if full_vin is None and crop_vin is None:
        return "both_null"
    if crop_vin is None:
        return "crop_null"
    return "crop_wrong"


def material_crop_regression(row: dict[str, Any]) -> bool:
    return bool(
        row.get("full_matches_truth")
        and not row.get("crop_matches_truth")
        and row.get("crop_vin") is not None
    )


def summarize(results: list[dict[str, Any]], *, n_tickets: int) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    full_failed = [r for r in ok if not r.get("full_matches_truth")]
    doc_types = sorted({r.get("document_type") for r in ok if r.get("document_type")})
    by_type: dict[str, int] = {}
    for row in full_failed:
        dt = row.get("document_type") or "unknown"
        by_type[dt] = by_type.get(dt, 0) + 1
    return {
        "n_tickets": n_tickets,
        "n_docs": len(results),
        "n_processed": len(ok),
        "n_errors": sum(1 for r in results if "error" in r),
        "document_types": doc_types,
        "full_matches_truth": sum(1 for r in ok if r.get("full_matches_truth")),
        "crop_matches_truth": sum(1 for r in ok if r.get("crop_matches_truth")),
        "crop_improved_vs_full": sum(1 for r in ok if r.get("crop_improved_vs_full")),
        "crop_null_vs_full_ok": sum(
            1 for r in ok if r.get("full_matches_truth") and not r.get("crop_matches_truth")
        ),
        "material_crop_regressions": sum(1 for r in ok if material_crop_regression(r)),
        "n_full_failed": len(full_failed),
        "crop_wins_on_full_failed": sum(1 for r in full_failed if r.get("crop_improved_vs_full")),
        "full_failed_by_doc_type": by_type,
    }


def render_md(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "## VIN crop A/B — ticket-sourced docs (gemini-3.1-flash-lite, QA prompt per doc type)",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Linear tickets (filtered) | {summary['n_tickets']} |",
        f"| Runnable docs | {summary['n_docs']} |",
        f"| Processed | {summary['n_processed']} |",
        f"| Errors | {summary['n_errors']} |",
        f"| Doc types | {', '.join(summary.get('document_types') or [])} |",
        f"| Full page matches truth | {summary['full_matches_truth']} |",
        f"| Crop matches truth | {summary['crop_matches_truth']} |",
        f"| **Crop wins** (full wrong, crop correct) | **{summary['crop_improved_vs_full']}** |",
        f"| Crop null (section miss; not regression) | {summary['crop_null_vs_full_ok']} |",
        f"| Material crop regressions | {summary['material_crop_regressions']} |",
        "",
        "| Ticket | Doc | Doc type | Partner | Truth | Full VIN | Crop VIN | Page | Crop win? |",
        "|--------|-----|----------|---------|-------|----------|----------|------|-----------|",
    ]
    for r in results:
        if "error" in r:
            lines.append(
                f"| {r['ticket']} | `{r['short_id']}` | {r.get('document_type','')} | {r['partner']} | "
                f"`{r.get('ground_truth','')}` | — | — | — | ERR: {r['error'][:50]} |"
            )
            continue
        win = "✓" if r.get("crop_improved_vs_full") else ("crop null" if crop_outcome(r) == "crop_null" else "—")
        lines.append(
            f"| [{r['ticket']}]({r.get('ticket_url','')}) | `{r['short_id']}` | {r.get('document_type')} | "
            f"{r['partner']} | `{r['ground_truth']}` | `{r.get('full_page_vin')}` | `{r.get('crop_vin')}` | "
            f"{r.get('page')} | {win} |"
        )
    return "\n".join(lines) + "\n"


def render_full_failed_md(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    ok = [r for r in results if "error" not in r]
    full_failed = [r for r in ok if not r.get("full_matches_truth")]
    passed = summary["full_matches_truth"]
    total_ok = len(ok)
    fix_rate = (
        f"{100 * summary['crop_wins_on_full_failed'] / summary['n_full_failed']:.1f}%"
        if summary["n_full_failed"]
        else "n/a"
    )

    lines = [
        "## VIN crop A/B — full-failed only (30-day ticket sweep)",
        "",
        f"{passed}/{total_ok} runnable docs passed full page and are excluded below.",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Linear tickets (filtered) | {summary['n_tickets']} |",
        f"| Runnable docs | {summary['n_docs']} |",
        f"| Skipped at discovery | see skipped.json |",
        f"| **Full-failed docs** | **{summary['n_full_failed']}** |",
        f"| Crop wins on full-failed | {summary['crop_wins_on_full_failed']} |",
        f"| Crop fix rate (full-failed) | {fix_rate} |",
        f"| Material crop regressions | {summary['material_crop_regressions']} |",
        "",
        "### Full-failed by doc type",
        "",
        "| Doc type | Full-failed count |",
        "|----------|-------------------|",
    ]
    for doc_type, count in sorted(summary.get("full_failed_by_doc_type", {}).items()):
        lines.append(f"| {doc_type} | {count} |")
    lines.extend(
        [
            "",
            "### Full-failed docs",
            "",
            "| Ticket | Doc | Doc type | Partner | Truth | Full VIN | Crop VIN | Crop outcome |",
            "|--------|-----|----------|---------|-------|----------|----------|--------------|",
        ]
    )
    for r in full_failed:
        lines.append(
            f"| [{r['ticket']}]({r.get('ticket_url','')}) | `{r['short_id']}` | {r.get('document_type')} | "
            f"{r['partner']} | `{r['ground_truth']}` | `{r.get('full_page_vin')}` | `{r.get('crop_vin')}` | "
            f"{crop_outcome(r)} |"
        )
    if not full_failed:
        lines.append("| — | — | — | — | — | — | — | none |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VIN crop A/B on ticket-sourced docs")
    parser.add_argument("--discover", action="store_true", help="Discover tickets from Linear cache + Looker")
    parser.add_argument("--since-days", type=int, default=30, help="Ticket/doc lookback window")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--max-docs-per-ticket",
        type=int,
        default=3,
        help="Max sample docs to resolve per ticket",
    )
    parser.add_argument(
        "--tickets-file",
        type=Path,
        help="Use prebuilt tickets.json instead of hardcoded list",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only run discovery/resolution; skip LLM A/B",
    )
    return parser.parse_args()


def main() -> None:
    global OUT_DIR
    args = parse_args()
    OUT_DIR = args.out_dir.resolve()
    profile = "prod"

    tickets: list[dict[str, Any]]
    n_linear_tickets = 0
    if args.discover:
        raw, tickets, skipped = discover_and_resolve(
            OUT_DIR,
            since_days=args.since_days,
            max_docs_per_ticket=args.max_docs_per_ticket,
        )
        n_linear_tickets = len(raw)
        print(f"Discovered {n_linear_tickets} Linear tickets -> {len(tickets)} runnable docs", flush=True)
        print(f"Skipped {len(skipped)} entries (see {OUT_DIR / 'skipped.json'})", flush=True)
        if args.discover_only:
            return
    elif args.tickets_file:
        tickets = json.loads(args.tickets_file.read_text()).get("tickets", [])
        summary_path = OUT_DIR / "discovery_summary.json"
        if summary_path.exists():
            n_linear_tickets = json.loads(summary_path.read_text()).get("n_linear_tickets", 0)
        else:
            n_linear_tickets = len({t["ticket"] for t in tickets})
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "tickets.json").write_text(json.dumps({"tickets": tickets}, indent=2))
    else:
        tickets = TICKETS
        n_linear_tickets = len({t["ticket"] for t in tickets})
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "tickets.json").write_text(json.dumps({"tickets": tickets}, indent=2))

    env = refresh_aws_env(copy.copy(os.environ))
    env["SKIP_LLM_CACHE"] = "1"
    env["AWS_PROFILE"] = profile
    env.pop("BUNDLE_PATH", None)

    log_path = OUT_DIR / "run.log"
    log_fh = log_path.open("w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    results: list[dict[str, Any]] = []
    for i, ticket in enumerate(tickets, 1):
        log(f"[{i}/{len(tickets)}] {ticket['ticket']} / {ticket['short']}")
        results.append(process_ticket(ticket, env, profile))

    summary = summarize(results, n_tickets=n_linear_tickets)
    out = {"summary": summary, "results": results}
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "analysis.md").write_text(render_md(results, summary))
    (OUT_DIR / "full_failed_analysis.md").write_text(render_full_failed_md(results, summary))
    log_fh.close()

    print("\n" + render_full_failed_md(results, summary))
    print(f"Wrote {OUT_DIR / 'results.json'}")
    print(f"Wrote {OUT_DIR / 'full_failed_analysis.md'}")


if __name__ == "__main__":
    main()
