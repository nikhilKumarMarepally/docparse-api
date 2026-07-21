#!/usr/bin/env python3
"""WA-513 V2: 500-doc title_application cross-partner A/B (baseline S3 vs NEW QA prompt)."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import yaml

LOOKER_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
LLM_CONFIG = Path(sys.argv[2]) if len(sys.argv) > 2 else None
OUT_FILE = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("wa513_v2_results.json")
MAX_WORKERS = int(os.environ.get("V2_WORKERS", "6"))
FIELDS = [
    "odometer",
    "sale_price",
    "elt_code",
    "applicants.applicant1.address.zip",
    "applicants.applicant2.address.zip",
]

ddb = boto3.Session(profile_name="prod").client("dynamodb", region_name="us-west-2")
s3 = boto3.Session(profile_name="prod").client("s3")
BUCKET = "informed-techno-core-prod-exchange"
MLLM_DIR = Path(__file__).resolve().parent.parent
PAYLOAD_DIR = MLLM_DIR / "payloads" / "wa513_v2"
PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)


def value_present(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and not v.strip():
        return False
    if isinstance(v, dict) and not any(value_present(x) for x in v.values()):
        return False
    return True


def deep_merge_first_wins(base, overlay):
    if overlay is None:
        return base
    if base is None:
        return copy.deepcopy(overlay)
    if isinstance(base, dict) and isinstance(overlay, dict):
        result = copy.deepcopy(base)
        for key, val in overlay.items():
            if key in result:
                result[key] = deep_merge_first_wins(result[key], val)
            else:
                result[key] = val
        return result
    return base if value_present(base) else overlay


def get_nested(d: dict, key: str):
    val = d
    for part in key.split("."):
        if not isinstance(val, dict):
            return None
        val = val.get(part)
    return val


def norm(field, val):
    if val is None:
        return None
    if field.endswith(".zip"):
        s = re.sub(r"[^0-9]", "", str(val))
        return s[:5] if len(s) >= 5 else s or None
    if field in ("odometer", "year"):
        s = re.sub(r"[^0-9.]", "", str(val))
        return s or None
    if field == "sale_price":
        s = re.sub(r"[^0-9.]", "", str(val))
        return s or None
    s = str(val).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s or None


def parse_looker(path: Path):
    docs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        docs.append(
            {
                "doc_id": row["documents.document_id"],
                "partner_name": row.get("documents.partner_name"),
            }
        )
    return docs


def get_doc_meta(doc_id: str):
    item = ddb.get_item(
        TableName="techno-core-prod-document-orchestrator",
        Key={"PK": {"S": doc_id}, "SK": {"S": "document"}},
        ProjectionExpression="partner_id, application_id, file_ids, parent_partition_params, document_type",
    )["Item"]
    pages = item["parent_partition_params"]["M"]["pages"]["L"][0]["M"]
    return {
        "partner_id": item["partner_id"]["S"],
        "app_id": item["application_id"]["S"],
        "file_id": item["file_ids"]["L"][0]["S"],
        "start": int(pages["start_page"]["N"]),
        "end": int(pages["end_page"]["N"]),
    }


def fetch_baseline(partner_id, app_id, doc_id):
    key = f"{partner_id}/{app_id}/raw_extracted_data_mllm/{doc_id}.json"
    try:
        data = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except Exception:
        return {}
    if isinstance(data, list):
        merged = None
        for page in data:
            if not isinstance(page, dict):
                continue
            clean = {k: v for k, v in page.items() if not str(k).startswith("_") and v is not None}
            merged = deep_merge_first_wins(merged, clean) if merged else copy.deepcopy(clean)
        return merged or {}
    return data if isinstance(data, dict) else {}


def run_mllm_page(doc_id, meta, page_num, page_offset, config):
    payload_path = PAYLOAD_DIR / f"{doc_id}_p{page_offset}.json"
    base = f"s3://{BUCKET}/{meta['partner_id']}/{meta['app_id']}"
    payload = {
        "detail": {
            "metadata": {},
            "data": {
                "partner_id": meta["partner_id"],
                "application_id": meta["app_id"],
                "application_data_uri": f"{base}/app_context/{meta['file_id']}.json",
                "document_id": doc_id,
                "document_type": "title_application",
                "image_uri": f"{base}/file/{meta['file_id']}/img/{meta['file_id']}-{page_num}.png",
            },
        },
        "config": config,
    }
    payload_path.write_text(json.dumps(payload))
    env = os.environ.copy()
    env["SKIP_LLM_CACHE"] = "1"
    env.pop("BUNDLE_PATH", None)
    proc = subprocess.run(
        [str(MLLM_DIR / "bin" / "mllm_test"), "--payload", str(payload_path)],
        cwd=str(MLLM_DIR),
        capture_output=True,
        text=True,
        env={**env, "AWS_PROFILE": "prod"},
    )
    out = proc.stdout
    start_idx = out.rfind("\n{")
    if start_idx < 0:
        start_idx = out.find("{")
    else:
        start_idx += 1
    if start_idx < 0:
        return {}
    return json.loads(out[start_idx:])


def merge_pages(page_results):
    merged = None
    for page in page_results:
        clean = {k: v for k, v in page.items() if not str(k).startswith("_") and v is not None}
        merged = deep_merge_first_wins(merged, clean) if merged else copy.deepcopy(clean)
    return merged or {}


def process_doc(doc, config):
    doc_id = doc["doc_id"]
    try:
        meta = get_doc_meta(doc_id)
        page_results = []
        for po in range(meta["end"] - meta["start"] + 1):
            page_num = meta["start"] + po
            page_results.append(run_mllm_page(doc_id, meta, page_num, po, config))
        new_data = merge_pages(page_results)
        base_data = fetch_baseline(meta["partner_id"], meta["app_id"], doc_id)
        row = {"doc_id": doc_id, "partner": doc.get("partner_name"), "fields": {}}
        for field in FIELDS:
            b = norm(field, get_nested(base_data, field) if "." in field else base_data.get(field))
            n = norm(field, get_nested(new_data, field) if "." in field else new_data.get(field))
            row["fields"][field] = {
                "baseline": b,
                "new": n,
                "agree": b == n,
                "baseline_only": b is not None and n is None,
                "new_only": b is None and n is not None,
                "disagree": b is not None and n is not None and b != n,
            }
        return row
    except Exception as e:
        return {"doc_id": doc_id, "error": str(e)}


def summarize(results):
    summary = {
        f: {"baseline_nn": 0, "new_nn": 0, "agree": 0, "new_only": 0, "baseline_only": 0, "disagree": 0}
        for f in FIELDS
    }
    for r in results:
        if "error" in r:
            continue
        for f, stats in r["fields"].items():
            if stats["baseline"] is not None:
                summary[f]["baseline_nn"] += 1
            if stats["new"] is not None:
                summary[f]["new_nn"] += 1
            if stats["agree"] and (stats["baseline"] is not None or stats["new"] is not None):
                summary[f]["agree"] += 1
            if stats["new_only"]:
                summary[f]["new_only"] += 1
            if stats["baseline_only"]:
                summary[f]["baseline_only"] += 1
            if stats["disagree"]:
                summary[f]["disagree"] += 1
    return summary


def main():
    if not LOOKER_FILE or not LLM_CONFIG:
        print("usage: wa513_v2_compare.py <looker.jsonl> <title_application.yml> [out.json]", file=sys.stderr)
        sys.exit(1)
    docs = parse_looker(LOOKER_FILE)
    config = yaml.safe_load(LLM_CONFIG.read_text())
    print(f"Processing {len(docs)} docs with {MAX_WORKERS} workers...", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_doc, d, config): d for d in docs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            results.append(fut.result())
            if done % 25 == 0:
                print(f"  {done}/{len(docs)}", flush=True)
    summary = summarize(results)
    OUT_FILE.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
