#!/usr/bin/env python3
"""VIN crop A/B on title_application docs sourced from Linear/Jira VIN tickets."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import boto3
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
MLLM = ROOT / "topologies/extraction/shared/mllm-invoker/bin/mllm_test"
MLLM_LOCAL = ROOT / "topologies/extraction/shared/mllm-invoker/bin/mllm_test_local_image.rb"
MLLM_DIR = MLLM.parent.parent
PAYLOAD_DIRS = [
    ROOT / "topologies/extraction/shared/mllm-invoker/payloads/wa577_v2",
    ROOT / "topologies/extraction/shared/mllm-invoker/payloads/wa513_577_v1",
]
QA_YAML = (
    ROOT.parent
    / "techno-configs/techno_configs/envs/qa/document_fields/extractions/llm_configs/title_application.yml"
)
SCHEMA_JSON = (
    ROOT.parent
    / "techno-configs/techno_configs/envs/qa/document_fields/serialization/v1/title_application.json"
)
OUT_DIR = ROOT / "wa577_gallery/vin_crop_tickets"
BUCKET = "informed-techno-core-prod-exchange"

# One doc per ticket — title_application VIN failures with retrievable doc IDs.
TICKETS: list[dict[str, Any]] = [
    {
        "ticket": "WA-577",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-577",
        "partner": "penair",
        "short": "87c842be",
        "doc_id": "87c842be-0578-452a-a9ba-99aefa0e0a4d",
        "ground_truth": "1FMEE8BH0TLA94758",
        "issue": "digit transposition 94758→97458",
    },
    {
        "ticket": "WA-577",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-577",
        "partner": "desertfinancial",
        "short": "4aeb9d2c",
        "doc_id": "4aeb9d2c-0328-4699-acc1-5993630c1258",
        "ground_truth": "1ft8w2bt5tee77678",
        "issue": "null VIN — fragmented/checkbox layout",
    },
    {
        "ticket": "WA-640",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-640",
        "partner": "cuofco",
        "short": "899d7cbd",
        "doc_id": "899d7cbd-ba36-46f2-88d3-8ef79282365f",
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
        "ground_truth": "1ftfw3ld4rfa81006",
        "issue": "b→8 OCR char swap",
    },
    {
        "ticket": "WA-489",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-489",
        "partner": "consumerscu",
        "short": "e9300398",
        "doc_id": "e9300398-823b-4acd-aba2-87263424da42",
        "ground_truth": "1gys4jkjxhr193767",
        "issue": "k↔j character transposition",
    },
    {
        "ticket": "WA-411",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-411",
        "partner": "autonationfinance",
        "short": "2565927a",
        "doc_id": "2565927a-ca67-4a8f-bb18-08ff5a45c9d9",
        "ground_truth": "JTND4MBE4P3206996",
        "issue": "leading 1 vs J misread",
    },
    {
        "ticket": "WA-411",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-411",
        "partner": "autonationfinance",
        "short": "5c09561a",
        "doc_id": "5c09561a-bde7-49e1-9574-40ad523f181e",
        "ground_truth": "KM8RKESA8TU067054",
        "issue": "B→8 single char error",
    },
    {
        "ticket": "WA-17",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-17",
        "partner": "mountainamericacu",
        "short": "4332254a",
        "doc_id": "4332254a-9061-47e8-aa19-825b1af3214f",
        "ground_truth": "5yjygdee4mf195635",
        "issue": "j/y character swap",
    },
    {
        "ticket": "WA-231",
        "source": "linear",
        "url": "https://linear.app/informediq/issue/WA-231",
        "partner": "desertfinancial",
        "short": "c26302a9",
        "doc_id": "c26302a9-ef68-46db-b815-ea7bffe17cc2",
        "ground_truth": "kmhrc8a36su406233",
        "issue": "RouteOne Decision Detail VIN garbling",
    },
    {
        "ticket": "LIV-545",
        "source": "jira",
        "url": "https://linear.app/informediq/issue/LIV-545",
        "jira_url": "https://informed-iq.atlassian.net/browse/LIV-545",
        "partner": "consumerscu",
        "short": "b1bc95a6",
        "doc_id": "b1bc95a6-fde4-4e29-9f73-2151b2c4004a",
        "ground_truth": "5lmpj8ja7tj049186",
        "issue": "char transposition (WA-489 triage fixable)",
        "note": "Week 20 QC batch; VIN case from linked WA-489 cluster",
    },
]

from wa577_vin_crop_helpers import (
    VIN_LABEL,
    VIN_TOKEN,
    pick_vin_section,
    score_vin_section,
    tight_vin_bounds,
)

_ddb = None
_s3 = None


def norm_vin(v: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(v or "")).upper()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(QA_YAML.read_text())
    schema_body = json.loads(SCHEMA_JSON.read_text())["definitions"]["extracted_data"]
    intro = config["model_info"]["payload_config"]["prompt_config"]["intro"]
    config["model_info"]["payload_config"]["prompt_config"]["intro"] = intro.replace(
        "$$_SCHEMA", json.dumps(schema_body)
    )
    return config


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
    return {
        "partner_id": item["partner_id"]["S"],
        "app_id": item["application_id"]["S"],
        "file_id": item["file_ids"]["L"][0]["S"],
        "start": int(pages["start_page"]["N"]),
        "end": int(pages["end_page"]["N"]),
    }


def find_payloads(short: str) -> list[Path]:
    found: list[Path] = []
    for d in PAYLOAD_DIRS:
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


def build_payload(doc_id: str, meta: dict[str, Any], page_num: int, page_offset: int, config: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload_path = OUT_DIR / meta["short"] / f"{meta['short']}_{page_offset}_payload.json"
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
                "document_type": "title_application",
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


def payloads_for_ticket(ticket: dict[str, Any], config: dict) -> list[tuple[str, Path, dict[str, Any]]]:
    short = ticket["short"]
    doc_id = ticket["doc_id"]
    existing = find_payloads(short)
    if existing:
        out = []
        for p in existing:
            page = p.stem.rsplit("_", 1)[-1]
            base = json.loads(p.read_text())
            base["config"] = copy.deepcopy(config)
            tmp = OUT_DIR / short / f"payload_{page}.json"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(base))
            out.append((page, tmp, base["detail"]["data"]))
        return out
    meta = get_doc_meta(doc_id)
    meta["short"] = short
    built = []
    for po in range(meta["end"] - meta["start"] + 1):
        page_num = meta["start"] + po
        page = f"p{po}"
        built.append((page, build_payload(doc_id, meta, page_num, po, config), {}))
    # fill data from built payload
    filled = []
    for page, path, _ in built:
        data = json.loads(path.read_text())["detail"]["data"]
        filled.append((page, path, data))
    return filled


def process_ticket(ticket: dict[str, Any], config: dict, env: dict[str, str], profile: str) -> dict[str, Any]:
    short = ticket["short"]
    truth = ticket["ground_truth"]
    print(f"=== {ticket['ticket']} / {short} ({ticket['partner']}) ===", flush=True)
    doc_dir = OUT_DIR / short
    doc_dir.mkdir(parents=True, exist_ok=True)

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
            gallery.save(doc_dir / f"{short}_{page}_vin_region.png")

            full_vin = run_mllm(payload_path, env).get("vin")
            crop_vin = run_mllm(payload_path, env, local_image=crop_path).get("vin")
            full_ok = norm_vin(full_vin) == norm_vin(truth)
            crop_ok = norm_vin(crop_vin) == norm_vin(truth)
            row = {
                "ticket": ticket["ticket"],
                "ticket_url": ticket.get("url"),
                "source": ticket.get("source"),
                "partner": ticket["partner"],
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
                "region_path": str(doc_dir / f"{short}_{page}_vin_region.png"),
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
                f"  {page}: full={full_vin!r} crop={crop_vin!r} "
                f"truth=({full_ok},{crop_ok})",
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
            "ground_truth": truth,
            "error": error or "no VIN section / no pages",
        }
    return best


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    return {
        "n_tickets": len(TICKETS),
        "n_processed": len(ok),
        "n_errors": sum(1 for r in results if "error" in r),
        "full_matches_truth": sum(1 for r in ok if r.get("full_matches_truth")),
        "crop_matches_truth": sum(1 for r in ok if r.get("crop_matches_truth")),
        "crop_improved_vs_full": sum(1 for r in ok if r.get("crop_improved_vs_full")),
        "crop_regressed_vs_full": sum(1 for r in ok if r.get("crop_regressed_vs_full")),
    }


def render_md(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "## VIN crop A/B — ticket-sourced docs (gemini-3.1-flash-lite, QA prompt)",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Tickets / docs | {summary['n_tickets']} |",
        f"| Processed | {summary['n_processed']} |",
        f"| Errors | {summary['n_errors']} |",
        f"| Full page matches truth | {summary['full_matches_truth']} |",
        f"| Crop matches truth | {summary['crop_matches_truth']} |",
        f"| **Crop wins** (full wrong, crop correct) | **{summary['crop_improved_vs_full']}** |",
        f"| Crop regressions | {summary['crop_regressed_vs_full']} |",
        "",
        "| Ticket | Doc | Partner | Truth | Full VIN | Crop VIN | Page | Crop win? |",
        "|--------|-----|---------|-------|----------|----------|------|-----------|",
    ]
    for r in results:
        if "error" in r:
            lines.append(
                f"| {r['ticket']} | `{r['short_id']}` | {r['partner']} | `{r.get('ground_truth','')}` | — | — | — | ERR: {r['error'][:40]} |"
            )
            continue
        win = "✓" if r.get("crop_improved_vs_full") else (
            "✗" if r.get("crop_regressed_vs_full") else "—"
        )
        lines.append(
            f"| [{r['ticket']}]({r.get('ticket_url','')}) | `{r['short_id']}` | {r['partner']} | "
            f"`{r['ground_truth']}` | `{r.get('full_page_vin')}` | `{r.get('crop_vin')}` | {r.get('page')} | {win} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    profile = "prod"
    env = refresh_aws_env(copy.copy(os.environ))
    env["SKIP_LLM_CACHE"] = "1"
    env["AWS_PROFILE"] = profile
    env.pop("BUNDLE_PATH", None)
    config = load_config()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tickets.json").write_text(json.dumps({"tickets": TICKETS}, indent=2))

    results = [process_ticket(t, config, env, profile) for t in TICKETS]
    summary = summarize(results)
    out = {"summary": summary, "results": results}
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "analysis.md").write_text(render_md(results, summary))
    print("\n" + render_md(results, summary))
    print(f"Wrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
