#!/usr/bin/env python3
"""Crop-only VIN extraction on Linear partner-closed / unsolvable ticket docs.

Uses prod S3 baseline (or ticket triage MLLM value) for full_page_vin — never reruns full-page MLLM.
Crop path: section bounds → enhance_vin_crop → MLLM (same as full_ne_crop_disagreements).
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import boto3
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import wa577_vin_full_ne_crop_disagreements as disag
import wa577_vin_ticket_crop_ab as tab
from wa577_vin_crop_helpers import pick_vin_section, tight_vin_bounds
from wa577_vin_ticket_discover import get_doc_meta, norm_vin, query_looker_vin_rows

OUT_DIR = ROOT / "wa577_gallery/vin_linear_partner_closed"
TICKET_AB_DIR = ROOT / "wa577_gallery/vin_ticket_crop_ab"
DESKTOP_HTML = Path.home() / "Desktop/vin_linear_partner_closed.html"
DESKTOP_ASSETS = Path.home() / "Desktop/vin_linear_partner_closed_assets"
ASSETS_PREFIX = "vin_linear_partner_closed_assets"
BUCKET = "informed-techno-core-prod-exchange"
QA_ROOT = ROOT.parent / "techno-configs/techno_configs/envs/qa/document_fields"

# Linear tickets closed/unsolvable with partner-side or low-fix-rate closure; extraction was wrong on samples.
TICKET_META: list[dict[str, Any]] = [
    {
        "ticket": "WA-153",
        "url": "https://linear.app/informediq/issue/WA-153",
        "assignee": "Nikhil Marepally",
        "status": "Canceled",
        "closure_reason": "unsolvable — prompt fix fixed <1% of issues (Nikhil comment 2026-06-11)",
        "partner": "stellantis",
        "document_type": "buyers_order",
        "verification_doc_type": "order_form",
        "question_code": "matches_contract_vin",
        "notes": "Triage: 72% B1 extraction-wrong; 13% B2 partner-decision. Samples below are B1 OCR/MLLM misreads.",
    },
    {
        "ticket": "WA-244",
        "url": "https://linear.app/informediq/issue/WA-244",
        "assignee": "Nishit Kumar",
        "status": "Done",
        "closure_reason": "unsolvable — no ground truth at extraction time (partner API null VIN, siblings parallel)",
        "partner": "suncoast",
        "document_type": "documentary_draft",
        "question_code": "vin",
        "notes": "Extraction OCR confusion (5/S, 0/D); closed as not worth pursuing. Docs from Looker fail/review.",
    },
]

# WA-153 triage sample docs (B1 — extraction wrong). prod_mllm from ticket investigation.
WA153_DOCS: list[dict[str, Any]] = [
    {
        "doc_id": "07bf3075-acda-46ff-b014-63bd6179020f",
        "ground_truth": "1c6pjtag9tl182140",
        "prod_mllm": "cc6pjtag9tl182140",
    },
    {
        "doc_id": "3f23ba5f-38bf-4a99-962a-893ba2874c52",
        "ground_truth": "3c7wrnal5tg231422",
        "prod_mllm": "307wrnal5tc231422",
    },
    {
        "doc_id": "0248f362-38f2-4479-9da5-f7347e2c8fc1",
        "ground_truth": "1c4rdjdg7tc222899",
        "prod_mllm": "1c4bd1067tc222899",
    },
    {
        "doc_id": "04ecf650-383d-4e8b-b35e-f0b7b867a2bc",
        "ground_truth": "1c4rjkbrxt8564840",
        "prod_mllm": "1c4ajkbrxt78564840",
    },
    {
        "doc_id": "073d17f5-53ca-4b73-a555-b70b0b731f43",
        "ground_truth": "jm3kfbdm5k0616366",
        "prod_mllm": "jm3kfrdm5k0616365",
    },
    {
        "doc_id": "08a39be0-b26e-4b72-b57d-cdca64e52816",
        "ground_truth": "1c4pjxfg1tw258439",
        "prod_mllm": None,
    },
]

_s3 = None


def load_config(doc_type: str) -> dict[str, Any]:
    yaml_path = QA_ROOT / f"extractions/llm_configs/{doc_type}.yml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"no QA llm_config for {doc_type}")
    config = yaml.safe_load(yaml_path.read_text())
    payload_cfg = config["model_info"]["payload_config"]
    payload_type = payload_cfg.get("type")
    if payload_type != "custom":
        return config
    schema_path = QA_ROOT / f"serialization/v1/{doc_type}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"no QA schema for {doc_type}")
    schema_body = json.loads(schema_path.read_text())["definitions"]["extracted_data"]
    schema_json = json.dumps(schema_body)
    if "prompt_config" in payload_cfg and "intro" in payload_cfg["prompt_config"]:
        intro = payload_cfg["prompt_config"]["intro"]
        payload_cfg["prompt_config"]["intro"] = intro.replace("$$_SCHEMA", schema_json)
    elif "prompt" in payload_cfg:
        payload_cfg["prompt"] = payload_cfg["prompt"].replace("$$_SCHEMA", schema_json)
    return config


def init_s3():
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


def looker_wa244_docs(limit: int = 10) -> list[dict[str, Any]]:
    rows = query_looker_vin_rows(
        since_days=90,
        partner="suncoast",
        verification_doc_type="documentary_draft",
        limit=50,
    )
    docs: list[dict[str, Any]] = []
    for row in rows:
        doc_id = row.get("questions_original.document_id")
        expected = row.get("questions_original.expected")
        answer = row.get("questions_original.answer")
        if not doc_id or not expected:
            continue
        gt = norm_vin(expected)
        ans = norm_vin(answer) if answer else None
        if ans == gt:
            continue
        docs.append({"doc_id": doc_id, "ground_truth": gt, "prod_mllm": ans})
        if len(docs) >= limit:
            break
    return docs


def build_ticket_docs(max_wa244: int = 10) -> list[dict[str, Any]]:
    tickets: list[dict[str, Any]] = []
    for spec in WA153_DOCS:
        doc_id = spec["doc_id"]
        short = doc_id.split("-")[0]
        meta = get_doc_meta(doc_id)
        tickets.append(
            {
                "ticket": "WA-153",
                "ticket_url": "https://linear.app/informediq/issue/WA-153",
                "partner": "stellantis",
                "partner_id": meta["partner_id"],
                "short": short,
                "doc_id": doc_id,
                "document_type": meta["document_type"] or "buyers_order",
                "ground_truth": norm_vin(spec["ground_truth"]),
                "prod_mllm_hint": spec.get("prod_mllm"),
                "closure_reason": "unsolvable (<1% fix rate)",
            }
        )
    for spec in looker_wa244_docs(limit=max_wa244):
        doc_id = spec["doc_id"]
        short = doc_id.split("-")[0]
        meta = get_doc_meta(doc_id)
        tickets.append(
            {
                "ticket": "WA-244",
                "ticket_url": "https://linear.app/informediq/issue/WA-244",
                "partner": "suncoast",
                "partner_id": meta["partner_id"],
                "short": short,
                "doc_id": doc_id,
                "document_type": meta["document_type"] or "documentary_draft",
                "ground_truth": spec["ground_truth"],
                "prod_mllm_hint": spec.get("prod_mllm"),
                "closure_reason": "unsolvable (no app-context VIN at extraction time)",
            }
        )
    return tickets


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
        return hv, "ticket_triage_mllm"

    return None, "missing"


def process_doc(ticket: dict[str, Any], env: dict[str, str], profile: str) -> dict[str, Any]:
    short = ticket["short"]
    truth = ticket["ground_truth"]
    doc_type = ticket["document_type"]
    doc_dir = OUT_DIR / short
    doc_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "ticket": ticket["ticket"],
        "ticket_url": ticket.get("ticket_url"),
        "closure_reason": ticket.get("closure_reason"),
        "partner": ticket["partner"],
        "short_id": short,
        "doc_id": ticket["doc_id"],
        "document_type": doc_type,
        "ground_truth": truth,
        "full_page_vin": None,
        "crop_vin_enhanced": None,
        "full_page_source": None,
        "crop_fixes": False,
        "error": None,
    }

    try:
        config = load_config(doc_type)
    except FileNotFoundError as exc:
        result["error"] = str(exc)
        return result

    try:
        meta = get_doc_meta(ticket["doc_id"])
        page_payloads = tab.payloads_for_ticket(
            {
                "doc_id": ticket["doc_id"],
                "short": short,
                "document_type": doc_type,
                "partner": ticket["partner"],
            },
            config,
        )
        best: dict[str, Any] | None = None
        for page, payload_path, data in page_payloads:
            image_path = doc_dir / f"{short}_{page}.png"
            tab.download_image(data["image_uri"], image_path, profile)
            sections_json = tab.ensure_sections(short, page, payload_path, image_path, profile)
            sections_data = json.loads(sections_json.read_text())
            vin_section = pick_vin_section(sections_data)
            if not vin_section:
                continue

            full_vin, full_src = resolve_full_vin(ticket, meta, doc_dir, short, page)
            full_img = Image.open(image_path).convert("RGB")
            bounds = tight_vin_bounds(vin_section, full_img, sections_data)
            section_bounds = vin_section["bounds"]
            section_crop_img = tab.crop_bounds(image_path, section_bounds)
            section_crop_path = doc_dir / f"{short}_{page}_section_crop.png"
            section_crop_img.save(section_crop_path)

            crop_vin, enhanced_path, _ = disag._run_enhanced_arm(
                payload_path, env, section_crop_img, doc_dir, short, page
            )
            crop_vin_norm = norm_vin(crop_vin) if crop_vin else None
            full_ok = norm_vin(full_vin) == norm_vin(truth) if full_vin else False
            crop_ok = crop_vin_norm == norm_vin(truth) if crop_vin_norm else False

            row = dict(result)
            row.update(
                {
                    "page": page,
                    "full_page_vin": full_vin,
                    "full_page_source": full_src,
                    "crop_vin_enhanced": crop_vin,
                    "full_matches_truth": full_ok,
                    "crop_matches_truth": crop_ok,
                    "crop_fixes": crop_ok and not full_ok,
                    "enhanced_crop_path": enhanced_path,
                    "section_crop_path": str(section_crop_path),
                    "image_path": str(image_path),
                }
            )
            if best is None or (row["crop_fixes"] and not best.get("crop_fixes")):
                best = row
            elif crop_ok and not best.get("crop_matches_truth"):
                best = row

            print(
                f"  {short} {page}: full={full_vin!r} crop={crop_vin!r} "
                f"gt={truth!r} fixes={row['crop_fixes']}",
                flush=True,
            )

        return best or {**result, "error": "no VIN section"}
    except Exception as exc:
        result["error"] = str(exc)
        return result


def vin_display(v: Any) -> str:
    return "null" if v is None else str(v)


def resolve_result_paths(row: dict[str, Any]) -> dict[str, Any]:
    """Resolve image/crop paths from results.json or gallery dirs."""
    r = dict(row)
    short = r.get("short_id") or (r.get("doc_id") or "")[:8]
    page = r.get("page", "p0")
    doc_dir = OUT_DIR / short

    for key in ("image_path", "enhanced_crop_path", "section_crop_path", "sections_overlay_path"):
        p = r.get(key)
        if p and Path(p).exists():
            r[key] = str(Path(p).resolve())

    if not r.get("image_path"):
        for candidate in (doc_dir / f"{short}_{page}.png", doc_dir / f"{short}_p0.png"):
            if candidate.exists():
                r["image_path"] = str(candidate.resolve())
                break

    if not r.get("enhanced_crop_path"):
        for candidate in (
            doc_dir / f"{short}_{page}_vin_crop_enhanced.png",
            doc_dir / f"{short}_p0_vin_crop_enhanced.png",
        ):
            if candidate.exists():
                r["enhanced_crop_path"] = str(candidate.resolve())
                break

    return r


def find_sections_json_for_row(short: str, page: str, image_path: Path) -> Path | None:
    """Sections JSON from partner-closed dir or vin_ticket_crop_ab fallback."""
    found = disag.find_sections_json(image_path)
    if found:
        return found
    for base in (OUT_DIR / short, TICKET_AB_DIR / short):
        candidate = base / f"{short}_{page}_sections" / f"{short}_{page}_sections.json"
        if candidate.exists():
            return candidate
    return None


def prepare_gallery_rows(results: list[dict[str, Any]], profile: str) -> list[dict[str, Any]]:
    """Ensure sections overlay paths for HTML gallery (no MLLM)."""
    rows = [resolve_result_paths(r) for r in results]
    overlay_rows = [r for r in rows if not r.get("error") and r.get("image_path")]
    for r in overlay_rows:
        image_path = Path(r["image_path"])
        short, page = r["short_id"], r.get("page", "p0")
        sections_json = find_sections_json_for_row(short, page, image_path)
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
            short,
            page,
            image_path,
            sections_data,
            picked if isinstance(picked, int) else None,
        )
        # Copy overlay into partner-closed doc dir for stable paths
        dest = OUT_DIR / short / f"{short}_{page}_sections.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if overlay_path != dest:
            shutil.copy2(overlay_path, dest)
        r["sections_overlay_path"] = str(dest.resolve())
        r["picked_section"] = picked
    return rows


def write_desktop_assets(rows: list[dict[str, Any]]) -> None:
    if DESKTOP_ASSETS.exists():
        shutil.rmtree(DESKTOP_ASSETS)
    DESKTOP_ASSETS.mkdir(parents=True)
    for r in rows:
        short = r["short_id"]
        if r.get("image_path") and Path(r["image_path"]).exists():
            disag.thumb_copy(Path(r["image_path"]), DESKTOP_ASSETS / f"{short}_full.png")
        if r.get("sections_overlay_path") and Path(r["sections_overlay_path"]).exists():
            disag.thumb_copy(
                Path(r["sections_overlay_path"]),
                DESKTOP_ASSETS / f"{short}_sections.png",
            )
        if r.get("enhanced_crop_path") and Path(r["enhanced_crop_path"]).exists():
            disag.thumb_copy(
                Path(r["enhanced_crop_path"]),
                DESKTOP_ASSETS / f"{short}_enhanced_crop.png",
                max_w=520,
            )


def render_html(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    ticket_meta: list[dict[str, Any]],
) -> str:
    meta_by_ticket = {m["ticket"]: m for m in ticket_meta}
    table_rows: list[str] = []
    cards: list[str] = []

    for r in results:
        short = html.escape(r["short_id"])
        ticket = html.escape(r.get("ticket") or "—")
        ticket_url = html.escape(r.get("ticket_url") or "#")
        dtype = html.escape(r.get("document_type") or "")
        partner = html.escape(str(r.get("partner") or "—"))
        full_v = html.escape(vin_display(r.get("full_page_vin")))
        crop_v = html.escape(vin_display(r.get("crop_vin_enhanced")))
        gt_v = html.escape(vin_display(r.get("ground_truth")))
        page = html.escape(r.get("page") or "—")
        err = r.get("error")

        if err:
            fix_label = "—"
            row_class = "muted"
        elif r.get("crop_fixes"):
            fix_label = "yes"
            row_class = "winner-crop"
        elif r.get("crop_matches_truth"):
            fix_label = "crop ok"
            row_class = "winner-crop"
        elif r.get("full_matches_truth"):
            fix_label = "full ok"
            row_class = "winner-full"
        else:
            fix_label = "no"
            row_class = "winner-both"

        table_rows.append(
            f"""      <tr>
        <td><a href="{ticket_url}">{ticket}</a></td>
        <td><code>{short}</code></td>
        <td>{partner}</td>
        <td class="vin-full"><code>{full_v}</code></td>
        <td class="vin-crop"><code>{crop_v}</code></td>
        <td class="vin-gt"><code>{gt_v}</code></td>
        <td class="{row_class}">{fix_label}</td>
      </tr>"""
        )

        if err:
            cards.append(
                f"""  <div class="doc-card error-card">
    <h2><a href="{ticket_url}">{ticket}</a> · <code>{short}</code> · {dtype} · {partner}</h2>
    <p class="muted">GT <code class="vin-gt">{gt_v}</code> · <span class="winner-both">ERR: {html.escape(str(err)[:120])}</span></p>
  </div>"""
            )
            continue

        sections_block = (
            f'<img src="{ASSETS_PREFIX}/{short}_sections.png" alt="sections overlay">'
            if r.get("sections_overlay_path") and Path(r["sections_overlay_path"]).exists()
            else '<p class="muted">No sections overlay</p>'
        )
        full_block = (
            f'<img src="{ASSETS_PREFIX}/{short}_full.png" alt="full page">'
            if r.get("image_path") and Path(r["image_path"]).exists()
            else '<p class="muted">No full page image</p>'
        )
        enhanced_block = (
            f'<img src="{ASSETS_PREFIX}/{short}_enhanced_crop.png" alt="enhanced crop">'
            if r.get("enhanced_crop_path") and Path(r["enhanced_crop_path"]).exists()
            else '<p class="muted">No enhanced crop</p>'
        )
        closure = html.escape((r.get("closure_reason") or "")[:80])
        full_src = html.escape(r.get("full_page_source") or "—")
        picked = r.get("picked_section")
        picked_note = f" · picked S{picked}" if picked is not None else ""

        cards.append(
            f"""  <div class="doc-card">
    <h2><a href="{ticket_url}">{ticket}</a> · <code>{short}</code> · {dtype} · {partner}</h2>
    <p class="muted">{closure}</p>
    <p>Page <code>{page}</code>{picked_note} · Full/prod <code class="vin-full">{full_v}</code> ({full_src}) · Crop <code class="vin-crop">{crop_v}</code> · GT <code class="vin-gt">{gt_v}</code> · <span class="{row_class}">{fix_label}</span></p>
    <div class="images images-3">
      <div class="img-block">
        <label>Full page</label>
        {full_block}
      </div>
      <div class="img-block">
        <label>Sections overlay</label>
        {sections_block}
      </div>
      <div class="img-block">
        <label>Enhanced crop</label>
        {enhanced_block}
      </div>
    </div>
  </div>"""
        )

    ticket_blurbs = []
    for m in ticket_meta:
        t = html.escape(m["ticket"])
        url = html.escape(m.get("url") or "#")
        reason = html.escape(m.get("closure_reason", ""))
        notes = html.escape(m.get("notes", ""))
        ticket_blurbs.append(
            f'<li><a href="{url}"><strong>{t}</strong></a> ({html.escape(m.get("partner", ""))}): '
            f"{reason}. {notes}</li>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Partner-closed VIN tickets — crop gallery</title>
  <style>
    :root {{ --bg:#0f1419; --surface:#1a2332; --border:#2d3a4d; --text:#e8edf4; --muted:#8b9cb3; --full:#fbbf24; --crop:#60a5fa; --gt:#34d399; }}
    body {{ margin:0; padding:2rem; font-family:-apple-system,sans-serif; background:var(--bg); color:var(--text); }}
    h1 {{ margin:0 0 .5rem; }}
    .subtitle {{ color:var(--muted); margin-bottom:1.5rem; }}
    .ticket-meta {{ color:var(--muted); margin-bottom:2rem; font-size:.95rem; }}
    .ticket-meta ul {{ margin:.5rem 0 0; padding-left:1.25rem; }}
    table {{ width:100%; border-collapse:collapse; margin-bottom:2rem; background:var(--surface); border-radius:8px; overflow:hidden; font-size:.92rem; }}
    th,td {{ padding:.65rem .85rem; text-align:left; border-bottom:1px solid var(--border); vertical-align:top; }}
    th {{ background:#243044; color:var(--muted); font-size:.78rem; text-transform:uppercase; }}
    code {{ font-family:monospace; background:#0d1117; padding:.1em .35em; border-radius:4px; word-break:break-all; }}
    a {{ color:#93c5fd; }}
    .vin-full {{ color:var(--full); }}
    .vin-crop {{ color:var(--crop); }}
    .vin-gt {{ color:var(--gt); }}
    .winner-crop {{ color:#34d399; font-weight:600; }}
    .winner-full {{ color:var(--full); }}
    .winner-both {{ color:#f87171; }}
    .muted {{ color:var(--muted); }}
    .doc-card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1.5rem; margin-bottom:2rem; }}
    .error-card {{ border-color:#7f1d1d; }}
    .images {{ display:grid; gap:1rem; }}
    .images-3 {{ grid-template-columns:1fr 1fr 1fr; }}
    @media (max-width:1100px) {{ .images-3 {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width:800px) {{ .images-3 {{ grid-template-columns:1fr; }} }}
    .img-block label {{ display:block; font-size:.78rem; font-weight:600; color:var(--muted); margin-bottom:.5rem; text-transform:uppercase; }}
    .img-block img {{ width:100%; border:1px solid var(--border); border-radius:6px; }}
  </style>
</head>
<body>
  <h1>Partner-closed VIN tickets — crop-only extraction</h1>
  <p class="subtitle">{summary['n_docs']} docs · {summary['n_tickets']} tickets · full wrong vs GT: {summary['full_wrong']} · crop fixes: {summary['crop_fixes']} · crop matches GT: {summary['crop_matches_gt']}</p>
  <div class="ticket-meta">
    <strong>Tickets</strong>
    <ul>
{chr(10).join(ticket_blurbs)}
    </ul>
  </div>
  <table>
    <thead><tr><th>Ticket</th><th>Doc</th><th>Partner</th><th>Prod/full VIN</th><th>Crop VIN</th><th>GT</th><th>Crop fixes?</th></tr></thead>
    <tbody>
{chr(10).join(table_rows)}
    </tbody>
  </table>
{chr(10).join(cards)}
</body>
</html>
"""


def render_md(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Partner-closed VIN tickets — crop-only extraction",
        "",
        "Full-page VIN from prod S3 baseline or ticket triage (no full-page MLLM rerun).",
        "Crop: section bounds + `enhance_vin_crop` + MLLM.",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Tickets | {summary['n_tickets']} |",
        f"| Docs run | {summary['n_docs']} |",
        f"| Errors | {summary['n_errors']} |",
        f"| Full wrong vs GT | {summary['full_wrong']} |",
        f"| **Crop fixes full** | **{summary['crop_fixes']}** |",
        f"| Crop matches GT | {summary['crop_matches_gt']} |",
        "",
        "| Ticket | Doc | Partner | Doc type | Prod/full VIN | Crop VIN | GT | Crop fixes? | Notes |",
        "|--------|-----|---------|----------|---------------|----------|-----|-------------|-------|",
    ]
    for r in results:
        if r.get("error"):
            lines.append(
                f"| {r['ticket']} | `{r['short_id']}` | {r['partner']} | {r.get('document_type','')} | "
                f"— | — | `{r.get('ground_truth','')}` | — | ERR: {r['error'][:40]} |"
            )
            continue
        fix = "yes" if r.get("crop_fixes") else ("no" if r.get("full_matches_truth") else "no")
        notes = r.get("closure_reason", "")[:50]
        lines.append(
            f"| {r['ticket']} | `{r['short_id']}` | {r['partner']} | {r.get('document_type','')} | "
            f"`{r.get('full_page_vin')}` | `{r.get('crop_vin_enhanced')}` | `{r.get('ground_truth')}` | "
            f"{fix} | {notes} |"
        )
    return "\n".join(lines) + "\n"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if not r.get("error")]
    return {
        "n_tickets": len({r["ticket"] for r in results}),
        "n_docs": len(results),
        "n_errors": sum(1 for r in results if r.get("error")),
        "full_wrong": sum(1 for r in ok if not r.get("full_matches_truth")),
        "crop_fixes": sum(1 for r in ok if r.get("crop_fixes")),
        "crop_matches_gt": sum(1 for r in ok if r.get("crop_matches_truth")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wa244", type=int, default=10, help="Max WA-244 docs from Looker")
    parser.add_argument(
        "--tickets-file",
        type=Path,
        help="Use prebuilt tickets.json instead of rebuilding from Looker",
    )
    parser.add_argument("--docs-only", action="store_true", help="Write tickets.json only")
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Regenerate desktop HTML/assets from results.json (no MLLM)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.html_only:
        results_path = OUT_DIR / "results.json"
        if not results_path.exists():
            raise SystemExit(f"Missing {results_path}")
        data = json.loads(results_path.read_text())
        results = data.get("results", [])
        summary = data.get("summary") or summarize(results)
        ticket_meta = data.get("ticket_meta") or TICKET_META
        profile = os.environ.get("AWS_PROFILE", "prod")
        gallery_rows = prepare_gallery_rows(results, profile)
        write_desktop_assets(gallery_rows)
        DESKTOP_HTML.write_text(render_html(gallery_rows, summary, ticket_meta))
        print(f"Wrote {len(gallery_rows)} doc cards to {DESKTOP_HTML}")
        print(f"Assets: {DESKTOP_ASSETS}")
        return
    (OUT_DIR / "tickets_meta.json").write_text(json.dumps({"tickets": TICKET_META}, indent=2))

    if args.tickets_file:
        docs = json.loads(args.tickets_file.read_text()).get("tickets", [])
    else:
        docs = build_ticket_docs(max_wa244=args.max_wa244)
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
    print("\n" + render_md(results, summary))
    print(f"Wrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
