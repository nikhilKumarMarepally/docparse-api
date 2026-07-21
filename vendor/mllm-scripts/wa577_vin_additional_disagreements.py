#!/usr/bin/env python3
"""Find 10 NEW full-page VIN != enhanced-crop disagreements beyond the existing pool."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from wa577_vin_crop_helpers import norm_vin  # noqa: E402

# Reuse processing + rendering from the main disagreements script.
MAIN = SCRIPT_DIR / "wa577_vin_full_ne_crop_disagreements.py"
spec = importlib.util.spec_from_file_location("wa577_main", MAIN)
disag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(disag)

OUT_DIR = ROOT / "wa577_gallery/vin_full_ne_crop_disagreements"
DESKTOP_HTML = Path("/Users/nikhilmarepally/Desktop/vin_full_ne_crop_disagreements_additional_10.html")
DESKTOP_ASSETS = Path("/Users/nikhilmarepally/Desktop/vin_full_ne_crop_disagreements_additional_10_assets")
TARGET = 10


def enhanced_disagrees(full_vin: Any, enhanced_vin: Any) -> bool:
    return norm_vin(full_vin) != norm_vin(enhanced_vin)


def existing_keys() -> set[tuple[str, str]]:
    path = OUT_DIR / "results.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    keys: set[tuple[str, str]] = set()
    for bucket in ("disagreements", "now_agree", "errors", "additional_disagreements"):
        for r in data.get(bucket, []):
            keys.add((r.get("short_id", ""), r.get("page", "p0")))
    return keys


def existing_shorts() -> set[str]:
    return {k[0] for k in existing_keys()}


def row_from_gallery_result(r: dict[str, Any], short: str, page: str, source: str) -> dict[str, Any]:
    return {
        "short_id": short,
        "doc_id": r.get("doc_id") or short,
        "document_type": r.get("document_type", "title_application"),
        "partner": r.get("partner"),
        "page": page,
        "ground_truth": r.get("ground_truth") or r.get("truth"),
        "prior_full_page_vin": r.get("full_page_vin"),
        "prior_crop_vin": r.get("crop_vin"),
        "image_path": r.get("image_path"),
        "old_crop_path": r.get("crop_path"),
        "source": source,
    }


def collect_gallery_candidates() -> list[dict[str, Any]]:
    """Prior disagreements from other wa577_gallery artifacts (not yet in main pool)."""
    skip = existing_keys()
    disagrees: dict[tuple[str, str], dict[str, Any]] = {}

    for p in sorted((ROOT / "wa577_gallery").rglob("results.json")):
        if "vin_full_ne_crop" in str(p):
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        rows = data.get("results", [])
        if not rows and ("wins" in data or "tried" in data):
            rows = data.get("wins", []) + data.get("tried", [])
        for r in rows:
            if norm_vin(r.get("full_page_vin")) == norm_vin(r.get("crop_vin")):
                continue
            short = r.get("short_id") or (r.get("doc_id") or "")[:8]
            page = r.get("page", "p0")
            key = (short, page)
            if key in skip:
                continue
            cand = row_from_gallery_result(r, short, page, str(p.relative_to(ROOT)))
            prev = disagrees.get(key)
            if prev is None or disag._row_priority(cand) > disag._row_priority(prev):
                disagrees[key] = cand

    wins_path = ROOT / "wa577_gallery/vin_crop_wins/wins.json"
    if wins_path.exists():
        data = json.loads(wins_path.read_text())
        for section in ("wins", "tried"):
            for r in data.get(section, []):
                if norm_vin(r.get("full_page_vin")) == norm_vin(r.get("crop_vin")):
                    continue
                short = r.get("short_id")
                page = r.get("page", "p0")
                key = (short, page)
                if key in skip:
                    continue
                cand = row_from_gallery_result(
                    r, short, page, f"wa577_gallery/vin_crop_wins/wins.json:{section}"
                )
                prev = disagrees.get(key)
                if prev is None or disag._row_priority(cand) > disag._row_priority(prev):
                    disagrees[key] = cand
    return list(disagrees.values())


def collect_wins_dir_candidates(limit: int) -> list[dict[str, Any]]:
    """Untested vin_crop_wins dirs with image + payload (p0 preferred)."""
    skip_shorts = existing_shorts()
    rows: list[dict[str, Any]] = []
    wins_base = ROOT / "wa577_gallery/vin_crop_wins"
    for d in sorted(wins_base.iterdir()):
        if not d.is_dir() or d.name in skip_shorts:
            continue
        short = d.name
        for page in ("p0", "p1"):
            img = d / f"{short}_{page}.png"
            if not img.exists():
                continue
            payloads = sorted(d.glob(f"payload*{page}*.json")) + sorted(d.glob("payload*.json"))
            if not payloads:
                continue
            key = (short, page)
            if key in existing_keys():
                continue
            rows.append(
                {
                    "short_id": short,
                    "doc_id": short,
                    "document_type": "title_application",
                    "partner": None,
                    "page": page,
                    "ground_truth": None,
                    "image_path": str(img),
                    "source": "wa577_gallery/vin_crop_wins/scan",
                }
            )
        if len(rows) >= limit:
            break
    return rows[:limit]


def score_candidate(row: dict[str, Any]) -> tuple[int, int, int]:
    both = int(row.get("prior_full_page_vin") is not None and row.get("prior_crop_vin") is not None)
    null_mismatch = int(
        (row.get("prior_full_page_vin") is None) != (row.get("prior_crop_vin") is None)
    )
    has_prior = int(bool(row.get("prior_full_page_vin") or row.get("prior_crop_vin")))
    return (both, null_mismatch, has_prior)


def build_candidate_pool(scan_limit: int) -> list[dict[str, Any]]:
    gallery = collect_gallery_candidates()
    gallery.sort(key=score_candidate, reverse=True)
    seen = {(r["short_id"], r["page"]) for r in gallery}
    pool = [disag.resolve_paths(r) for r in gallery if r.get("short_id")]

    for row in collect_wins_dir_candidates(scan_limit):
        key = (row["short_id"], row["page"])
        if key in seen:
            continue
        seen.add(key)
        pool.append(disag.resolve_paths(row))
    return [r for r in pool if r.get("image_path")]


def write_additional_assets(rows: list[dict[str, Any]]) -> None:
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


def render_additional_md(rows: list[dict[str, Any]], scanned: int) -> str:
    lines = [
        "# Additional 10 — Full-page VIN ≠ enhanced crop",
        "",
        f"Scanned **{scanned}** new docs (excluded existing pool of 15). "
        f"Selected **{len(rows)}** where `norm(full_page_vin) != norm(crop_vin_enhanced)`.",
        "",
        "| Doc | Type | Partner | Full VIN | Enhanced VIN | Verdict |",
        "|-----|------|---------|----------|--------------|---------|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['short_id']}` | {r.get('document_type', '')} | {r.get('partner') or '—'} | "
            f"`{disag.vin_display(r.get('full_page_vin'))}` | "
            f"`{disag.vin_display(r.get('crop_vin_enhanced'))}` | "
            f"**{r.get('verdict', 'pending')}** |"
        )
    return "\n".join(lines) + "\n"


def auto_verdict(row: dict[str, Any]) -> None:
    """Heuristic verdict when vision review not pre-seeded."""
    full = row.get("full_page_vin")
    enh = row.get("crop_vin_enhanced")
    gt = row.get("ground_truth")
    if gt:
        ng = norm_vin(gt)
        full_ok = norm_vin(full) == ng
        enh_ok = norm_vin(enh) == ng
        if enh_ok and not full_ok:
            row["verdict"] = "crop_wins"
        elif full_ok and not enh_ok:
            row["verdict"] = "full_wins"
        elif not full_ok and not enh_ok:
            row["verdict"] = "both_wrong"
        else:
            row["verdict"] = "agree"
        return
    if full and not enh:
        row["verdict"] = "crop_null_full_correct"
    elif not full and enh:
        row["verdict"] = "crop_wins"
    elif full and enh:
        row["verdict"] = "pending"
    else:
        row["verdict"] = "both_null"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--scan-limit", type=int, default=80, help="Max vin_crop_wins dirs to scan")
    parser.add_argument("--html-only", action="store_true")
    parser.add_argument(
        "--crop-only",
        action="store_true",
        help="Skip full-page MLLM; require cached full_extraction.json or prior full_page_vin",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = os.environ.get("AWS_PROFILE", "prod")

    if args.html_only:
        data = json.loads((OUT_DIR / "results.json").read_text())
        rows = data.get("additional_disagreements", [])
        for r in rows:
            overlay = disag.resolve_sections_overlay(r)
            if overlay:
                r["sections_overlay_path"] = str(overlay.resolve())
        write_additional_assets(rows)
        DESKTOP_HTML.write_text(disag.render_html(rows, len(rows), rows))
        (OUT_DIR / "additional_10_results.md").write_text(
            render_additional_md(rows, data.get("summary", {}).get("additional_scanned", 0))
        )
        print(f"Wrote {DESKTOP_HTML}")
        return

    pool = build_candidate_pool(args.scan_limit)
    disag.backfill_full_extractions_from_results()
    env = disag.refresh_aws_env(dict(os.environ))
    env["AWS_PROFILE"] = profile
    env["SKIP_LLM_CACHE"] = "1"

    found: list[dict[str, Any]] = []
    seen_shorts: set[str] = set()
    scanned = 0
    for i, row in enumerate(pool, 1):
        if len(found) >= args.target:
            break
        short, page = row["short_id"], row["page"]
        if short in seen_shorts:
            continue
        print(f"[{i}/{len(pool)}] {short} {page}", flush=True)
        result = disag.process_row(row, env, profile, crop_only=args.crop_only)
        scanned += 1
        disagree = enhanced_disagrees(result.get("full_page_vin"), result.get("crop_vin_enhanced"))
        print(
            f"  {'DISAGREE' if disagree else 'skip'} full={result.get('full_page_vin')!r} "
            f"enhanced={result.get('crop_vin_enhanced')!r} err={result.get('error')}",
            flush=True,
        )
        if result.get("error") or not disagree:
            continue
        result = disag.merge_s0_rerun(result)
        auto_verdict(result)
        found.append(result)
        seen_shorts.add(short)

    if len(found) < args.target:
        print(f"WARNING: only found {len(found)}/{args.target} disagreements after {scanned} scans")

    disag.prepare_sections_overlays(found, profile)
    for r in found:
        overlay = disag.resolve_sections_overlay(r)
        if overlay:
            r["sections_overlay_path"] = str(overlay.resolve())

    # Append to results.json
    results_path = OUT_DIR / "results.json"
    data = json.loads(results_path.read_text()) if results_path.exists() else {"summary": {}, "disagreements": []}
    data.setdefault("summary", {})["additional_scanned"] = scanned
    data.setdefault("summary", {})["additional_disagreements_count"] = len(found)
    data["additional_disagreements"] = found
    results_path.write_text(json.dumps(data, indent=2))

    (OUT_DIR / "additional_10_results.md").write_text(render_additional_md(found, scanned))
    write_additional_assets(found)
    html = disag.render_html(found, scanned, found)
    html = html.replace(
        "vin_full_ne_crop_disagreements_assets",
        "vin_full_ne_crop_disagreements_additional_10_assets",
    )
    html = html.replace(
        "Full-page VIN ≠ crop VIN (post-fix + enhanced)",
        "Additional 10 — Full-page VIN ≠ enhanced crop",
    )
    DESKTOP_HTML.write_text(html)
    print(f"\nFound {len(found)} additional disagreements (scanned {scanned})")
    print(f"Wrote {results_path}")
    print(f"Wrote {OUT_DIR / 'additional_10_results.md'}")
    print(f"Wrote {DESKTOP_HTML}")


if __name__ == "__main__":
    main()
