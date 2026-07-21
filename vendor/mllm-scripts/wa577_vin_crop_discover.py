#!/usr/bin/env python3
"""Find title_application docs where VIN crop fixes a wrong full-page extraction."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

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
V2_RESULTS = (
    ROOT
    / "topologies/extraction/shared/mllm-invoker/payloads/wa577_v2/wa577_v2_results.json"
)
QA_YAML = (
    ROOT.parent
    / "techno-configs/techno_configs/envs/qa/document_fields/extractions/llm_configs/title_application.yml"
)
SCHEMA_JSON = (
    ROOT.parent
    / "techno-configs/techno_configs/envs/qa/document_fields/serialization/v1/title_application.json"
)
OUT_DIR = ROOT / "wa577_gallery/vin_crop_wins"
TARGET_WINS = int(os.environ.get("TARGET_WINS", "10"))
SKIP_SHORTS: set[str] = set()

from wa577_vin_crop_helpers import (
    VIN_LABEL,
    VIN_TOKEN,
    norm_vin,
    pick_vin_section,
    score_vin_section,
    tight_vin_bounds,
)

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


def download_image(uri: str, dest: Path, profile: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    key = uri.removeprefix("s3://informed-techno-core-prod-exchange/")
    subprocess.run(
        ["aws", "s3", "cp", f"s3://informed-techno-core-prod-exchange/{key}", str(dest), "--profile", profile],
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


def load_candidates() -> list[dict[str, Any]]:
    data = json.loads(V2_RESULTS.read_text())
    cands: dict[str, dict[str, Any]] = {}

    # All V2 docs with prod baseline VIN (broader than disagree-only).
    for doc in data["results"]:
        vin = doc["fields"]["vin"]
        truth = (vin.get("baseline") or "").strip()
        if not truth or len(norm_vin(truth)) < 11:
            continue
        short = doc["doc_id"].split("-")[0]
        if short in SKIP_SHORTS:
            continue
        cands[short] = {
            "short": short,
            "partner": doc.get("partner"),
            "truth": truth,
            "v2_new": vin.get("new"),
            "disagree": bool(vin.get("disagree")),
        }

    # Prioritize VIN disagrees, then the rest.
    ordered = sorted(cands.values(), key=lambda c: (not c["disagree"], c["short"]))
    ticket = [
        {"short": "87c842be", "partner": "penair", "truth": "1FMEE8BH0TLA94758", "disagree": True},
        {"short": "899d7cbd", "partner": "cuofco", "truth": "JF2SJAEC6XJ500527", "disagree": True},
        {"short": "4aeb9d2c", "partner": "desertfinancial", "truth": "3C4NJDCB4MT527789", "disagree": True},
    ]
    for t in ticket:
        if t["short"] not in SKIP_SHORTS:
            cands[t["short"]] = {**t, "v2_new": None}
            ordered = sorted(cands.values(), key=lambda c: (not c["disagree"], c["short"]))
    return ordered


def process_doc(meta: dict[str, Any], config: dict[str, Any], env: dict[str, str], profile: str) -> dict[str, Any] | None:
    short = meta["short"]
    truth = meta["truth"]
    payloads = find_payloads(short)
    if not payloads:
        return None

    best: dict[str, Any] | None = None
    for payload_path in payloads:
        page = payload_path.stem.rsplit("_", 1)[-1]
        base_payload = json.loads(payload_path.read_text())
        data = base_payload["detail"]["data"]
        doc_dir = OUT_DIR / short
        doc_dir.mkdir(parents=True, exist_ok=True)
        image_path = doc_dir / f"{short}_{page}.png"
        try:
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

            full_payload = copy.deepcopy(base_payload)
            full_payload["config"] = copy.deepcopy(config)
            full_tmp = doc_dir / f"payload_{page}_full.json"
            full_tmp.write_text(json.dumps(full_payload))
            crop_payload = copy.deepcopy(base_payload)
            crop_payload["config"] = copy.deepcopy(config)
            crop_tmp = doc_dir / f"payload_{page}_crop.json"
            crop_tmp.write_text(json.dumps(crop_payload))

            full_vin = run_mllm(full_tmp, env).get("vin")
            full_ok = norm_vin(full_vin) == norm_vin(truth)
            if full_ok:
                continue
            crop_vin = run_mllm(crop_tmp, env, local_image=crop_path).get("vin")
            full_ok = norm_vin(full_vin) == norm_vin(truth)
            crop_ok = norm_vin(crop_vin) == norm_vin(truth)
            row = {
                "short_id": short,
                "partner": meta["partner"],
                "page": page,
                "truth": truth,
                "full_page_vin": full_vin,
                "crop_vin": crop_vin,
                "full_matches_truth": full_ok,
                "crop_matches_truth": crop_ok,
                "crop_improved_vs_full": crop_ok and not full_ok,
                "vin_section_index": vin_section["index"],
                "crop_size": list(crop_img.size),
                "image_path": str(image_path),
                "crop_path": str(crop_path),
                "region_path": str(doc_dir / f"{short}_{page}_vin_region.png"),
            }
            if row["crop_improved_vs_full"]:
                return row
            if best is None or (not row["full_matches_truth"] and row["crop_matches_truth"]):
                best = row
        except Exception as exc:
            print(f"  {short}/{page} error: {exc}", flush=True)
    return best


def render_md(wins: list[dict[str, Any]]) -> str:
    lines = [
        "## VIN crop wins — full page wrong, crop correct (gemini-3.1-flash-lite, QA prompt)",
        "",
        f"Found **{len(wins)}** docs where crop fixes a wrong full-page VIN.",
        "",
        "| # | Doc | Partner | Page | Ground truth | Full page VIN | Crop VIN |",
        "|---|-----|---------|------|--------------|---------------|----------|",
    ]
    for i, r in enumerate(wins, 1):
        lines.append(
            f"| {i} | `{r['short_id']}` | {r['partner']} | {r['page']} | `{r['truth']}` | `{r['full_page_vin']}` | `{r['crop_vin']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    profile = "prod"
    env = refresh_aws_env(copy.copy(os.environ))
    env["SKIP_LLM_CACHE"] = "1"
    env["AWS_PROFILE"] = profile
    env.pop("BUNDLE_PATH", None)
    config = load_config()

    wins: list[dict[str, Any]] = []
    tried: list[dict[str, Any]] = []
    existing = OUT_DIR / "wins.json"
    if existing.exists():
        prev = json.loads(existing.read_text())
        wins = [w for w in prev.get("wins", []) if w.get("crop_improved_vs_full")]
        tried = prev.get("tried", [])
        SKIP_SHORTS.update(w["short_id"] for w in tried)
        print(f"Resuming with {len(wins)} wins, skipping {len(SKIP_SHORTS)} tried docs", flush=True)

    candidates = load_candidates()
    print(f"Candidates: {len(candidates)}", flush=True)

    for meta in candidates:
        if len(wins) >= TARGET_WINS:
            break
        print(f"=== {meta['short']} ({meta['partner']}) ===", flush=True)
        row = process_doc(meta, config, env, profile)
        if row:
            tried.append(row)
            if row.get("crop_improved_vs_full"):
                wins.append(row)
                print(
                    f"  WIN full={row['full_page_vin']!r} crop={row['crop_vin']!r} page={row['page']}",
                    flush=True,
                )
            else:
                print(
                    f"  skip full={row.get('full_page_vin')!r} crop={row.get('crop_vin')!r}",
                    flush=True,
                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"target": TARGET_WINS, "wins": wins, "tried": tried}
    (OUT_DIR / "wins.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "analysis.md").write_text(render_md(wins))
    print("\n" + render_md(wins))
    print(f"Wrote {OUT_DIR / 'wins.json'} ({len(wins)}/{TARGET_WINS} wins)")


if __name__ == "__main__":
    main()
