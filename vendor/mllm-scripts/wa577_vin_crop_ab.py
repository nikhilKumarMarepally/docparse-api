#!/usr/bin/env python3
"""VIN crop A/B: full-page vs VIN-section crop on title_application issue docs."""

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
from PIL import Image

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
BATCH_DIR = ROOT / "wa577_gallery/vin_sections_batch"
OUT_DIR = ROOT / "wa577_gallery/vin_crop_ab"

# 10 title_application docs with VIN extraction issues
CURATED = [
    {"short": "87c842be", "partner": "penair", "prod": None, "ground_truth": "1FMEE8BH0TLA94758"},
    {"short": "feddebac", "partner": "cuofco", "prod": "1fmsk8jh0mgc19927", "ground_truth": "1fmsk8jh0mgc19927"},
    {"short": "fbd240ba", "partner": "chase", "prod": "4s4ghud62t3730412", "ground_truth": "4s4ghud62t3730412"},
    {"short": "df26e460", "partner": "chase", "prod": "jn8az3be9t9720535", "ground_truth": "jn8az3be9t9720535"},
    {"short": "cbef7557", "partner": "chase", "prod": "2t36crav3tw048303", "ground_truth": "2t36crav3tw048303"},
    {"short": "c9bef3f0", "partner": "stellantislease", "prod": "1c6srfjt3tn295698", "ground_truth": "1c6srfjt3tn295698"},
    {"short": "b933127a", "partner": "desertfinancial", "prod": "3n8ap6ce0tl308016", "ground_truth": "3n8ap6ce0tl308016"},
    {"short": "b8cbf6ca", "partner": "cinchauto", "prod": "1fmee4dp4tla88703", "ground_truth": "1fmee4dp4tla88703"},
    {"short": "b7b84eac", "partner": "exeter", "prod": "5xyp5dhc2mg172164", "ground_truth": "5xyp5dhc2mg172164"},
    {"short": "e99c9299", "partner": "cps", "prod": "3fmcr9b69prd88098", "ground_truth": "3fmcr9b69prd88098"},
]

from wa577_vin_crop_helpers import norm_vin, pick_vin_section, score_vin_section

VIN_TOKEN = re.compile(r"[A-HJ-NPR-Z0-9]{11,17}", re.I)


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(QA_YAML.read_text())
    schema_body = json.loads(SCHEMA_JSON.read_text())["definitions"]["extracted_data"]
    intro = config["model_info"]["payload_config"]["prompt_config"]["intro"]
    config["model_info"]["payload_config"]["prompt_config"]["intro"] = intro.replace(
        "$$_SCHEMA", json.dumps(schema_body)
    )
    return config


def find_payload(short: str) -> Path:
    for d in PAYLOAD_DIRS:
        hits = sorted(d.glob(f"{short}*_p0.json"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no p0 payload for {short}")


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


def ensure_sections(short: str, payload: Path, image: Path, profile: str) -> Path:
    sections_json = BATCH_DIR / short / f"{short}_p0_sections" / f"{short}_p0_sections.json"
    if sections_json.exists():
        return sections_json
    out = OUT_DIR / short / f"{short}_p0_sections"
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
    return out / f"{short}_p0_sections.json"


def crop_section(image_path: Path, bounds: dict[str, float], pad: int = 24) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    x0 = max(0, int(bounds["min_x"]) - pad)
    y0 = max(0, int(bounds["min_y"]) - pad)
    x1 = min(img.width, int(bounds["max_x"]) + pad)
    y1 = min(img.height, int(bounds["max_y"]) + pad)
    return img.crop((x0, y0, x1, y1))


def run_mllm(
    payload_path: Path,
    env: dict[str, str],
    *,
    local_image: Path | None = None,
) -> dict[str, Any]:
    run_env = dict(env)
    if local_image:
        cmd = ["ruby", str(MLLM_LOCAL), "--payload", str(payload_path)]
        run_env["LOCAL_IMAGE_PATH"] = str(local_image)
    else:
        cmd = [str(MLLM), "--payload", str(payload_path)]
    proc = subprocess.run(
        cmd,
        cwd=str(MLLM_DIR),
        capture_output=True,
        text=True,
        env=run_env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:])
    text = proc.stdout
    start = text.rfind("\n{")
    if start < 0:
        start = text.find("{")
    else:
        start += 1
    return json.loads(text[start:])


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


def vin_match(got: Any, truth: str | None) -> bool | None:
    if not truth:
        return None
    return norm_vin(got) == norm_vin(truth)


def main() -> None:
    profile = "prod"
    env = refresh_aws_env(copy.copy(os.environ))
    env["SKIP_LLM_CACHE"] = "1"
    env["AWS_PROFILE"] = profile
    env.pop("BUNDLE_PATH", None)

    config = load_config()
    results = []
    for meta in CURATED:
        short = meta["short"]
        print(f"=== {short} ({meta['partner']}) ===", flush=True)
        doc_dir = OUT_DIR / short
        doc_dir.mkdir(parents=True, exist_ok=True)

        payload_path = find_payload(short)
        base_payload = json.loads(payload_path.read_text())
        data = base_payload["detail"]["data"]
        image_path = doc_dir / f"{short}_p0.png"
        download_image(data["image_uri"], image_path, profile)

        sections_json = ensure_sections(short, payload_path, image_path, profile)
        sections_data = json.loads(sections_json.read_text())
        vin_section = pick_vin_section(sections_data)
        if not vin_section:
            raise RuntimeError("no VIN section found")
        crop_img = crop_section(image_path, vin_section["bounds"])
        crop_path = doc_dir / f"{short}_vin_crop.png"
        crop_img.save(crop_path)

        from PIL import ImageDraw

        gallery = Image.open(image_path).convert("RGB")
        b = vin_section["bounds"]
        draw = ImageDraw.Draw(gallery)
        draw.rectangle(
            [(b["min_x"], b["min_y"]), (b["max_x"], b["max_y"])],
            outline=(20, 150, 60),
            width=4,
        )
        gallery.save(doc_dir / f"{short}_vin_region.png")

        full_payload = copy.deepcopy(base_payload)
        full_payload["config"] = copy.deepcopy(config)
        full_tmp = doc_dir / "payload_full.json"
        full_tmp.write_text(json.dumps(full_payload))

        crop_payload = copy.deepcopy(base_payload)
        crop_payload["config"] = copy.deepcopy(config)
        crop_tmp = doc_dir / "payload_crop.json"
        crop_tmp.write_text(json.dumps(crop_payload))

        full_ext = run_mllm(full_tmp, env)
        crop_ext = run_mllm(crop_tmp, env, local_image=crop_path)
        full_vin = full_ext.get("vin")
        crop_vin = crop_ext.get("vin")

        truth = meta.get("ground_truth") or meta.get("prod")
        prod = meta.get("prod")
        row = {
            "short_id": short,
            "partner": meta["partner"],
            "doc_id": data["document_id"],
            "ground_truth": truth,
            "prod_baseline": prod,
            "vin_section_index": vin_section["index"],
            "vin_section_preview": vin_section["text"][:160].replace("\n", " | "),
            "crop_path": str(crop_path),
            "crop_size": list(crop_img.size),
            "full_page_vin": full_vin,
            "crop_vin": crop_vin,
            "full_matches_truth": vin_match(full_vin, truth),
            "crop_matches_truth": vin_match(crop_vin, truth),
            "full_matches_prod": vin_match(full_vin, prod),
            "crop_matches_prod": vin_match(crop_vin, prod),
            "crop_improved_vs_full": norm_vin(crop_vin) == norm_vin(truth)
            and norm_vin(full_vin) != norm_vin(truth)
            if truth
            else norm_vin(crop_vin) == norm_vin(prod)
            and norm_vin(full_vin) != norm_vin(prod)
            if prod
            else None,
        }
        results.append(row)
        print(
            f"  full={full_vin!r} crop={crop_vin!r} "
            f"truth_ok=({row['full_matches_truth']},{row['crop_matches_truth']})",
            flush=True,
        )

    summary = summarize(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps({"results": results, "summary": summary}, indent=2))
    (OUT_DIR / "analysis.md").write_text(render_md(results, summary))
    print("\n" + render_md(results, summary))
    print(f"\nWrote {OUT_DIR / 'results.json'}")


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    with_truth = [r for r in results if r["ground_truth"]]
    return {
        "n_docs": n,
        "full_matches_truth": sum(1 for r in with_truth if r["full_matches_truth"]),
        "crop_matches_truth": sum(1 for r in with_truth if r["crop_matches_truth"]),
        "crop_improved_vs_full": sum(1 for r in results if r.get("crop_improved_vs_full")),
        "crop_regressed_vs_full": sum(
            1
            for r in results
            if r.get("full_matches_prod") and not r.get("crop_matches_prod")
        ),
        "full_matches_prod": sum(1 for r in results if r.get("full_matches_prod")),
        "crop_matches_prod": sum(1 for r in results if r.get("crop_matches_prod")),
    }


def render_md(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "## VIN crop A/B — full page vs VIN-section crop (gemini-3.1-flash-lite, QA prompt)",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Docs | {summary['n_docs']} |",
        f"| Full page matches ground truth | {summary['full_matches_truth']} |",
        f"| Crop matches ground truth | {summary['crop_matches_truth']} |",
        f"| Crop improved vs full (truth) | {summary['crop_improved_vs_full']} |",
        f"| Full matches prod baseline | {summary['full_matches_prod']} |",
        f"| Crop matches prod baseline | {summary['crop_matches_prod']} |",
        "",
        "| Doc | Partner | Prod / truth | Full page VIN | Crop VIN | Crop win? |",
        "|-----|---------|--------------|---------------|----------|-----------|",
    ]
    for r in results:
        ref = r["ground_truth"] or r["prod_baseline"] or "—"
        win = "✓" if r.get("crop_improved_vs_full") else ("—" if r["full_matches_truth"] == r["crop_matches_truth"] else "✗")
        lines.append(
            f"| `{r['short_id']}` | {r['partner']} | `{ref}` | `{r['full_page_vin']}` | `{r['crop_vin']}` | {win} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
