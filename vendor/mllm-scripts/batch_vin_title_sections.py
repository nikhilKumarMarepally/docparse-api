#!/usr/bin/env python3
"""Batch-run OCR line→section clustering on title_application pages with VIN."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "wa577_gallery/vin_sections_batch"
PAYLOAD_DIRS = [
    ROOT / "topologies/extraction/shared/mllm-invoker/payloads/wa513_577_v1",
    ROOT / "topologies/extraction/shared/mllm-invoker/payloads/wa577_v2",
]

# V1 ticket docs (all title_application with VIN checks) + diverse V2 samples
CURATED_SHORT_IDS = {
    # WA-513 / WA-577 tickets
    "2fe77315",  # gofi SC Form 400
    "d4eb2430",  # crescentbank SC Form 400
    "863cb231",  # cypruscu UT
    "f44356d8",  # consumerscu MI
    "ef6d31b2",  # crescentbank IL
    "75163cff",  # chase lease
    "87c842be",  # penair FL
    "4aeb9d2c",  # desertfinancial AZ
    "899d7cbd",  # cuofco CO (multi-page)
    # V2 cross-partner samples
    "f0942208",  # desertfinancial
    "0e66117c",  # chase
    "756a4670",  # mountainamericacu
    "92f5af77",  # gmfinancial
    "8fdcb3d8",  # suncoast
    "44d324a2",  # autonationfinance
}


def parse_s3(uri: str) -> str:
    return uri.removeprefix("s3://informed-techno-core-prod-exchange/")


def download(s3_key: str, dest: Path, profile: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    subprocess.run(
        ["aws", "s3", "cp", f"s3://informed-techno-core-prod-exchange/{s3_key}", str(dest), "--profile", profile],
        check=True,
        capture_output=True,
        text=True,
    )


def find_payloads(short_id: str) -> list[Path]:
    found: list[Path] = []
    for d in PAYLOAD_DIRS:
        if not d.exists():
            continue
        found.extend(sorted(d.glob(f"{short_id}*_p*.json")))
    # dedupe by page stem
    seen: set[str] = set()
    unique: list[Path] = []
    for p in found:
        stem = p.stem.rsplit("_", 1)[-1]  # p0, p1, ...
        key = f"{short_id}_{stem}"
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return sorted(unique)


def run_one(payload: Path, profile: str) -> dict:
    data = json.loads(payload.read_text())["detail"]["data"]
    short = data["document_id"].split("-")[0]
    page = payload.stem.rsplit("_", 1)[-1]
    doc_dir = OUT_ROOT / short
    image = doc_dir / f"{short}_{page}.png"
    out = doc_dir / f"{short}_{page}_sections"
    download(parse_s3(data["image_uri"]), image, profile)

    proc = subprocess.run(
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
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:] or proc.stdout[-500:])

    sections_json = out / f"{short}_{page}_sections.json"
    summary = json.loads(sections_json.read_text())
    vin_sections = []
    for s in summary["sections"]:
        t = s["text"].upper()
        if "VIN" in t or "VEHICLE IDENTIFICATION" in t or "IDENTIFICATION NUMBER" in t:
            vin_sections.append(s["index"])
    return {
        "doc_id": data["document_id"],
        "short_id": short,
        "page": page,
        "partner_id": data.get("partner_id"),
        "image": str(image),
        "sections_png": str(out / f"{short}_{page}_sections.png"),
        "line_count": summary["line_count"],
        "section_count": summary["section_count"],
        "gap_threshold": summary["gap_stats"]["threshold"],
        "vin_section_indices": vin_sections,
    }


def main() -> None:
    profile = "prod"
    subprocess.run(
        ["aws", "configure", "export-credentials", "--profile", profile, "--format", "env"],
        check=True,
        capture_output=True,
        text=True,
    )

    index: list[dict] = []
    errors: list[dict] = []

    for short_id in sorted(CURATED_SHORT_IDS):
        payloads = find_payloads(short_id)
        if not payloads:
            errors.append({"short_id": short_id, "error": "no payloads found"})
            continue
        for payload in payloads:
            print(f"=== {payload.name} ===", flush=True)
            try:
                row = run_one(payload, profile)
                index.append(row)
                print(
                    f"  OK lines={row['line_count']} sections={row['section_count']} "
                    f"vin_sections={row['vin_section_indices']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"payload": str(payload), "error": str(exc)})
                print(f"  FAIL {exc}", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "index.json").write_text(json.dumps({"results": index, "errors": errors}, indent=2))
    print(f"\nDone: {len(index)} pages, {len(errors)} errors -> {OUT_ROOT / 'index.json'}")


if __name__ == "__main__":
    main()
