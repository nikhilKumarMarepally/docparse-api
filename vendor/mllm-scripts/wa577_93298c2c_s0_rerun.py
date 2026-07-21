#!/usr/bin/env python3
"""One-off: S0 full vs S0 VIN-row crop + MLLM for 93298c2c p0."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from wa577_vin_crop_helpers import (  # noqa: E402
    _section_ocr_lines,
    _union_bounds,
    norm_vin,
)

TICKET_AB = SCRIPT_DIR / "wa577_vin_ticket_crop_ab.py"
spec = importlib.util.spec_from_file_location("wa577_ticket_ab", TICKET_AB)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SHORT = "93298c2c"
PAGE = "p0"
IMAGE = ROOT / f"wa577_gallery/vin_crop_wins/{SHORT}/{SHORT}_{PAGE}.png"
SECTIONS_JSON = (
    ROOT / f"wa577_gallery/vin_crop_wins/{SHORT}/{SHORT}_{PAGE}_sections/{SHORT}_{PAGE}_sections.json"
)
OUT_DIR = ROOT / f"wa577_gallery/vin_full_ne_crop_disagreements/{SHORT}"
FULL_CROP = OUT_DIR / f"{SHORT}_{PAGE}_s0_full_crop.png"
VINROW_CROP = OUT_DIR / f"{SHORT}_{PAGE}_s0_vinrow_crop.png"
RESULT_JSON = OUT_DIR / "s0_rerun.json"
DESKTOP_FULL = Path.home() / "Desktop" / f"{SHORT}_s0_full_crop.png"
DESKTOP_VINROW = Path.home() / "Desktop" / f"{SHORT}_s0_vinrow_crop.png"

VIN_TOKEN = re.compile(r"[A-HJ-NPR-Z0-9]{11,17}", re.I)


def vinrow_bounds(section: dict, sections_data: dict, image: Image.Image) -> dict[str, float]:
    """Crop to VEHICLE IDENTIFICATION label + handwritten VIN line within S0."""
    fallback = section["bounds"]
    section_line_data = _section_ocr_lines(section, sections_data, image)
    if not section_line_data:
        raise RuntimeError("no OCR lines for section 0")
    section_line_data.sort(key=lambda x: x[0])

    label_idx: int | None = None
    vin_idx: int | None = None
    for i, (_, text, _) in enumerate(section_line_data):
        if "VEHICLE IDENTIFICATION" in text.upper():
            label_idx = i
            break

    if label_idx is None:
        raise RuntimeError("no VEHICLE IDENTIFICATION label line in S0")

    for j in range(label_idx + 1, min(label_idx + 4, len(section_line_data))):
        compact = norm_vin(section_line_data[j][1])
        if "MMVABDM" in compact or VIN_TOKEN.search(compact):
            vin_idx = j
            break

    if vin_idx is None:
        vin_idx = label_idx + 1

    if label_idx is None or vin_idx is None:
        raise RuntimeError(f"could not find VIN row lines label={label_idx} vin={vin_idx}")

    lo, hi = sorted((label_idx, vin_idx))
    boxes = [section_line_data[j][2] for j in range(lo, hi + 1)]
    return _union_bounds(boxes, fallback, image.height)


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sections_data = json.loads(SECTIONS_JSON.read_text())
    sections = sections_data["sections"]
    s0 = next(s for s in sections if s["index"] == 0)

    full_img = Image.open(IMAGE).convert("RGB")
    full_bounds = s0["bounds"]
    vinrow_bounds_dict = vinrow_bounds(s0, sections_data, full_img)

    full_crop_img = mod.crop_bounds(IMAGE, full_bounds)
    vinrow_crop_img = mod.crop_bounds(IMAGE, vinrow_bounds_dict)
    full_crop_img.save(FULL_CROP)
    vinrow_crop_img.save(VINROW_CROP)
    shutil.copy2(FULL_CROP, DESKTOP_FULL)
    shutil.copy2(VINROW_CROP, DESKTOP_VINROW)

    payload_path = ROOT / f"wa577_gallery/vin_crop_wins/{SHORT}/payload_{PAGE}_full.json"
    env = refresh_aws_env(copy.copy(os.environ))
    env["SKIP_LLM_CACHE"] = "1"
    env["AWS_PROFILE"] = "prod"
    env.pop("BUNDLE_PATH", None)

    full_ext = mod.run_mllm(payload_path, env, local_image=FULL_CROP)
    vinrow_ext = mod.run_mllm(payload_path, env, local_image=VINROW_CROP)
    full_page_ext = mod.run_mllm(payload_path, env)

    s0_full_vin = mod.extract_vin(full_ext)
    s0_vinrow_vin = mod.extract_vin(vinrow_ext)
    full_page_vin = mod.extract_vin(full_page_ext)

    result = {
        "doc_id": SHORT,
        "section_pick": 0,
        "s0_full_crop_vin": s0_full_vin,
        "s0_vinrow_crop_vin": s0_vinrow_vin,
        "full_page_vin": full_page_vin,
        "printed_truth": "7MMVABDM2SN307242",
        "note": "S0 section pick correct; prior tight_vin_bounds hit header (ONLY VIN FEE); OCR 367 vs printed 307 in SN segment",
        "s0_full_bounds": full_bounds,
        "s0_vinrow_bounds": vinrow_bounds_dict,
        "crop_sizes": {"full": list(full_crop_img.size), "vinrow": list(vinrow_crop_img.size)},
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
