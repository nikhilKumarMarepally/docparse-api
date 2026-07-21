#!/usr/bin/env python3
"""Re-run full vs section-enhanced VIN A/B; collect remaining disagreements."""

from __future__ import annotations

import copy
import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from wa577_vin_crop_helpers import (  # noqa: E402
    enhance_vin_crop,
    norm_vin,
    pick_vin_section,
)
from ocr_line_to_sections import draw_sections_overlay  # noqa: E402

TICKET_AB = SCRIPT_DIR / "wa577_vin_ticket_crop_ab.py"
OUT_DIR = ROOT / "wa577_gallery/vin_full_ne_crop_disagreements"
DESKTOP_HTML = Path("/Users/nikhilmarepally/Desktop/vin_full_ne_crop_disagreements.html")
DESKTOP_ASSETS = Path("/Users/nikhilmarepally/Desktop/vin_full_ne_crop_disagreements_assets")

spec = importlib.util.spec_from_file_location("wa577_ticket_ab", TICKET_AB)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def row_disagrees(full_vin: Any, crop_vin: Any) -> bool:
    return norm_vin(full_vin) != norm_vin(crop_vin)


def vins_agree(a: Any, b: Any) -> bool:
    return norm_vin(a) == norm_vin(b)


def load_pool_from_results() -> list[dict[str, Any]]:
    """Re-run disagreements + errors from the prior results.json."""
    path = OUT_DIR / "results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for key in ("disagreements", "errors"):
        for r in data.get(key, []):
            row = {k: v for k, v in r.items() if not k.startswith("prior_")}
            rows.append(row)
    return [resolve_paths(r) for r in rows if r.get("short_id")]


def collect_pool() -> list[dict[str, Any]]:
    """Gather unique (short_id, page) from prior gallery disagreements."""
    disagrees: dict[tuple[str, str], dict[str, Any]] = {}
    skip = {ROOT / "wa577_gallery/vin_full_ne_crop_disagreements/results.json"}
    for p in sorted((ROOT / "wa577_gallery").rglob("results.json")):
        if p in skip or "vin_full_ne_crop" in p.name:
            continue
        data = json.loads(p.read_text())
        for r in data.get("results", []):
            if not row_disagrees(r.get("full_page_vin"), r.get("crop_vin")):
                continue
            short = r.get("short_id") or (r.get("doc_id") or "")[:8]
            page = r.get("page", "p0")
            key = (short, page)
            cand = _row_from_result(r, short, page, str(p.relative_to(ROOT)))
            prev = disagrees.get(key)
            if prev is None or _row_priority(cand) > _row_priority(prev):
                disagrees[key] = cand

    wins_path = ROOT / "wa577_gallery/vin_crop_wins/wins.json"
    if wins_path.exists():
        data = json.loads(wins_path.read_text())
        for section in ("wins", "tried"):
            for r in data.get(section, []):
                if not row_disagrees(r.get("full_page_vin"), r.get("crop_vin")):
                    continue
                short = r.get("short_id")
                page = r.get("page", "p0")
                key = (short, page)
                cand = _row_from_result(
                    r, short, page, f"wa577_gallery/vin_crop_wins/wins.json:{section}"
                )
                prev = disagrees.get(key)
                if prev is None or _row_priority(cand) > _row_priority(prev):
                    disagrees[key] = cand
    return list(disagrees.values())


def _row_priority(row: dict[str, Any]) -> tuple[int, int]:
    source = row.get("source") or ""
    if "wins.json" in source:
        tier = 3
    elif "vin_ticket_crop_ab" in source or "vin_crop_tickets" in source:
        tier = 2
    elif "vin_crop_ab" in source:
        tier = 0
    else:
        tier = 1
    has_crop = int(bool(row.get("old_crop_path")))
    return (tier, has_crop)


def _row_from_result(r: dict[str, Any], short: str, page: str, source: str) -> dict[str, Any]:
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


def resolve_paths(row: dict[str, Any]) -> dict[str, Any]:
    short, page = row["short_id"], row["page"]
    for key in ("image_path", "old_crop_path"):
        p = row.get(key)
        if p and Path(p).exists():
            row[key] = str(Path(p).resolve())

    if row.get("image_path"):
        return row

    source = row.get("source") or ""
    if "vin_crop_wins" in source:
        bases = [ROOT / "wa577_gallery/vin_crop_wins" / short]
    elif "vin_ticket_crop_ab_30d" in source:
        bases = [ROOT / "wa577_gallery/vin_ticket_crop_ab_30d" / short]
    elif "vin_ticket_crop_ab" in source:
        bases = [ROOT / "wa577_gallery/vin_ticket_crop_ab" / short]
    elif "vin_crop_tickets" in source:
        bases = [ROOT / "wa577_gallery/vin_crop_tickets" / short]
    elif "vin_crop_ab" in source:
        bases = [ROOT / "wa577_gallery/vin_crop_ab" / short]
    else:
        bases = [
            ROOT / "wa577_gallery/vin_crop_wins" / short,
            ROOT / "wa577_gallery/vin_ticket_crop_ab_30d" / short,
            ROOT / "wa577_gallery/vin_ticket_crop_ab" / short,
            ROOT / "wa577_gallery/vin_crop_tickets" / short,
            ROOT / "wa577_gallery/vin_crop_ab" / short,
        ]
    for base in bases:
        if not base.exists():
            continue
        img = base / f"{short}_{page}.png"
        if img.exists():
            row["image_path"] = str(img.resolve())
        if not row.get("old_crop_path"):
            for crop in (base / f"{short}_{page}_vin_crop.png", base / f"{short}_vin_crop.png"):
                if crop.exists():
                    row["old_crop_path"] = str(crop.resolve())
                    break
    return row


def find_sections_json(image_path: Path) -> Path | None:
    stem = image_path.stem
    parent = image_path.parent
    for c in [
        parent / f"{stem}_sections" / f"{stem}_sections.json",
        *parent.glob(f"**/{stem}_sections.json"),
    ]:
        if c.exists():
            return c
    return None


def find_payload(short: str, page: str, image_path: Path) -> Path | None:
    parent = image_path.parent
    page_num = page.replace("p", "")
    candidates = [
        *parent.glob("payload*.json"),
        *parent.glob(f"*{page}_payload.json"),
        *parent.glob(f"*{page_num}_payload.json"),
        *parent.glob(f"payload_{page}.json"),
    ]
    for c in candidates:
        if c.exists():
            return c
    for gallery in (ROOT / "wa577_gallery").iterdir():
        base = gallery / short
        if not base.is_dir():
            continue
        hits = sorted(base.glob(f"*{page_num}_payload.json")) + sorted(base.glob("payload*.json"))
        if hits:
            return hits[0]
    return None


def find_sections_png(image_path: Path) -> Path | None:
    stem = image_path.stem
    parent = image_path.parent
    for c in [
        parent / f"{stem}_sections" / f"{stem}_sections.png",
        parent / f"{stem}_sections.png",
        *parent.glob(f"**/{stem}_sections.png"),
    ]:
        if c.exists():
            return c
    return None


def _section_by_index(sections_data: dict[str, Any], index: int) -> dict[str, Any] | None:
    for section in sections_data.get("sections", []):
        if section.get("index") == index:
            return section
    return None


def ensure_sections_overlay(
    short: str,
    page: str,
    image_path: Path,
    sections_data: dict[str, Any],
    picked_section_idx: int | None,
) -> Path:
    """Return path to multi-color sections overlay PNG (generate if missing)."""
    doc_dir = OUT_DIR / short
    doc_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = doc_dir / f"{short}_{page}_sections.png"

    full_img = Image.open(image_path).convert("RGB")
    overlay = draw_sections_overlay(
        full_img,
        sections_data.get("sections", []),
        picked_section_idx,
    )
    overlay.save(overlay_path)
    return overlay_path


def ensure_sections_json(
    short: str, page: str, payload: Path, image: Path, profile: str
) -> Path:
    sections_json = image.parent / f"{short}_{page}_sections" / f"{short}_{page}_sections.json"
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


def vin_display(v: Any) -> str:
    return "null" if v is None else str(v)


def thumb_copy(src: Path, dest: Path, max_w: int = 560) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, max(1, int(img.height * ratio))), Image.Resampling.LANCZOS)
    img.save(dest, quality=88)
    return dest


def full_extraction_path(doc_dir: Path, short: str, page: str) -> Path:
    return doc_dir / f"{short}_{page}_full_extraction.json"


def save_full_extraction(
    doc_dir: Path, short: str, page: str, raw: dict[str, Any], full_vin: Any
) -> Path:
    path = full_extraction_path(doc_dir, short, page)
    path.write_text(
        json.dumps(
            {
                "full_page_vin": full_vin,
                "mllm_response": raw,
                "source": "mllm",
            },
            indent=2,
        )
    )
    return path


def backfill_full_extraction_stub(
    doc_dir: Path, short: str, page: str, full_vin: Any
) -> Path | None:
    """Write stub from results.json so future runs skip full-page MLLM."""
    path = full_extraction_path(doc_dir, short, page)
    if path.exists():
        return None
    doc_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "full_page_vin": full_vin,
                "mllm_response": {"vehicle": {"vin": full_vin}},
                "source": "backfill_from_results",
            },
            indent=2,
        )
    )
    return path


def load_cached_full_page_vin(
    doc_dir: Path, short: str, page: str, row: dict[str, Any]
) -> tuple[Any, str] | None:
    """Return (vin, source) when full-page MLLM can be skipped."""
    path = full_extraction_path(doc_dir, short, page)
    if path.exists():
        saved = json.loads(path.read_text())
        vin = saved.get("full_page_vin")
        if vin is None:
            vin = mod.extract_vin(saved.get("mllm_response") or saved)
        return vin, "full_extraction.json"

    prior = row.get("prior_full_page_vin")
    if prior is not None:
        backfill_full_extraction_stub(doc_dir, short, page, prior)
        return prior, "prior_full_page_vin"

    row_vin = row.get("full_page_vin")
    if row_vin is not None:
        backfill_full_extraction_stub(doc_dir, short, page, row_vin)
        return row_vin, "results.json"

    return None


def backfill_full_extractions_from_results() -> int:
    """Seed full_extraction.json stubs from existing results.json rows."""
    path = OUT_DIR / "results.json"
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    written = 0
    for bucket in ("disagreements", "now_agree", "errors", "additional_disagreements"):
        for r in data.get(bucket, []):
            vin = r.get("full_page_vin") or r.get("prior_full_page_vin")
            if vin is None:
                continue
            short = r.get("short_id")
            if not short:
                continue
            page = r.get("page", "p0")
            doc_dir = OUT_DIR / short
            if backfill_full_extraction_stub(doc_dir, short, page, vin):
                written += 1
    return written


def resolve_full_page_vin(
    row: dict[str, Any],
    doc_dir: Path,
    short: str,
    page: str,
    payload_path: Path,
    env: dict[str, str],
    *,
    crop_only: bool,
) -> tuple[Any, str]:
    """Run full-page MLLM once per doc; reuse cached extraction on reruns."""
    cached = load_cached_full_page_vin(doc_dir, short, page, row)
    if cached is not None:
        return cached

    if crop_only:
        raise RuntimeError(
            f"crop-only: no cached full_page_vin for {short} {page} "
            f"(missing {short}_{page}_full_extraction.json)"
        )

    full_ext = mod.run_mllm(payload_path, env)
    full_vin = mod.extract_vin(full_ext)
    save_full_extraction(doc_dir, short, page, full_ext, full_vin)
    return full_vin, "mllm"


def _run_enhanced_arm(
    payload_path: Path,
    env: dict[str, str],
    crop_img: Image.Image,
    doc_dir: Path,
    short: str,
    page: str,
    *,
    suffix: str = "",
) -> tuple[str | None, str, list[int]]:
    tag = f"_{suffix}" if suffix else ""
    enhanced_path = doc_dir / f"{short}_{page}_vin_crop_enhanced{tag}.png"
    enhanced_img = enhance_vin_crop(crop_img)
    enhanced_img.save(enhanced_path)
    enhanced_ext = mod.run_mllm(payload_path, env, local_image=enhanced_path)
    return (
        mod.extract_vin(enhanced_ext),
        str(enhanced_path.resolve()),
        list(enhanced_img.size),
    )


def process_row(
    row: dict[str, Any],
    env: dict[str, str],
    profile: str,
    *,
    crop_only: bool = False,
) -> dict[str, Any]:
    short = row["short_id"]
    page = row["page"]
    doc_dir = OUT_DIR / short
    doc_dir.mkdir(parents=True, exist_ok=True)

    result = dict(row)
    result.update(
        {
            "full_page_vin": None,
            "crop_vin": None,
            "crop_vin_enhanced": None,
            "section_crop_path": None,
            "enhanced_crop_path": None,
            "sections_overlay_path": None,
            "picked_section": None,
            "enhanced_crop_variant": None,
            "full_page_source": None,
            "error": None,
        }
    )

    image_path = Path(row["image_path"])
    if not image_path.exists():
        result["error"] = "image missing"
        return result

    payload_path = find_payload(short, page, image_path)
    if not payload_path:
        result["error"] = "payload missing"
        return result

    try:
        sections_json = find_sections_json(image_path)
        if not sections_json:
            sections_json = ensure_sections_json(short, page, payload_path, image_path, profile)
        sections_data = json.loads(sections_json.read_text())
        vin_section = pick_vin_section(sections_data)
        if not vin_section:
            result["error"] = "no VIN section"
            return result

        section_bounds = vin_section["bounds"]
        section_crop_img = mod.crop_bounds(image_path, section_bounds)
        section_crop_path = doc_dir / f"{short}_{page}_section_crop.png"
        section_crop_img.save(section_crop_path)

        sections_overlay_path = ensure_sections_overlay(
            short,
            page,
            image_path,
            sections_data,
            vin_section["index"],
        )

        full_vin, full_source = resolve_full_page_vin(
            row,
            doc_dir,
            short,
            page,
            payload_path,
            env,
            crop_only=crop_only,
        )

        enhanced_vin, enhanced_path, enhanced_size = _run_enhanced_arm(
            payload_path,
            env,
            section_crop_img,
            doc_dir,
            short,
            page,
        )

        result.update(
            {
                "full_page_vin": full_vin,
                "crop_vin": enhanced_vin,
                "crop_vin_enhanced": enhanced_vin,
                "section_crop_path": str(section_crop_path.resolve()),
                "enhanced_crop_path": enhanced_path,
                "sections_overlay_path": str(sections_overlay_path.resolve()),
                "picked_section": vin_section["index"],
                "section_crop_size": list(section_crop_img.size),
                "enhanced_crop_size": enhanced_size,
                "enhanced_crop_variant": "section_full",
                "full_page_source": full_source,
                "image_path": str(image_path.resolve()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)[-400:]

    return result


def merge_s0_rerun(row: dict[str, Any]) -> dict[str, Any]:
    """Merge per-doc s0_rerun.json when present (93298c2c investigation)."""
    short = row.get("short_id")
    if not short:
        return row
    s0_path = OUT_DIR / short / "s0_rerun.json"
    if not s0_path.exists():
        return row
    s0 = json.loads(s0_path.read_text())
    page = row.get("page", "p0")
    merged = dict(row)
    for key in ("s0_full_crop_vin", "s0_vinrow_crop_vin", "printed_truth", "full_page_vin"):
        if s0.get(key) is not None:
            merged[key] = s0[key]
    doc_dir = OUT_DIR / short
    for suffix, field in (
        ("s0_full_crop", "s0_full_crop_path"),
        ("s0_vinrow_crop", "s0_vinrow_crop_path"),
    ):
        p = doc_dir / f"{short}_{page}_{suffix}.png"
        if p.exists():
            merged[field] = str(p.resolve())
    if s0.get("s0_vinrow_crop_vin") is not None and merged.get("crop_vin") is None:
        merged["crop_vin"] = s0["s0_vinrow_crop_vin"]
        vinrow_path = doc_dir / f"{short}_{page}_s0_vinrow_crop.png"
        if vinrow_path.exists():
            merged["new_crop_path"] = str(vinrow_path.resolve())
    if s0.get("s0_full_crop_vin") is not None:
        merged.setdefault("crop_vin_enhanced", s0["s0_full_crop_vin"])
        merged.setdefault("enhanced_crop_variant", "s0_full")
    if s0.get("note"):
        merged.setdefault("notes", s0["note"])
    return merged


def resolve_sections_overlay(row: dict[str, Any]) -> Path | None:
    """Best available multi-color sections overlay for a disagreement row."""
    short = row.get("short_id", "")
    page = row.get("page", "p0")
    for candidate in (
        row.get("sections_overlay_path"),
        str(Path.home() / "Desktop" / f"vin_sections_{short}.png"),
        str(OUT_DIR / short / f"{short}_{page}_sections.png"),
    ):
        if candidate and Path(candidate).exists():
            return Path(candidate)

    image_path = row.get("image_path")
    if not image_path:
        return None
    img = Path(image_path)
    found = find_sections_png(img)
    if found:
        return found

    sections_json = find_sections_json(img)
    if not sections_json:
        return None
    sections_data = json.loads(sections_json.read_text())
    picked = row.get("picked_section")
    idx = picked if isinstance(picked, int) else None
    return ensure_sections_overlay(short, page, img, sections_data, idx)


def printed_ambiguity_digits(printed_truth: str | None) -> str:
    """Three-digit SN segment from printed_truth (e.g. 307 in SN307242)."""
    if not printed_truth:
        return "307"
    m = re.search(r"SN(\d{3})", printed_truth, re.I)
    return m.group(1) if m else printed_truth[-6:-3]


def render_s0_rerun_block(r: dict[str, Any]) -> str:
    """Optional S0 rerun note + crop images appended below the standard card grid."""
    if not (r.get("s0_full_crop_vin") or r.get("s0_vinrow_crop_vin")):
        return ""
    short = html.escape(r["short_id"])
    s0_full_v = html.escape(vin_display(r.get("s0_full_crop_vin")))
    s0_vinrow_v = html.escape(vin_display(r.get("s0_vinrow_crop_vin")))
    printed = html.escape(printed_ambiguity_digits(r.get("printed_truth")))
    s0_full_block = (
        f'<img src="vin_full_ne_crop_disagreements_assets/{short}_s0_full_crop.png" alt="S0 full crop">'
        if r.get("s0_full_crop_path") and Path(r["s0_full_crop_path"]).exists()
        else '<p class="muted">No S0 full crop</p>'
    )
    s0_vinrow_block = (
        f'<img src="vin_full_ne_crop_disagreements_assets/{short}_s0_vinrow_crop.png" alt="S0 VIN row crop">'
        if r.get("s0_vinrow_crop_path") and Path(r["s0_vinrow_crop_path"]).exists()
        else '<p class="muted">No S0 VIN row crop</p>'
    )
    return f"""    <p><strong>S0 full → <code class="vin-enhanced">{s0_full_v}</code> · S0 VIN row → <code class="vin-enhanced">{s0_vinrow_v}</code></strong> <span class="muted">(printed looks like <code>{printed}</code>)</span></p>
    <div class="images images-2 s0-rerun">
      <div class="img-block">
        <label>S0 full crop</label>
        {s0_full_block}
      </div>
      <div class="img-block">
        <label>S0 VIN row crop</label>
        {s0_vinrow_block}
      </div>
    </div>"""


def vin_char_len(v: Any) -> int:
    return len(norm_vin(v))


def enforce_valid_enhanced_verdict(row: dict[str, Any]) -> None:
    """Never label crop_wins when enhanced VIN is not exactly 17 characters."""
    enhanced = row.get("crop_vin_enhanced")
    if enhanced is None:
        return
    enh_len = vin_char_len(enhanced)
    if enh_len == 17:
        return

    prev = row.get("verdict")
    if prev != "crop_wins":
        return

    full_len = vin_char_len(row.get("full_page_vin"))
    notes: list[str] = [f"overrode {prev}: enhanced VIN is {enh_len} chars (must be 17)"]

    if full_len == 17:
        row["verdict"] = "mllm_read_error"
        notes.append(
            "enhanced crop shows printed VIN but MLLM dropped char(s); full page has valid 17-char VIN"
        )
    elif enh_len < 17:
        row["verdict"] = "crop_truncated"
        notes.append(f"enhanced VIN truncated to {enh_len} chars")
    else:
        row["verdict"] = "mllm_read_error"
        notes.append(f"enhanced VIN is {enh_len} chars (invalid length)")

    extra = "; ".join(notes)
    existing = row.get("notes") or ""
    if extra and extra not in existing:
        row["notes"] = f"{existing} | {extra}" if existing else extra


def classify_verdict(row: dict[str, Any]) -> str:
    """Quick vision-informed verdict bucket (filled after review)."""
    return row.get("verdict") or "pending"


def enhanced_summary_metrics(all_rerun: list[dict[str, Any]]) -> dict[str, int]:
    ok = [r for r in all_rerun if not r.get("error")]
    enhanced_agree_crop = sum(
        1 for r in ok if vins_agree(r.get("crop_vin"), r.get("crop_vin_enhanced"))
    )
    enhanced_agree_full = sum(
        1 for r in ok if vins_agree(r.get("full_page_vin"), r.get("crop_vin_enhanced"))
    )
    enhanced_fix_nulls = sum(
        1
        for r in ok
        if r.get("crop_vin") is None and r.get("crop_vin_enhanced") is not None
    )
    enhanced_changed = sum(
        1
        for r in ok
        if not vins_agree(r.get("crop_vin"), r.get("crop_vin_enhanced"))
    )
    return {
        "enhanced_agree_crop": enhanced_agree_crop,
        "enhanced_agree_full": enhanced_agree_full,
        "enhanced_fix_nulls": enhanced_fix_nulls,
        "enhanced_changed_outcome": enhanced_changed,
    }


def count_cached_full_page_rows(rows: list[dict[str, Any]]) -> int:
    """Docs whose full VIN came from cache (not a fresh full-page MLLM call)."""
    count = 0
    for r in rows:
        source = r.get("full_page_source")
        if source and source != "mllm":
            count += 1
            continue
        short = r.get("short_id")
        if not short:
            continue
        page = r.get("page", "p0")
        if full_extraction_path(OUT_DIR / short, short, page).exists():
            count += 1
    return count


def render_md(rows: list[dict[str, Any]], pool_size: int, all_rerun: list[dict[str, Any]]) -> str:
    agreed = sum(
        1
        for r in all_rerun
        if not r.get("error") and not row_disagrees(r.get("full_page_vin"), r.get("crop_vin"))
    )
    errors = sum(1 for r in all_rerun if r.get("error"))
    null_mismatch = sum(
        1
        for r in rows
        if (r.get("full_page_vin") is None) != (r.get("crop_vin") is None)
    )
    both_nonnull = sum(
        1
        for r in rows
        if r.get("full_page_vin") is not None and r.get("crop_vin") is not None
    )
    verdict_counts: dict[str, int] = {}
    for r in rows:
        v = r.get("verdict", "pending")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    enh = enhanced_summary_metrics(all_rerun)

    cached_full = count_cached_full_page_rows(all_rerun)
    lines = [
        "# Full-page VIN ≠ image enhancement — full section crop + enhance",
        "",
        f"Re-ran **{len(all_rerun)}** disagreement/error docs (full picked-section crop + "
        f"`enhance_vin_crop`). **{len(rows)}** still disagree full vs image enhancement; "
        f"**{agreed}** now agree full vs image enhancement; **{errors}** errors.",
        "",
        f"Full VIN from cached extraction (not re-run) on **{cached_full}** / "
        f"**{len(all_rerun)}** docs; crop/enhanced MLLM re-run per doc.",
        "",
        "## Summary metrics",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| Re-run pool | {pool_size} |",
        f"| **Still disagree (full vs image enhancement)** | **{len(rows)}** |",
        f"| Now agree (full vs image enhancement) | {agreed} |",
        f"| Errors | {errors} |",
        f"| Both non-null disagree | {both_nonnull} |",
        f"| Null mismatch | {null_mismatch} |",
        f"| Enhanced agrees with full | {enh['enhanced_agree_full']} |",
        "",
        "## Verdict breakdown",
        "",
        "| Verdict | Count |",
        "|---------|------:|",
    ]
    for v, c in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {v} | {c} |")

    lines.extend(
        [
            "",
            "## Disagreements",
            "",
            "| Doc | Type | Partner | Full VIN | Image enhancement | Verdict |",
            "|-----|------|---------|----------|-------------------|---------|",
        ]
    )
    for r in rows:
        lines.append(
            f"| `{r['short_id']}` | {r.get('document_type', '')} | {r.get('partner') or '—'} | "
            f"`{vin_display(r.get('full_page_vin'))}` | "
            f"`{vin_display(r.get('crop_vin_enhanced'))}` | "
            f"**{r.get('verdict', 'pending')}** |"
        )

    agreed_rows = [r for r in all_rerun if not r.get("error") and not row_disagrees(r.get("full_page_vin"), r.get("crop_vin"))]
    if agreed_rows:
        lines.extend(
            [
                "",
                "## Now agree (full = image enhancement)",
                "",
                "| Doc | Type | Partner | Full VIN | Image enhancement | Verdict |",
                "|-----|------|---------|----------|-------------------|---------|",
            ]
        )
        for r in agreed_rows:
            lines.append(
                f"| `{r['short_id']}` | {r.get('document_type', '')} | {r.get('partner') or '—'} | "
                f"`{vin_display(r.get('full_page_vin'))}` | "
                f"`{vin_display(r.get('crop_vin_enhanced'))}` | "
                f"**{r.get('verdict', 'agree')}** |"
            )
    return "\n".join(lines) + "\n"


def render_html(rows: list[dict[str, Any]], pool_size: int, all_rerun: list[dict[str, Any]]) -> str:
    agreed_full_enh = sum(
        1
        for r in all_rerun
        if not r.get("error")
        and vins_agree(r.get("full_page_vin"), r.get("crop_vin_enhanced"))
    )
    cards = []
    table_rows = []
    for r in rows:
        short = html.escape(r["short_id"])
        dtype = html.escape(r.get("document_type") or "")
        partner = html.escape(str(r.get("partner") or "—"))
        full_v = html.escape(vin_display(r.get("full_page_vin")))
        crop_enh = html.escape(vin_display(r.get("crop_vin_enhanced")))
        s0_full_v = html.escape(vin_display(r.get("s0_full_crop_vin")))
        s0_vinrow_v = html.escape(vin_display(r.get("s0_vinrow_crop_vin")))
        verdict = html.escape(r.get("verdict", "pending"))
        page = html.escape(r.get("page") or "p0")
        wclass = {
            "crop_wins": "winner-crop",
            "full_wins": "winner-full",
            "both_wrong": "winner-both",
            "crop_null_full_correct": "winner-null-ok",
            "crop_null_full_wrong": "winner-null-bad",
            "B2_app_context": "winner-b2",
            "mllm_read_error": "winner-mllm-error",
            "crop_truncated": "winner-truncated",
        }.get(r.get("verdict", ""), "muted")

        enh_cell = crop_enh
        if r.get("s0_full_crop_vin"):
            enh_cell = f'{s0_full_v} <span class="muted">(S0 full)</span>'
            if r.get("s0_vinrow_crop_vin"):
                enh_cell += f' · {s0_vinrow_v} <span class="muted">(S0 vinrow)</span>'

        table_rows.append(
            f"""      <tr>
        <td><code>{short}</code></td>
        <td>{dtype}</td>
        <td>{partner}</td>
        <td class="vin-full"><code>{full_v}</code></td>
        <td class="vin-enhanced">{enh_cell}</td>
        <td class="{wclass}"><code>{verdict}</code></td>
      </tr>"""
        )

        sections_overlay = r.get("sections_overlay_path") or ""
        enhanced_crop = r.get("enhanced_crop_path") or ""
        sections_block = (
            f'<img src="vin_full_ne_crop_disagreements_assets/{short}_sections.png" alt="sections overlay">'
            if sections_overlay and Path(sections_overlay).exists()
            else '<p class="muted">No sections overlay</p>'
        )
        enh_label = "Image enhancement"
        enhanced_block = (
            f'<img src="vin_full_ne_crop_disagreements_assets/{short}_enhanced_crop.png" alt="image enhancement">'
            if enhanced_crop and Path(enhanced_crop).exists()
            else '<p class="muted">No image enhancement</p>'
        )
        picked = r.get("picked_section")
        picked_note = f" · picked S{picked}" if picked is not None else ""
        s0_block = render_s0_rerun_block(r)

        cards.append(
            f"""  <div class="doc-card">
    <h2><code>{short}</code> · {dtype} · {partner}</h2>
    <p>Page <code>{page}</code>{picked_note} · Full <code class="vin-full">{full_v}</code> · Image enhancement <code class="vin-enhanced">{crop_enh}</code> · <span class="{wclass}">{verdict}</span></p>
    <div class="images images-3">
      <div class="img-block">
        <label>Full page</label>
        <img src="vin_full_ne_crop_disagreements_assets/{short}_full.png" alt="full">
      </div>
      <div class="img-block">
        <label>Sections overlay</label>
        {sections_block}
      </div>
      <div class="img-block">
        <label>{enh_label}</label>
        {enhanced_block}
      </div>
    </div>
{s0_block}
  </div>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VIN full ≠ image enhancement — section crop disagreements</title>
  <style>
    :root {{ --bg:#0f1419; --surface:#1a2332; --border:#2d3a4d; --text:#e8edf4; --muted:#8b9cb3; --full:#fbbf24; --enhanced:#34d399; }}
    body {{ margin:0; padding:2rem; font-family:-apple-system,sans-serif; background:var(--bg); color:var(--text); }}
    h1 {{ margin:0 0 .5rem; }}
    .subtitle {{ color:var(--muted); margin-bottom:2rem; }}
    table {{ width:100%; border-collapse:collapse; margin-bottom:2rem; background:var(--surface); border-radius:8px; overflow:hidden; font-size:.92rem; }}
    th,td {{ padding:.65rem .85rem; text-align:left; border-bottom:1px solid var(--border); vertical-align:top; }}
    th {{ background:#243044; color:var(--muted); font-size:.78rem; text-transform:uppercase; }}
    code {{ font-family:monospace; background:#0d1117; padding:.1em .35em; border-radius:4px; word-break:break-all; }}
    .vin-full {{ color:var(--full); }}
    .vin-enhanced {{ color:var(--enhanced); }}
    .winner-crop {{ color:#60a5fa; }}
    .winner-full {{ color:#fbbf24; }}
    .winner-both {{ color:#f87171; }}
    .winner-mllm-error {{ color:#f472b6; }}
    .winner-truncated {{ color:#c084fc; }}
    .winner-null-ok {{ color:#4ade80; }}
    .winner-null-bad {{ color:#fb923c; }}
    .winner-b2 {{ color:#a78bfa; }}
    .muted {{ color:var(--muted); }}
    .doc-card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1.5rem; margin-bottom:2rem; }}
    .images {{ display:grid; gap:1rem; }}
    .images-2 {{ grid-template-columns:1fr 1fr; }}
    .images-3 {{ grid-template-columns:1fr 1fr 1fr; }}
    .s0-rerun {{ margin-top:1rem; }}
    @media (max-width:1100px) {{ .images-2, .images-3 {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width:800px) {{ .images-2, .images-3 {{ grid-template-columns:1fr; }} }}
    .img-block label {{ display:block; font-size:.78rem; font-weight:600; color:var(--muted); margin-bottom:.5rem; text-transform:uppercase; }}
    .img-block img {{ width:100%; border:1px solid var(--border); border-radius:6px; }}
  </style>
</head>
<body>
  <h1>Full-page VIN ≠ image enhancement (full section crop)</h1>
  <p class="subtitle">{len(rows)} disagreements · {len(all_rerun)} re-run · {agreed_full_enh} full=enhanced agree</p>
  <table>
    <thead><tr><th>Doc</th><th>Type</th><th>Partner</th><th>Full VIN</th><th>Image enhancement</th><th>Verdict</th></tr></thead>
    <tbody>
{chr(10).join(table_rows)}
    </tbody>
  </table>
{chr(10).join(cards)}
</body>
</html>
"""


def refresh_crop_preview_assets(row: dict[str, Any]) -> None:
    """Re-crop section bounds and re-enhance PNGs (no MLLM)."""
    image_path = row.get("image_path")
    if not image_path or not Path(image_path).exists():
        return
    short, page = row["short_id"], row["page"]
    doc_dir = OUT_DIR / short
    doc_dir.mkdir(parents=True, exist_ok=True)
    sections_json = find_sections_json(Path(image_path))
    if not sections_json:
        return
    sections_data = json.loads(sections_json.read_text())
    picked = row.get("picked_section")
    idx = picked if isinstance(picked, int) else None
    vin_section = _section_by_index(sections_data, idx) if idx is not None else None
    if vin_section is None:
        vin_section = pick_vin_section(sections_data)
    if not vin_section:
        return
    section_crop_img = mod.crop_bounds(Path(image_path), vin_section["bounds"])
    section_crop_path = doc_dir / f"{short}_{page}_section_crop.png"
    section_crop_img.save(section_crop_path)
    enhanced_path = doc_dir / f"{short}_{page}_vin_crop_enhanced.png"
    enhance_vin_crop(section_crop_img).save(enhanced_path)
    row["section_crop_path"] = str(section_crop_path.resolve())
    row["enhanced_crop_path"] = str(enhanced_path.resolve())
    row["section_crop_size"] = list(section_crop_img.size)


def write_sanity_desktop_assets(short_ids: list[str], rows_by_id: dict[str, dict[str, Any]]) -> None:
    """Copy sanity-check doc assets to desktop (includes now_agree docs)."""
    for short in short_ids:
        r = rows_by_id.get(short)
        if not r:
            continue
        if r.get("image_path") and Path(r["image_path"]).exists():
            thumb_copy(Path(r["image_path"]), DESKTOP_ASSETS / f"{short}_full.png")
        if r.get("sections_overlay_path") and Path(r["sections_overlay_path"]).exists():
            thumb_copy(Path(r["sections_overlay_path"]), DESKTOP_ASSETS / f"{short}_sections.png")
        if r.get("enhanced_crop_path") and Path(r["enhanced_crop_path"]).exists():
            thumb_copy(
                Path(r["enhanced_crop_path"]),
                DESKTOP_ASSETS / f"{short}_enhanced_crop.png",
                max_w=520,
            )


def prepare_sections_overlays(rows: list[dict[str, Any]], profile: str) -> None:
    """Ensure multi-color sections overlay PNG exists for each gallery row."""
    for r in rows:
        image_path = r.get("image_path")
        if not image_path or not Path(image_path).exists():
            continue
        short, page = r["short_id"], r["page"]
        sections_json = find_sections_json(Path(image_path))
        if not sections_json:
            payload_path = find_payload(short, page, Path(image_path))
            if not payload_path:
                continue
            try:
                sections_json = ensure_sections_json(short, page, payload_path, Path(image_path), profile)
            except subprocess.CalledProcessError:
                continue
        sections_data = json.loads(sections_json.read_text())
        picked = r.get("picked_section")
        idx = picked if isinstance(picked, int) else None
        overlay_path = ensure_sections_overlay(
            short,
            page,
            Path(image_path),
            sections_data,
            idx,
        )
        r["sections_overlay_path"] = str(overlay_path.resolve())


def write_desktop_assets(rows: list[dict[str, Any]]) -> None:
    if DESKTOP_ASSETS.exists():
        shutil.rmtree(DESKTOP_ASSETS)
    DESKTOP_ASSETS.mkdir(parents=True)
    for r in rows:
        short = r["short_id"]
        if r.get("image_path") and Path(r["image_path"]).exists():
            thumb_copy(Path(r["image_path"]), DESKTOP_ASSETS / f"{short}_full.png")
        if r.get("sections_overlay_path") and Path(r["sections_overlay_path"]).exists():
            thumb_copy(
                Path(r["sections_overlay_path"]),
                DESKTOP_ASSETS / f"{short}_sections.png",
            )
        if r.get("enhanced_crop_path") and Path(r["enhanced_crop_path"]).exists():
            thumb_copy(
                Path(r["enhanced_crop_path"]),
                DESKTOP_ASSETS / f"{short}_enhanced_crop.png",
                max_w=520,
            )
        for src_key, asset_name, max_w in (
            ("s0_full_crop_path", f"{short}_s0_full_crop.png", 560),
            ("s0_vinrow_crop_path", f"{short}_s0_vinrow_crop.png", 560),
        ):
            p = r.get(src_key)
            if p and Path(p).exists():
                thumb_copy(Path(p), DESKTOP_ASSETS / asset_name, max_w=max_w)


def apply_vision_reviews(rows: list[dict[str, Any]]) -> None:
    """Vision review verdicts — updated after image inspection."""
    reviews: dict[str, dict[str, str]] = {
        "87c842be": {"verdict": "crop_wins", "notes": "FL Section 2; full swapped 947/974"},
        "d0877660": {"verdict": "crop_wins", "notes": "Vehicle Inspection; full dropped second 7"},
        "ae35905f": {"verdict": "crop_wins", "notes": "NY Section 2 boxed VIN; full truncated"},
        "3422862e": {"verdict": "crop_wins", "notes": "Same pattern as ae35905f"},
        "57f91148": {"verdict": "crop_wins", "notes": "FL Section 2; full dropped second 7"},
        "18bc23db": {"verdict": "crop_wins", "notes": "NY Section 2; full null; crop matches printed"},
        "93298c2c": {
            "verdict": "agree",
            "notes": "OR S0; full=crop=enhanced; user confirmed extraction correct (S0 full/vinrow both right)",
        },
        "915df3ff": {"verdict": "crop_wins", "notes": "MO DOR-108; full null; crop matches printed"},
        "dcf91cab": {"verdict": "crop_wins", "notes": "AZ title; full null; crop matches printed"},
        "2b896599": {
            "verdict": "full_wins",
            "notes": "TX 130-U S0; full matches printed; crop/enhanced drop L (16-char MLLM read)",
        },
        "7d6bae3e": {"verdict": "both_wrong", "notes": "MO DOR-108; full truncated; crop extra 0"},
        "5a15a826": {"verdict": "crop_wins", "notes": "OR DMV; full extra digit; crop matches printed"},
        "6f54cca6": {"verdict": "crop_wins", "notes": "WI MV1 Section B; full null; crop matches printed"},
        "6fddd398": {
            "verdict": "agree",
            "notes": "PA MV-1 S1 Section A VIN; S2 trade-in block was wrongly picked before",
        },
    }
    for r in rows:
        rev = reviews.get(r["short_id"])
        if rev:
            r.update(rev)
        enforce_valid_enhanced_verdict(r)
        if rev:
            continue
        if r.get("full_page_vin") and r.get("crop_vin"):
            r.setdefault("verdict", "pending")
            r.setdefault("notes", "needs vision review")
        elif r.get("full_page_vin") and not r.get("crop_vin"):
            r.setdefault("verdict", "crop_null_full_correct")
            r.setdefault("notes", "crop null; full has value")
        elif not r.get("full_page_vin") and r.get("crop_vin"):
            r.setdefault("verdict", "full_null_crop_has")
            r.setdefault("notes", "full null; crop has value")
        else:
            r.setdefault("verdict", "both_null")
            r.setdefault("notes", "both null")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Regenerate sections overlays + desktop HTML/assets from results.json (no MLLM)",
    )
    parser.add_argument(
        "--crop-only",
        action="store_true",
        help="Skip full-page MLLM; require cached full_extraction.json or prior full_page_vin",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run.log").touch(exist_ok=True)

    profile = os.environ.get("AWS_PROFILE", "prod")

    if args.html_only:
        results_path = OUT_DIR / "results.json"
        if not results_path.exists():
            raise SystemExit(f"Missing {results_path}")
        data = json.loads(results_path.read_text())
        disagreements = [merge_s0_rerun(resolve_paths(r)) for r in data.get("disagreements", [])]
        all_rerun = disagreements + data.get("now_agree", []) + data.get("errors", [])
        pool_size = data.get("summary", {}).get("pool_size", len(all_rerun))
        apply_vision_reviews(disagreements)
        overlay_rows = [
            merge_s0_rerun(resolve_paths(r))
            for r in data.get("disagreements", [])
            + data.get("now_agree", [])
            if not r.get("error")
        ]
        for r in overlay_rows:
            refresh_crop_preview_assets(r)
        prepare_sections_overlays(overlay_rows, profile)
        rows_by_id = {r["short_id"]: r for r in overlay_rows}
        for r in disagreements:
            synced = rows_by_id.get(r["short_id"])
            if synced:
                r["sections_overlay_path"] = synced.get("sections_overlay_path")
                r["enhanced_crop_path"] = synced.get("enhanced_crop_path")
                r["section_crop_path"] = synced.get("section_crop_path")
        write_desktop_assets(disagreements)
        write_sanity_desktop_assets(["93298c2c"], rows_by_id)
        (OUT_DIR / "results.md").write_text(render_md(disagreements, pool_size, all_rerun))
        DESKTOP_HTML.write_text(render_html(disagreements, pool_size, all_rerun))
        print(f"Wrote {len(disagreements)} disagreement cards to {DESKTOP_HTML}")
        return

    pool = load_pool_from_results()
    if not pool:
        pool = [resolve_paths(r) for r in collect_pool()]
    pool = [r for r in pool if r.get("image_path")]
    pool_size = len(pool)

    backfilled = backfill_full_extractions_from_results()
    if backfilled:
        print(f"Backfilled {backfilled} full_extraction.json stub(s) from results.json", flush=True)

    env = refresh_aws_env(dict(os.environ))
    env["AWS_PROFILE"] = profile
    env["SKIP_LLM_CACHE"] = "1"

    all_rerun: list[dict[str, Any]] = []
    for i, row in enumerate(pool, 1):
        print(f"[{i}/{pool_size}] {row['short_id']} {row['page']}", flush=True)
        result = process_row(row, env, profile, crop_only=args.crop_only)
        disagree = row_disagrees(result.get("full_page_vin"), result.get("crop_vin"))
        enh_changed = not vins_agree(result.get("crop_vin"), result.get("crop_vin_enhanced"))
        status = "DISAGREE" if disagree else "agree"
        print(
            f"  {status} full={result.get('full_page_vin')!r} crop={result.get('crop_vin')!r} "
            f"enhanced={result.get('crop_vin_enhanced')!r}"
            + (" CHANGED" if enh_changed else ""),
            flush=True,
        )
        all_rerun.append(result)

    disagreements = [
        merge_s0_rerun(r)
        for r in all_rerun
        if not r.get("error") and row_disagrees(r.get("full_page_vin"), r.get("crop_vin"))
    ]
    apply_vision_reviews(disagreements)
    overlay_rows = [r for r in all_rerun if not r.get("error")]
    prepare_sections_overlays(overlay_rows, profile)

    agreed = [
        r
        for r in all_rerun
        if not r.get("error") and not row_disagrees(r.get("full_page_vin"), r.get("crop_vin"))
    ]
    enh = enhanced_summary_metrics(all_rerun)
    cached_full = count_cached_full_page_rows(all_rerun)

    out = {
        "summary": {
            "pool_size": pool_size,
            "rerun": len(all_rerun),
            "still_disagree": len(disagreements),
            "now_agree": len(agreed),
            "errors": sum(1 for r in all_rerun if r.get("error")),
            "full_page_cached": cached_full,
            "comparison": "norm(full_page_vin) != norm(crop_vin); null vs non-null is disagree",
            "crop_logic": "pick_vin_section full bounds + enhance_vin_crop",
            "model": "gemini-3.1-flash-lite",
            "skip_llm_cache": True,
            **enh,
        },
        "disagreements": disagreements,
        "now_agree": agreed,
        "errors": [r for r in all_rerun if r.get("error")],
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "results.md").write_text(render_md(disagreements, pool_size, all_rerun))
    write_desktop_assets(disagreements)
    DESKTOP_HTML.write_text(render_html(disagreements, pool_size, all_rerun))

    print(
        f"\nPool: {pool_size} | Still disagree: {len(disagreements)} | Now agree: {len(agreed)}"
    )
    print(
        f"Enhanced: agree_crop={enh['enhanced_agree_crop']} agree_full={enh['enhanced_agree_full']} "
        f"fix_nulls={enh['enhanced_fix_nulls']} changed={enh['enhanced_changed_outcome']}"
    )
    print(f"Wrote {OUT_DIR / 'results.json'}")
    print(f"Wrote {DESKTOP_HTML}")


if __name__ == "__main__":
    main()
