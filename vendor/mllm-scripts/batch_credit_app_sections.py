#!/usr/bin/env python3
"""Batch OCR section overlays for RouteOne / DealerTrack credit_application pages.

Uses local ca_appctx clustering cache (page.png + Gemini OCR JSON). Converts
Gemini token bboxes to word-level vision format, clusters lines into sections,
and writes colored S0/S1/... overlays (no inner crop box).

Outputs:
  wa577_gallery/credit_app_sections/{routeone,dealertrack}/{short_id}/
  wa577_gallery/credit_app_sections/manifest.json
  ~/Desktop/credit_app_sections_gallery.html
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ocr_line_to_sections import (  # noqa: E402
    draw_sections_overlay,
    lines_to_sections,
)
from ocr_word_to_line_boxes import load_words, words_to_lines  # noqa: E402

from section_content_classifier import (  # noqa: E402
    classify_sections_payload,
    load_default_model,
)

OUT_ROOT = ROOT / "wa577_gallery/credit_app_sections"
CA_APPCTX = Path("/tmp/ca_appctx")
PER_PARTNER = 18
SEED = 577
SHORT_ID_RE = re.compile(r"^[0-9a-f]{8}$", re.I)

# Curated short IDs from prior credit-app audits (mix of layouts / page counts)
CURATED_ROUTEONE = [
    "92aef038", "ae8f79bf", "5e04851d", "c91e88b7", "a8090b02", "64645831",
    "fd6e6874", "76f49e0e", "a9aa59a4", "32cb13f4", "dafec7b5", "1faaab55",
    "090e5000", "ef82b8d9", "efd65fe9", "435bfa21", "37c66a58", "fbc37f1f",
]
CURATED_DEALERTRACK = [
    "46501a8b", "4c383c09", "1f2b4939", "da9b4627", "a439eaf0", "602a5000",
    "fb03cc41", "e9e0b408", "08dfdf13", "b00fd237", "2b5fe3cb", "33ae5db2",
    "50e50f0e", "62104c22", "98ba38e3", "6bf89f71", "005f9437", "3a04577c",
]


def short_id(doc_id: str) -> str:
    return doc_id.split("-")[0].lower()


def gemini_ocr_to_vision(ocr_payload: dict[str, Any], img_w: int, img_h: int) -> dict[str, Any]:
    """Adapt local Gemini OCR JSON to vision document_text.words for line clustering."""
    od = ocr_payload.get("ocr_data") or {}
    full_text = od.get("text") or ""
    pages = od.get("pages") or []
    words: list[dict[str, Any]] = []
    if not pages:
        return {"ocr_data": {"document_text": {"words": words}}}

    for tok in (pages[0] or {}).get("tokens") or []:
        layout = tok.get("layout") or {}
        segs = (layout.get("text_anchor") or {}).get("text_segments") or []
        if not segs:
            continue
        start = int(segs[0].get("start_index", 0))
        end = int(segs[0].get("end_index", 0))
        text = full_text[start:end].strip()
        if not text:
            continue

        poly = (layout.get("bounding_poly") or {})
        verts = poly.get("vertices") or []
        if not verts:
            norm = poly.get("normalized_vertices") or []
            if norm and img_w and img_h:
                verts = [
                    {"x": (v.get("x") or 0) * img_w, "y": (v.get("y") or 0) * img_h}
                    for v in norm
                    if isinstance(v, dict)
                ]
        if not verts:
            continue
        xs = [float(v.get("x", 0)) for v in verts if isinstance(v, dict)]
        ys = [float(v.get("y", 0)) for v in verts if isinstance(v, dict)]
        if not xs or not ys:
            continue
        words.append(
            {
                "text": text,
                "bounds": {
                    "min_x": min(xs),
                    "min_y": min(ys),
                    "max_x": max(xs),
                    "max_y": max(ys),
                },
            }
        )

    return {"ocr_data": {"document_text": {"words": words}}}


def pick_applicant_section(sections: list[dict[str, Any]]) -> int | None:
    """Best-effort highlight for applicant block (credit app has no VIN section)."""
    markers = (
        "credit application: applicant",
        "section a",
        "applicant information",
        "primary applicant",
    )
    best_idx: int | None = None
    best_score = 0
    for section in sections:
        text = (section.get("text") or "").lower()
        score = sum(1 for m in markers if m in text)
        if score > best_score:
            best_score = score
            best_idx = int(section["index"])
    return best_idx if best_score else None


def metadata_by_short(work_dir: Path) -> dict[str, dict[str, Any]]:
    meta_path = work_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    raw = json.loads(meta_path.read_text())
    out: dict[str, dict[str, Any]] = {}
    for doc_id, meta in raw.items():
        out[short_id(doc_id)] = {**meta, "doc_id": doc_id}
    return out


def asset_paths(work_dir: Path, meta: dict[str, Any], page: int) -> tuple[Path, Path]:
    asset = f"{meta['doc_id']}_{meta['file_id']}_p{page}"
    image = work_dir / "clustering" / "assets" / asset / "page.png"
    ocr = work_dir / "clustering" / "ocr" / f"{asset}.json"
    return image, ocr


def list_cached_pages(work_dir: Path) -> dict[str, list[int]]:
    """short_id -> sorted page numbers with both image and OCR on disk."""
    assets_dir = work_dir / "clustering" / "assets"
    ocr_dir = work_dir / "clustering" / "ocr"
    by_short: dict[str, set[int]] = {}
    if not assets_dir.exists():
        return {}
    for asset_dir in assets_dir.iterdir():
        if not asset_dir.is_dir():
            continue
        image = asset_dir / "page.png"
        ocr = ocr_dir / f"{asset_dir.name}.json"
        if not image.exists() or not ocr.exists():
            continue
        m = re.match(r"^([0-9a-f]{8})-[0-9a-f-]+_[0-9a-f-]+_p(\d+)$", asset_dir.name, re.I)
        if not m:
            continue
        sid, page = m.group(1).lower(), int(m.group(2))
        by_short.setdefault(sid, set()).add(page)
    return {sid: sorted(pages) for sid, pages in by_short.items()}


def select_docs(partner: str, work_dir: Path, curated: list[str]) -> list[tuple[str, int]]:
    """Return (short_id, page) pairs — prefer curated docs, then backfill from cache."""
    by_short_meta = metadata_by_short(work_dir)
    cached = list_cached_pages(work_dir)
    picks: list[tuple[str, int]] = []
    seen: set[str] = set()

    def pick_page(sid: str) -> int | None:
        pages = cached.get(sid)
        if not pages:
            return None
        meta = by_short_meta.get(sid, {})
        start = int(meta.get("start_page") or 1)
        end = int(meta.get("end_page") or start)
        if start <= 1 <= end and 1 in pages:
            return 1
        for p in pages:
            if start <= p <= end:
                return p
        return pages[0]

    for sid in curated:
        if sid in seen:
            continue
        page = pick_page(sid)
        if page is None:
            continue
        picks.append((sid, page))
        seen.add(sid)
        if len(picks) >= PER_PARTNER:
            return picks

    remaining = [s for s in sorted(cached) if s not in seen]
    random.Random(SEED + hash(partner) % 10000).shuffle(remaining)
    for sid in remaining:
        page = pick_page(sid)
        if page is None:
            continue
        picks.append((sid, page))
        seen.add(sid)
        if len(picks) >= PER_PARTNER:
            break
    return picks


# Content-type colors for gallery overlay legend
CONTENT_TYPE_COLORS = {
    "personal_identity": "#e6194b",
    "contact_info": "#3cb44b",
    "residential_address": "#4363d8",
    "mailing_address": "#911eb4",
    "employment_income": "#f58231",
    "business_entity": "#42d4f4",
    "joint_intent": "#f032e6",
    "vehicle_description": "#bfef45",
    "trade_in_vehicle": "#9a6324",
    "financial_disclosure": "#800000",
    "itemization": "#aaffc3",
    "insurance_product": "#808000",
    "signature_authorization": "#ffd8b1",
    "signature_consent": "#e6beff",
    "dealer_seller_info": "#469990",
    "form_metadata": "#a9a9a9",
}


def process_page(
    partner: str,
    short: str,
    page: int,
    image: Path,
    ocr_path: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
    doc_dir = OUT_ROOT / partner / short
    doc_dir.mkdir(parents=True, exist_ok=True)
    page_tag = f"p{page}"
    stem = f"{short}_{page_tag}"

    full_copy = doc_dir / f"{stem}_full.png"
    if not full_copy.exists():
        full_copy.write_bytes(image.read_bytes())

    sections_dir = doc_dir / f"{stem}_sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    sections_json = sections_dir / f"{stem}_sections.json"
    sections_png = sections_dir / f"{stem}_sections.png"

    with Image.open(image) as img:
        img_w, img_h = img.size

    ocr_payload = json.loads(ocr_path.read_text())
    vision = gemini_ocr_to_vision(ocr_payload, img_w, img_h)
    words = load_words(vision)
    lines = words_to_lines(words, page_width=img_w, full_width=True)
    sections, stats = lines_to_sections(lines)
    section_dicts = [s.to_dict() for s in sections]
    picked = pick_applicant_section(section_dicts)

    model = load_default_model()
    classified_sections = classify_sections_payload(
        section_dicts,
        document_type="credit_application",
        model=model,
    )

    payload = {
        "image": str(full_copy),
        "vision": str(ocr_path),
        "partner": partner,
        "doc_id": meta.get("doc_id"),
        "short_id": short,
        "page": page_tag,
        "document_type": "credit_application",
        "line_count": len(lines),
        "section_count": len(sections),
        "gap_stats": stats.to_dict(),
        "picked_section": picked,
        "sections": classified_sections,
    }
    sections_json.write_text(json.dumps(payload, indent=2))

    classified_json = sections_dir / f"{stem}_sections_classified.json"
    classified_json.write_text(
        json.dumps(
            {
                "document_type": "credit_application",
                "short_id": short,
                "page": page_tag,
                "sections": classified_sections,
            },
            indent=2,
        )
    )

    overlay = draw_sections_overlay(
        Image.open(full_copy).convert("RGB"),
        section_dicts,
        picked,
        crop_bounds=None,
    )
    overlay.save(sections_png)

    return {
        "partner": partner,
        "doc_id": meta.get("doc_id"),
        "short_id": short,
        "page": page_tag,
        "page_num": page,
        "partner_name": meta.get("partner_name"),
        "section_count": len(sections),
        "line_count": len(lines),
        "picked_section": picked,
        "content_types_summary": _summarize_content_types(classified_sections),
        "full_png": str(full_copy.resolve()),
        "sections_png": str(sections_png.resolve()),
        "sections_json": str(sections_json.resolve()),
        "sections_classified_json": str(classified_json.resolve()),
    }


def _summarize_content_types(sections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for section in sections:
        cc = section.get("content_classification") or {}
        for ct in cc.get("content_types") or []:
            counts[ct] = counts.get(ct, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def build_html(manifest: list[dict[str, Any]], out_path: Path) -> None:
    legend_items = "".join(
        f'<span class="legend" style="border-color:{color}">{ct}</span>'
        for ct, color in sorted(CONTENT_TYPE_COLORS.items())
    )

    def cards_for(partner: str) -> str:
        rows = [m for m in manifest if m["partner"] == partner]
        parts: list[str] = []
        for m in rows:
            rel_full = Path(m["full_png"]).name
            rel_sections = Path(m["sections_png"]).name
            base = f"{partner}/{m['short_id']}"
            picked = m.get("picked_section")
            picked_txt = f"S{picked}" if picked is not None else "—"
            ctypes = m.get("content_types_summary") or {}
            ctype_txt = ", ".join(f"{k}({v})" for k, v in list(ctypes.items())[:4]) or "—"
            parts.append(
                f"""
        <div class="card">
          <header>
            <span class="doc-id">{m['short_id']}</span>
            <span class="meta">{m['page']} · {m['section_count']} sections · picked {picked_txt}</span>
            <div class="ctypes">{ctype_txt}</div>
          </header>
          <motion-pair>
            <figure>
              <img src="file://{m['full_png']}" alt="full {m['short_id']}" loading="lazy" />
              <figcaption>full page</figcaption>
            </figure>
            <figure>
              <img src="file://{m['sections_png']}" alt="sections {m['short_id']}" loading="lazy" />
              <figcaption>sections overlay</figcaption>
            </figure>
          </motion-pair>
        </div>""".replace("<motion-pair>", '<div class="pair">').replace("</motion-pair>", "</div>")
            )
        return "\n".join(parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Credit App Section Overlays</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; background: #f4f4f8; color: #1a1a2e; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 40px; border-bottom: 2px solid #ccc; padding-bottom: 6px; }}
    .sub {{ color: #555; margin-bottom: 24px; }}
    .grid {{ display: flex; flex-direction: column; gap: 28px; }}
    .card {{ background: #fff; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,.08); padding: 16px; }}
    .doc-id {{ font-family: ui-monospace, monospace; font-weight: 700; font-size: 15px; }}
    .meta {{ color: #666; margin-left: 12px; font-size: 13px; }}
    .ctypes {{ color: #444; font-size: 12px; margin-top: 4px; font-family: ui-monospace, monospace; }}
    .legend-wrap {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 24px; }}
    .legend {{ font-size: 11px; padding: 2px 6px; border-left: 4px solid #999; background: #fff; border-radius: 3px; }}
    header {{ margin-bottom: 12px; }}
    .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
    figcaption {{ text-align: center; font-size: 12px; color: #777; margin-top: 4px; }}
    @media (max-width: 900px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Credit Application — OCR Section Overlays</h1>
  <p class="sub">RouteOne &amp; DealerTrack · gap-based line clustering · universal content types · {len(manifest)} pages</p>
  <div class="legend-wrap">{legend_items}</div>

  <h2>RouteOne ({sum(1 for m in manifest if m['partner']=='routeone')})</h2>
  <div class="grid">{cards_for('routeone')}</div>

  <h2>DealerTrack ({sum(1 for m in manifest if m['partner']=='dealertrack')})</h2>
  <div class="grid">{cards_for('dealertrack')}</div>
</body>
</html>"""
    out_path.write_text(html)


def main() -> None:
    manifest: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    cohorts = [
        ("routeone", CA_APPCTX / "routeone", CURATED_ROUTEONE),
        ("dealertrack", CA_APPCTX / "dealertrack", CURATED_DEALERTRACK),
    ]

    for partner, work_dir, curated in cohorts:
        if not work_dir.exists():
            errors.append({"partner": partner, "error": f"missing work dir {work_dir}"})
            continue
        by_short = metadata_by_short(work_dir)
        for sid, page in select_docs(partner, work_dir, curated):
            meta = by_short.get(sid)
            if not meta:
                # Synthesize minimal meta from asset dir name when metadata row missing.
                matches = list((work_dir / "clustering" / "assets").glob(f"{sid}-*_*_p{page}"))
                if not matches:
                    errors.append({"partner": partner, "short_id": sid, "error": "no metadata"})
                    continue
                asset_name = matches[0].name
                doc_id = asset_name.rsplit("_", 2)[0]
                meta = {"doc_id": doc_id, "file_id": asset_name.split("_")[1]}
            image, ocr = asset_paths(work_dir, meta, page)
            if not image.exists() or not ocr.exists():
                errors.append(
                    {
                        "partner": partner,
                        "short_id": sid,
                        "page": page,
                        "error": f"missing image={image.exists()} ocr={ocr.exists()}",
                    }
                )
                continue
            try:
                row = process_page(partner, sid, page, image, ocr, meta)
                manifest.append(row)
                print(
                    f"OK {partner} {sid} {row['page']} sections={row['section_count']} "
                    f"picked={row['picked_section']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"partner": partner, "short_id": sid, "error": str(exc)})
                print(f"FAIL {partner} {sid}: {exc}", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_manifest = {
        "results": manifest,
        "errors": errors,
        "routeone_count": sum(1 for m in manifest if m["partner"] == "routeone"),
        "dealertrack_count": sum(1 for m in manifest if m["partner"] == "dealertrack"),
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(out_manifest, indent=2))

    gallery = Path.home() / "Desktop" / "credit_app_sections_gallery.html"
    build_html(manifest, gallery)

    print(
        f"\nDone: {len(manifest)} pages "
        f"(RO={out_manifest['routeone_count']} DT={out_manifest['dealertrack_count']}), "
        f"{len(errors)} errors",
        flush=True,
    )
    print(f"manifest: {OUT_ROOT / 'manifest.json'}", flush=True)
    print(f"gallery:  {gallery}", flush=True)


if __name__ == "__main__":
    main()
