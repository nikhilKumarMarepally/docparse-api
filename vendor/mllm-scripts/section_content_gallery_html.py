#!/usr/bin/env python3
"""Build section content + field classifier HTML gallery from credit_app_sections manifest."""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_credit_app_sections import CONTENT_TYPE_COLORS, OUT_ROOT  # noqa: E402

GALLERY_ROOT = OUT_ROOT
CROP_ROOT = ROOT / "wa577_gallery" / "section_classifier" / "gallery_crops"
DESKTOP_HTML = Path.home() / "Desktop" / "section_classifier_gallery.html"
REPO_HTML = ROOT / "wa577_gallery" / "section_classifier" / "gallery.html"
MANIFEST_PATH = GALLERY_ROOT / "manifest.json"
PICK_COUNT = 10

INTEREST_CONTENT_TYPES = (
    "business_entity",
    "personal_identity",
    "employment_income",
    "signature_authorization",
    "residential_address",
    "vehicle_description",
    "contact_info",
)


def file_url(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


def bounds_summary(bounds: dict[str, Any] | None) -> str:
    if not bounds:
        return "—"
    return (
        f"[{bounds.get('min_x', 0):.0f},{bounds.get('min_y', 0):.0f}]"
        f"→[{bounds.get('max_x', 0):.0f},{bounds.get('max_y', 0):.0f}]"
    )


def section_classification(section: dict[str, Any]) -> dict[str, Any]:
    return section.get("content_classification") or {}


def section_fields(section: dict[str, Any]) -> list[str]:
    cc = section_classification(section)
    fields = cc.get("fields") or section.get("fields") or []
    return sorted(set(str(f) for f in fields if f))


def section_content_types(section: dict[str, Any]) -> list[str]:
    cc = section_classification(section)
    return list(cc.get("content_types") or [])


def passes_crop_filter(section: dict[str, Any]) -> bool:
    return bool(section_fields(section) or section_content_types(section))


def filter_reason(section: dict[str, Any]) -> str:
    fields = section_fields(section)
    ctypes = section_content_types(section)
    if fields and ctypes:
        return "fields + content_types"
    if fields:
        return "fields"
    if ctypes:
        return "content_types"
    return "filtered out"


def row_tags(row: dict[str, Any]) -> set[str]:
    tags: set[str] = {row.get("partner") or "unknown"}
    summary = row.get("content_types_summary") or {}
    for ct in INTEREST_CONTENT_TYPES:
        if summary.get(ct, 0) > 0:
            tags.add(ct)
    section_count = int(row.get("section_count") or 0)
    if section_count >= 12:
        tags.add("high_sections")
    elif section_count <= 3:
        tags.add("low_sections")
    else:
        tags.add("mid_sections")
    if row.get("page", "p1") != "p1":
        tags.add("multi_page")
    return tags


def pick_diverse_docs(manifest: list[dict[str, Any]], n: int = PICK_COUNT) -> list[dict[str, Any]]:
    if len(manifest) <= n:
        return list(manifest)

    annotated = [(row, row_tags(row)) for row in manifest]
    covered: set[str] = set()
    picked: list[dict[str, Any]] = []
    remaining = list(annotated)

    def score(item: tuple[dict[str, Any], set[str]]) -> tuple[int, int, int]:
        _row, tags = item
        new_tags = tags - covered
        partner_bonus = 0
        if _row.get("partner") == "routeone":
            partner_bonus = max(0, 5 - sum(1 for p in picked if p.get("partner") == "routeone"))
        elif _row.get("partner") == "dealertrack":
            partner_bonus = max(0, 5 - sum(1 for p in picked if p.get("partner") == "dealertrack"))
        return (len(new_tags) + partner_bonus, int(_row.get("section_count") or 0), hash(_row["short_id"]) % 1000)

    while len(picked) < n and remaining:
        remaining.sort(key=score, reverse=True)
        row, tags = remaining.pop(0)
        picked.append(row)
        covered |= tags

    return picked


def crop_section_thumb(
    full_png: Path,
    bounds: dict[str, Any],
    out_path: Path,
    *,
    max_width: int = 180,
    padding: int = 6,
) -> bool:
    if not full_png.exists() or not bounds:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(full_png) as img:
        x0 = max(0, int(bounds.get("min_x", 0)) - padding)
        y0 = max(0, int(bounds.get("min_y", 0)) - padding)
        x1 = min(img.width, int(bounds.get("max_x", 0)) + padding)
        y1 = min(img.height, int(bounds.get("max_y", 0)) + padding)
        if x1 <= x0 or y1 <= y0:
            return False
        crop = img.crop((x0, y0, x1, y1))
        if crop.width > max_width:
            ratio = max_width / crop.width
            crop = crop.resize((max_width, max(1, int(crop.height * ratio))), Image.Resampling.LANCZOS)
        crop.save(out_path)
    return True


def load_sections(row: dict[str, Any]) -> list[dict[str, Any]]:
    sections_json = Path(row.get("sections_json") or "")
    if not sections_json.exists():
        classified = Path(row.get("sections_classified_json") or "")
        if classified.exists():
            sections_json = classified
        else:
            return []
    payload = json.loads(sections_json.read_text())
    return payload.get("sections") or []


def build_card(row: dict[str, Any], sections: list[dict[str, Any]]) -> str:
    short_id = row["short_id"]
    partner = row["partner"]
    page = row.get("page", "p1")
    full_png = Path(row["full_png"])
    sections_png = Path(row["sections_png"])
    crop_dir = CROP_ROOT / short_id

    kept = [s for s in sections if passes_crop_filter(s)]
    all_fields: set[str] = set()
    all_ctypes: set[str] = set()
    for section in kept:
        all_fields.update(section_fields(section))
        all_ctypes.update(section_content_types(section))

    table_rows: list[str] = []
    for section in kept:
        idx = section.get("index", "?")
        ctypes = section_content_types(section)
        fields = section_fields(section)
        bounds = section.get("bounds")
        reason = filter_reason(section)
        ctype_badges = "".join(
            f'<span class="ctype" style="border-color:{CONTENT_TYPE_COLORS.get(ct, "#999")}">{html.escape(ct)}</span>'
            for ct in ctypes
        ) or '<span class="muted">—</span>'
        fields_txt = html.escape(", ".join(fields)) if fields else '<span class="muted">—</span>'

        thumb_html = '<span class="muted">—</span>'
        thumb_path = crop_dir / f"s{idx}.png"
        if crop_section_thumb(full_png, bounds or {}, thumb_path):
            thumb_html = (
                f'<img class="thumb" src="{file_url(thumb_path)}" '
                f'alt="S{idx} crop" title="S{idx}" />'
            )

        table_rows.append(
            f"""<tr>
  <td class="mono">S{idx}</td>
  <td class="mono bounds">{html.escape(bounds_summary(bounds))}</td>
  <td>{ctype_badges}</td>
  <td class="fields">{fields_txt}</td>
  <td class="reason">{html.escape(reason)}</td>
  <td class="thumb-cell">{thumb_html}</td>
</tr>"""
        )

    if not table_rows:
        table_rows.append(
            '<tr><td colspan="6" class="muted">No sections passed crop filter</td></tr>'
        )

    final_ctypes = ", ".join(sorted(all_ctypes)) or "—"
    final_fields = ", ".join(sorted(all_fields)) or "—"
    filtered_out = len(sections) - len(kept)

    return f"""
<article class="card" id="{html.escape(short_id)}">
  <header>
    <div class="title-row">
      <span class="doc-id">{html.escape(short_id)}</span>
      <span class="badge partner">{html.escape(partner)}</span>
      <span class="badge">{html.escape(page)}</span>
      <span class="meta">{len(sections)} sections · {len(kept)} kept · {filtered_out} filtered out</span>
    </div>
    <div class="summary-types">{html.escape(str(row.get('content_types_summary') or {}))}</div>
  </header>

  <div class="pair">
    <figure>
      <img src="{file_url(full_png)}" alt="full {html.escape(short_id)}" loading="lazy" />
      <figcaption>Full page</figcaption>
    </figure>
    <figure>
      <img src="{file_url(sections_png)}" alt="sections {html.escape(short_id)}" loading="lazy" />
      <figcaption>Sections overlay</figcaption>
    </figure>
  </div>

  <h3>Crop filtering — kept sections</h3>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>S-index</th>
          <th>Bounds</th>
          <th>Content types</th>
          <th>Fields</th>
          <th>Filter reason</th>
          <th>Crop</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </div>

  <h3>Final values (aggregated across kept sections)</h3>
  <div class="final-values">
    <p><strong>Content types:</strong> <span class="mono">{html.escape(final_ctypes)}</span></p>
    <p><strong>Fields:</strong> <span class="mono fields-block">{html.escape(final_fields)}</span></p>
  </div>
</article>"""


def build_html(picked: list[dict[str, Any]], out_path: Path) -> None:
    legend_items = "".join(
        f'<span class="legend" style="border-color:{color}">{html.escape(ct)}</span>'
        for ct, color in sorted(CONTENT_TYPE_COLORS.items())
    )
    cards: list[str] = []
    for row in picked:
        sections = load_sections(row)
        cards.append(build_card(row, sections))

    doc_list = ", ".join(f"{r['partner']}/{r['short_id']} {r.get('page','p1')}" for r in picked)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Section Content + Field Classifier Gallery</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 24px;
      background: #f0f2f8;
      color: #1a1a2e;
      line-height: 1.45;
    }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 36px; border-bottom: 2px solid #ccd; padding-bottom: 6px; }}
    h3 {{ margin: 18px 0 8px; font-size: 15px; color: #334; }}
    .sub {{ color: #555; margin-bottom: 16px; max-width: 1100px; }}
    .doc-list {{ font-family: ui-monospace, monospace; font-size: 12px; color: #444; margin-bottom: 20px; }}
    .legend-wrap {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 24px; }}
    .legend {{
      font-size: 11px; padding: 2px 6px; border-left: 4px solid #999;
      background: #fff; border-radius: 3px;
    }}
    .grid {{ display: flex; flex-direction: column; gap: 32px; }}
    .card {{
      background: #fff; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,.08);
      padding: 18px;
    }}
    .title-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 6px; }}
    .doc-id {{ font-family: ui-monospace, monospace; font-weight: 700; font-size: 16px; }}
    .badge {{
      font-size: 11px; padding: 2px 8px; border-radius: 999px;
      background: #e8ecf4; color: #334;
    }}
    .badge.partner {{ background: #dbeafe; color: #1e3a8a; }}
    .meta {{ color: #666; font-size: 13px; }}
    .summary-types {{ font-family: ui-monospace, monospace; font-size: 11px; color: #555; }}
    .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 14px 0; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
    figcaption {{ text-align: center; font-size: 12px; color: #777; margin-top: 4px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border: 1px solid #e2e6ef; padding: 6px 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f6f8fc; font-weight: 600; }}
    tr:nth-child(even) td {{ background: #fafbfd; }}
    .mono {{ font-family: ui-monospace, monospace; }}
    .bounds {{ font-size: 11px; white-space: nowrap; }}
    .fields {{ font-size: 11px; max-width: 360px; word-break: break-word; }}
    .fields-block {{ display: block; margin-top: 4px; word-break: break-word; }}
    .reason {{ font-size: 11px; color: #555; white-space: nowrap; }}
    .ctype {{
      display: inline-block; font-size: 10px; padding: 1px 5px; margin: 1px 2px 1px 0;
      border-left: 3px solid #999; background: #f8f9fb; border-radius: 2px;
    }}
    .thumb {{ max-width: 160px; max-height: 80px; border: 1px solid #ccc; border-radius: 3px; }}
    .thumb-cell {{ width: 170px; }}
    .muted {{ color: #999; }}
    .final-values {{ background: #f6f8fc; border-radius: 6px; padding: 10px 12px; font-size: 13px; }}
    @media (max-width: 900px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Section Content + Field Classifier</h1>
  <p class="sub">
    Credit application pages — full page, section overlays, crop-filtered sections
    (non-empty <code>fields</code> or <code>content_types</code>), and aggregated final values.
  </p>
  <p class="doc-list">Selected ({len(picked)}): {html.escape(doc_list)}</p>
  <div class="legend-wrap">{legend_items}</div>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"Missing manifest: {MANIFEST_PATH}")
        sys.exit(1)

    manifest_data = json.loads(MANIFEST_PATH.read_text())
    manifest: list[dict[str, Any]] = manifest_data.get("results") or []
    if not manifest:
        print("Manifest has no results")
        sys.exit(1)

    picked = pick_diverse_docs(manifest, PICK_COUNT)
    build_html(picked, DESKTOP_HTML)
    REPO_HTML.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DESKTOP_HTML, REPO_HTML)

    print(f"Selected {len(picked)} docs:")
    for row in picked:
        sections = load_sections(row)
        kept = sum(1 for s in sections if passes_crop_filter(s))
        print(
            f"  {row['partner']:12} {row['short_id']} {row.get('page','p1'):4} "
            f"sections={row.get('section_count')} kept={kept}"
        )
    print(f"\nDesktop HTML: {DESKTOP_HTML}")
    print(f"Repo copy:    {REPO_HTML}")
    print(f"Crop dir:     {CROP_ROOT}")


if __name__ == "__main__":
    main()
