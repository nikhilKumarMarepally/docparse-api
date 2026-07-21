#!/usr/bin/env python3
"""Collect docs where norm(full_page_vin) != norm(crop_vin) from wa577_gallery artifacts."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
OUT_DIR = ROOT / "wa577_gallery/vin_full_ne_crop_20"
DESKTOP_HTML = Path("/Users/nikhilmarepally/Desktop/vin_full_ne_crop_20.html")
TARGET = 20


def norm_vin(v: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(v or "")).upper()


def row_disagrees(r: dict[str, Any]) -> bool:
    return norm_vin(r.get("full_page_vin")) != norm_vin(r.get("crop_vin"))


def collect_rows() -> list[dict[str, Any]]:
    disagrees: dict[tuple[str, str], dict[str, Any]] = {}
    skip = (ROOT / "wa577_gallery/vin_full_ne_crop_20/results.json",)
    for p in sorted((ROOT / "wa577_gallery").rglob("results.json")):
        if p in skip:
            continue
        data = json.loads(p.read_text())
        for r in data.get("results", []):
            if not row_disagrees(r):
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
                if not row_disagrees(r):
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
    """Prefer wins/ticket artifacts and rows that already have crop_path recorded."""
    source = row.get("source") or ""
    if "wins.json" in source:
        tier = 3
    elif "vin_ticket_crop_ab" in source or "vin_crop_tickets" in source:
        tier = 2
    elif "vin_crop_ab" in source:
        tier = 0
    else:
        tier = 1
    has_crop = int(bool(row.get("crop_path")))
    return (tier, has_crop)


def _row_from_result(r: dict[str, Any], short: str, page: str, source: str) -> dict[str, Any]:
    return {
        "short_id": short,
        "doc_id": r.get("doc_id") or short,
        "document_type": r.get("document_type", "title_application"),
        "partner": r.get("partner"),
        "page": page,
        "full_page_vin": r.get("full_page_vin"),
        "crop_vin": r.get("crop_vin"),
        "image_path": r.get("image_path"),
        "crop_path": r.get("crop_path"),
        "source": source,
    }


def resolve_paths(row: dict[str, Any]) -> dict[str, Any]:
    """Keep source-recorded paths when files exist; only search galleries as fallback."""
    short, page = row["short_id"], row["page"]

    for key in ("image_path", "crop_path"):
        p = row.get(key)
        if p and Path(p).exists():
            row[key] = str(Path(p).resolve())

    if row.get("image_path") and row.get("crop_path"):
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
        if not row.get("image_path"):
            img = base / f"{short}_{page}.png"
            if img.exists():
                row["image_path"] = str(img.resolve())
        if not row.get("crop_path"):
            crop = base / f"{short}_{page}_vin_crop.png"
            if not crop.exists():
                crop = base / f"{short}_vin_crop.png"
            if crop.exists():
                row["crop_path"] = str(crop.resolve())
    return row


def score_row(r: dict[str, Any]) -> tuple[int, int, int]:
    both = int(r.get("full_page_vin") is not None and r.get("crop_vin") is not None)
    imgs = int(bool(r.get("image_path")) and bool(r.get("crop_path")))
    dtype = 0 if r.get("document_type") == "title_application" else 1
    return (both, imgs, dtype)


def select_rows(rows: list[dict[str, Any]], target: int = TARGET) -> list[dict[str, Any]]:
    resolved = [resolve_paths(r) for r in rows]
    resolved.sort(key=score_row, reverse=True)
    selected: list[dict[str, Any]] = []
    seen_short: set[str] = set()
    for r in resolved:
        if len(selected) >= target:
            break
        if r["short_id"] in seen_short:
            continue
        if not r.get("image_path"):
            continue
        selected.append(r)
        seen_short.add(r["short_id"])
    return selected


def vin_display(v: Any) -> str:
    if v is None:
        return "null"
    return str(v)


def render_table_md(rows: list[dict[str, Any]], pool_size: int) -> str:
    lines = [
        "# Full-page VIN ≠ crop VIN (normalized)",
        "",
        f"**{len(rows)}** docs selected from **{pool_size}** disagreements in `wa577_gallery` "
        "(null vs non-null counts as disagree). gemini-3.1-flash-lite + QA prompt per doc type.",
        "",
        "| Doc | Doc type | Partner | Full-page VIN | Crop VIN | Full image | Crop image |",
        "|-----|----------|---------|---------------|----------|------------|------------|",
    ]
    for r in rows:
        img = r.get("image_path") or ""
        crop = r.get("crop_path") or ""
        img_link = f"[full](file://{img})" if img else "—"
        crop_link = f"[crop](file://{crop})" if crop else "—"
        lines.append(
            f"| `{r['short_id']}` | {r['document_type']} | {r.get('partner') or '—'} | "
            f"`{vin_display(r.get('full_page_vin'))}` | `{vin_display(r.get('crop_vin'))}` | "
            f"{img_link} | {crop_link} |"
        )
    return "\n".join(lines) + "\n"


def load_vision_review() -> dict[str, dict[str, str]]:
    path = OUT_DIR / "vision_review.md"
    if not path.exists():
        return {}
    review: dict[str, dict[str, str]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 6 or parts[0] == "Doc":
            continue
        short = parts[0].strip("`")
        winner = parts[5].strip("*").strip()
        review[short] = {
            "printed_vin": parts[3].strip("`"),
            "winner": winner,
        }
    return review


def winner_class(winner: str) -> str:
    return {
        "crop_wins": "winner-crop",
        "full_wins": "winner-full",
        "both_wrong": "winner-both",
        "crop_null_full_correct": "winner-null-ok",
        "crop_null_full_wrong": "winner-null-bad",
    }.get(winner, "muted")


def vision_subtitle(review: dict[str, dict[str, str]]) -> str:
    if not review:
        return ""
    counts: dict[str, int] = {}
    for row in review.values():
        counts[row["winner"]] = counts.get(row["winner"], 0) + 1
    parts = [
        f"crop wins {counts.get('crop_wins', 0)}",
        f"full wins {counts.get('full_wins', 0)}",
        f"both wrong {counts.get('both_wrong', 0)}",
        f"crop null + full correct {counts.get('crop_null_full_correct', 0)}",
        f"crop null + full wrong {counts.get('crop_null_full_wrong', 0)}",
    ]
    return f'<p class="subtitle" style="margin-top:-1rem">Vision review: {" · ".join(parts)}</p>'


def render_html(rows: list[dict[str, Any]], pool_size: int) -> str:
    review = load_vision_review()
    table_rows = []
    cards = []
    for r in rows:
        short = html.escape(r["short_id"])
        dtype = html.escape(r["document_type"])
        partner = html.escape(str(r.get("partner") or "—"))
        full_v = html.escape(vin_display(r.get("full_page_vin")))
        crop_v = html.escape(vin_display(r.get("crop_vin")))
        img = r.get("image_path") or ""
        crop = r.get("crop_path") or ""
        page = html.escape(r.get("page") or "p0")
        vr = review.get(r["short_id"], {})
        printed = html.escape(vr.get("printed_vin", "—"))
        winner = vr.get("winner", "—")
        wclass = winner_class(winner) if winner != "—" else "muted"
        winner_cell = (
            f'<td class="{wclass}"><code>{html.escape(winner)}</code></td>'
            if review
            else ""
        )
        printed_cell = (
            f'<td class="printed-vin"><code>{printed}</code></td>' if review else ""
        )
        table_rows.append(
            f"""      <tr>
        <td><code>{short}</code></td>
        <td>{dtype}</td>
        <td>{partner}</td>
        <td class="vin-full"><code>{full_v}</code></td>
        <td class="vin-crop"><code>{crop_v}</code></td>
        {printed_cell}
        {winner_cell}
        <td><a href="file://{html.escape(img)}" style="color:var(--accent)">full</a></td>
        <td><a href="file://{html.escape(crop)}" style="color:var(--accent)">crop</a></td>
      </tr>"""
        )
        crop_img = (
            f'<img src="file://{html.escape(crop)}" alt="{short} VIN crop">'
            if crop
            else '<p class="muted">No crop image</p>'
        )
        cards.append(
            f"""  <div class="doc-card">
    <h2><code>{short}</code> · {dtype} · {partner}</h2>
    <p>Page <code>{page}</code> · Full <code class="vin-full">{full_v}</code> · Crop <code class="vin-crop">{crop_v}</code></p>
    <div class="images">
      <div class="img-block">
        <label>Full page ({page})</label>
        <img src="file://{html.escape(img)}" alt="{short} full page">
      </div>
      <div class="img-block">
        <label>VIN crop (MLLM input)</label>
        {crop_img}
      </div>
    </div>
  </div>"""
        )

    extra_headers = ""
    if review:
        extra_headers = "\n        <th>Printed VIN</th>\n        <th>Winner</th>"
    extra_styles = ""
    if review:
        extra_styles = """
    .winner-crop { color: #60a5fa; }
    .winner-full { color: #fbbf24; }
    .winner-both { color: #f87171; }
    .winner-null-ok { color: #4ade80; }
    .winner-null-bad { color: #fb923c; }
    .printed-vin { color: #a78bfa; }
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VIN full-page ≠ crop — 20 docs</title>
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --border: #2d3a4d;
      --text: #e8edf4;
      --muted: #8b9cb3;
      --full: #fbbf24;
      --crop: #60a5fa;
      --accent: #a78bfa;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 2rem;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    h1 {{ margin: 0 0 0.5rem; font-size: 1.5rem; }}
    .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 2.5rem;
      background: var(--surface);
      border-radius: 8px;
      overflow: hidden;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 0.65rem 0.85rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      background: #243044;
      font-weight: 600;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: var(--muted);
    }}
    tr:last-child td {{ border-bottom: none; }}
    code {{
      font-family: "SF Mono", Menlo, Monaco, monospace;
      font-size: 0.88em;
      background: #0d1117;
      padding: 0.12em 0.35em;
      border-radius: 4px;
      word-break: break-all;
    }}
    .vin-full {{ color: var(--full); }}
    .vin-crop {{ color: var(--crop); }}
    .doc-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.5rem;
      margin-bottom: 2rem;
    }}
    .doc-card h2 {{ margin: 0 0 0.75rem; font-size: 1.1rem; }}
    .images {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }}
    @media (max-width: 900px) {{ .images {{ grid-template-columns: 1fr; }} }}
    .img-block label {{
      display: block;
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      margin-bottom: 0.5rem;
    }}
    .img-block img {{
      width: 100%;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #000;
    }}
    .muted {{ color: var(--muted); }}{extra_styles}
  </style>
</head>
<body>
  <h1>Full-page VIN ≠ crop VIN</h1>
  <p class="subtitle">{len(rows)} docs · normalized comparison · pool of {pool_size} disagreements from wa577_gallery</p>
{vision_subtitle(review)}

  <table>
    <thead>
      <tr>
        <th>Doc ID</th>
        <th>Doc type</th>
        <th>Partner</th>
        <th>Full-page VIN</th>
        <th>Crop VIN</th>{extra_headers}
        <th>Full image</th>
        <th>Crop image</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(table_rows)}
    </tbody>
  </table>

{chr(10).join(cards)}
</body>
</html>
"""


def main() -> None:
    pool = collect_rows()
    selected = select_rows(pool, TARGET)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = {
        "summary": {
            "pool_disagreements": len(pool),
            "selected": len(selected),
            "target": TARGET,
            "comparison": "norm(full_page_vin) != norm(crop_vin); null vs non-null is disagree",
            "model": "gemini-3.1-flash-lite",
            "prompt": "QA per doc type",
        },
        "results": selected,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "results_table.md").write_text(render_table_md(selected, len(pool)))
    DESKTOP_HTML.write_text(render_html(selected, len(pool)))

    print(f"Pool disagreements: {len(pool)}")
    print(f"Selected: {len(selected)}")
    print(f"Wrote {OUT_DIR / 'results.json'}")
    print(f"Wrote {OUT_DIR / 'results_table.md'}")
    print(f"Wrote {DESKTOP_HTML}")


if __name__ == "__main__":
    main()
