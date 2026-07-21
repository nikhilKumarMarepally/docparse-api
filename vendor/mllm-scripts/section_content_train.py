#!/usr/bin/env python3
"""Mine weak-labeled section crops and train generic FastText content-type model."""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_credit_app_sections import gemini_ocr_to_vision  # noqa: E402
from ocr_line_to_sections import lines_to_sections  # noqa: E402
from ocr_word_to_line_boxes import load_words, words_to_lines  # noqa: E402
from section_content_classifier import preprocess_for_fasttext  # noqa: E402
from section_content_heuristics import (  # noqa: E402
    _LABEL_ANCHORS,
    classify_content_types_heuristic,
)
from section_content_taxonomy import TARGET_DOC_TYPES  # noqa: E402

ROOT = SCRIPT_DIR.parents[4]
OUT_DIR = ROOT / "wa577_gallery" / "section_classifier"
CORPUS_DIR = OUT_DIR / "corpus"
MODEL_DIR = OUT_DIR / "models"
CA_APPCTX = Path("/tmp/ca_appctx")

# Binary has_form_data vs boilerplate gate: see section_boilerplate_train.py

# Doc type -> gallery / OCR source roots (best-effort offline mining).
DOC_SOURCES: dict[str, list[Path]] = {
    "credit_application": [
        ROOT / "wa577_gallery" / "credit_app_sections",
    ],
    "title_application": [
        ROOT / "wa577_gallery" / "vin_sections_batch",
    ],
}


def _iter_sections_json(paths: list[Path], *, max_files: int = 120) -> Iterator[dict[str, Any]]:
    seen_files = 0
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*_sections.json")):
            if seen_files >= max_files:
                return
            seen_files += 1
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            doc_type = payload.get("document_type") or _infer_doc_type(path)
            for section in payload.get("sections") or []:
                text = (section.get("text") or "").strip()
                if len(text) < 12:
                    continue
                yield {
                    "document_type": doc_type,
                    "text": text,
                    "line_count": section.get("line_count"),
                    "source": str(path),
                    "section_index": section.get("index"),
                }


def _infer_doc_type(path: Path) -> str:
    p = str(path).lower()
    if "credit_app" in p:
        return "credit_application"
    if "vin_" in p or "title" in p:
        return "title_application"
    return "unknown"


def _mine_from_ca_appctx(doc_type: str = "credit_application", limit: int = 500) -> list[dict[str, Any]]:
    """Mine sections from /tmp/ca_appctx OCR when gallery JSON is sparse."""
    rows: list[dict[str, Any]] = []
    for cohort in ("routeone", "dealertrack"):
        work = CA_APPCTX / cohort
        ocr_dir = work / "clustering" / "ocr"
        assets_dir = work / "clustering" / "assets"
        if not ocr_dir.exists():
            continue
        ocr_files = sorted(ocr_dir.glob("*.json"))[: limit // 2]
        for ocr_path in ocr_files:
            m = re.match(r"^([0-9a-f]{8})-.+_p(\d+)\.json$", ocr_path.name, re.I)
            if not m:
                continue
            asset = ocr_path.stem
            image = assets_dir / asset / "page.png"
            if not image.exists():
                continue
            try:
                from PIL import Image  # noqa: PLC0415

                with Image.open(image) as img:
                    w, h = img.size
                payload = json.loads(ocr_path.read_text())
                vision = gemini_ocr_to_vision(payload, w, h)
                words = load_words(vision)
                lines = words_to_lines(words, page_width=w, full_width=True)
                sections, _ = lines_to_sections(lines)
                for section in sections:
                    text = section.text.strip()
                    if len(text) < 12:
                        continue
                    rows.append(
                        {
                            "document_type": doc_type,
                            "text": text,
                            "line_count": len(section.lines),
                            "source": str(ocr_path),
                            "section_index": section.index,
                        }
                    )
            except Exception:
                continue
    return rows


def weak_label_row(row: dict[str, Any]) -> list[str]:
    present, _ = classify_content_types_heuristic(
        row["text"],
        line_count=row.get("line_count"),
        threshold=0.30,
    )
    return present


def _synthetic_anchor_rows() -> list[dict[str, Any]]:
    """Synthetic cross-doc snippets from generic label anchors."""
    rows: list[dict[str, Any]] = []
    doc_types = [
        "credit_application",
        "title_application",
        "retail_installment_sales_contract",
        "gap_binder",
        "buyers_order",
        "odometer_disclosure_statement_retail",
        "vehicle_service_contract",
    ]
    for i, (ctype, anchors) in enumerate(_LABEL_ANCHORS.items()):
        doc_type = doc_types[i % len(doc_types)]
        for phrase, _weight in anchors[:3]:
            text = f"{phrase.replace('_', ' ').title()}\nSample value line for training"
            rows.append(
                {
                    "document_type": doc_type,
                    "text": text,
                    "line_count": 2,
                    "source": "synthetic_anchor",
                    "section_index": 0,
                    "labels": [ctype],
                }
            )
    return rows


def build_corpus(
    *,
    doc_types: tuple[str, ...] | None = None,
    extra_limit: int = 200,
    skip_ca_appctx: bool = False,
) -> tuple[list[dict[str, Any]], Path, Path]:
    wanted = set(doc_types or TARGET_DOC_TYPES)
    rows: list[dict[str, Any]] = []
    for doc_type, roots in DOC_SOURCES.items():
        if doc_type not in wanted:
            continue
        for item in _iter_sections_json(roots):
            item["document_type"] = doc_type
            rows.append(item)

    if "credit_application" in wanted and not skip_ca_appctx:
        rows.extend(_mine_from_ca_appctx("credit_application", limit=extra_limit))

    rows.extend(_synthetic_anchor_rows())

    # Deduplicate by normalized text prefix.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = preprocess_for_fasttext(row["text"])[:120]
        if key in seen:
            continue
        seen.add(key)
        labeled = dict(row)
        if row.get("labels"):
            labeled["labels"] = row["labels"]
        else:
            labeled["labels"] = weak_label_row(row)
        if labeled["labels"]:
            deduped.append(labeled)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    all_path = CORPUS_DIR / "sections_labeled.jsonl"
    with all_path.open("w") as fh:
        for row in deduped:
            fh.write(json.dumps(row) + "\n")

    by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in deduped:
        by_doc.setdefault(row["document_type"], []).append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    rng = random.Random(577)
    for doc_type, doc_rows in by_doc.items():
        rng.shuffle(doc_rows)
        split = max(1, int(len(doc_rows) * 0.15))
        val_rows.extend(doc_rows[:split])
        train_rows.extend(doc_rows[split:])

    train_path = CORPUS_DIR / "train.txt"
    val_path = CORPUS_DIR / "val.txt"
    _write_fasttext(train_path, train_rows)
    _write_fasttext(val_path, val_rows)
    return deduped, train_path, val_path


def _write_fasttext(path: Path, rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for row in rows:
        labels = row.get("labels") or weak_label_row(row)
        if not labels:
            continue
        label_str = " ".join(f"__label__{l}" for l in sorted(set(labels)))
        text = preprocess_for_fasttext(row["text"])
        if not text:
            continue
        lines.append(f"{label_str} {text}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def train_fasttext(
    train_path: Path,
    val_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    import fasttext  # noqa: PLC0415

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = out_path or MODEL_DIR / "content_types.bin"
    model = fasttext.train_supervised(
        input=str(train_path),
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        minCount=1,
        loss="ova",
    )
    if val_path and val_path.exists() and val_path.read_text().strip():
        n, prec, rec = model.test(str(val_path), k=-1, threshold=0.25)
        print(f"val n={n} precision={prec:.4f} recall={rec:.4f}")
    model.save_model(str(out))
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train generic section content FastText model")
    ap.add_argument("--build-corpus-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--extra-limit", type=int, default=200)
    ap.add_argument("--skip-ca-appctx", action="store_true", help="Use gallery JSON only (faster)")
    args = ap.parse_args()

    if not args.train_only:
        print("Building corpus...", flush=True)
        rows, train_path, val_path = build_corpus(
            extra_limit=args.extra_limit,
            skip_ca_appctx=args.skip_ca_appctx,
        )
        print(f"Corpus: {len(rows)} sections -> {train_path} ({sum(1 for _ in train_path.open())} lines)")
        if args.build_corpus_only:
            return
    else:
        train_path = CORPUS_DIR / "train.txt"
        val_path = CORPUS_DIR / "val.txt"

    model_path = train_fasttext(train_path, val_path)
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
