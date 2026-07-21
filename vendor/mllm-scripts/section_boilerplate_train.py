#!/usr/bin/env python3
"""Train FastText binary classifier: has_form_data vs boilerplate sections."""

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

from section_boilerplate_classifier import (  # noqa: E402
    LABEL_BOILERPLATE,
    LABEL_HAS_FORM_DATA,
    is_boilerplate_heuristic,
    _has_strong_form_rows,
)
from section_content_heuristics import preprocess_for_fasttext  # noqa: E402
from section_content_heuristics import normalize_ocr_text  # noqa: E402
from section_field_classifier import has_extractable_values  # noqa: E402

ROOT = SCRIPT_DIR.parents[4]
OUT_DIR = ROOT / "wa577_gallery" / "section_classifier"
CORPUS_DIR = OUT_DIR / "corpus" / "has_data"
MODEL_DIR = OUT_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "section_has_data.bin"

DOC_SOURCES: dict[str, list[Path]] = {
    "credit_application": [
        ROOT / "wa577_gallery" / "credit_app_sections",
    ],
    "title_application": [
        ROOT / "wa577_gallery" / "vin_sections_batch",
    ],
}


def _iter_sections_json(paths: list[Path], *, max_files: int = 200) -> Iterator[dict[str, Any]]:
    seen_files = 0
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*_sections.json")):
            if "_classified" in path.name:
                continue
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
                if len(text) < 20:
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


def weak_label_row(row: dict[str, Any]) -> str:
    text = row["text"]
    norm = normalize_ocr_text(text)
    if row.get("label") in (LABEL_BOILERPLATE, LABEL_HAS_FORM_DATA):
        return row["label"]
    if is_boilerplate_heuristic(text):
        return LABEL_BOILERPLATE
    if _has_strong_form_rows(text, norm):
        return LABEL_HAS_FORM_DATA
    if has_extractable_values(text):
        return LABEL_HAS_FORM_DATA
    return LABEL_BOILERPLATE


def _synthetic_rows() -> list[dict[str, Any]]:
    """Hand-authored snippets including legal disclosure consent prose."""
    boilerplate_samples = [
        (
            'The words "we," "us," "our" and "ours" as used below refer to us, the dealer, '
            "and to the financial institution(s) selected to receive your application. You "
            "understand and agree that you are applying for credit by providing the information "
            "to complete and submit this credit application. In accordance with the Fair Credit "
            "Reporting Act, you authorize that such financial institutions may submit your "
            "applications to other financial institutions. You agree that we may obtain a "
            "consumer credit report periodically from one or more consumer reporting agencies "
            "(credit bureaus). You agree that the dealer and the financial institutions may "
            "verify your employment, pay, assets and debts. The dealer and the financial "
            "institutions may monitor and record telephone calls regarding your account for "
            "quality assurance, compliance, training, or similar purposes."
        ),
        (
            "FEDERAL NOTICES\nIMPORTANT INFORMATION ABOUT PROCEDURES FOR OPENING A NEW ACCOUNT "
            "To help the government fight the funding of terrorism and money laundering activities, "
            "Federal law requires all financial institutions to obtain, verify, and record "
            "information that identifies each person who opens an account."
        ),
        (
            "STATE NOTICES\nCalifornia Residents: An applicant, if married, may apply for a separate account.\n"
            "Married Wisconsin Residents: complete Section A about yourself and Section B about your spouse."
        ),
        (
            "You consent to receive autodialed, pre-recorded and artificial voice telemarketing "
            "calls from the dealer at the following number(s) (843)377-7187 including any cell "
            "phone number you provide."
        ),
        (
            "BY SIGNING BELOW YOU CERTIFY THAT YOU HAVE READ AND AGREE TO THE TERMS\n"
            "APPLICANT'S SIGNATURE DATE\nPage 1 of 4\n©2026 Dealertrack, Inc. All rights reserved."
        ),
    ]
    form_samples = [
        (
            "Last Name First Name Middle Initial Social Security Number Birth Date\n"
            "SMITH JOHN A 123-45-6789 01/15/1990"
        ),
        (
            "Present Address City State Zip\n"
            "123 Main Street Stamford CT 06907"
        ),
        (
            "Employer Occupation Length of Employment\n"
            "ACME CORP ENGINEER 24 months"
        ),
        (
            "Year Make Vehicle Identification Number\n"
            "2020 TOYOTA 1HGBH41JXMN109186"
        ),
        (
            "B. CO-APPLICANT INFORMATION\n"
            "Last Name First Name SSN\n"
            "DOE JANE 242-39-2813"
        ),
        (
            "Home Phone Cell Phone Email\n"
            "(555)123-4567 (555)987-6543 jane.doe@example.com"
        ),
    ]
    rows: list[dict[str, Any]] = []
    for i, text in enumerate(boilerplate_samples):
        rows.append(
            {
                "document_type": "credit_application",
                "text": text,
                "line_count": text.count("\n") + 1,
                "source": "synthetic_boilerplate",
                "section_index": i,
                "label": LABEL_BOILERPLATE,
            }
        )
    for i, text in enumerate(form_samples):
        rows.append(
            {
                "document_type": "credit_application",
                "text": text,
                "line_count": text.count("\n") + 1,
                "source": "synthetic_form",
                "section_index": i,
                "label": LABEL_HAS_FORM_DATA,
            }
        )
    return rows


def build_corpus(
    *,
    max_files: int = 200,
) -> tuple[list[dict[str, Any]], Path, Path]:
    rows: list[dict[str, Any]] = []
    for doc_type, roots in DOC_SOURCES.items():
        for item in _iter_sections_json(roots, max_files=max_files):
            item["document_type"] = doc_type
            rows.append(item)
    rows.extend(_synthetic_rows())

    seen: set[str] = set()
    labeled: list[dict[str, Any]] = []
    for row in rows:
        key = preprocess_for_fasttext(row["text"])[:160]
        if key in seen:
            continue
        seen.add(key)
        label = weak_label_row(row)
        labeled.append({**row, "label": label})

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    all_path = CORPUS_DIR / "sections_labeled.jsonl"
    with all_path.open("w") as fh:
        for row in labeled:
            fh.write(json.dumps(row) + "\n")

    rng = random.Random(577)
    rng.shuffle(labeled)
    split = max(1, int(len(labeled) * 0.15))
    val_rows = labeled[:split]
    train_rows = labeled[split:]

    train_path = CORPUS_DIR / "train.txt"
    val_path = CORPUS_DIR / "val.txt"
    _write_fasttext(train_path, train_rows)
    _write_fasttext(val_path, val_rows)
    return labeled, train_path, val_path


def _write_fasttext(path: Path, rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for row in rows:
        label = row.get("label") or weak_label_row(row)
        text = preprocess_for_fasttext(row["text"])
        if not text:
            continue
        lines.append(f"__label__{label} {text}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def train_fasttext(
    train_path: Path,
    val_path: Path | None = None,
    out_path: Path | None = None,
) -> tuple[Path, dict[str, float]]:
    import fasttext  # noqa: PLC0415

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = out_path or DEFAULT_MODEL_PATH
    model = fasttext.train_supervised(
        input=str(train_path),
        lr=0.5,
        epoch=30,
        wordNgrams=2,
        minCount=1,
        loss="softmax",
    )
    metrics: dict[str, float] = {}
    for split_name, split_path in (("train", train_path), ("val", val_path)):
        if split_path and split_path.exists() and split_path.read_text().strip():
            n, prec, rec = model.test(str(split_path))
            metrics[f"{split_name}_n"] = float(n)
            metrics[f"{split_name}_precision"] = float(prec)
            metrics[f"{split_name}_recall"] = float(rec)
            print(f"{split_name} n={n} precision={prec:.4f} recall={rec:.4f}")
    model.save_model(str(out))
    return out, metrics


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train section has_form_data FastText model")
    ap.add_argument("--build-corpus-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--max-files", type=int, default=200)
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = ap.parse_args()

    if not args.train_only:
        print("Building corpus...", flush=True)
        rows, train_path, val_path = build_corpus(max_files=args.max_files)
        print(
            f"Corpus: {len(rows)} sections -> {train_path} "
            f"({sum(1 for _ in train_path.open())} train lines)"
        )
        bp = sum(1 for r in rows if r["label"] == LABEL_BOILERPLATE)
        fd = sum(1 for r in rows if r["label"] == LABEL_HAS_FORM_DATA)
        print(f"  boilerplate={bp} has_form_data={fd}")
        if args.build_corpus_only:
            return
    else:
        train_path = CORPUS_DIR / "train.txt"
        val_path = CORPUS_DIR / "val.txt"

    model_path, metrics = train_fasttext(train_path, val_path, args.out)
    print(f"Model: {model_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
