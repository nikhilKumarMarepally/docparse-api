#!/usr/bin/env python3
"""Train FastText multi-label section field classifier (field paths or empty)."""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from section_boilerplate_classifier import is_boilerplate_heuristic  # noqa: E402
from section_content_classifier import preprocess_for_fasttext  # noqa: E402
from section_field_classifier import (  # noqa: E402
    LABEL_EMPTY,
    FastTextFieldClassifier,
    _classify_section_fields_heuristic,
    _section_is_label_only,
    classify_section_fields,
    has_extractable_values,
)

ROOT = SCRIPT_DIR.parents[4]
OUT_DIR = ROOT / "wa577_gallery" / "section_classifier"
CORPUS_DIR = OUT_DIR / "corpus" / "fields"
MODEL_DIR = OUT_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "section_fields.bin"

DOC_SOURCES: dict[str, list[Path]] = {
    "credit_application": [
        ROOT / "wa577_gallery" / "credit_app_sections",
    ],
}

EVAL_DOC_IDS = ("6bf89f71", "6f99d76d")


def _iter_sections_json(paths: list[Path], *, max_files: int = 500) -> Iterator[dict[str, Any]]:
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
            doc_type = payload.get("document_type") or "credit_application"
            short_id = payload.get("short_id") or ""
            for section in payload.get("sections") or []:
                text = (section.get("text") or "").strip()
                if len(text) < 12:
                    continue
                yield {
                    "document_type": doc_type,
                    "text": text,
                    "line_count": section.get("line_count"),
                    "source": str(path),
                    "short_id": short_id,
                    "section_index": section.get("index"),
                }


def _synthetic_rows() -> list[dict[str, Any]]:
    """Hand-authored negatives (consent / legal prose) and positive form rows."""
    negatives = [
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
            "calls from the dealer at the following number(s) including any cell phone number you provide."
        ),
        (
            "BY SIGNING BELOW YOU CERTIFY THAT YOU HAVE READ AND AGREE TO THE TERMS\n"
            "APPLICANT'S SIGNATURE DATE\nPage 1 of 4\n©2026 Dealertrack, Inc. All rights reserved."
        ),
        "Last Name\nSSN\nDate of Birth",
        "Yrs. Mos.\nPrevious Full Address (if less than 2 years)City State Zip",
        (
            "Married Wisconsin Residents : complete Section A and Section B about your spouse."
        ),
    ]
    positives = [
        (
            "Last Name First Name Middle Initial Social Security Number Birth Date\n"
            "SMITH JOHN A 123-45-6789 01/15/1990"
        ),
        "Present Address City State Zip\n123 Main Street Stamford CT 06907",
        "Employer Occupation Length of Employment\nACME CORP ENGINEER 24 months",
        "Year Make Vehicle Identification Number\n2020 TOYOTA 1HGBH41JXMN109186",
        (
            "B. CO-APPLICANT INFORMATION\n"
            "Last Name First Name SSN\n"
            "DOE JANE 242-39-2813"
        ),
        "Home Phone Cell Phone Email\n(555)123-4567 (555)987-6543 jane.doe@example.com",
        (
            "Trade Name of Business Tax ID\n"
            "Springdale Developers LLC 33-3229251"
        ),
    ]
    rows: list[dict[str, Any]] = []
    for i, text in enumerate(negatives):
        rows.append(
            {
                "document_type": "credit_application",
                "text": text,
                "line_count": text.count("\n") + 1,
                "source": "synthetic_negative",
                "section_index": i,
                "labels": [LABEL_EMPTY],
            }
        )
    for i, text in enumerate(positives):
        rows.append(
            {
                "document_type": "credit_application",
                "text": text,
                "line_count": text.count("\n") + 1,
                "source": "synthetic_positive",
                "section_index": i,
            }
        )
    return rows


def weak_label_fields(row: dict[str, Any]) -> list[str]:
    if row.get("labels"):
        return list(row["labels"])
    text = row["text"]
    doc_type = row.get("document_type") or "credit_application"
    if is_boilerplate_heuristic(text):
        return [LABEL_EMPTY]
    if _section_is_label_only(text):
        return [LABEL_EMPTY]
    if not has_extractable_values(text):
        return [LABEL_EMPTY]
    fields = _classify_section_fields_heuristic(text, doc_type)
    return fields if fields else [LABEL_EMPTY]


def build_corpus(*, max_files: int = 500) -> tuple[list[dict[str, Any]], Path, Path]:
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
        labels = weak_label_fields(row)
        labeled.append({**row, "labels": labels})

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
        labels = row.get("labels") or weak_label_fields(row)
        if not labels:
            continue
        label_str = " ".join(f"__label__{label}" for label in sorted(set(labels)))
        text = preprocess_for_fasttext(row["text"])
        if not text:
            continue
        lines.append(f"{label_str} {text}")
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
        epoch=35,
        wordNgrams=2,
        minCount=1,
        loss="ova",
    )
    metrics: dict[str, float] = {}
    for split_name, split_path in (("train", train_path), ("val", val_path)):
        if split_path and split_path.exists() and split_path.read_text().strip():
            n, prec, rec = model.test(str(split_path), k=-1, threshold=0.35)
            metrics[f"{split_name}_n"] = float(n)
            metrics[f"{split_name}_precision"] = float(prec)
            metrics[f"{split_name}_recall"] = float(rec)
            print(f"{split_name} n={n} precision={prec:.4f} recall={rec:.4f}")
    model.save_model(str(out))
    return out, metrics


def _parse_fasttext_labels(line: str) -> tuple[set[str], str]:
    parts = line.strip().split()
    labels: set[str] = set()
    text_parts: list[str] = []
    for part in parts:
        if part.startswith("__label__"):
            labels.add(part.replace("__label__", ""))
        else:
            text_parts.append(part)
    return labels, " ".join(text_parts)


def per_label_metrics(
    model_path: Path,
    val_path: Path,
    *,
    threshold: float = 0.35,
) -> dict[str, dict[str, float]]:
    import fasttext  # noqa: PLC0415

    model = fasttext.load_model(str(model_path))
    label_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
    )
    for line in val_path.read_text().splitlines():
        if not line.strip():
            continue
        gold_labels, text = _parse_fasttext_labels(line)
        if not text:
            continue
        for label in gold_labels:
            label_stats[label]["support"] += 1
        labels, probs = model.predict(text, k=-1)
        pred_labels = {
            lbl.replace("__label__", "")
            for lbl, prob in zip(labels, probs)
            if float(prob) >= threshold
        }
        all_labels = set(label_stats.keys()) | gold_labels | pred_labels
        for label in all_labels:
            in_gold = label in gold_labels
            in_pred = label in pred_labels
            if in_gold and in_pred:
                label_stats[label]["tp"] += 1
            elif in_pred:
                label_stats[label]["fp"] += 1
            elif in_gold:
                label_stats[label]["fn"] += 1

    out: dict[str, dict[str, float]] = {}
    for label, stats in sorted(label_stats.items()):
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": stats["support"],
        }
    return out


def tune_threshold(
    model_path: Path,
    val_path: Path,
    *,
    thresholds: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55),
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.35
    best_f1 = -1.0
    best_metrics: dict[str, float] = {}
    for threshold in thresholds:
        per_label = per_label_metrics(model_path, val_path, threshold=threshold)
        if not per_label:
            continue
        macro_f1 = sum(v["f1"] for v in per_label.values()) / len(per_label)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_threshold = threshold
            best_metrics = {
                "macro_f1": round(macro_f1, 4),
                "empty_f1": per_label.get(LABEL_EMPTY, {}).get("f1", 0.0),
            }
    return best_threshold, best_metrics


def eval_gallery_docs(
    model_path: Path,
    *,
    threshold: float = 0.35,
    doc_ids: tuple[str, ...] = EVAL_DOC_IDS,
) -> list[dict[str, Any]]:
    """Before/after examples on held-out gallery docs."""
    ft = FastTextFieldClassifier(model_path)
    examples: list[dict[str, Any]] = []
    gallery = ROOT / "wa577_gallery" / "credit_app_sections"
    for path in sorted(gallery.rglob("*_sections.json")):
        if "_classified" in path.name:
            continue
        payload = json.loads(path.read_text())
        short_id = payload.get("short_id") or ""
        if short_id not in doc_ids and not any(d in str(path) for d in doc_ids):
            continue
        for section in payload.get("sections") or []:
            text = section.get("text") or ""
            before = _classify_section_fields_heuristic(text, "credit_application")
            after = classify_section_fields(
                text,
                "credit_application",
                field_model=ft,
                field_threshold=threshold,
            )
            examples.append(
                {
                    "short_id": short_id,
                    "section_index": section.get("index"),
                    "text_preview": text[:120].replace("\n", " "),
                    "heuristic_fields": before,
                    "fasttext_fields": after,
                }
            )
    return examples


def write_results(
    model_path: Path,
    corpus_stats: dict[str, Any],
    val_metrics: dict[str, float],
    per_label: dict[str, dict[str, float]],
    threshold: float,
    examples: list[dict[str, Any]],
) -> Path:
    results_path = OUT_DIR / "field_classifier_results.md"
    lines = [
        "# Section Field Classifier — FastText direct field prediction",
        "",
        f"Model: `{model_path}`",
        f"Threshold: **{threshold}**",
        "",
        "## Corpus",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Sections | {corpus_stats.get('total', 0)} |",
        f"| Empty labels | {corpus_stats.get('empty', 0)} |",
        f"| Field labels | {corpus_stats.get('with_fields', 0)} |",
        "",
        "## Validation (multi-label OVA)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| N | {val_metrics.get('val_n', 0):.0f} |",
        f"| Precision | {val_metrics.get('val_precision', 0):.4f} |",
        f"| Recall | {val_metrics.get('val_recall', 0):.4f} |",
        "",
        "## Per-label metrics",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|-----|---------|",
    ]
    for label, stats in sorted(per_label.items(), key=lambda kv: (-kv[1]["support"], kv[0])):
        lines.append(
            f"| {label} | {stats['precision']:.3f} | {stats['recall']:.3f} | "
            f"{stats['f1']:.3f} | {stats['support']} |"
        )
    lines.extend(["", "## Gallery eval (6bf89f71, 6f99d76d, legal sections)", ""])
    for ex in examples:
        lines.append(
            f"- **{ex['short_id']}** s{ex['section_index']}: "
            f"heuristic={ex['heuristic_fields'] or '[]'} → fasttext={ex['fasttext_fields'] or '[]'}"
        )
        lines.append(f"  - _{ex['text_preview']}_")
    results_path.write_text("\n".join(lines) + "\n")
    return results_path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train section field-path FastText model")
    ap.add_argument("--build-corpus-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--max-files", type=int, default=500)
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    train_path = CORPUS_DIR / "train.txt"
    val_path = CORPUS_DIR / "val.txt"

    if args.eval_only:
        if not args.out.exists():
            raise SystemExit(f"Model not found: {args.out}")
        threshold = args.threshold or tune_threshold(args.out, val_path)[0]
        per_label = per_label_metrics(args.out, val_path, threshold=threshold)
        examples = eval_gallery_docs(args.out, threshold=threshold)
        print(json.dumps(per_label, indent=2))
        print(json.dumps(examples[:8], indent=2))
        return

    if not args.train_only:
        print("Building corpus...", flush=True)
        rows, train_path, val_path = build_corpus(max_files=args.max_files)
        empty = sum(1 for r in rows if r["labels"] == [LABEL_EMPTY])
        with_fields = sum(1 for r in rows if r["labels"] != [LABEL_EMPTY])
        print(
            f"Corpus: {len(rows)} sections -> {train_path} "
            f"({sum(1 for _ in train_path.open())} train lines)"
        )
        print(f"  empty={empty} with_fields={with_fields}")
        if args.build_corpus_only:
            return
    else:
        rows = [
            json.loads(line)
            for line in (CORPUS_DIR / "sections_labeled.jsonl").read_text().splitlines()
            if line.strip()
        ]
        empty = sum(1 for r in rows if r.get("labels") == [LABEL_EMPTY])
        with_fields = sum(1 for r in rows if r.get("labels") != [LABEL_EMPTY])

    print("Training...", flush=True)
    model_path, metrics = train_fasttext(train_path, val_path, args.out)
    threshold, tune_metrics = tune_threshold(model_path, val_path)
    if args.threshold is not None:
        threshold = args.threshold
    per_label = per_label_metrics(model_path, val_path, threshold=threshold)
    examples = eval_gallery_docs(model_path, threshold=threshold)
    results_path = write_results(
        model_path,
        {"total": len(rows), "empty": empty, "with_fields": with_fields},
        metrics,
        per_label,
        threshold,
        examples,
    )
    print(f"Model: {model_path}")
    print(f"Best threshold: {threshold} ({json.dumps(tune_metrics)})")
    print(f"Results: {results_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
