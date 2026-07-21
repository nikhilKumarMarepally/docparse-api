#!/usr/bin/env python3
"""Evaluate generic section content classifier: within-type and LOOTO."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from section_content_classifier import (  # noqa: E402
    FastTextContentClassifier,
    classify_section,
    load_default_model,
)
from section_content_heuristics import classify_content_types_heuristic  # noqa: E402
from section_content_taxonomy import CONTENT_TYPES, TARGET_DOC_TYPES  # noqa: E402

ROOT = SCRIPT_DIR.parents[4]
OUT_DIR = ROOT / "wa577_gallery" / "section_classifier"
CORPUS_PATH = OUT_DIR / "corpus" / "sections_labeled.jsonl"
RESULTS_PATH = OUT_DIR / "results.md"
GOLD_PATH = OUT_DIR / "gold_labels.jsonl"


def _load_corpus() -> list[dict[str, Any]]:
    if not CORPUS_PATH.exists():
        from section_content_train import build_corpus  # noqa: PLC0415

        build_corpus()
    rows: list[dict[str, Any]] = []
    for line in CORPUS_PATH.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_gold() -> list[dict[str, Any]]:
    if GOLD_PATH.exists():
        return [json.loads(ln) for ln in GOLD_PATH.read_text().splitlines() if ln.strip()]
    return _bootstrap_gold()


def _bootstrap_gold() -> list[dict[str, Any]]:
    """Seed gold labels from high-confidence heuristic rows (calibration set)."""
    rows = _load_corpus()
    gold: list[dict[str, Any]] = []
    for row in rows:
        labels = row.get("labels") or []
        if not labels:
            continue
        present, scores = classify_content_types_heuristic(row["text"], line_count=row.get("line_count"))
        if present == labels and max(scores.values()) >= 0.75:
            gold.append(
                {
                    "document_type": row["document_type"],
                    "text": row["text"],
                    "labels": labels,
                    "source": row.get("source"),
                    "section_index": row.get("section_index"),
                }
            )
        if len(gold) >= 400:
            break
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_PATH.open("w") as fh:
        for g in gold:
            fh.write(json.dumps(g) + "\n")
    return gold


def _per_label_metrics(
    y_true: list[set[str]],
    y_pred: list[set[str]],
    labels: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, int]] = {
        lb: {"tp": 0, "fp": 0, "fn": 0} for lb in labels
    }
    for truth, pred in zip(y_true, y_pred):
        for lb in labels:
            t = lb in truth
            p = lb in pred
            if t and p:
                stats[lb]["tp"] += 1
            elif p and not t:
                stats[lb]["fp"] += 1
            elif t and not p:
                stats[lb]["fn"] += 1
    out: dict[str, dict[str, float]] = {}
    for lb, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lb] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def _macro_f1(metrics: dict[str, dict[str, float]]) -> float:
    f1s = [m["f1"] for m in metrics.values() if m["support"] > 0]
    return sum(f1s) / len(f1s) if f1s else 0.0


def predict_rows(
    rows: list[dict[str, Any]],
    *,
    use_model: bool = False,
    model: FastTextContentClassifier | None = None,
) -> list[set[str]]:
    preds: list[set[str]] = []
    for row in rows:
        if use_model and model is not None:
            result = classify_section(
                row["text"],
                document_type=row.get("document_type"),
                line_count=row.get("line_count"),
                model=model,
            )
            preds.append(set(result.content_types))
        else:
            present, _ = classify_content_types_heuristic(
                row["text"], line_count=row.get("line_count")
            )
            preds.append(set(present))
    return preds


def eval_within_type(
    gold: list[dict[str, Any]],
    *,
    model: FastTextContentClassifier | None,
) -> dict[str, Any]:
    ca_rows = [g for g in gold if g["document_type"] == "credit_application"]
    if not ca_rows:
        return {}
    split = max(1, int(len(ca_rows) * 0.2))
    test = ca_rows[:split]
    y_true = [set(r["labels"]) for r in test]
    h_pred = predict_rows(test, use_model=False)
    m_pred = predict_rows(test, use_model=True, model=model) if model else h_pred
    return {
        "n": len(test),
        "heuristic_macro_f1": _macro_f1(_per_label_metrics(y_true, h_pred, CONTENT_TYPES)),
        "model_macro_f1": _macro_f1(_per_label_metrics(y_true, m_pred, CONTENT_TYPES)),
        "heuristic_metrics": _per_label_metrics(y_true, h_pred, CONTENT_TYPES),
        "model_metrics": _per_label_metrics(y_true, m_pred, CONTENT_TYPES),
    }


def eval_looto(
    gold: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Leave-one-doc-type-out using heuristic labels as proxy when no per-type model."""
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold:
        by_doc[row["document_type"]].append(row)

    results: list[dict[str, Any]] = []
    doc_types = sorted(by_doc.keys())
    for held_out in doc_types:
        test = by_doc[held_out]
        if not test:
            continue
        y_true = [set(r["labels"]) for r in test]
        y_pred = predict_rows(test, use_model=False)
        metrics = _per_label_metrics(y_true, y_pred, CONTENT_TYPES)
        results.append(
            {
                "held_out": held_out,
                "n": len(test),
                "macro_f1": _macro_f1(metrics),
                "metrics": metrics,
            }
        )
    return results


def eval_looto_fasttext(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retrain FastText per held-out doc type and evaluate."""
    import fasttext  # noqa: PLC0415

    from section_content_train import _write_fasttext  # noqa: PLC0415

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in corpus:
        by_doc[row["document_type"]].append(row)

    results: list[dict[str, Any]] = []
    tmp_dir = OUT_DIR / "corpus" / "looto"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for held_out in sorted(by_doc.keys()):
        train_rows = [r for dt, rows in by_doc.items() if dt != held_out for r in rows]
        test_rows = by_doc[held_out]
        if len(train_rows) < 20 or not test_rows:
            continue
        train_path = tmp_dir / f"train_no_{held_out}.txt"
        _write_fasttext(train_path, train_rows)
        model = fasttext.train_supervised(
            input=str(train_path),
            lr=0.5,
            epoch=20,
            wordNgrams=2,
            minCount=1,
            loss="ova",
        )
        ft = FastTextContentClassifier.__new__(FastTextContentClassifier)
        ft.model = model
        ft.model_path = train_path
        y_true = [set(r["labels"]) for r in test_rows]
        y_pred = predict_rows(test_rows, use_model=True, model=ft)
        h_pred = predict_rows(test_rows, use_model=False)
        metrics = _per_label_metrics(y_true, y_pred, CONTENT_TYPES)
        h_metrics = _per_label_metrics(y_true, h_pred, CONTENT_TYPES)
        results.append(
            {
                "held_out": held_out,
                "n": len(test_rows),
                "model_macro_f1": _macro_f1(metrics),
                "heuristic_macro_f1": _macro_f1(h_metrics),
                "delta_pp": round((_macro_f1(metrics) - _macro_f1(h_metrics)) * 100, 1),
                "metrics": metrics,
            }
        )
    return results


def write_results_md(
    within: dict[str, Any],
    looto_heur: list[dict[str, Any]],
    looto_ft: list[dict[str, Any]],
) -> None:
    lines = [
        "# Generic Section Content Classifier — Eval Results",
        "",
        "## Within-type (credit_application held-out 20%)",
        "",
        "| Metric | Heuristic | Heuristic+FastText |",
        "|--------|-----------|-------------------|",
    ]
    if within:
        lines.append(
            f"| Macro F1 | {within['heuristic_macro_f1']:.3f} | {within.get('model_macro_f1', 0):.3f} |"
        )
        lines.append(f"| N test sections | {within['n']} | {within['n']} |")
    lines.extend(["", "## Leave-one-doc-type-out (heuristic)", ""])
    lines.append("| Held-out doc type | N | Macro F1 |")
    lines.append("|-------------------|---|----------|")
    for row in looto_heur:
        lines.append(f"| {row['held_out']} | {row['n']} | {row['macro_f1']:.3f} |")

    lines.extend(["", "## Leave-one-doc-type-out (FastText retrained)", ""])
    lines.append("| Held-out | N | Heuristic F1 | Model F1 | Delta (pp) |")
    lines.append("|----------|---|--------------|----------|------------|")
    for row in looto_ft:
        lines.append(
            f"| {row['held_out']} | {row['n']} | {row['heuristic_macro_f1']:.3f} | "
            f"{row['model_macro_f1']:.3f} | {row['delta_pp']:+.1f} |"
        )

    if within and within.get("model_metrics"):
        lines.extend(["", "## Per content-type (within-type, model)", ""])
        lines.append("| Content type | Precision | Recall | F1 | Support |")
        lines.append("|--------------|-----------|--------|-----|---------|")
        for ct in CONTENT_TYPES:
            m = within["model_metrics"].get(ct, {})
            if m.get("support", 0) == 0:
                continue
            lines.append(
                f"| {ct} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {int(m['support'])} |"
            )

    avg_delta = (
        sum(r["delta_pp"] for r in looto_ft) / len(looto_ft) if looto_ft else 0.0
    )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- LOOTO average model vs heuristic delta: **{avg_delta:+.1f} pp** macro-F1",
            "- Ship bar: model beats heuristic by ≥10pp on LOOTO macro-F1",
            f"- Status: **{'PASS' if avg_delta >= 10 else 'FOLLOW-UP'}** (heuristic baseline strong; FastText adds value on unseen layouts when delta positive)",
        ]
    )
    RESULTS_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    corpus = _load_corpus()
    gold = _load_gold()
    model = load_default_model()

    within = eval_within_type(gold, model=model)
    looto_heur = eval_looto(gold)
    looto_ft = eval_looto_fasttext(corpus)
    write_results_md(within, looto_heur, looto_ft)
    print(f"Wrote {RESULTS_PATH}")
    if within:
        print(
            f"Within-type macro-F1 heuristic={within['heuristic_macro_f1']:.3f} "
            f"model={within.get('model_macro_f1', 0):.3f}"
        )


if __name__ == "__main__":
    main()
