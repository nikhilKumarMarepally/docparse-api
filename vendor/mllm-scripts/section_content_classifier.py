#!/usr/bin/env python3
"""Unified section content classifier: heuristics + optional FastText fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from string import punctuation
from typing import Any

from section_content_heuristics import (
    DEFAULT_THRESHOLD,
    classify_content_types_heuristic,
    normalize_ocr_text,
    preprocess_for_fasttext,
    score_content_types,
)
from section_content_taxonomy import (
    CONTENT_TYPES,
    load_registry,
    route_content_types_to_sections,
)
from section_field_classifier import classify_section_fields
from section_preprocess import annotate_preprocess, filter_sections

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]
DEFAULT_MODEL_PATH = ROOT / "wa577_gallery" / "section_classifier" / "models" / "content_types.bin"


@dataclass
class ContentTypeResult:
    content_types: list[str]
    scores: dict[str, float]
    method: str
    doc_sections: list[str] = field(default_factory=list)
    raw_scores: dict[str, float] = field(default_factory=dict)
    fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_types": self.content_types,
            "scores": self.scores,
            "method": self.method,
            "doc_sections": self.doc_sections,
            "raw_scores": self.raw_scores,
            "fields": self.fields,
        }


def preprocess_for_fasttext(text: str) -> str:
    """Backward-compatible re-export; see section_content_heuristics."""
    from section_content_heuristics import preprocess_for_fasttext as _prep

    return _prep(text)


class FastTextContentClassifier:
    def __init__(self, model_path: Path | str) -> None:
        import fasttext  # noqa: PLC0415

        self.model_path = Path(model_path)
        self.model = fasttext.load_model(str(self.model_path))

    def predict(
        self,
        text: str,
        *,
        threshold: float = 0.25,
        k: int | None = None,
    ) -> tuple[list[str], dict[str, float]]:
        proc = preprocess_for_fasttext(text)
        if not proc:
            return [], {ct: 0.0 for ct in CONTENT_TYPES}
        k = k or len(self.model.labels)
        labels, probs = self.model.predict(proc, k=k)
        scores: dict[str, float] = {ct: 0.0 for ct in CONTENT_TYPES}
        for label, prob in zip(labels, probs):
            ctype = label.replace("__label__", "")
            if ctype in scores:
                scores[ctype] = float(prob)
        present = [ct for ct, s in scores.items() if s >= threshold]
        present.sort(key=lambda c: (-scores[c], c))
        return present, scores


def _fuse_scores(
    heuristic: dict[str, float],
    model: dict[str, float] | None,
    *,
    heuristic_weight: float = 0.45,
) -> dict[str, float]:
    max_h = max(heuristic.values()) if heuristic else 0.0
    if max_h <= 0 and not model:
        return {ct: 0.0 for ct in CONTENT_TYPES}
    norm_h = {ct: (heuristic.get(ct, 0.0) / max_h if max_h else 0.0) for ct in CONTENT_TYPES}
    if not model or max(model.values()) <= 0:
        return {ct: round(norm_h.get(ct, 0.0), 4) for ct in CONTENT_TYPES}
    # When heuristics are confident, keep them primary; use model to break ties only.
    if max_h >= 2.5:
        heuristic_weight = 0.75
    fused: dict[str, float] = {}
    for ct in CONTENT_TYPES:
        h = norm_h.get(ct, 0.0)
        m = model.get(ct, 0.0)
        fused[ct] = heuristic_weight * h + (1.0 - heuristic_weight) * m
    return {ct: round(v, 4) for ct, v in fused.items()}


def classify_section(
    text: str,
    *,
    bounds: dict[str, Any] | None = None,
    document_type: str | None = None,
    line_count: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    model: FastTextContentClassifier | str | Path | None = None,
    registry: dict[str, Any] | None = None,
) -> ContentTypeResult:
    """Classify a geometric section crop into universal content types."""
    _ = bounds  # reserved for future image+text tier
    lc = line_count
    if lc is None and text:
        lc = len([ln for ln in text.splitlines() if ln.strip()])

    raw = score_content_types(text, line_count=lc)
    h_present, h_norm = classify_content_types_heuristic(
        text, line_count=lc, threshold=threshold
    )

    ft_classifier: FastTextContentClassifier | None = None
    if model is not None:
        if isinstance(model, FastTextContentClassifier):
            ft_classifier = model
        else:
            path = Path(model)
            if path.exists():
                ft_classifier = FastTextContentClassifier(path)

    m_present: list[str] = []
    m_scores: dict[str, float] = {ct: 0.0 for ct in CONTENT_TYPES}
    method = "heuristic"
    if ft_classifier is not None:
        m_present, m_scores = ft_classifier.predict(text, threshold=threshold * 0.85)
        method = "heuristic+fasttext"

    fused = _fuse_scores(raw, m_scores if ft_classifier else None)
    present = [ct for ct, s in fused.items() if s >= threshold]
    if not present and h_present:
        present = h_present
    if not present and m_present:
        present = m_present
    present.sort(key=lambda c: (-fused.get(c, 0.0), c))

    doc_sections: list[str] = []
    if document_type and present:
        doc_sections = route_content_types_to_sections(
            present, document_type, registry=registry
        )

    fields: list[str] = []
    if document_type:
        fields = classify_section_fields(
            text,
            document_type,
            content_types=present or None,
        )

    return ContentTypeResult(
        content_types=present,
        scores=fused,
        method=method,
        doc_sections=doc_sections,
        raw_scores=raw,
        fields=fields,
    )


def classify_sections_payload(
    sections: list[dict[str, Any]],
    *,
    document_type: str | None = None,
    model: FastTextContentClassifier | str | Path | None = None,
    registry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Annotate each section dict with preprocess gate + content classification."""
    doc_type = document_type or "credit_application"
    kept, dropped = filter_sections(sections, document_type=doc_type)
    dropped_by_index = {s.get("index"): s for s in dropped}

    classified_kept: dict[Any, dict[str, Any]] = {}
    for section in kept:
        result = classify_section(
            section.get("text") or "",
            bounds=section.get("bounds"),
            document_type=document_type,
            line_count=section.get("line_count"),
            model=model,
            registry=registry,
        )
        row = dict(section)
        row["content_classification"] = result.to_dict()
        row["fields"] = result.fields
        classified_kept[section.get("index")] = row

    out: list[dict[str, Any]] = []
    for section in sections:
        idx = section.get("index")
        if idx in classified_kept:
            out.append(classified_kept[idx])
            continue
        row = dropped_by_index.get(idx) or annotate_preprocess(section, document_type=doc_type)
        row["content_classification"] = {
            "content_types": [],
            "scores": {},
            "method": "preprocess_skipped",
            "doc_sections": [],
            "raw_scores": {},
            "fields": [],
        }
        row["fields"] = []
        out.append(row)
    return out


def load_default_model() -> FastTextContentClassifier | None:
    if DEFAULT_MODEL_PATH.exists():
        return FastTextContentClassifier(DEFAULT_MODEL_PATH)
    return None


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Classify section text into content types")
    ap.add_argument("--text", help="Inline OCR text")
    ap.add_argument("--sections-json", type=Path, help="sections.json from gallery")
    ap.add_argument("--document-type", default="credit_application")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = ap.parse_args()

    model = FastTextContentClassifier(args.model) if args.model.exists() else None
    reg = load_registry()

    if args.sections_json:
        payload = json.loads(args.sections_json.read_text())
        classified = classify_sections_payload(
            payload.get("sections") or [],
            document_type=args.document_type,
            model=model,
            registry=reg,
        )
        print(json.dumps(classified, indent=2))
        return

    text = args.text or ""
    result = classify_section(
        text,
        document_type=args.document_type,
        model=model,
        registry=reg,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
