#!/usr/bin/env python3
"""Section preprocessor — drop boilerplate before field detection and Gemini.

Pipeline position (credit-app section crops):
  OCR line clustering → **section_preprocess** → content classification →
  field detection → Gemini per-section extraction

Sections are KEPT only when OCR shows placeholder/form signals:
  - Filled values (SSN, phone, names, addresses, dates, dollars), OR
  - Visible form structure (label rows with blanks/underscores/checkboxes)

Dropped sections (legal disclaimers, federal/state notices, consent prose,
copyright footers) are annotated with ``preprocess: {kept: false, reason}`` and
skip downstream classifiers / LLM calls.
"""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from section_boilerplate_classifier import (
    DEFAULT_MODEL_PATH as BOILERPLATE_MODEL_PATH,
    DEFAULT_THRESHOLD as BOILERPLATE_THRESHOLD,
    LABEL_BOILERPLATE,
    LABEL_HAS_FORM_DATA,
    FastTextBoilerplateClassifier,
    _has_strong_form_rows,
    _resolve_model as _resolve_boilerplate_model,
)
from section_content_heuristics import fuzzy_contains, normalize_ocr_text
from section_field_classifier import (
    DEFAULT_FIELD_MODEL_PATH,
    DEFAULT_FIELD_THRESHOLD,
    FastTextFieldClassifier,
    _resolve_field_model,
    has_extractable_values,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]


class DropReason(str, Enum):
    """Why a section was removed before field detection / Gemini."""

    DISCLAIMER = "disclaimer"
    LEGAL_NOTICE = "legal_notice"
    FOOTER = "footer"
    NO_FORM_STRUCTURE = "no_form_structure"


# Re-export patterns aligned with section_boilerplate_classifier.
_RE_LEGAL_AGREEMENT = re.compile(
    r"\b(?:agreement|fair credit reporting act|"
    r'words\s*["\']?(?:we|us|our)|'
    r"as used (?:in this application|below))\b",
    re.I,
)
_RE_LEGAL_NOTICE = re.compile(
    r"\b(?:federal notices?|state notices?|"
    r"important information about procedures|"
    r"residents? of (?:california|wisconsin|new york|maine|tennessee|new hampshire))\b",
    re.I,
)
_RE_CONSENT_PROSE = re.compile(
    r"\b(?:monitor and record telephone|"
    r"consumer credit report periodically|"
    r"authorize (?:us|that such financial institutions) to submit|"
    r"consent to receive autodialed|"
    r"pre-?recorded and artificial voice|"
    r"obtain,? verify,? and record information|"
    r"false statements may subject you)\b",
    re.I,
)
_RE_FOOTER_BOILERPLATE = re.compile(
    r"\b(?:all rights reserved|dealertrack inc|©\s*\d{4}|"
    r"page\s+\d+\s+of\s+\d+)\b",
    re.I,
)
_RE_FORM_BLANK = re.compile(
    r"(?:_{2,}|x{3,}|/s/|\[ ?\]|\[x\]|sign\s+here|initial\s+here)",
    re.I,
)
_RE_FORM_HEADER = re.compile(
    r"\b(?:last name|first name|middle initial|social security|birth date|"
    r"present address|previous.*address|employer|occupation|"
    r"year\s+make|vehicle identification|co-?applicant information|"
    r"driver'?s license|home phone|cell phone|e-?mail address)\b",
    re.I,
)
_RE_CHECKBOX = re.compile(r"[\u2610\u2611\u2612\u25a1\u25a2]|☐|☑|□")
_RE_ADDRESS_GRID = re.compile(
    r"\b(?:city|state|zip)\b",
    re.I,
)


def _line_looks_like_form_header(line: str) -> bool:
    """True for dedicated label rows, not legal prose mentioning field names."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if re.search(
        r"\b(?:notices?|residents?|pursuant|accordance|certify|understand|authorize)\b",
        stripped,
        re.I,
    ):
        return False
    hits = len(_RE_FORM_HEADER.findall(stripped))
    if hits >= 2:
        return True
    if hits == 1 and len(stripped.split()) <= 12:
        return True
    return False


def has_form_structure(text: str, norm: str | None = None) -> bool:
    """True when OCR shows form labels/blanks/checkboxes without filled values."""
    norm = norm if norm is not None else normalize_ocr_text(text)
    if not norm:
        return False

    if _RE_FORM_BLANK.search(text) or _RE_CHECKBOX.search(text):
        return True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    header_lines = [ln for ln in lines if _line_looks_like_form_header(ln)]
    if header_lines and len(lines) >= 2:
        return True

    if _RE_ADDRESS_GRID.search(norm) and (
        fuzzy_contains(norm, "address") or fuzzy_contains(norm, "previous")
    ):
        # Require column-style address grid, not prose mentioning "address".
        if any(_line_looks_like_form_header(ln) for ln in lines):
            return True
        compact = re.sub(r"\s+", " ", norm)
        if re.search(r"(?:city|state|zip).{0,40}(?:city|state|zip)", compact):
            return True

    if fuzzy_contains(norm, "employer") and (
        fuzzy_contains(norm, "occupation") or fuzzy_contains(norm, "salary")
    ):
        if any(_line_looks_like_form_header(ln) for ln in lines):
            return True

    return False


def _explicit_drop_reason(text: str, norm: str) -> DropReason | None:
    """Drop legal/footer/disclaimer prose before loose form-structure heuristics."""
    if _has_strong_form_rows(text, norm):
        return None
    if has_extractable_values(text):
        return None
    if _RE_FOOTER_BOILERPLATE.search(norm) and not _RE_FORM_HEADER.search(norm):
        return DropReason.FOOTER
    if _RE_LEGAL_NOTICE.search(norm):
        return DropReason.LEGAL_NOTICE
    if _RE_LEGAL_AGREEMENT.search(norm) or _RE_CONSENT_PROSE.search(norm):
        return DropReason.DISCLAIMER
    return None


def _heuristic_drop_reason(text: str, norm: str) -> DropReason | None:
    """Classify known boilerplate types; None => defer to FastText / default."""
    return _explicit_drop_reason(text, norm)


def _fasttext_says_boilerplate(
    text: str,
    *,
    boilerplate_model: FastTextBoilerplateClassifier | None,
    threshold: float,
) -> bool:
    if boilerplate_model is None:
        return False
    label, conf = boilerplate_model.predict(text)
    return label == LABEL_BOILERPLATE and conf >= threshold


def _fasttext_has_fields(
    text: str,
    *,
    field_model: FastTextFieldClassifier | None,
    threshold: float,
) -> bool:
    if field_model is None:
        return False
    _fields, _scores, is_empty = field_model.predict(text, threshold=threshold)
    return not is_empty


def should_keep_section(
    text: str,
    *,
    document_type: str = "credit_application",
    boilerplate_model: FastTextBoilerplateClassifier | str | Path | None = None,
    field_model: FastTextFieldClassifier | str | Path | None = None,
    boilerplate_threshold: float = BOILERPLATE_THRESHOLD,
    field_threshold: float = DEFAULT_FIELD_THRESHOLD,
) -> tuple[bool, DropReason | None]:
    """Return (kept, drop_reason). drop_reason is set only when kept is False."""
    _ = document_type  # reserved for doc-type-specific gates
    norm = normalize_ocr_text(text)
    if not norm:
        return False, DropReason.NO_FORM_STRUCTURE

    if _has_strong_form_rows(text, norm):
        return True, None
    if has_extractable_values(text):
        return True, None

    explicit = _explicit_drop_reason(text, norm)
    if explicit is not None:
        return False, explicit

    if has_form_structure(text, norm):
        return True, None

    heuristic_reason = _heuristic_drop_reason(text, norm)
    if heuristic_reason is not None:
        return False, heuristic_reason

    bp_model = _resolve_boilerplate_model(boilerplate_model)
    ft_model = _resolve_field_model(field_model)

    if _fasttext_has_fields(text, field_model=ft_model, threshold=field_threshold):
        if bp_model is None or not _fasttext_says_boilerplate(
            text, boilerplate_model=bp_model, threshold=boilerplate_threshold
        ):
            return True, None

    if bp_model is not None:
        label, conf = bp_model.predict(text)
        if label == LABEL_HAS_FORM_DATA and conf >= boilerplate_threshold:
            return True, None
        if label == LABEL_BOILERPLATE and conf >= boilerplate_threshold:
            if _RE_FOOTER_BOILERPLATE.search(norm):
                return False, DropReason.FOOTER
            if _RE_LEGAL_NOTICE.search(norm):
                return False, DropReason.LEGAL_NOTICE
            if _RE_LEGAL_AGREEMENT.search(norm) or _RE_CONSENT_PROSE.search(norm):
                return False, DropReason.DISCLAIMER
            return False, DropReason.NO_FORM_STRUCTURE

    return False, DropReason.NO_FORM_STRUCTURE


def annotate_preprocess(
    section: dict[str, Any],
    *,
    document_type: str = "credit_application",
    boilerplate_model: FastTextBoilerplateClassifier | str | Path | None = None,
    field_model: FastTextFieldClassifier | str | Path | None = None,
) -> dict[str, Any]:
    """Return a copy of *section* with ``preprocess`` metadata."""
    text = section.get("text") or ""
    kept, reason = should_keep_section(
        text,
        document_type=document_type,
        boilerplate_model=boilerplate_model,
        field_model=field_model,
    )
    row = dict(section)
    if kept:
        row["preprocess"] = {"kept": True}
    else:
        row["preprocess"] = {"kept": False, "reason": reason.value if reason else DropReason.NO_FORM_STRUCTURE.value}
    return row


def filter_sections(
    sections: list[dict[str, Any]],
    *,
    document_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(kept_sections, dropped_sections)`` with drop_reason on each dropped."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for section in sections:
        row = annotate_preprocess(section, document_type=document_type)
        if row["preprocess"]["kept"]:
            kept.append(row)
        else:
            dropped.append(row)
    return kept, dropped


def section_passes_preprocess(section: dict[str, Any]) -> bool:
    """True when section was not dropped by the preprocessor."""
    prep = section.get("preprocess") or {}
    return bool(prep.get("kept", True))


@lru_cache(maxsize=1)
def default_models_exist() -> bool:
    return BOILERPLATE_MODEL_PATH.exists() and DEFAULT_FIELD_MODEL_PATH.exists()
