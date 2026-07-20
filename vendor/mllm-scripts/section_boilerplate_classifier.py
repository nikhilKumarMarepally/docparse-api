#!/usr/bin/env python3
"""Binary section gate: boilerplate/legal prose vs extractable form data."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from section_content_heuristics import normalize_ocr_text, preprocess_for_fasttext

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODEL_PATH = (
    ROOT / "wa577_gallery" / "section_classifier" / "models" / "section_has_data.bin"
)
DEFAULT_THRESHOLD = 0.65

LABEL_BOILERPLATE = "boilerplate"
LABEL_HAS_FORM_DATA = "has_form_data"

_RE_LEGAL_AGREEMENT = re.compile(
    r"\b(?:agreement|fair credit reporting act|"
    r'words\s*["\']?(?:we|us|our)|'
    r"as used (?:in this application|below))\b",
    re.I,
)
_RE_LEGAL_NOTICE = re.compile(
    r"\b(?:federal notices?|state notices?|"
    r"important information about procedures|"
    r"residents? of (?:california|wisconsin|new york))\b",
    re.I,
)
_RE_CONSENT_PROSE = re.compile(
    r"\b(?:monitor and record telephone|"
    r"consumer credit report periodically|"
    r"authorize (?:us|that such financial institutions) to submit|"
    r"consent to receive autodialed|"
    r"pre-?recorded and artificial voice|"
    r"obtain,? verify,? and record information)\b",
    re.I,
)
_RE_FOOTER_BOILERPLATE = re.compile(
    r"\b(?:all rights reserved|dealertrack inc|©\s*\d{4}|"
    r"page\s+\d+\s+of\s+\d+)\b",
    re.I,
)
_RE_STRONG_BUSINESS_ROW = re.compile(
    r"(?:trade name of business|legal business name|business name|tax id).*\n\s*.+",
    re.I | re.S,
)
_RE_FORM_FIELD_HEADER = re.compile(
    r"\b(?:last name|first name|social security|"
    r"present address|employer|occupation|"
    r"year\s+make|vehicle identification|co-?applicant information)\b",
    re.I,
)
_RE_STRONG_SSN_ROW = re.compile(
    r"(?:last name|first name|social security).*\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    re.I | re.S,
)
_RE_STRONG_NAME_ADDR = re.compile(
    r"(?:last name|first name).*\n\s*[A-Z][A-Za-z' -]{1,30}\s+[A-Z][A-Za-z' -]{1,30}",
    re.I | re.S,
)
_RE_STRONG_VIN_ROW = re.compile(
    r"(?:year|make|vin|vehicle identification).*\b[A-HJ-NPR-Z0-9]{17}\b",
    re.I | re.S,
)


def is_boilerplate_heuristic(text: str) -> bool:
    """True when section text is legal/footer/consent prose without form rows."""
    norm = normalize_ocr_text(text)
    if not norm:
        return True

    if _has_strong_form_rows(text, norm):
        return False

    if _RE_LEGAL_AGREEMENT.search(norm):
        return True
    if _RE_LEGAL_NOTICE.search(norm):
        return True
    if _RE_CONSENT_PROSE.search(norm):
        return True
    if _RE_FOOTER_BOILERPLATE.search(norm) and not _RE_FORM_FIELD_HEADER.search(norm):
        return True

    return False


def _has_strong_form_rows(text: str, norm: str) -> bool:
    if _RE_STRONG_SSN_ROW.search(text) or _RE_STRONG_SSN_ROW.search(norm):
        return True
    if _RE_STRONG_NAME_ADDR.search(text):
        return True
    if _RE_STRONG_VIN_ROW.search(text) or _RE_STRONG_VIN_ROW.search(norm):
        return True
    if _RE_STRONG_BUSINESS_ROW.search(text):
        value_line = text.splitlines()[-1] if text.splitlines() else ""
        if re.search(r"\b\d{2}[-\s]?\d{7}\b", value_line) or re.search(
            r"\b(?:llc|inc|corp|l\.?l\.?c\.?|company)\b", value_line, re.I
        ):
            return True
    if _RE_FORM_FIELD_HEADER.search(norm):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 2:
            value_line = lines[1]
            if re.search(r"\d{3}[-\s]?\d{2}[-\s]?\d{4}", value_line):
                return True
            if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", value_line):
                return True
            tokens = re.findall(r"[A-Za-z']+", value_line)
            if len(tokens) >= 2 and not all(t.isupper() and len(t) <= 3 for t in tokens):
                return True
    return False


def should_skip_field_matching(
    text: str,
    *,
    model: FastTextBoilerplateClassifier | str | Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Fused heuristic + FastText gate: True => return fields=[]."""
    from section_field_classifier import has_extractable_values  # noqa: PLC0415

    norm = normalize_ocr_text(text)
    if _has_strong_form_rows(text, norm):
        return False
    if is_boilerplate_heuristic(text):
        return True
    if not has_extractable_values(text):
        return True

    ft = _resolve_model(model)
    if ft is None:
        return False

    label, conf = ft.predict(text)
    return label == LABEL_BOILERPLATE and conf >= threshold


class FastTextBoilerplateClassifier:
    def __init__(self, model_path: Path | str) -> None:
        import fasttext  # noqa: PLC0415

        self.model_path = Path(model_path)
        self.model = fasttext.load_model(str(self.model_path))

    def predict(self, text: str) -> tuple[str, float]:
        proc = preprocess_for_fasttext(text)
        if not proc:
            return LABEL_BOILERPLATE, 1.0
        labels, probs = self.model.predict(proc, k=1)
        label = labels[0].replace("__label__", "")
        return label, float(probs[0])


@lru_cache(maxsize=1)
def _cached_default_model() -> FastTextBoilerplateClassifier | None:
    if DEFAULT_MODEL_PATH.exists():
        return FastTextBoilerplateClassifier(DEFAULT_MODEL_PATH)
    return None


def _resolve_model(
    model: FastTextBoilerplateClassifier | str | Path | None,
) -> FastTextBoilerplateClassifier | None:
    if model is None:
        return _cached_default_model()
    if isinstance(model, FastTextBoilerplateClassifier):
        return model
    path = Path(model)
    if path.exists():
        return FastTextBoilerplateClassifier(path)
    return None


def load_default_boilerplate_model() -> FastTextBoilerplateClassifier | None:
    return _cached_default_model()


def classify_has_form_data(
    text: str,
    *,
    model: FastTextBoilerplateClassifier | str | Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Return diagnostic payload for debugging / gallery."""
    from section_field_classifier import has_extractable_values  # noqa: PLC0415

    heuristic_bp = is_boilerplate_heuristic(text)
    has_values = has_extractable_values(text)
    ft_label = ""
    ft_conf = 0.0
    ft = _resolve_model(model)
    if ft is not None:
        ft_label, ft_conf = ft.predict(text)
    skip = should_skip_field_matching(text, model=model, threshold=threshold)
    return {
        "skip_field_matching": skip,
        "heuristic_boilerplate": heuristic_bp,
        "has_extractable_values": has_values,
        "fasttext_label": ft_label,
        "fasttext_confidence": round(ft_conf, 4),
        "predicted": LABEL_HAS_FORM_DATA if not skip else LABEL_BOILERPLATE,
    }
