#!/usr/bin/env python3
"""Generic OCR-tolerant heuristics for universal section content types."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from section_content_taxonomy import CONTENT_TYPES

# Homoglyph / OCR confusion normalization (Latin + common Cyrillic lookalikes).
_HOMOGLYPHS = str.maketrans(
    {
        "О": "O",
        "о": "o",
        "І": "I",
        "і": "i",
        "С": "C",
        "с": "c",
        "А": "A",
        "а": "a",
        "Е": "E",
        "е": "e",
        "Р": "P",
        "р": "p",
        "Х": "X",
        "х": "x",
        "Т": "T",
        "т": "t",
        "Н": "H",
        "н": "h",
        "К": "K",
        "к": "k",
        "М": "M",
        "м": "m",
        "В": "B",
        "в": "b",
        "У": "Y",
        "у": "y",
        "0": "0",
        "O": "O",
        "o": "o",
        "1": "1",
        "l": "l",
        "I": "I",
    }
)

# Value-shape detectors (doc-agnostic).
_RE_SSN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
_RE_PHONE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+", re.I)
_RE_VIN = re.compile(r"\b[A-HJ-NPR-Z0-9IO]{11,17}\b", re.I)
_RE_VIN_LABEL = re.compile(
    r"vehicle identification|identification number|\bvin\b|serial\s*(?:number|#)",
    re.I,
)
_RE_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_RE_STATE = re.compile(r"\b[A-Z]{2}\b")
_RE_DOLLAR = re.compile(r"\$\s?[\d,]+\.?\d*")
_RE_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
_RE_EIN = re.compile(r"\b\d{2}[-\s]?\d{7}\b")
_RE_EMP_LEN = re.compile(r"\b\d+\s*(?:mos|months|yrs|years)\b", re.I)
_RE_APR = re.compile(r"\b(?:apr|annual percentage rate)\b", re.I)
_RE_TIL = re.compile(r"truth.in.lending|finance charge|amount financed|total of payments", re.I)
_RE_TRADE = re.compile(
    r"trade[- ]?in|vehicle traded|additional trade",
    re.I,
)
_RE_SIGNATURE = re.compile(
    r"\b(?:sign(?:ed|ature)?|by signing|i hereby apply|certify that)\b",
    re.I,
)
_RE_FOOTER = re.compile(
    r"\b(?:printed:|creation time|page\s+\d+\s+of|credit application creation)\b",
    re.I,
)
_RE_LIENHOLDER = re.compile(r"lienholder|secured party|first lien", re.I)
_RE_LESSEE = re.compile(r"\blessee\b|lessor\s*/\s*lessee", re.I)

# Generic label anchors: (content_type, [(phrase, weight), ...])
_LABEL_ANCHORS: dict[str, list[tuple[str, float]]] = {
    "personal_identity": [
        ("last name", 2.0),
        ("first name", 2.0),
        ("middle name", 1.5),
        ("social security", 2.5),
        ("date of birth", 2.0),
        ("dob", 1.0),
        ("suffix", 0.8),
    ],
    "contact_info": [
        ("email", 1.5),
        ("phone", 1.5),
        ("home phone", 2.0),
        ("cell phone", 2.0),
        ("e-mail", 1.5),
    ],
    "residential_address": [
        ("present address", 2.5),
        ("current address", 2.0),
        ("street address", 2.0),
        ("city", 1.0),
        ("zip", 1.0),
        ("time at present address", 2.0),
        ("monthly rent", 1.5),
        ("mortgage", 1.0),
    ],
    "mailing_address": [
        ("mailing address", 3.0),
    ],
    "employment_income": [
        ("employer", 2.0),
        ("occupation", 2.0),
        ("employment", 1.5),
        ("salary", 2.0),
        ("gross income", 2.0),
        ("length of employment", 2.0),
        ("monthly income", 2.0),
    ],
    "business_entity": [
        ("trade name of business", 3.0),
        ("legal business name", 2.5),
        ("tax id", 2.0),
        ("principal", 1.5),
        ("years in business", 2.0),
    ],
    "joint_intent": [
        ("joint credit", 2.5),
        ("joint intent", 2.5),
        ("individual credit", 1.5),
        ("community property", 1.5),
    ],
    "vehicle_description": [
        ("vehicle identification", 3.0),
        ("year make model", 2.5),
        ("odometer", 2.0),
        ("serial number", 1.5),
        ("new used demo", 1.5),
    ],
    "trade_in_vehicle": [
        ("trade in", 2.5),
        ("trade-in", 2.5),
        ("vehicle traded", 2.5),
    ],
    "financial_disclosure": [
        ("truth in lending", 3.0),
        ("annual percentage rate", 2.5),
        ("finance charge", 2.0),
        ("amount financed", 2.0),
        ("total of payments", 2.0),
    ],
    "itemization": [
        ("cash price", 2.0),
        ("down payment", 2.0),
        ("sales tax", 1.5),
        ("total due", 1.5),
        ("itemization", 2.5),
        ("subtotal", 1.5),
    ],
    "insurance_product": [
        ("gap", 2.0),
        ("guaranteed asset protection", 2.5),
        ("credit life", 2.0),
        ("credit disability", 2.0),
        ("single interest insurance", 2.0),
        ("optional insurance", 1.5),
    ],
    "signature_authorization": [
        ("credit application signature", 3.0),
        ("applicant signature", 2.5),
        ("buyer signature", 2.0),
        ("by signing below", 2.0),
        ("signature of applicant", 2.5),
        ("agree to the terms", 2.0),
    ],
    "signature_consent": [
        ("optional consent", 2.5),
        ("joint intent signature", 2.5),
        ("consent to share", 2.0),
        ("initial here", 1.5),
        ("initials", 1.0),
    ],
    "dealer_seller_info": [
        ("seller creditor", 2.5),
        ("dealer name", 2.0),
        ("authorized dealer", 1.5),
        ("lender address", 2.0),
    ],
    "form_metadata": [
        ("form number", 2.0),
        ("revision date", 1.5),
        ("document language", 2.0),
    ],
}

DEFAULT_THRESHOLD = 0.35


def normalize_ocr_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_HOMOGLYPHS)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fuzzy_contains(haystack: str, needle: str, *, min_ratio: float = 0.82) -> bool:
    if needle in haystack:
        return True
    if len(needle) < 8 or len(haystack) > 800:
        return False
    window = len(needle) + 4
    for i in range(0, max(1, len(haystack) - len(needle) + 1)):
        chunk = haystack[i : i + window]
        if SequenceMatcher(None, chunk, needle).ratio() >= min_ratio:
            return True
    return False


def _shape_scores(text: str) -> dict[str, float]:
    scores = {ct: 0.0 for ct in CONTENT_TYPES}
    if _RE_SSN.search(text):
        scores["personal_identity"] += 2.5
    if _RE_DATE.search(text):
        scores["personal_identity"] += 0.8
    if _RE_PHONE.search(text):
        scores["contact_info"] += 2.0
    if _RE_EMAIL.search(text):
        scores["contact_info"] += 2.5
    if _RE_ZIP.search(text) or _RE_STATE.search(text):
        scores["residential_address"] += 1.2
    if _RE_EMP_LEN.search(text):
        scores["employment_income"] += 1.5
        scores["residential_address"] += 0.5
    if _RE_DOLLAR.search(text):
        scores["employment_income"] += 0.8
        scores["itemization"] += 0.8
        scores["financial_disclosure"] += 0.5
    if _RE_EIN.search(text):
        scores["business_entity"] += 2.0
    if _RE_VIN.search(text) or _RE_VIN_LABEL.search(text):
        scores["vehicle_description"] += 2.5
    if _RE_TRADE.search(text):
        scores["trade_in_vehicle"] += 2.0
        scores["vehicle_description"] -= 0.5
    if _RE_APR.search(text) or _RE_TIL.search(text):
        scores["financial_disclosure"] += 2.5
    if _RE_SIGNATURE.search(text):
        scores["signature_authorization"] += 1.5
    return scores


def _anchor_scores(text: str) -> dict[str, float]:
    scores = {ct: 0.0 for ct in CONTENT_TYPES}
    for ctype, anchors in _LABEL_ANCHORS.items():
        for phrase, weight in anchors:
            if fuzzy_contains(text, phrase):
                scores[ctype] += weight
    return scores


def _penalties(text: str, line_count: int) -> dict[str, float]:
    pen = {ct: 0.0 for ct in CONTENT_TYPES}
    if _RE_FOOTER.search(text):
        pen["signature_authorization"] += 4.0
        pen["form_metadata"] += 1.0
    if _RE_LIENHOLDER.search(text):
        pen["vehicle_description"] += 2.0
        pen["personal_identity"] += 1.0
    if _RE_LESSEE.search(text):
        pen["vehicle_description"] += 1.5
    if line_count <= 2 and _RE_SIGNATURE.search(text) and not _RE_FOOTER.search(text):
        pen["signature_authorization"] -= 0.5  # boost short sig rows
    return pen


def score_content_types(
    text: str,
    *,
    line_count: int | None = None,
) -> dict[str, float]:
    """Return raw scores per universal content type."""
    norm = normalize_ocr_text(text)
    if not norm:
        return {ct: 0.0 for ct in CONTENT_TYPES}
    lc = line_count if line_count is not None else norm.count("\n") + 1
    scores = {ct: 0.0 for ct in CONTENT_TYPES}
    for src in (_anchor_scores(norm), _shape_scores(norm)):
        for ct, val in src.items():
            scores[ct] += val
    for ct, val in _penalties(norm, lc).items():
        scores[ct] -= val
    return {ct: max(0.0, v) for ct, v in scores.items()}


def classify_content_types_heuristic(
    text: str,
    *,
    line_count: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[str], dict[str, float]]:
    """Multi-label classification via normalized heuristic scores."""
    raw = score_content_types(text, line_count=line_count)
    if not any(raw.values()):
        return [], {ct: 0.0 for ct in CONTENT_TYPES}
    max_score = max(raw.values())
    if max_score <= 0:
        return [], {k: 0.0 for k, v in raw.items()}
    norm_scores = {ct: round(v / max_score, 4) for ct, v in raw.items()}
    present = [ct for ct, s in norm_scores.items() if s >= threshold and raw[ct] > 0]
    present.sort(key=lambda c: (-norm_scores[c], c))
    return present, norm_scores


def preprocess_for_fasttext(text: str) -> str:
    """Normalize OCR text for FastText training / inference."""
    from string import punctuation

    t = normalize_ocr_text(text)
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"\$\s*\S+", " _dollars ", t)
    t = "".join(c for c in t if c in {"_"} or c not in punctuation)
    t = re.sub(r"\s+", " ", t).strip()
    return t
