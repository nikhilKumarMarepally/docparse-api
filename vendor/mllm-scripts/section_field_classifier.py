#!/usr/bin/env python3
"""Field-path detection for geometric section OCR crops (FastText + heuristics)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from section_content_heuristics import (
    fuzzy_contains,
    normalize_ocr_text,
    preprocess_for_fasttext,
)
from section_content_taxonomy import (
    field_path_to_content_types,
    load_document_field_paths,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]
DEFAULT_FIELD_MODEL_PATH = (
    ROOT / "wa577_gallery" / "section_classifier" / "models" / "section_fields.bin"
)
DEFAULT_FIELD_THRESHOLD = 0.35
LABEL_EMPTY = "empty"

# Reuse value-shape detectors from content heuristics module.
_RE_SSN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
_RE_PHONE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+", re.I)
_RE_VIN = re.compile(r"\b[A-HJ-NPR-Z0-9IO]{11,17}\b", re.I)
_RE_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_RE_STATE = re.compile(r"\b[A-Z]{2}\b")
_RE_DOLLAR = re.compile(r"\$\s?[\d,]+\.?\d*")
_RE_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
_RE_EIN = re.compile(r"\b\d{2}[-\s]?\d{7}\b")
_RE_EMP_LEN = re.compile(r"\b\d+\s*(?:mos|months|yrs|years)\b", re.I)
_RE_SIGNATURE = re.compile(
    r"\b(?:sign(?:ed|ature)?|by signing|credit application signature|applicant signature)\b",
    re.I,
)
_RE_TRADE = re.compile(r"trade[- ]?in|vehicle traded", re.I)

_RE_CO_APPLICANT = re.compile(
    r"\b(?:co[- ]?applicant(?:\s+information)?|co[- ]?buyer|guarantor|joint applicant)\b|"
    r"(?:^|\n)\s*b\.\s+co[- ]?applicant",
    re.I | re.M,
)
_RE_PRIMARY_APPLICANT = re.compile(
    r"(?:^|\n)\s*a\.\s+(?:primary\s+)?applicant\b|"
    r"\b(?:primary applicant|main applicant|buyer information|"
    r"credit application:\s*applicant)\b",
    re.I | re.M,
)
# Standalone "applicant information" header — exclude when part of "co-applicant information".
_RE_APPLICANT_INFO = re.compile(r"(?<!co[- ])\bapplicant information\b", re.I)
_RE_BUSINESS = re.compile(
    r"\b(?:trade name of business|legal business name|business applicant|"
    r"commercial credit|business credit|principal officer|years in business)\b",
    re.I,
)
_RE_MAILING = re.compile(r"\bmailing address\b", re.I)
_RE_PRESENT_ADDR = re.compile(
    r"\b(?:present address|current address|street address|address line)\b",
    re.I,
)
_RE_JOINT_INTENT = re.compile(
    r"\b(?:joint credit|individual credit|joint intent|community property)\b",
    re.I,
)
_RE_JOINT_SIG = re.compile(r"\bjoint intent\b", re.I)
_RE_CONSENT = re.compile(r"\boptional consent\b", re.I)
_RE_VEHICLE = re.compile(
    r"\b(?:year make model|vehicle identification|odometer|description of vehicle)\b",
    re.I,
)
_RE_FORM_DATE = re.compile(
    r"\b(?:\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{2,4})\b"
)
_RE_SIG_BLOCK = re.compile(
    r"(?:_{3,}|x{3,}|/s/|sign\s+here|applicant\s*:?\s*_{2,})",
    re.I,
)
_RE_FOOTER_META = re.compile(
    r"\b(?:all rights reserved|dealertrack inc|©\s*\d{4})\b",
    re.I,
)
_RE_STREET_NUMBER = re.compile(
    r"\b\d{1,5}\s+(?!(?:yrs?|mos|months|years)\b)"
    r"(?:[NSEW]\.?\s+)?[A-Za-z]{4,}",
    re.I,
)
_RE_NAME_VALUE_LINE = re.compile(
    r"(?:^|\n)\s*(?:[A-Z][A-Z'\-]+(?:\s+[A-Z][A-Z'\-]+){1,4}|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*(?:\n|$)",
    re.M,
)

_STREET_WORD_DENYLIST = frozenset(
    {
        "a", "an", "and", "applied", "accord", "at", "by", "for", "in", "of", "on",
        "or", "stat", "the", "to", "under", "wis",
    }
)
_STATE_DENYLIST = frozenset(
    {
        "AN", "AS", "AT", "BE", "BY", "DT", "IF", "IN", "IS", "IT", "NO", "OF",
        "ON", "OR", "TO", "US", "WE",
    }
)
_RE_DECIMAL_AMOUNT = re.compile(r"\b[\d,]+\.\d{2}\b")

_PERSONAL_LEAVES = frozenset(
    {
        "first_name",
        "middle_name",
        "last_name",
        "suffix",
        "dob",
        "ssn",
        "email",
        "phone",
    }
)
_ADDRESS_PARENT_SUFFIXES = (".address", ".mailing_address")

# Leaf label anchors (normalized OCR text).
_LEAF_LABELS: dict[str, tuple[str, ...]] = {
    "first_name": ("first name", "given name"),
    "middle_name": ("middle name", "middle initial", "mi"),
    "last_name": ("last name", "surname", "family name"),
    "suffix": ("suffix", "jr", "sr", "ii", "iii"),
    "ssn": ("social security", "ssn"),
    "dob": ("date of birth", "dob", "birth date"),
    "phone": ("phone", "home phone", "cell phone", "telephone"),
    "employer_name": ("employer", "employer name", "name of employer"),
    "occupation": ("occupation", "job title", "position"),
    "employment_type": ("employment type", "full time", "part time", "self employed"),
    "employment_status": ("employment status", "employed", "unemployed", "retired"),
    "employment_length_months": (
        "length of employment",
        "time employed",
        "employment length",
    ),
    "residence_length_months": (
        "time at present address",
        "length of residence",
        "residence length",
        "months at address",
    ),
    "monthly_housing_cost": (
        "monthly rent",
        "mortgage payment",
        "housing cost",
        "rent payment",
    ),
    "business_name": (
        "trade name of business",
        "legal business name",
        "business name",
        "company name",
    ),
    "tax_id": ("tax id", "ein", "federal tax id", "employer identification"),
    "street_address": ("street address", "present address", "address line 1"),
    "street_2": ("address line 2", "apt", "suite", "unit"),
    "city": ("city",),
    "state": ("state",),
    "zip": ("zip", "zip code", "postal code"),
    "amount": ("salary", "gross income", "monthly income", "income", "wages"),
    "period": ("per month", "monthly", "bi-weekly", "weekly", "annual", "yearly"),
    "vin": ("vin", "vehicle identification", "identification number"),
    "year": ("year", "model year"),
    "make": ("make",),
    "model": ("model",),
    "odometer": ("odometer", "mileage"),
    "form_number": ("form number", "form no", "form #"),
    "document_language": ("document language", "language"),
    "joint_intent_checked": ("joint credit", "individual credit", "joint intent"),
    "section_present": ("signature", "sign here", "applicant signature"),
    "signature_present": ("signature", "signed", "sign here"),
    "signature_date": ("signature date", "date signed"),
    "e_signed": ("electronically signed", "e-signed", "/s/"),
}


@lru_cache(maxsize=32)
def _cached_document_field_paths(document_type: str) -> tuple[str, ...]:
    return tuple(load_document_field_paths(document_type))


def _detect_context(norm: str) -> dict[str, bool]:
    co_applicant = bool(_RE_CO_APPLICANT.search(norm))
    primary_applicant = bool(
        _RE_PRIMARY_APPLICANT.search(norm) or _RE_APPLICANT_INFO.search(norm)
    )
    return {
        "applicant1": primary_applicant and not (co_applicant and not primary_applicant),
        "applicant2": co_applicant,
        "business": bool(_RE_BUSINESS.search(norm)),
        "mailing": bool(_RE_MAILING.search(norm)),
        "present_address": bool(_RE_PRESENT_ADDR.search(norm)),
        "joint_intent": bool(_RE_JOINT_INTENT.search(norm)),
        "joint_sig": bool(_RE_JOINT_SIG.search(norm)),
        "consent": bool(_RE_CONSENT.search(norm)),
        "vehicle": bool(_RE_VEHICLE.search(norm) or _RE_VIN.search(norm)),
        "trade_in": bool(_RE_TRADE.search(norm)),
        "signature": bool(_RE_SIG_BLOCK.search(norm)),
    }


def _applicant_prefixes(ctx: dict[str, bool]) -> list[str]:
    if ctx["business"] and not ctx["applicant1"] and not ctx["applicant2"]:
        return ["applicants.applicant1"]
    prefixes: list[str] = []
    if ctx["applicant2"] and not ctx["applicant1"]:
        prefixes.append("applicants.applicant2")
    elif ctx["applicant1"] and not ctx["applicant2"]:
        prefixes.append("applicants.applicant1")
    elif ctx["applicant2"]:
        prefixes.extend(["applicants.applicant1", "applicants.applicant2"])
    else:
        prefixes.append("applicants.applicant1")
    return prefixes


def _path_allowed_for_context(field_path: str, ctx: dict[str, bool], norm: str) -> bool:
    leaf = field_path.split(".")[-1]
    if "applicants.applicant2." in field_path and not ctx["applicant2"]:
        if not field_path.startswith("signatures."):
            return False
    if "applicants.applicant1." in field_path and ctx["applicant2"] and not ctx["applicant1"]:
        return False
    if "applicants.applicant2." in field_path and ctx["applicant1"] and not ctx["applicant2"]:
        return False
    if "applicant2" in field_path and not ctx["applicant2"]:
        if not field_path.startswith("signatures."):
            return False
    if "applicant1" in field_path and ctx["applicant2"] and not ctx["applicant1"]:
        if not field_path.startswith("signatures."):
            return False
    if ctx["business"] and leaf in _PERSONAL_LEAVES:
        if not any(fuzzy_contains(norm, phrase) for phrase in _LEAF_LABELS.get(leaf, ())):
            return False
    if ctx["trade_in"] and field_path.startswith("vehicle.") and "trade" not in field_path:
        if leaf in ("vin", "year", "make", "model"):
            return False
    if "mailing_address" in field_path and not ctx["mailing"]:
        return False
    return True


def _duration_value_evidence(norm: str) -> bool:
    for m in _RE_EMP_LEN.finditer(norm):
        prefix = norm[max(0, m.start() - 30) : m.start()]
        if re.search(r"\b(?:less than|if|within|under|minimum|at least)\s*$", prefix):
            continue
        return True
    return False


def _street_number_evidence(text: str) -> bool:
    for m in re.finditer(r"\b(\d{1,6})\s+([A-Za-z]+)", text):
        if m.start() > 0 and text[m.start() - 1] == ".":
            continue
        word = m.group(2)
        word_l = word.lower()
        if word_l in _STREET_WORD_DENYLIST:
            continue
        if word_l in {"yrs", "yr", "mos", "months", "years", "dealertrack", "stat"}:
            continue
        if len(word) < 4 and word_l not in {
            "st", "ave", "rd", "ln", "dr", "ct", "nw", "ne", "sw", "se",
        }:
            continue
        try:
            num = int(m.group(1))
            if 1900 <= num <= 2100:
                continue
        except ValueError:
            pass
        return True
    return False


def _name_value_line_evidence(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _line_is_label_header(stripped):
            continue
        if re.search(r"\b(?:notices?|residents?|information|copyright)\b", stripped, re.I):
            continue
        if _RE_NAME_VALUE_LINE.search(f"\n{stripped}\n"):
            return True
    return False


def _line_is_field_label(line: str) -> bool:
    if _line_is_label_header(line):
        return True
    lower = re.sub(r"\s+", " ", line.lower().strip())
    single_labels = (
        "last name",
        "first name",
        "middle name",
        "middle initial",
        "ssn",
        "dob",
        "date of birth",
        "social security",
        "social security number",
        "phone",
        "home phone",
        "cell phone",
        "employer",
        "occupation",
        "address",
        "city",
        "state",
        "zip",
        "zip code",
    )
    return lower in single_labels


def _section_is_label_only(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    return all(_line_is_field_label(ln) for ln in lines)


def _decimal_amount_evidence(text: str) -> bool:
    for m in _RE_DECIMAL_AMOUNT.finditer(text):
        prefix = text[max(0, m.start() - 20) : m.start()]
        if re.search(r"(?:§|\bstat\.?)\s*$", prefix, re.I):
            continue
        window = text[max(0, m.start() - 8) : m.end() + 8]
        if "$" in window:
            return True
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        if line_end == -1:
            line_end = len(text)
        if _RE_DOLLAR.search(text[line_start:line_end]):
            return True
    return False


def _vin_value_evidence(text: str, norm: str) -> bool:
    if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", text):
        return True
    if not (_RE_VEHICLE.search(norm) or fuzzy_contains(norm, "vehicle identification")):
        return False
    for m in _RE_VIN.finditer(text):
        token = m.group(0)
        if len(token) >= 11 and not token.isalpha():
            return True
    return False


def has_extractable_values(text: str) -> bool:
    """True when OCR shows filled form data, not labels/legal prose alone."""
    norm = normalize_ocr_text(text)
    if not norm:
        return False
    if _section_is_label_only(text):
        return False
    if _RE_FOOTER_META.search(norm) and len(norm.split()) < 25:
        if not any(
            (
                _RE_SSN.search(text),
                _RE_PHONE.search(text),
                _RE_EMAIL.search(text),
                _RE_DOLLAR.search(text),
                _RE_FORM_DATE.search(text),
                _RE_SIG_BLOCK.search(text),
            )
        ):
            return False
    if _RE_SSN.search(text):
        return True
    if _RE_PHONE.search(text):
        return True
    if _RE_EMAIL.search(text):
        return True
    if _RE_ZIP.search(text):
        return True
    if _RE_EIN.search(text):
        return True
    if _RE_DOLLAR.search(text):
        return True
    if _decimal_amount_evidence(text):
        return True
    if _duration_value_evidence(norm):
        return True
    if _vin_value_evidence(text, norm):
        return True
    if _RE_FORM_DATE.search(text):
        return True
    if _street_number_evidence(text):
        return True
    if _name_value_line_evidence(text):
        return True
    if _RE_SIG_BLOCK.search(text) or _RE_SIG_BLOCK.search(norm):
        return True
    return False


def _section_has_form_values(text: str, norm: str) -> bool:
    """Backward-compatible alias for has_extractable_values."""
    _ = norm
    return has_extractable_values(text)


def _state_code_evidence(text: str, norm: str) -> bool:
    codes = [m.group(0) for m in _RE_STATE.finditer(text)]
    codes = [c for c in codes if c not in _STATE_DENYLIST]
    if not codes:
        return False
    if _RE_ZIP.search(text):
        return True
    if _street_number_evidence(text):
        return True
    if _leaf_label_hit("state", norm) and any(
        re.search(rf"\b{re.escape(code)}\b", text) for code in codes
    ):
        return True
    return False


def _signature_block_evidence(text: str, norm: str) -> bool:
    return bool(_RE_SIG_BLOCK.search(text) or _RE_SIG_BLOCK.search(norm))


def _address_subfield_evidence(
    leaf: str,
    text: str,
    norm: str,
    ctx: dict[str, bool],
) -> bool:
    if leaf == "zip":
        return bool(_RE_ZIP.search(text))
    if leaf == "state":
        return _state_code_evidence(text, norm)
    if leaf == "street_address":
        return _street_number_evidence(text)
    if leaf == "city":
        return bool(_street_number_evidence(text) or _RE_ZIP.search(text))
    return False


def _name_field_evidence(leaf: str, text: str, norm: str) -> bool:
    if not _leaf_label_hit(leaf, norm):
        return False
    if _name_value_line_evidence(text):
        return True
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _line_is_label_header(stripped):
            continue
        tokens = re.findall(r"[A-Za-z']+", stripped)
        if len(tokens) >= 2 and not all(t.isupper() and len(t) <= 3 for t in tokens):
            return True
    return False


def _line_is_label_header(line: str) -> bool:
    lower = line.lower()
    label_markers = (
        "last name",
        "first name",
        "middle",
        "social security",
        "birth date",
        "address",
        "city",
        "state",
        "zip",
        "phone",
        "employer",
        "occupation",
        "signature",
        "previous",
    )
    hits = sum(1 for marker in label_markers if marker in lower)
    return hits >= 2 or (hits >= 1 and len(line) > 40)


def _leaf_label_hit(leaf: str, norm: str) -> bool:
    for phrase in _LEAF_LABELS.get(leaf, ()):
        if phrase == "date":
            continue
        if len(phrase) <= 3:
            if re.search(rf"\b{re.escape(phrase)}\b", norm):
                return True
            continue
        if fuzzy_contains(norm, phrase):
            return True
    return False


def _address_header_row(norm: str) -> bool:
    return fuzzy_contains(norm, "address") and (
        fuzzy_contains(norm, "city")
        or fuzzy_contains(norm, "zip")
        or fuzzy_contains(norm, "state")
    )


def _housing_payment_context(norm: str) -> bool:
    return any(
        fuzzy_contains(norm, phrase)
        for phrase in ("rent", "mortgage", "housing cost", "mtg", "rent/mtg")
    )


def _income_context(norm: str) -> bool:
    return any(fuzzy_contains(norm, phrase) for phrase in _LEAF_LABELS.get("amount", ()))


def _leaf_shape_hit(leaf: str, text: str, norm: str) -> bool:
    if leaf == "ssn":
        return bool(_RE_SSN.search(text))
    if leaf == "phone":
        return bool(_RE_PHONE.search(text))
    if leaf == "email":
        return bool(_RE_EMAIL.search(text))
    if leaf == "dob":
        return bool(_RE_FORM_DATE.search(text))
    if leaf == "zip":
        return bool(_RE_ZIP.search(text))
    if leaf == "state":
        return _state_code_evidence(text, norm)
    if leaf == "tax_id":
        return bool(_RE_EIN.search(text))
    if leaf == "monthly_housing_cost":
        return bool(_RE_DOLLAR.search(text)) and _housing_payment_context(norm)
    if leaf == "amount":
        has_amount = bool(_RE_DOLLAR.search(text) or _RE_DECIMAL_AMOUNT.search(text))
        return has_amount and _income_context(norm) and not _housing_payment_context(norm)
    if leaf == "employment_length_months":
        if not _duration_value_evidence(norm):
            return False
        if _leaf_label_hit("residence_length_months", norm) and not _leaf_label_hit(
            "employment_length_months", norm
        ):
            return False
        return True
    if leaf == "residence_length_months":
        return _duration_value_evidence(norm) and (
            _leaf_label_hit("residence_length_months", norm)
            or fuzzy_contains(norm, "time at address")
        )
    if leaf == "vin":
        return bool(_RE_VIN.search(text))
    if leaf in ("section_present", "signature_present", "e_signed"):
        return _signature_block_evidence(text, norm)
    if leaf == "signature_date":
        return _signature_block_evidence(text, norm) and _leaf_label_hit("signature_date", norm)
    return False


def _leaf_detected(leaf: str, text: str, norm: str, ctx: dict[str, bool]) -> bool:
    if _leaf_shape_hit(leaf, text, norm):
        return True
    if leaf in ("first_name", "last_name", "middle_name", "suffix"):
        return _name_field_evidence(leaf, text, norm)
    if leaf in ("street_address", "city", "state", "zip"):
        return _address_subfield_evidence(leaf, text, norm, ctx)
    if leaf == "address":
        return any(
            _address_subfield_evidence(child, text, norm, ctx)
            for child in ("street_address", "city", "state", "zip")
        )
    if leaf in ("employer_name", "occupation", "employment_type", "employment_status"):
        if not _leaf_label_hit(leaf, norm):
            return False
        return bool(
            _RE_DOLLAR.search(text)
            or _duration_value_evidence(norm)
            or _RE_PHONE.search(text)
            or _name_value_line_evidence(text)
        )
    if leaf in ("period", "amount", "monthly_housing_cost", "tax_id"):
        return _leaf_shape_hit(leaf, text, norm)
    if leaf == "business_name":
        if not _leaf_label_hit(leaf, norm):
            return False
        return bool(
            _name_value_line_evidence(text)
            or _RE_EIN.search(text)
            or re.search(r"\b(?:llc|inc|corp|l\.?l\.?c\.?|company)\b", norm)
        )
    if leaf in ("employment_length_months", "residence_length_months"):
        return _leaf_shape_hit(leaf, text, norm)
    if leaf in ("joint_intent_checked",):
        return _leaf_shape_hit(leaf, text, norm) or (
            _leaf_label_hit(leaf, norm) and _signature_block_evidence(text, norm)
        )
    if leaf in ("section_present", "signature_present", "signature_date", "e_signed"):
        return _signature_block_evidence(text, norm)
    if _leaf_label_hit(leaf, norm):
        return False
    return False


def _signature_role_allowed(field_path: str, ctx: dict[str, bool]) -> bool:
    if "applicant2" in field_path:
        return ctx["applicant2"]
    if "applicant1" in field_path:
        return ctx["applicant1"]
    if ".dealer." in field_path:
        return ctx["signature"]
    return True


def _signature_section_for_path(field_path: str, ctx: dict[str, bool]) -> bool:
    if not field_path.startswith("signatures."):
        return True
    if not ctx["signature"]:
        return False
    parts = field_path.split(".")
    if len(parts) < 3:
        return True
    section = parts[1]
    if section == "joint_intent":
        return ctx["joint_sig"] or ctx["joint_intent"]
    if section == "optional_consent":
        return ctx["consent"]
    if section in ("credit_application", "initialed_pages"):
        return _signature_role_allowed(field_path, ctx)
    return True


def _content_type_boost(
    field_path: str,
    content_types: list[str] | None,
) -> bool:
    if not content_types:
        return True
    ctypes = field_path_to_content_types(field_path)
    return bool(ctypes & set(content_types))


def _add_parent_paths(paths: set[str], candidates: set[str]) -> set[str]:
    out = set(paths)
    for path in list(paths):
        if ".address." in path:
            parent = path.split(".address.", 1)[0] + ".address"
            if parent in candidates or any(c.startswith(parent + ".") for c in candidates):
                out.add(parent)
        if ".mailing_address." in path:
            parent = path.split(".mailing_address.", 1)[0] + ".mailing_address"
            if parent in candidates or any(c.startswith(parent + ".") for c in candidates):
                out.add(parent)
        if path.endswith(".address") or path.endswith(".mailing_address"):
            out.add(path)
    return out


class FastTextFieldClassifier:
    """Multi-label FastText model: field paths or ``empty`` for section OCR text."""

    def __init__(self, model_path: Path | str) -> None:
        import fasttext  # noqa: PLC0415

        self.model_path = Path(model_path)
        self.model = fasttext.load_model(str(self.model_path))

    def predict(
        self,
        text: str,
        *,
        threshold: float = DEFAULT_FIELD_THRESHOLD,
        k: int | None = None,
    ) -> tuple[list[str], dict[str, float], bool]:
        """Return (field_paths, scores, is_empty).

        ``is_empty`` is True when the model confidently predicts no extractable fields.
        """
        proc = preprocess_for_fasttext(text)
        if not proc:
            return [], {LABEL_EMPTY: 1.0}, True

        k = k or len(self.model.labels)
        labels, probs = self.model.predict(proc, k=k)
        scores: dict[str, float] = {
            label.replace("__label__", ""): float(prob)
            for label, prob in zip(labels, probs)
        }
        empty_score = scores.get(LABEL_EMPTY, 0.0)
        field_hits = [
            name
            for name, score in scores.items()
            if name != LABEL_EMPTY and score >= threshold
        ]
        if field_hits:
            best_field = max(scores[name] for name in field_hits)
            if empty_score >= threshold and empty_score > best_field:
                return [], scores, True
            return sorted(field_hits), scores, False
        if empty_score >= threshold:
            return [], scores, True
        return [], scores, False


@lru_cache(maxsize=1)
def _cached_default_field_model() -> FastTextFieldClassifier | None:
    if DEFAULT_FIELD_MODEL_PATH.exists():
        return FastTextFieldClassifier(DEFAULT_FIELD_MODEL_PATH)
    return None


def _resolve_field_model(
    model: FastTextFieldClassifier | str | Path | None,
) -> FastTextFieldClassifier | None:
    if model is None:
        return _cached_default_field_model()
    if isinstance(model, FastTextFieldClassifier):
        return model
    path = Path(model)
    if path.exists():
        return FastTextFieldClassifier(path)
    return None


def load_default_field_model() -> FastTextFieldClassifier | None:
    return _cached_default_field_model()


def _filter_field_candidates(
    fields: list[str],
    document_type: str,
) -> list[str]:
    candidates = set(_cached_document_field_paths(document_type))
    if not candidates:
        return []
    return sorted(f for f in fields if f in candidates)


def _classify_section_fields_heuristic(
    text: str,
    document_type: str,
    *,
    content_types: list[str] | None = None,
) -> list[str]:
    """Heuristic-only field detection (training labels + FastText fallback)."""
    norm = normalize_ocr_text(text)
    if not norm:
        return []
    if _RE_FOOTER_META.search(norm) and not _section_has_form_values(text, norm):
        return []
    if not _section_has_form_values(text, norm):
        return []

    candidates = set(_cached_document_field_paths(document_type))
    if not candidates:
        return []

    ctx = _detect_context(norm)
    matched: set[str] = set()

    for field_path in candidates:
        if not _path_allowed_for_context(field_path, ctx, norm):
            continue
        if not _signature_section_for_path(field_path, ctx):
            continue

        leaf = field_path.split(".")[-1]
        if field_path.endswith(_ADDRESS_PARENT_SUFFIXES):
            if _leaf_detected("address", text, norm, ctx):
                matched.add(field_path)
            continue

        if not _leaf_detected(leaf, text, norm, ctx):
            continue
        if content_types and not _content_type_boost(field_path, content_types):
            if not (_leaf_label_hit(leaf, norm) or _leaf_shape_hit(leaf, text, norm)):
                continue
        matched.add(field_path)

    if not ctx["applicant1"] and not ctx["applicant2"] and not ctx["business"]:
        for leaf in _PERSONAL_LEAVES | {"employer_name", "occupation"}:
            if not _leaf_detected(leaf, text, norm, ctx):
                continue
            for prefix in _applicant_prefixes(ctx):
                candidate = f"{prefix}.{leaf}"
                if candidate in candidates:
                    matched.add(candidate)
                addr_leaf = f"{prefix}.address.{leaf}"
                if leaf in {"street_address", "city", "state", "zip"} and addr_leaf in candidates:
                    matched.add(addr_leaf)

    matched = _add_parent_paths(matched, candidates)
    return sorted(matched)


def _filter_fields_by_context(
    fields: list[str],
    text: str,
    document_type: str,
) -> list[str]:
    norm = normalize_ocr_text(text)
    ctx = _detect_context(norm)
    candidates = set(_cached_document_field_paths(document_type))
    kept: list[str] = []
    for field_path in fields:
        if field_path not in candidates:
            continue
        if not _path_allowed_for_context(field_path, ctx, norm):
            continue
        if not _signature_section_for_path(field_path, ctx):
            continue
        kept.append(field_path)
    return sorted(kept)


def _heuristic_override_empty(text: str, heuristic_fields: list[str]) -> bool:
    """True when heuristics found form fields the model should not suppress."""
    if not heuristic_fields or not has_extractable_values(text):
        return False
    try:
        from section_boilerplate_classifier import is_boilerplate_heuristic  # noqa: PLC0415
    except ImportError:
        return True
    return not is_boilerplate_heuristic(text)


def classify_section_fields(
    text: str,
    document_type: str,
    *,
    content_types: list[str] | None = None,
    field_model: FastTextFieldClassifier | str | Path | None = None,
    field_threshold: float = DEFAULT_FIELD_THRESHOLD,
    min_field_confidence: float = 0.20,
) -> list[str]:
    """Return dotted field paths likely present in section OCR text."""
    norm = normalize_ocr_text(text)
    if not norm:
        return []

    heuristic_fields = _classify_section_fields_heuristic(
        text,
        document_type,
        content_types=content_types,
    )

    ft = _resolve_field_model(field_model)
    if ft is None:
        return heuristic_fields

    raw_fields, scores, is_empty = ft.predict(text, threshold=field_threshold)
    empty_score = scores.get(LABEL_EMPTY, 0.0)
    best_field_score = max(
        (score for name, score in scores.items() if name != LABEL_EMPTY),
        default=0.0,
    )

    if is_empty and empty_score >= field_threshold:
        if _heuristic_override_empty(text, heuristic_fields):
            return heuristic_fields
        return []

    filtered = _filter_fields_by_context(raw_fields, text, document_type)
    if filtered:
        return filtered

    if best_field_score < min_field_confidence and empty_score < field_threshold:
        return heuristic_fields
    if raw_fields and not filtered:
        return heuristic_fields
    if is_empty:
        return []
    return heuristic_fields


def classify_section_fields_payload(
    sections: list[dict[str, Any]],
    *,
    document_type: str,
    field_model: FastTextFieldClassifier | str | Path | None = None,
    field_threshold: float = DEFAULT_FIELD_THRESHOLD,
) -> list[dict[str, Any]]:
    """Annotate section dicts with a top-level fields list (also inside content_classification)."""
    out: list[dict[str, Any]] = []
    for section in sections:
        row = dict(section)
        content_types = None
        if isinstance(row.get("content_classification"), dict):
            content_types = row["content_classification"].get("content_types")
        fields = classify_section_fields(
            row.get("text") or "",
            document_type,
            content_types=content_types,
            field_model=field_model,
            field_threshold=field_threshold,
        )
        row["fields"] = fields
        cc = dict(row.get("content_classification") or {})
        cc["fields"] = fields
        row["content_classification"] = cc
        out.append(row)
    return out
