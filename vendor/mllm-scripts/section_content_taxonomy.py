#!/usr/bin/env python3
"""Universal content-type taxonomy and doc-type section routing registry.

Parses QA techno-configs llm_configs YAMLs to map doc-specific MLLM sections
to universal content types used by the cross-doc section classifier.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent  # DocExtract repo root (standalone deploy)
DEFAULT_CONFIGS_ROOT = (
    ROOT.parent / "techno-configs" / "techno_configs" / "envs" / "qa" / "document_fields" / "extractions" / "llm_configs"
)
DEFAULT_REGISTRY_PATH = ROOT / "wa577_gallery" / "section_classifier" / "section_content_registry.json"

CONTENT_TYPES: tuple[str, ...] = (
    "personal_identity",
    "contact_info",
    "residential_address",
    "mailing_address",
    "employment_income",
    "business_entity",
    "joint_intent",
    "vehicle_description",
    "trade_in_vehicle",
    "financial_disclosure",
    "itemization",
    "insurance_product",
    "signature_authorization",
    "signature_consent",
    "dealer_seller_info",
    "form_metadata",
)

# Field-path leaf / prefix hints -> universal content types (doc-agnostic).
_FIELD_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(^|\.)(first_name|middle_name|last_name|suffix|dob|ssn)$"), "personal_identity"),
    (re.compile(r"(^|\.)(email|phone|home_phone|cell_phone)$"), "contact_info"),
    (re.compile(r"mailing_address", re.I), "mailing_address"),
    (re.compile(r"(^|\.)(street_address|street_2|city|state|zip|county)$"), "residential_address"),
    (re.compile(r"\.address\.", re.I), "residential_address"),
    (re.compile(r"residence_length|monthly_housing|housing_cost", re.I), "residential_address"),
    (
        re.compile(
            r"(employer|occupation|employment_|income\.|salary|gross_income|net_income)",
            re.I,
        ),
        "employment_income",
    ),
    (re.compile(r"(business_name|tax_id|ein)", re.I), "business_entity"),
    (re.compile(r"joint_intent", re.I), "joint_intent"),
    (
        re.compile(
            r"(^|\.)(vin|odometer|vehicle\.|year|make|model|body_type|color|mileage)$",
            re.I,
        ),
        "vehicle_description",
    ),
    (re.compile(r"trade_in", re.I), "trade_in_vehicle"),
    (
        re.compile(
            r"(amount_financed|finance_charge|apr|annual_percentage|truth.in.lending|total_of_payments|total_sale_price)",
            re.I,
        ),
        "financial_disclosure",
    ),
    (
        re.compile(
            r"(cash_price|itemization|down_?payment|sales_tax|total_balance|fees|subtotal|amount_due)",
            re.I,
        ),
        "itemization",
    ),
    (
        re.compile(r"(gap|credit_life|credit_disability|vsi|insurance_product|policy_number|warranty)", re.I),
        "insurance_product",
    ),
    (re.compile(r"signatures\.(credit_application|document)\.", re.I), "signature_authorization"),
    (re.compile(r"signatures\.joint_intent\.", re.I), "signature_consent"),
    (re.compile(r"signatures\.optional_consent\.", re.I), "signature_consent"),
    (re.compile(r"signatures\.initialed_pages\.", re.I), "signature_consent"),
    (re.compile(r"signatures\.", re.I), "signature_authorization"),
    (re.compile(r"(dealer|seller|creditor|lender)\.", re.I), "dealer_seller_info"),
    (re.compile(r"(form_number|document_language|revision_date|form_version)", re.I), "form_metadata"),
]

# Section id / description keyword hints for section_aware YAMLs.
_SECTION_KEYWORD_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"joint.?intent|joint.?credit", re.I), "joint_intent"),
    (re.compile(r"buyer|co.?buyer|applicant|guarantor", re.I), "personal_identity"),
    (re.compile(r"vehicle|vin|odometer|description of property", re.I), "vehicle_description"),
    (re.compile(r"trade.?in", re.I), "trade_in_vehicle"),
    (re.compile(r"truth.in.lending|federal disclosure|amount financed", re.I), "financial_disclosure"),
    (re.compile(r"itemization|cash price|downpayment", re.I), "itemization"),
    (re.compile(r"gap|insurance|vsi|credit life", re.I), "insurance_product"),
    (re.compile(r"signature|agree to terms|assignment", re.I), "signature_authorization"),
    (re.compile(r"consent|optional", re.I), "signature_consent"),
    (re.compile(r"seller|creditor|dealer", re.I), "dealer_seller_info"),
    (re.compile(r"document level|form number|metadata", re.I), "form_metadata"),
]

TARGET_DOC_TYPES: tuple[str, ...] = (
    "credit_application",
    "title_application",
    "retail_installment_sales_contract",
    "gap_binder",
    "buyers_order",
    "odometer_disclosure_statement_retail",
    "vehicle_service_contract",
)


def field_path_to_content_types(field_path: str) -> set[str]:
    """Map a dotted extraction field path to universal content type(s)."""
    fp = field_path.strip()
    if not fp:
        return set()
    out: set[str] = set()
    for pattern, ctype in _FIELD_RULES:
        if pattern.search(fp):
            out.add(ctype)
    return out


def section_keywords_to_content_types(section_id: str, description: str = "") -> set[str]:
    blob = f"{section_id} {description}"
    out: set[str] = set()
    for pattern, ctype in _SECTION_KEYWORD_RULES:
        if pattern.search(blob):
            out.add(ctype)
    return out


def _normalize_field_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.split(":")[0].strip()
    return str(entry)


_COMPOSITE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "address": ("street_address", "street_2", "city", "state", "zip"),
    "mailing_address": ("street_address", "street_2", "city", "state", "zip"),
    "income": ("amount", "period"),
    "signature_block": ("section_present", "signature_present", "signature_date", "e_signed"),
}


def expand_field_mapping_entry(entry: str) -> list[str]:
    """Expand YAML field_mapping aliases (address, income, signature_block) to dotted paths."""
    raw = entry.strip()
    if not raw:
        return []
    if ":" not in raw:
        base = raw
        if re.match(r"signatures\.[^.]+\.(applicant1|applicant2|dealer)$", base):
            return [f"{base}.{suffix}" for suffix in _COMPOSITE_SUFFIXES["signature_block"]]
        return [base]
    base, alias = (part.strip() for part in raw.split(":", 1))
    keys = _COMPOSITE_SUFFIXES.get(alias)
    if not keys:
        return [base]
    if alias in ("address", "mailing_address"):
        return [base, *(f"{base}.{key}" for key in keys)]
    return [f"{base}.{key}" for key in keys]


def _coerce_mapping_entry(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry.strip() or None
    if isinstance(entry, dict) and len(entry) == 1:
        key, value = next(iter(entry.items()))
        return f"{key}:{value}"
    return None


def field_paths_from_llm_config(path: Path) -> list[str]:
    """Collect dotted field paths from a QA llm_config YAML."""
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    data = yaml.safe_load(path.read_text()) or {}
    payload = (data.get("model_info") or {}).get("payload_config") or {}
    paths: set[str] = set()

    field_mapping = (payload.get("field_mapping") or {}).get("default") or {}
    if isinstance(field_mapping, dict):
        for entries in field_mapping.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                coerced = _coerce_mapping_entry(entry)
                if coerced:
                    paths.update(expand_field_mapping_entry(coerced))

    prompt_config = payload.get("prompt_config") or {}
    field_prompts = prompt_config.get("field_prompts") or {}
    if isinstance(field_prompts, dict):
        for key in field_prompts:
            fp = str(key).strip()
            if fp:
                paths.add(fp)
                if fp.startswith("signatures.") and fp.count(".") == 2:
                    paths.update(
                        f"{fp}.{suffix}" for suffix in _COMPOSITE_SUFFIXES["signature_block"]
                    )

    return sorted(paths)


def load_document_field_paths(
    document_type: str,
    *,
    configs_root: Path | None = None,
) -> list[str]:
    """Load extraction field paths for a document type from QA llm_config YAML."""
    root = configs_root or DEFAULT_CONFIGS_ROOT
    yml = root / f"{document_type}.yml"
    if yml.exists():
        return field_paths_from_llm_config(yml)
    if document_type == "credit_application":
        try:
            import sys

            repo_root = ROOT
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from ca_appctx_field_stats import ALL_FIELDS  # noqa: PLC0415

            return list(ALL_FIELDS)
        except ImportError:
            pass
    return []


def _fields_from_mapping_block(block: Any) -> list[str]:
    if not isinstance(block, dict):
        return []
    fields: list[str] = []
    for _section_id, entries in block.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            fields.append(_normalize_field_entry(entry))
    return fields


def _infer_custom_doc_groups(field_prompts: dict[str, Any]) -> dict[str, list[str]]:
    """Group custom prompt keys into pseudo-sections for routing."""
    groups: dict[str, set[str]] = {
        "applicants": set(),
        "vehicle": set(),
        "signatures": set(),
        "financial": set(),
        "dealer": set(),
        "form": set(),
    }
    for key in field_prompts:
        fp = key.strip()
        if fp.startswith("applicants."):
            groups["applicants"].add(fp)
        elif re.search(r"\b(vin|vehicle|odometer|trade_in)\b", fp, re.I):
            groups["vehicle"].add(fp)
        elif fp.startswith("signatures."):
            groups["signatures"].add(fp)
        elif re.search(
            r"(amount_financed|total_|cash_price|itemization|finance|price|fee)",
            fp,
            re.I,
        ):
            groups["financial"].add(fp)
        elif re.search(r"(dealer|seller|lender|creditor)", fp, re.I):
            groups["dealer"].add(fp)
        elif re.search(r"(form_number|document_language)", fp, re.I):
            groups["form"].add(fp)
    return {k: sorted(v) for k, v in groups.items() if v}


def parse_llm_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML required: pip install pyyaml")
    data = yaml.safe_load(path.read_text()) or {}
    payload = (data.get("model_info") or {}).get("payload_config") or {}
    doc_type = path.stem
    result: dict[str, Any] = {
        "document_type": doc_type,
        "payload_type": payload.get("type", "unknown"),
        "section_to_content_types": {},
        "content_type_to_sections": {ct: [] for ct in CONTENT_TYPES},
    }

    if payload.get("type") == "section_aware":
        sections = ((payload.get("section_detection") or {}).get("sections")) or []
        field_mapping = (payload.get("field_mapping") or {}).get("default") or {}
        for section in sections:
            if not isinstance(section, dict):
                continue
            sid = section.get("id") or ""
            desc = section.get("description") or ""
            ctypes: set[str] = set(section_keywords_to_content_types(sid, desc))
            entries = field_mapping.get(sid) or []
            for entry in entries:
                ctypes |= field_path_to_content_types(_normalize_field_entry(entry))
            if ctypes:
                result["section_to_content_types"][sid] = sorted(ctypes)
                for ct in ctypes:
                    if sid not in result["content_type_to_sections"][ct]:
                        result["content_type_to_sections"][ct].append(sid)
    else:
        prompt_config = payload.get("prompt_config") or {}
        field_prompts = prompt_config.get("field_prompts") or {}
        groups = _infer_custom_doc_groups(field_prompts)
        pseudo_map = {
            "applicants": "applicant_fields",
            "vehicle": "vehicle_fields",
            "signatures": "signature_fields",
            "financial": "financial_fields",
            "dealer": "dealer_fields",
            "form": "form_fields",
        }
        for group, fields in groups.items():
            sid = pseudo_map[group]
            ctypes: set[str] = set()
            for fp in fields:
                ctypes |= field_path_to_content_types(fp)
            if group == "signatures":
                ctypes.add("signature_authorization")
            if ctypes:
                result["section_to_content_types"][sid] = sorted(ctypes)
                for ct in ctypes:
                    if sid not in result["content_type_to_sections"][ct]:
                        result["content_type_to_sections"][ct].append(sid)

    return result


def build_registry(
    configs_root: Path | None = None,
    doc_types: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    root = configs_root or DEFAULT_CONFIGS_ROOT
    wanted = set(doc_types or TARGET_DOC_TYPES)
    registry: dict[str, Any] = {
        "content_types": list(CONTENT_TYPES),
        "document_types": {},
    }
    for yml in sorted(root.glob("*.yml")):
        if yml.stem not in wanted:
            continue
        try:
            registry["document_types"][yml.stem] = parse_llm_config(yml)
        except Exception as exc:  # noqa: BLE001
            registry["document_types"][yml.stem] = {"error": str(exc)}
    return registry


def save_registry(path: Path | None = None, **kwargs: Any) -> Path:
    out = path or DEFAULT_REGISTRY_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    registry = build_registry(**kwargs)
    out.write_text(json.dumps(registry, indent=2))
    return out


def load_registry(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_REGISTRY_PATH
    if not p.exists():
        save_registry(p)
    return json.loads(p.read_text())


def route_content_types_to_sections(
    content_types: list[str],
    document_type: str,
    registry: dict[str, Any] | None = None,
) -> list[str]:
    """Map predicted universal types to doc-specific section ids (best-effort)."""
    reg = registry or load_registry()
    doc = (reg.get("document_types") or {}).get(document_type) or {}
    section_map: dict[str, list[str]] = doc.get("section_to_content_types") or {}
    scores: dict[str, int] = {}
    for section_id, ctypes in section_map.items():
        scores[section_id] = len(set(content_types) & set(ctypes))
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [sid for sid, score in ranked if score > 0]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build section content registry from QA YAMLs")
    ap.add_argument("--configs-root", type=Path, default=DEFAULT_CONFIGS_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_REGISTRY_PATH)
    ap.add_argument("--doc-type", action="append", dest="doc_types")
    args = ap.parse_args()
    doc_types = tuple(args.doc_types) if args.doc_types else TARGET_DOC_TYPES
    out = save_registry(args.out, configs_root=args.configs_root, doc_types=doc_types)
    print(f"Wrote {out} ({len(json.loads(out.read_text())['document_types'])} doc types)")


if __name__ == "__main__":
    main()
