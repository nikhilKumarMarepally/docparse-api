#!/usr/bin/env python3
"""Discover VIN extraction tickets (Linear/Jira) and resolve runnable docs."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3

ROOT = Path(__file__).resolve().parents[5]
QA_ROOT = ROOT.parent / "techno-configs/techno_configs/envs/qa/document_fields"

VIN_QUESTION_CODES = [
    "matches_contract_vin",
    "matches_vehicle_vin",
    "matches_application_vehicle_vin",
    "matches_applicant_vin",
]

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.I,
)
VIN_17_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.I)
TITLE_DOC_TYPE_RE = re.compile(r"\[([a-z0-9_]+)\]", re.I)
PARTNER_LABEL_RE = re.compile(r"partner:([a-z0-9_]+)", re.I)

EXCLUDE_TITLE_PHRASES = (
    "credit stacking",
    "glue job",
    "config-studio",
    "remove vin",
    "cr-906",
    "glossary",
    "data-agent",
    "straw purchase",
    "funded a strict superset",
)

_ddb = None


def norm_vin(v: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(v or "")).upper()


def ensure_looker_env() -> None:
    mapping = {
        "LOOKERSDK_BASE_URL": "LOOKER_BASE_URL",
        "LOOKERSDK_CLIENT_ID": "LOOKER_CLIENT_ID",
        "LOOKERSDK_CLIENT_SECRET": "LOOKER_CLIENT_SECRET",
    }
    for sdk_var, env_var in mapping.items():
        if sdk_var not in os.environ and env_var in os.environ:
            os.environ[sdk_var] = os.environ[env_var]
    os.environ.setdefault("LOOKERSDK_API_VERSION", "4.0")


def init_aws():
    global _ddb
    if _ddb is None:
        session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "prod"), region_name="us-west-2")
        _ddb = session.client("dynamodb")


def has_qa_yaml(doc_type: str) -> bool:
    return (QA_ROOT / f"extractions/llm_configs/{doc_type}.yml").exists()


def parse_iso_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def within_days(value: str | None, since_days: int) -> bool:
    dt = parse_iso_date(value)
    if not dt:
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    return dt >= cutoff


def is_vin_extraction_ticket(issue: dict[str, Any]) -> bool:
    title = (issue.get("title") or "").lower()
    desc = (issue.get("description") or "").lower()
    status_type = (issue.get("statusType") or "").lower()

    if status_type == "canceled":
        return False
    if any(phrase in title for phrase in EXCLUDE_TITLE_PHRASES):
        return False
    if "remove vin number verification" in title:
        return False

    if any(code in title for code in VIN_QUESTION_CODES):
        return True
    if any(code in desc for code in VIN_QUESTION_CODES):
        return True
    if "incorrect vin extraction" in title:
        return True
    if "failure to extract vehicle vin" in title:
        return True
    if "vin extraction" in title and "bug" in title:
        return True

    labels = issue.get("labels") or []
    label_text = " ".join(labels) if isinstance(labels, list) else str(labels)
    if "category:extraction" in label_text and "vin" in title:
        return True
    if issue.get("id", "").startswith("WA-") and "vin" in title and "qc" in label_text.lower():
        return True
    if "qc" in label_text.lower() and "extraction bug" in title:
        if "vin" in desc or any(code in desc for code in VIN_QUESTION_CODES):
            return True

    # QC doc tickets with explicit VIN items in description
    if "matches_contract_vin" in desc or "matches_vehicle_vin" in desc:
        return True
    if re.search(r"matches_\w+_vin", desc):
        return True

    return False


def parse_verification_doc_type(title: str) -> str | None:
    match = TITLE_DOC_TYPE_RE.search(title or "")
    return match.group(1) if match else None


def parse_question_code(title: str, description: str | None) -> str:
    text = f"{title}\n{description or ''}"
    for code in VIN_QUESTION_CODES:
        if code in text:
            return code
    return "matches_contract_vin"


def parse_partner(issue: dict[str, Any]) -> str | None:
    labels = issue.get("labels") or []
    for label in labels:
        text = label if isinstance(label, str) else str(label)
        match = PARTNER_LABEL_RE.search(text)
        if match:
            return match.group(1)
    desc = issue.get("description") or ""
    for line in desc.splitlines():
        line = line.strip().lower()
        if line.startswith("partner:"):
            return line.split(":", 1)[1].strip().split()[0]
        if "**partner:**" in line.lower():
            part = line.split(":", 1)[-1].strip().strip("*")
            if part:
                return part.split()[0].lower()
    return None


def extract_doc_ids(text: str) -> list[str]:
    found: list[str] = []
    for match in UUID_RE.finditer(text or ""):
        uid = match.group(0).lower()
        if uid not in found:
            found.append(uid)
    return found


EXPECTED_RE = re.compile(r"\*\*Expected:\*\*\s*`([^`]+)`", re.I)


def is_probable_doc_id(doc_id: str) -> bool:
    try:
        get_doc_meta(doc_id)
        return True
    except Exception:
        return False


def section_is_vin_related(section: str) -> bool:
    lower = section.lower()
    return any(
        token in lower
        for token in (
            "matches_contract_vin",
            "matches_vehicle_vin",
            "matches_application_vehicle_vin",
            "matches_applicant_vin",
            "contract's vin",
            "vehicle vin",
        )
    )


def extract_doc_ground_truth_pairs(text: str) -> list[tuple[str, str | None]]:
    """Parse QC-style blocks with Document + Expected VIN nearby."""
    pairs: list[tuple[str, str | None]] = []
    if not text:
        return pairs

    sections = re.split(r"\n###\s+", text)
    if len(sections) == 1:
        sections = [text]

    for section in sections:
        if not section_is_vin_related(section):
            continue
        lines = section.splitlines()
        for i, line in enumerate(lines):
            if "document" not in line.lower():
                continue
            doc_ids = extract_doc_ids(line)
            if not doc_ids:
                continue
            doc_id = doc_ids[0]
            if not is_probable_doc_id(doc_id):
                continue
            window = "\n".join(lines[i : i + 10])
            expected = EXPECTED_RE.search(window)
            truth = norm_vin(expected.group(1)) if expected else None
            if not truth:
                vins = [norm_vin(v) for v in VIN_17_RE.findall(window)]
                truth = next((v for v in vins if len(v) >= 11), None)
            pairs.append((doc_id, truth))

    if not pairs:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "document" not in line.lower():
                continue
            doc_ids = extract_doc_ids(line)
            if not doc_ids:
                continue
            doc_id = doc_ids[0]
            if not is_probable_doc_id(doc_id):
                continue
            window = "\n".join(lines[i : i + 8])
            if not section_is_vin_related(window):
                continue
            expected = EXPECTED_RE.search(window)
            truth = norm_vin(expected.group(1)) if expected else None
            if not truth:
                vins = [norm_vin(v) for v in VIN_17_RE.findall(window)]
                truth = next((v for v in vins if len(v) >= 11), None)
            pairs.append((doc_id, truth))

    if not pairs:
        for doc_id in extract_doc_ids(text):
            if is_probable_doc_id(doc_id):
                pairs.append((doc_id, None))
    return pairs


def is_staging_only(text: str) -> bool:
    lower = (text or "").lower()
    has_staging = ".staging.informediq" in lower or "verifyiq.staging" in lower
    has_prod = ".prod.informediq" in lower or "verifyiq.prod" in lower
    return has_staging and not has_prod


def load_linear_cache(cache_path: Path) -> list[dict[str, Any]]:
    if not cache_path.exists():
        return []
    data = json.loads(cache_path.read_text())
    return data.get("issues", data if isinstance(data, list) else [])


def discover_linear_tickets(cache_path: Path, since_days: int) -> list[dict[str, Any]]:
    issues = load_linear_cache(cache_path)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for issue in issues:
        ident = issue.get("id") or issue.get("identifier")
        if not ident or ident in seen:
            continue
        if not within_days(issue.get("createdAt"), since_days):
            continue
        if not is_vin_extraction_ticket(issue):
            continue
        seen.add(ident)
        selected.append(issue)
    return sorted(selected, key=lambda x: x.get("createdAt", ""), reverse=True)


def query_looker_vin_rows(
    *,
    since_days: int,
    partner: str | None = None,
    verification_doc_type: str | None = None,
    question_code: str | None = None,
    document_id: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ensure_looker_env()
    import looker_sdk
    from looker_sdk import models40 as models

    sdk = looker_sdk.init40()
    filters: dict[str, str] = {
        "questions_original.question_code": question_code or ",".join(VIN_QUESTION_CODES),
        "questions_original.status": "fail,review",
        "documents.created_date": f"{since_days} days",
    }
    if partner:
        filters["documents.partner_name"] = partner
    if verification_doc_type:
        filters["documents.document_type"] = verification_doc_type
    if document_id:
        filters["questions_original.document_id"] = document_id

    body = models.WriteQuery(
        model="redshift_dw",
        view="questions_original",
        fields=[
            "questions_original.document_id",
            "documents.partner_name",
            "documents.document_type",
            "questions_original.question_code",
            "questions_original.answer",
            "questions_original.expected",
            "questions_original.status",
        ],
        filters=filters,
        sorts=["questions_original.document_id"],
        limit=str(limit),
    )
    return json.loads(sdk.run_inline_query("json", body))


def get_doc_meta(doc_id: str) -> dict[str, Any]:
    init_aws()
    resp = _ddb.get_item(
        TableName="techno-core-prod-document-orchestrator",
        Key={"PK": {"S": doc_id}, "SK": {"S": "document"}},
        ProjectionExpression="partner_id, application_id, file_ids, parent_partition_params, document_type",
    )
    item = resp.get("Item")
    if not item:
        raise RuntimeError(f"document {doc_id} not found in DynamoDB")
    pages = item["parent_partition_params"]["M"]["pages"]["L"][0]["M"]
    file_ids = item.get("file_ids", {}).get("L", [])
    if not file_ids:
        raise RuntimeError(f"document {doc_id} has no file_ids in DynamoDB")
    return {
        "partner_id": item["partner_id"]["S"],
        "app_id": item["application_id"]["S"],
        "file_id": file_ids[0]["S"],
        "document_type": item.get("document_type", {}).get("S"),
        "start": int(pages["start_page"]["N"]),
        "end": int(pages["end_page"]["N"]),
    }


def ground_truth_for_doc(doc_id: str, question_code: str, hint: str | None = None) -> str | None:
    if hint and len(norm_vin(hint)) >= 11:
        return norm_vin(hint)
    rows = query_looker_vin_rows(since_days=90, document_id=doc_id, question_code=question_code, limit=3)
    for row in rows:
        expected = row.get("questions_original.expected")
        if expected and len(norm_vin(expected)) >= 11:
            return norm_vin(expected)
    return None


def resolve_ticket_docs(
    issue: dict[str, Any],
    *,
    since_days: int,
    max_docs_per_ticket: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (runnable_docs, skipped_entries)."""
    ticket = issue.get("id") or issue.get("identifier")
    title = issue.get("title") or ""
    desc = issue.get("description") or ""
    url = issue.get("url") or f"https://linear.app/informediq/issue/{ticket}"
    verification_doc_type = parse_verification_doc_type(title)
    question_code = parse_question_code(title, desc)
    partner = parse_partner(issue)

    if is_staging_only(desc):
        return [], [{"ticket": ticket, "reason": "staging-only links", "url": url}]

    runnable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_docs: set[str] = set()

    candidates: list[tuple[str, str | None]] = extract_doc_ground_truth_pairs(desc)
    if not candidates:
        for doc_id in extract_doc_ids(desc)[:max_docs_per_ticket]:
            candidates.append((doc_id, None))

    if not candidates and partner and verification_doc_type:
        try:
            rows = query_looker_vin_rows(
                since_days=since_days,
                partner=partner,
                verification_doc_type=verification_doc_type,
                question_code=question_code,
                limit=max_docs_per_ticket,
            )
            for row in rows:
                doc_id = row.get("questions_original.document_id")
                expected = row.get("questions_original.expected")
                if doc_id:
                    candidates.append((doc_id, expected))
        except Exception as exc:
            skipped.append({"ticket": ticket, "reason": f"looker fallback failed: {exc}", "url": url})

    for doc_id, hint_truth in candidates[:max_docs_per_ticket]:
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        short = doc_id.split("-")[0]
        try:
            meta = get_doc_meta(doc_id)
            doc_type = meta.get("document_type")
            if not doc_type:
                skipped.append({"ticket": ticket, "doc_id": doc_id, "reason": "no document_type in DynamoDB"})
                continue
            if not has_qa_yaml(doc_type):
                skipped.append(
                    {
                        "ticket": ticket,
                        "doc_id": doc_id,
                        "document_type": doc_type,
                        "reason": f"no QA llm_config for {doc_type}",
                    }
                )
                continue
            truth = ground_truth_for_doc(doc_id, question_code, hint_truth)
            if not truth:
                skipped.append({"ticket": ticket, "doc_id": doc_id, "reason": "no ground_truth from ticket/Looker"})
                continue
            runnable.append(
                {
                    "ticket": ticket,
                    "source": "linear",
                    "url": url,
                    "partner": meta["partner_id"],
                    "short": short,
                    "doc_id": doc_id,
                    "document_type": doc_type,
                    "verification_doc_type": verification_doc_type,
                    "question_code": question_code,
                    "ground_truth": truth,
                    "issue": title[:120],
                }
            )
        except Exception as exc:
            skipped.append({"ticket": ticket, "doc_id": doc_id, "reason": str(exc)})

    if not runnable and not skipped:
        skipped.append({"ticket": ticket, "reason": "no resolvable doc_id", "url": url})
    return runnable, skipped


def discover_and_resolve(
    out_dir: Path,
    *,
    since_days: int = 30,
    cache_path: Path | None = None,
    max_docs_per_ticket: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_path or (out_dir / "linear_issues_cache.json")
    tickets_raw = discover_linear_tickets(cache, since_days)
    (out_dir / "linear_tickets_filtered.json").write_text(json.dumps(tickets_raw, indent=2))

    runnable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for issue in tickets_raw:
        docs, skips = resolve_ticket_docs(
            issue, since_days=since_days, max_docs_per_ticket=max_docs_per_ticket
        )
        runnable.extend(docs)
        skipped.extend(skips)

    # Dedupe by doc_id — keep first ticket association
    deduped: list[dict[str, Any]] = []
    seen_doc: set[str] = set()
    for doc in runnable:
        if doc["doc_id"] in seen_doc:
            continue
        seen_doc.add(doc["doc_id"])
        deduped.append(doc)

    (out_dir / "tickets.json").write_text(json.dumps({"tickets": deduped}, indent=2))
    (out_dir / "skipped.json").write_text(json.dumps({"skipped": skipped}, indent=2))
    (out_dir / "discovery_summary.json").write_text(
        json.dumps(
            {
                "n_linear_tickets": len(tickets_raw),
                "n_runnable_docs": len(deduped),
                "n_skipped": len(skipped),
                "since_days": since_days,
            },
            indent=2,
        )
    )
    return tickets_raw, deduped, skipped
