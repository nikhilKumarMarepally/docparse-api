"""SQLite users, credits, and extraction ledger — no AWS required."""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.paths import JOB_ROOT

logger = logging.getLogger(__name__)

_TABLE = os.environ.get("DOC_EXTRACT_USERS_TABLE", "docparse_users")
_LEDGER = "credit_ledger"
INITIAL_CREDITS = int(os.environ.get("DOC_EXTRACT_INITIAL_CREDITS", "2"))
CREDITS_PER_DOCUMENT = int(os.environ.get("DOC_EXTRACT_CREDITS_PER_DOC", "2"))


class InsufficientCreditsError(Exception):
    def __init__(self, balance: int) -> None:
        self.balance = balance
        super().__init__(f"insufficient credits (balance={balance})")


def _db_path() -> Path:
    override = os.environ.get("DOC_EXTRACT_USERS_DB", "").strip()
    if override:
        return Path(override).expanduser()
    data_dir = Path(os.environ.get("DOC_EXTRACT_DATA_DIR", str(JOB_ROOT.parent / "data")))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "users.sqlite"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def ensure_users_table() -> None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (_TABLE,),
        ).fetchone()
        if row:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({_TABLE})").fetchall()}
            if "user_id" not in cols:
                legacy = f"{_TABLE}_legacy"
                conn.execute(f"ALTER TABLE {_TABLE} RENAME TO {legacy}")
                conn.commit()

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                user_id TEXT PRIMARY KEY NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT,
                google_sub TEXT UNIQUE,
                name TEXT,
                picture_url TEXT,
                credits INTEGER NOT NULL DEFAULT {INITIAL_CREDITS},
                created_at TEXT NOT NULL,
                last_login_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_LEDGER} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                job_id TEXT,
                delta INTEGER NOT NULL,
                balance_after INTEGER NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _row_to_public(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "user_id": row["user_id"],
        "email": row["email"],
        "name": row["name"],
        "picture": row["picture_url"],
        "credits": int(row["credits"]),
    }


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    ensure_users_table()
    with _connect() as conn:
        row = conn.execute(
            f"SELECT * FROM {_TABLE} WHERE user_id = ? LIMIT 1",
            (user_id,),
        ).fetchone()
    return _row_to_public(row) if row else None


def get_user_by_email(email: str) -> sqlite3.Row | None:
    ensure_users_table()
    with _connect() as conn:
        return conn.execute(
            f"SELECT * FROM {_TABLE} WHERE email = ? COLLATE NOCASE LIMIT 1",
            (_normalize_email(email),),
        ).fetchone()


def register_email_user(*, email: str, password_hash: str) -> dict[str, Any]:
    ensure_users_table()
    norm = _normalize_email(email)
    if get_user_by_email(norm):
        raise ValueError("email_already_registered")
    user_id = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            f"""
            INSERT INTO {_TABLE}
            (user_id, email, password_hash, google_sub, name, picture_url, credits, created_at, last_login_at)
            VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?, ?)
            """,
            (user_id, norm, password_hash, INITIAL_CREDITS, now, now),
        )
        conn.execute(
            f"""
            INSERT INTO {_LEDGER} (user_id, job_id, delta, balance_after, reason, created_at)
            VALUES (?, NULL, ?, ?, ?, ?)
            """,
            (user_id, INITIAL_CREDITS, INITIAL_CREDITS, "signup_bonus", now),
        )
        conn.commit()
    user = get_user_by_id(user_id)
    assert user is not None
    return user


def upsert_google_user(
    *,
    google_sub: str,
    email: str,
    name: str | None,
    picture_url: str | None,
) -> dict[str, Any]:
    ensure_users_table()
    norm = _normalize_email(email)
    now = _now()
    with _connect() as conn:
        existing = conn.execute(
            f"SELECT * FROM {_TABLE} WHERE email = ? COLLATE NOCASE OR google_sub = ? LIMIT 1",
            (norm, google_sub),
        ).fetchone()
        if existing:
            user_id = existing["user_id"]
            conn.execute(
                f"""
                UPDATE {_TABLE}
                SET email = ?, google_sub = ?, name = ?, picture_url = ?, last_login_at = ?
                WHERE user_id = ?
                """,
                (norm, google_sub, name, picture_url, now, user_id),
            )
        else:
            user_id = f"google:{google_sub}"
            conn.execute(
                f"""
                INSERT INTO {_TABLE}
                (user_id, email, password_hash, google_sub, name, picture_url, credits, created_at, last_login_at)
                VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, norm, google_sub, name, picture_url, INITIAL_CREDITS, now, now),
            )
            conn.execute(
                f"""
                INSERT INTO {_LEDGER} (user_id, job_id, delta, balance_after, reason, created_at)
                VALUES (?, NULL, ?, ?, ?, ?)
                """,
                (user_id, INITIAL_CREDITS, INITIAL_CREDITS, "signup_bonus", now),
            )
        conn.commit()
    user = get_user_by_id(user_id)
    assert user is not None
    return user


def verify_email_login(email: str, password_hash_check) -> dict[str, Any] | None:
    """password_hash_check(plain, stored_hash) -> bool"""
    row = get_user_by_email(email)
    if not row or not row["password_hash"]:
        return None
    if not password_hash_check(row["password_hash"]):
        return None
    now = _now()
    with _connect() as conn:
        conn.execute(
            f"UPDATE {_TABLE} SET last_login_at = ? WHERE user_id = ?",
            (now, row["user_id"]),
        )
        conn.commit()
    return get_user_by_id(row["user_id"])


def spend_credits_for_job(user_id: str, job_id: str, *, amount: int | None = None) -> int:
    cost = amount if amount is not None else CREDITS_PER_DOCUMENT
    ensure_users_table()
    now = _now()
    with _connect() as conn:
        row = conn.execute(
            f"SELECT credits FROM {_TABLE} WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            raise InsufficientCreditsError(0)
        balance = int(row["credits"])
        if balance < cost:
            raise InsufficientCreditsError(balance)
        new_balance = balance - cost
        conn.execute(
            f"UPDATE {_TABLE} SET credits = ? WHERE user_id = ?",
            (new_balance, user_id),
        )
        conn.execute(
            f"""
            INSERT INTO {_LEDGER} (user_id, job_id, delta, balance_after, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, job_id, -cost, new_balance, "document_extraction", now),
        )
        conn.commit()
    return new_balance


def ensure_account_from_claims(
    *,
    user_id: str,
    email: str | None,
    name: str | None = None,
    picture: str | None = None,
) -> dict[str, Any] | None:
    """Create a missing account for a valid JWT (e.g. after disk wipe)."""
    ensure_users_table()
    existing = get_user_by_id(user_id)
    if existing:
        return existing
    if not email:
        return None
    norm = _normalize_email(email)
    now = _now()
    google_sub = user_id.removeprefix("google:") if user_id.startswith("google:") else None
    with _connect() as conn:
        by_email = conn.execute(
            f"SELECT * FROM {_TABLE} WHERE email = ? COLLATE NOCASE LIMIT 1",
            (norm,),
        ).fetchone()
        if by_email:
            return get_user_by_id(by_email["user_id"])
        conn.execute(
            f"""
            INSERT INTO {_TABLE}
            (user_id, email, password_hash, google_sub, name, picture_url, credits, created_at, last_login_at)
            VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, norm, google_sub, name, picture, INITIAL_CREDITS, now, now),
        )
        conn.execute(
            f"""
            INSERT INTO {_LEDGER} (user_id, job_id, delta, balance_after, reason, created_at)
            VALUES (?, NULL, ?, ?, ?, ?)
            """,
            (user_id, INITIAL_CREDITS, INITIAL_CREDITS, "signup_bonus", now),
        )
        conn.commit()
    return get_user_by_id(user_id)


def users_db_status() -> dict[str, Any]:
    path = _db_path()
    table_ok = False
    try:
        ensure_users_table()
        with _connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (_TABLE,),
            ).fetchone()
            table_ok = bool(row)
    except Exception as exc:  # noqa: BLE001 — status endpoint must stay up
        logger.warning("users_db_status failed: %s", exc)
    return {
        "backend": "sqlite",
        "configured": True,
        "path": str(path),
        "table": _TABLE,
        "exists": path.is_file() and table_ok,
        "initial_credits": INITIAL_CREDITS,
        "credits_per_document": CREDITS_PER_DOCUMENT,
    }
