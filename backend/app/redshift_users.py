from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _redshift_configured() -> bool:
    return bool(
        os.environ.get("REDSHIFT_HOST")
        and os.environ.get("REDSHIFT_DATABASE")
        and os.environ.get("REDSHIFT_USER")
        and os.environ.get("REDSHIFT_PASSWORD")
    )


def _connect():
    import redshift_connector

    return redshift_connector.connect(
        host=os.environ["REDSHIFT_HOST"],
        port=int(os.environ.get("REDSHIFT_PORT", "5439")),
        database=os.environ["REDSHIFT_DATABASE"],
        user=os.environ["REDSHIFT_USER"],
        password=os.environ["REDSHIFT_PASSWORD"],
        ssl=os.environ.get("REDSHIFT_SSL", "true").lower() in {"1", "true", "yes"},
    )


def ensure_users_table() -> None:
    if not _redshift_configured():
        return
    schema = os.environ.get("REDSHIFT_SCHEMA", "public")
    table = os.environ.get("REDSHIFT_USERS_TABLE", "docparse_users")
    qualified = f"{schema}.{table}" if "." not in table else table
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {qualified} (
                    google_sub VARCHAR(255) NOT NULL,
                    email VARCHAR(320) NOT NULL,
                    name VARCHAR(512),
                    picture_url VARCHAR(2048),
                    created_at TIMESTAMP DEFAULT GETDATE(),
                    last_login_at TIMESTAMP DEFAULT GETDATE()
                )
                DISTSTYLE AUTO
                """
            )
        conn.commit()


def upsert_user(*, google_sub: str, email: str, name: str | None, picture_url: str | None) -> None:
    if not _redshift_configured():
        logger.warning("Redshift not configured — user %s not persisted", email)
        return

    schema = os.environ.get("REDSHIFT_SCHEMA", "public")
    table = os.environ.get("REDSHIFT_USERS_TABLE", "docparse_users")
    qualified = f"{schema}.{table}" if "." not in table else table
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {qualified} WHERE google_sub = %s LIMIT 1", (google_sub,))
            exists = cur.fetchone() is not None
            if exists:
                cur.execute(
                    f"""
                    UPDATE {qualified}
                    SET email = %s, name = %s, picture_url = %s, last_login_at = %s
                    WHERE google_sub = %s
                    """,
                    (email, name, picture_url, now, google_sub),
                )
            else:
                cur.execute(
                    f"""
                    INSERT INTO {qualified} (google_sub, email, name, picture_url, created_at, last_login_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (google_sub, email, name, picture_url, now, now),
                )
        conn.commit()


def redshift_status() -> dict[str, Any]:
    return {
        "configured": _redshift_configured(),
        "schema": os.environ.get("REDSHIFT_SCHEMA", "public"),
        "table": os.environ.get("REDSHIFT_USERS_TABLE", "docparse_users"),
    }
