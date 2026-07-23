from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from fastapi import Header, HTTPException
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from app.users_db import (
    CREDITS_PER_DOCUMENT,
    INITIAL_CREDITS,
    register_email_user,
    upsert_google_user,
    verify_email_login,
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def auth_required() -> bool:
    return os.environ.get("DOC_EXTRACT_AUTH_REQUIRED", "true").lower() not in {
        "0",
        "false",
        "no",
    }


def google_client_id() -> str | None:
    return os.environ.get("GOOGLE_OAUTH_CLIENT_ID") or os.environ.get("VITE_GOOGLE_CLIENT_ID")


def jwt_secret() -> str:
    secret = os.environ.get("DOC_EXTRACT_JWT_SECRET")
    if secret:
        return secret
    if not auth_required():
        return "docparse-dev-insecure-secret"
    raise HTTPException(status_code=503, detail="DOC_EXTRACT_JWT_SECRET is not configured")


def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _check_password(stored_hash: str, plain: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), stored_hash.encode("utf-8"))
    except ValueError:
        return False


def _session_user_payload(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "sub": user["user_id"],
        "email": user["email"],
        "name": user.get("name"),
        "picture": user.get("picture"),
        "credits": user.get("credits", 0),
    }


def issue_session_token(user: dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user["sub"],
        "email": user["email"],
        "name": user.get("name"),
        "picture": user.get("picture"),
        "credits": user.get("credits", 0),
        "iat": now,
        "exp": now + timedelta(days=7),
    }
    return jwt.encode(payload, jwt_secret(), algorithm="HS256")


def verify_session_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, jwt_secret(), algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Session expired — please sign in again") from exc
    return payload


def auth_response_from_user(user: dict[str, Any]) -> dict[str, Any]:
    session_user = _session_user_payload(user)
    return {
        "token": issue_session_token(session_user),
        "user": {
            "email": user["email"],
            "name": user.get("name"),
            "picture": user.get("picture"),
            "credits": user.get("credits", 0),
        },
        "credits_per_document": CREDITS_PER_DOCUMENT,
        "initial_credits": INITIAL_CREDITS,
    }


def register_with_email(email: str, password: str) -> dict[str, Any]:
    if not _EMAIL_RE.match(email.strip()):
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    try:
        user = register_email_user(email=email, password_hash=_hash_password(password))
    except ValueError as exc:
        if str(exc) == "email_already_registered":
            raise HTTPException(status_code=409, detail="An account with this email already exists") from exc
        raise
    return auth_response_from_user(user)


def login_with_email(email: str, password: str) -> dict[str, Any]:
    user = verify_email_login(email, lambda stored: _check_password(stored, password))
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return auth_response_from_user(user)


def login_with_google(id_token_str: str) -> dict[str, Any]:
    client_id = google_client_id()
    if not client_id:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured")
    try:
        google_user = id_token.verify_oauth2_token(
            id_token_str, google_requests.Request(), client_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid Google sign-in") from exc
    if not google_user.get("email"):
        raise HTTPException(status_code=401, detail="Google account has no email")
    user = upsert_google_user(
        google_sub=str(google_user["sub"]),
        email=str(google_user["email"]),
        name=google_user.get("name"),
        picture_url=google_user.get("picture"),
    )
    return auth_response_from_user(user)


def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any] | None:
    if not auth_required():
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in required to extract documents")
    payload = verify_session_token(authorization.removeprefix("Bearer ").strip())
    from app.users_db import ensure_account_from_claims

    fresh = ensure_account_from_claims(
        user_id=str(payload["sub"]),
        email=payload.get("email"),
        name=payload.get("name"),
        picture=payload.get("picture"),
    )
    if fresh:
        payload["credits"] = fresh["credits"]
    return payload


def me_from_token_payload(payload: dict[str, Any]) -> dict[str, Any]:
    from app.users_db import ensure_account_from_claims

    user_id = str(payload["sub"])
    fresh = ensure_account_from_claims(
        user_id=user_id,
        email=payload.get("email"),
        name=payload.get("name"),
        picture=payload.get("picture"),
    )
    credits = fresh["credits"] if fresh else int(payload.get("credits") or 0)
    return {
        "email": payload.get("email"),
        "name": payload.get("name"),
        "picture": payload.get("picture"),
        "credits": credits,
        "credits_per_document": CREDITS_PER_DOCUMENT,
        "initial_credits": INITIAL_CREDITS,
    }
