from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Header, HTTPException
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from app.redshift_users import ensure_users_table, upsert_user


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


def verify_google_id_token(token: str) -> dict[str, Any]:
    client_id = google_client_id()
    if not client_id:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured")
    try:
        payload = id_token.verify_oauth2_token(token, google_requests.Request(), client_id)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid Google sign-in") from exc
    if not payload.get("email"):
        raise HTTPException(status_code=401, detail="Google account has no email")
    return payload


def issue_session_token(user: dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user["sub"],
        "email": user["email"],
        "name": user.get("name"),
        "picture": user.get("picture"),
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


def login_with_google(id_token_str: str) -> dict[str, Any]:
    google_user = verify_google_id_token(id_token_str)
    ensure_users_table()
    upsert_user(
        google_sub=str(google_user["sub"]),
        email=str(google_user["email"]),
        name=google_user.get("name"),
        picture_url=google_user.get("picture"),
    )
    session = issue_session_token(google_user)
    return {
        "token": session,
        "user": {
            "email": google_user["email"],
            "name": google_user.get("name"),
            "picture": google_user.get("picture"),
        },
    }


def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any] | None:
    if not auth_required():
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in required to extract documents")
    return verify_session_token(authorization.removeprefix("Bearer ").strip())
