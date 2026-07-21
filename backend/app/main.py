from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.jobs import create_job, get_job, job_file_path
from app.paths import JOB_ROOT, TOOL_ROOT, configure_web_env, is_cloud_deploy
from app.auth import (
    auth_required,
    current_user,
    google_client_id,
    login_with_email,
    login_with_google,
    me_from_token_payload,
    register_with_email,
    verify_session_token,
)
from app.users_db import (
    CREDITS_PER_DOCUMENT,
    InsufficientCreditsError,
    get_user_by_id,
    spend_credits_for_job,
)
from app.users_db import users_db_status

_CRED_SOURCE = configure_web_env()


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(title="Document Extract Web", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

SAMPLE_DOCS: dict[str, str] = {
    "invoice": "hk_group_invoice",
    "receipt": "contoso_invoice",
    "passport": "farview_invoice",
    "contract": "commercial_invoice_sectioning",
    "bank": "farview_invoice",
}


@app.get("/api/health")
def health() -> dict:
    import os

    ocr = "docai" if (
        (os.environ.get("GCP_PROJECT_NUMBER") or os.environ.get("GCP_PROJECT_ID"))
        and (os.environ.get("DOCUMENT_AI_PROCESSOR_ID") or os.environ.get("PROCESSOR_ID"))
    ) else "vision"
    has_gcp = bool(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.environ.get("GOOGLE_CLOUD_KEYFILE_JSON")
        or os.environ.get("GOOGLE_CLOUD_API_KEY")
    )
    has_gemini = bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or (
            not is_cloud_deploy()
            and (
                os.environ.get("DOC_EXTRACT_GEMINI_PROJECT_ID")
                or os.environ.get("GCP_PROJECT_ID")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
            )
            and (
                os.environ.get("DOC_EXTRACT_GEMINI_KEYFILE_JSON")
                or os.environ.get("GOOGLE_CLOUD_KEYFILE_JSON")
                or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            )
        )
    )
    gemini_auth = "missing"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        gemini_auth = "api_key"
    elif not is_cloud_deploy() and (
        (
            os.environ.get("DOC_EXTRACT_GEMINI_PROJECT_ID")
            or os.environ.get("GCP_PROJECT_ID")
        )
        and (
            os.environ.get("DOC_EXTRACT_GEMINI_KEYFILE_JSON")
            or os.environ.get("GOOGLE_CLOUD_KEYFILE_JSON")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
    ):
        gemini_auth = "vertex"
    gemini_project = None
    if gemini_auth == "vertex":
        gemini_project = (
            os.environ.get("DOC_EXTRACT_GEMINI_PROJECT_ID")
            or os.environ.get("GCP_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
    return {
        "status": "ok",
        "credentials": _CRED_SOURCE,
        "cred_mode": "personal_deploy" if is_cloud_deploy() else _CRED_SOURCE,
        "ocr_backend": ocr if has_gcp else "missing_credentials",
        "extraction": "gemini" if has_gemini else "missing_credentials",
        "gemini_auth": gemini_auth,
        "gemini_project": gemini_project,
        "auth_required": auth_required(),
        "google_oauth": bool(google_client_id()),
        "users_db": users_db_status(),
    }


@app.get("/api/auth/config")
def auth_config() -> dict:
    from app.users_db import INITIAL_CREDITS

    return {
        "auth_required": auth_required(),
        "google_client_id": google_client_id(),
        "initial_credits": INITIAL_CREDITS,
        "credits_per_document": CREDITS_PER_DOCUMENT,
    }


@app.post("/api/auth/register")
def auth_register(body: dict) -> dict:
    email = body.get("email")
    password = body.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    return register_with_email(str(email), str(password))


@app.post("/api/auth/login")
def auth_login(body: dict) -> dict:
    email = body.get("email")
    password = body.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")
    return login_with_email(str(email), str(password))


@app.get("/api/auth/me")
def auth_me(authorization: str | None = Header(default=None)) -> dict:
    if not auth_required():
        return {"auth_required": False}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in required")
    payload = verify_session_token(authorization.removeprefix("Bearer ").strip())
    return me_from_token_payload(payload)


@app.post("/api/auth/google")
def auth_google(body: dict) -> dict:
    token = body.get("id_token") or body.get("credential")
    if not token:
        raise HTTPException(status_code=400, detail="Missing Google id_token")
    return login_with_google(str(token))


@app.get("/api/samples/{doc_type}.png")
def sample_image(doc_type: str) -> FileResponse:
    folder = SAMPLE_DOCS.get(doc_type, "hk_group_invoice")
    path = TOOL_ROOT / "demo_output" / folder / "source.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/samples/{doc_type}/overlay.png")
def sample_overlay(doc_type: str) -> FileResponse:
    folder = SAMPLE_DOCS.get(doc_type, "hk_group_invoice")
    for name in ("overlay_sections.png", "overlay.png", "source.png"):
        path = TOOL_ROOT / "demo_output" / folder / name
        if path.exists():
            return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Sample not found")


@app.post("/api/jobs")
async def upload_job(
    file: UploadFile = File(...),
    skip_llm: bool = False,
    user: dict | None = Depends(current_user),
) -> dict[str, str | int]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    credits_remaining: int | None = None
    if auth_required() and user:
        uid = str(user["sub"])
        account = get_user_by_id(uid)
        balance = account["credits"] if account else 0
        if balance < CREDITS_PER_DOCUMENT:
            raise HTTPException(
                status_code=402,
                detail=(
                    "You ran out of credits. Each document costs 2 credits; "
                    "new accounts start with 2 credits (one free document)."
                ),
            )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 20 MB limit")

    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = JOB_ROOT / f"_upload_{file.filename}"
    tmp.write_bytes(data)
    try:
        job = create_job(tmp, file.filename, skip_llm=skip_llm)
    finally:
        if tmp.exists():
            tmp.unlink()

    if auth_required() and user:
        try:
            credits_remaining = spend_credits_for_job(str(user["sub"]), job.job_id)
        except InsufficientCreditsError:
            raise HTTPException(
                status_code=402,
                detail=(
                    "You ran out of credits. Each document costs 2 credits; "
                    "new accounts start with 2 credits (one free document)."
                ),
            )

    payload: dict[str, str | int] = {"job_id": job.job_id}
    if credits_remaining is not None:
        payload["credits_remaining"] = credits_remaining
    return payload


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        result_path = JOB_ROOT / job_id / "result.json"
        if result_path.exists():
            import json

            payload = json.loads(result_path.read_text())
            payload["status"] = "completed"
            return payload
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/pages/{page_index}/overlay.png")
def page_overlay(job_id: str, page_index: int) -> FileResponse:
    rel = f"page_{page_index:03d}/overlay.png"
    path = job_file_path(job_id, rel)
    if path is None:
        path = JOB_ROOT / job_id / rel
    if not path.exists():
        raise HTTPException(status_code=404, detail="Overlay not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/pages/{page_index}/sections/{section_index}/crop.png")
def section_crop(job_id: str, page_index: int, section_index: int) -> FileResponse:
    rel = f"page_{page_index:03d}/crops/s{section_index}.png"
    path = job_file_path(job_id, rel)
    if path is None:
        path = JOB_ROOT / job_id / rel
    if not path.exists():
        raise HTTPException(status_code=404, detail="Crop not found")
    return FileResponse(path, media_type="image/png")
