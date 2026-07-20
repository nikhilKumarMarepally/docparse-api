from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = BACKEND_ROOT.parent
MLLM_SCRIPTS = TOOL_ROOT / "vendor" / "mllm-scripts"
# DocExtract deploy uses Render dashboard env vars only — no company cred files.
ENV_LOCAL = Path("/nonexistent/company/mllm-invoker/.env.local")
PERSONAL_ENV = TOOL_ROOT / ".env.local"
JOB_ROOT = Path(os.environ.get("DOC_EXTRACT_JOB_ROOT", "/tmp/doc-extract-web"))

_COMPANY_CRED_KEYS = (
    "GOOGLE_CLOUD_KEYFILE_JSON",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "DOCUMENT_AI_PROCESSOR_ID",
    "PROCESSOR_ID",
    "GCP_PROJECT_NUMBER",
)
_PERSONAL_CRED_KEYS = (
    "GOOGLE_CLOUD_API_KEY",
    "GOOGLE_VISION_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)
_COMPANY_GEMINI_OVERLAY_KEYS = (
    "DOC_EXTRACT_GEMINI_PROJECT_ID",
    "DOC_EXTRACT_GEMINI_KEYFILE_JSON",
)
_COMPANY_VERTEX_KEYS = (
    "GOOGLE_CLOUD_KEYFILE_JSON",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_LOCATION",
)


def _strip_env_keys(keys: tuple[str, ...]) -> None:
    for key in keys:
        os.environ.pop(key, None)


def ensure_script_path() -> None:
    scripts = str(MLLM_SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_dotenv_override(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value.lower() in {"", "unset", "none"}:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def is_cloud_deploy() -> bool:
    mode = (os.environ.get("DOC_EXTRACT_CRED_MODE") or "").strip().lower()
    if mode in {"personal", "personal_only"}:
        return True
    if mode in {"company", "local"}:
        return False
    return bool(os.environ.get("RENDER"))


def _strip_company_vertex_overlay() -> None:
    for key in _COMPANY_GEMINI_OVERLAY_KEYS + _COMPANY_VERTEX_KEYS:
        os.environ.pop(key, None)


def configure_web_env() -> str:
    _strip_env_keys(_COMPANY_CRED_KEYS)
    if is_cloud_deploy():
        _strip_company_vertex_overlay()
        return "personal_deploy"
    if PERSONAL_ENV.exists():
        load_dotenv_override(PERSONAL_ENV)
        return "personal"
    return "env"


def configure_company_env() -> str:
    _strip_env_keys(_PERSONAL_CRED_KEYS)
    if PERSONAL_ENV.exists():
        load_dotenv_override(PERSONAL_ENV)
    return "personal"


def configure_env() -> str:
    return configure_web_env()
