"""OpenAI-compatible chat API for self-hosted gate models (Ollama, vLLM, llama.cpp)."""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import requests
from PIL import Image

from app.extract.gemini import parse_json_response

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_GATE_MODEL = "qwen2.5:3b"


def local_gate_base_url() -> str:
    raw = (
        os.environ.get("DOC_EXTRACT_GATE_BASE_URL")
        or os.environ.get("OLLAMA_HOST")
        or DEFAULT_BASE_URL
    ).strip().rstrip("/")
    if not raw:
        return DEFAULT_BASE_URL
    if not raw.endswith("/v1"):
        # Ollama default host is http://host:11434 without /v1
        if raw.endswith(":11434") or raw.endswith(":11434/"):
            raw = f"{raw.rstrip('/')}/v1"
    return raw


def local_gate_model() -> str:
    return os.environ.get("DOC_EXTRACT_GATE_MODEL") or DEFAULT_LOCAL_GATE_MODEL


def local_gate_api_key() -> str:
    return os.environ.get("DOC_EXTRACT_GATE_API_KEY") or "ollama"


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def chat_completion_json(
    prompt: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    image: Image.Image | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Call /v1/chat/completions and parse the assistant message as JSON."""
    model = model or local_gate_model()
    base = (base_url or local_gate_base_url()).rstrip("/")
    url = f"{base}/chat/completions"
    timeout = timeout_s or float(os.environ.get("DOC_EXTRACT_GATE_TIMEOUT_S", "120"))

    if image is not None:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
        ]
    else:
        content = prompt

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "stream": False,
    }
    if os.environ.get("DOC_EXTRACT_GATE_JSON_MODE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {local_gate_api_key()}",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("local gate request failed: %s", exc)
        return {}

    try:
        data = resp.json()
    except ValueError:
        logger.warning("local gate returned non-JSON HTTP body")
        return {}

    choices = data.get("choices") or []
    if not choices:
        return {}
    message = choices[0].get("message") or {}
    text = message.get("content") or ""
    if not text and isinstance(message.get("content"), list):
        text = "".join(
            part.get("text", "")
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return parse_json_response(text)
