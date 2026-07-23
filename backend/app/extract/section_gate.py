"""Per-section gate before full extraction — self-hosted small LLM or Gemini."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image

from app.extract.gemini import call_gemini, parse_json_response, setup_gemini_client
from app.extract.local_llm import chat_completion_json, local_gate_model

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_GATE_MODEL = os.environ.get("DOC_EXTRACT_GEMINI_GATE_MODEL", "gemini-2.0-flash-lite")

SECTION_GATE_PROMPT = """Should this document section be sent to field extraction?
Return JSON only: {{"extractable": true|false, "confidence": <number 0.0-1.0>}}

extractable=true for PII, addresses, phones, filled form values, checkboxes, table rows, payment lines, signatures.
extractable=false for headers, titles-only, disclaimers, legal notices, terms, footers, or empty label blocks.
confidence is how sure you are (1.0 = certain).

{layout_line}OCR text:
{ocr_text}
"""


@dataclass(frozen=True)
class SectionGateResult:
    extractable: bool
    confidence: float
    provider: str = "local"
    model: str = ""

    def passes_threshold(self, min_confidence: float) -> bool:
        return self.extractable and self.confidence >= min_confidence

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "extractable": self.extractable,
            "confidence": round(self.confidence, 4),
            "provider": self.provider,
        }
        if self.model:
            out["model"] = self.model
        return out


def gate_min_confidence() -> float:
    raw = os.environ.get("DOC_EXTRACT_GATE_MIN_CONFIDENCE", "0.5").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.5


def section_gate_enabled() -> bool:
    raw = (os.environ.get("DOC_EXTRACT_SECTION_GATE") or "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def gate_provider() -> str:
    """local = Ollama/vLLM (default); gemini = cloud API."""
    return (os.environ.get("DOC_EXTRACT_GATE_PROVIDER") or "local").strip().lower()


def section_gate_use_image() -> bool:
    raw = os.environ.get("DOC_EXTRACT_GATE_USE_IMAGE", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    # Default: send section crop when the gate model is vision-capable (e.g. Qwen2.5-VL).
    model = local_gate_model().lower()
    return any(tag in model for tag in ("vl", "vision", "llava", "moondream"))


def _parse_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(1.0, float(value.strip())))
        except ValueError:
            pass
    return 0.5


def parse_gate_response(
    parsed: dict[str, Any],
    *,
    provider: str,
    model: str = "",
) -> SectionGateResult:
    extractable = parsed.get("extractable")
    if not isinstance(extractable, bool):
        extractable = True
    return SectionGateResult(
        extractable=extractable,
        confidence=_parse_confidence(parsed.get("confidence")),
        provider=provider,
        model=model,
    )


def _gate_prompt(ocr_text: str, layout_kind: str | None) -> str:
    layout_line = f"Layout hint: {layout_kind}\n" if layout_kind else ""
    return SECTION_GATE_PROMPT.format(layout_line=layout_line, ocr_text=ocr_text or "(empty)")


def _downscale_for_gate(crop_img: Image.Image, *, max_side: int = 512) -> Image.Image:
    w, h = crop_img.size
    if max(w, h) <= max_side:
        return crop_img
    scale = max_side / float(max(w, h))
    return crop_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)


def _fallback_result(*, provider: str, model: str) -> SectionGateResult:
    return SectionGateResult(
        True,
        0.0,
        provider=provider,
        model=model,
    )


class LocalSectionContentGate:
    """Self-hosted small LLM via OpenAI-compatible API (Ollama, vLLM, Mistral, Qwen, etc.)."""

    def __init__(self, *, model: str | None = None) -> None:
        self.model = model or local_gate_model()

    def classify(
        self,
        ocr_text: str,
        *,
        crop_img: Image.Image | None = None,
        layout_kind: str | None = None,
    ) -> SectionGateResult:
        prompt = _gate_prompt(ocr_text, layout_kind)
        image = _downscale_for_gate(crop_img) if section_gate_use_image() and crop_img else None
        if section_gate_use_image() and crop_img is None:
            logger.debug("gate USE_IMAGE set but no crop; OCR text only")
        parsed = chat_completion_json(prompt, model=self.model, image=image)
        if not parsed:
            return _fallback_result(provider="local", model=self.model)
        return parse_gate_response(parsed, provider="local", model=self.model)


class GeminiSectionContentGate:
    """Optional cloud gate (DOC_EXTRACT_GATE_PROVIDER=gemini)."""

    def __init__(
        self,
        *,
        model: str | None = None,
        client: Any | None = None,
        use_image: bool | None = None,
    ) -> None:
        self.model = model or DEFAULT_GEMINI_GATE_MODEL
        self._client = client
        self.use_image = section_gate_use_image() if use_image is None else use_image

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = setup_gemini_client(model=self.model)
        return self._client

    def classify(
        self,
        ocr_text: str,
        *,
        crop_img: Image.Image | None = None,
        layout_kind: str | None = None,
    ) -> SectionGateResult:
        prompt = _gate_prompt(ocr_text, layout_kind)
        if self.use_image and crop_img is not None:
            parsed = call_gemini(self.client, self.model, _downscale_for_gate(crop_img), prompt)
        else:
            parsed = self._call_gemini_text(prompt)
        if not parsed:
            return _fallback_result(provider="gemini", model=self.model)
        return parse_gate_response(parsed, provider="gemini", model=self.model)

    def _call_gemini_text(self, prompt: str) -> dict[str, Any]:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        text = getattr(response, "text", None) or ""
        if not text and getattr(response, "candidates", None):
            parts = response.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") or "" for p in parts)
        return parse_json_response(text)


SectionContentGate = LocalSectionContentGate | GeminiSectionContentGate

_gate: SectionContentGate | None = None


def get_section_gate() -> SectionContentGate:
    global _gate
    if _gate is None:
        provider = gate_provider()
        if provider in {"gemini", "google", "vertex"}:
            _gate = GeminiSectionContentGate()
        else:
            _gate = LocalSectionContentGate()
    return _gate
