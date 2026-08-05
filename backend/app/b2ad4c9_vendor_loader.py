"""
Load frozen ``vendor/mllm-scripts-b2ad4c9`` (@ commit b2ad4c9) without replacing
live ``vendor/mllm-scripts`` imports used by later pipeline steps.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from app.paths import ensure_script_path, TOOL_ROOT

ensure_script_path()

B2AD4C9_VENDOR_DIR = TOOL_ROOT / "vendor" / "mllm-scripts-b2ad4c9"
_WORD_PATH = B2AD4C9_VENDOR_DIR / "ocr_word_to_line_boxes.py"
_LINE_PATH = B2AD4C9_VENDOR_DIR / "ocr_line_to_sections.py"

_MODULE_PREFIX = "b2ad4c9_frozen"
_word_module: ModuleType | None = None
_line_module: ModuleType | None = None


def _exec_vendor_module(module_name: str, path: Path) -> ModuleType:
    full_name = f"{_MODULE_PREFIX}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load b2ad4c9 vendor module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _load_word_module() -> ModuleType:
    global _word_module
    if _word_module is None:
        if not _WORD_PATH.is_file():
            raise FileNotFoundError(
                f"missing frozen b2ad4c9 vendor file: {_WORD_PATH}"
            )
        _word_module = _exec_vendor_module("ocr_word_to_line_boxes", _WORD_PATH)
    return _word_module


def _load_line_module() -> ModuleType:
    global _line_module
    if _line_module is None:
        if not _LINE_PATH.is_file():
            raise FileNotFoundError(
                f"missing frozen b2ad4c9 vendor file: {_LINE_PATH}"
            )
        word_mod = _load_word_module()
        saved_word = sys.modules.get("ocr_word_to_line_boxes")
        sys.modules["ocr_word_to_line_boxes"] = word_mod
        try:
            _line_module = _exec_vendor_module("ocr_line_to_sections", _LINE_PATH)
        finally:
            if saved_word is not None:
                sys.modules["ocr_word_to_line_boxes"] = saved_word
            else:
                sys.modules.pop("ocr_word_to_line_boxes", None)
    return _line_module


def get_b2ad4c9_vendor_modules() -> tuple[Any, Any]:
    """Return ``(ocr_line_to_sections, ocr_word_to_line_boxes)`` frozen at b2ad4c9."""
    return _load_line_module(), _load_word_module()
