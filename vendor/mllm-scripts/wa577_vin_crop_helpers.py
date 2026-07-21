"""Shared VIN section scoring and tight-crop bounds for WA-577 crop scripts."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

from ocr_word_to_line_boxes import Box, Line, load_vision, load_words, words_to_lines

MIN_VIN_CROP_WIDTH = 2000
MIN_VIN_CROP_HEIGHT = 200
VIN_CROP_PAD_TRIGGER_HEIGHT = 120

VIN_TOKEN = re.compile(r"[A-HJ-NPR-Z0-9]{11,17}", re.I)
VIN17 = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
# OCR-tolerant pattern for crop geometry only (O/I confusion).
VIN_TOKEN_OCR = re.compile(r"[A-HJ-NP-Z0-9IO]{11,20}", re.I)
VIN_LABEL = re.compile(
    r"vehicle identification|vehicle information|identification number|\bvin\b",
    re.I,
)
# Header/fee lines that contain "VIN" but are not the vehicle-ID field label.
_FALSE_VIN_LABEL = re.compile(
    r"\bONLY\s+VIN\b|\bVIN\s+FEE\b|\bMPG\s+VIN\b|VIN\s+INSPECTION",
    re.I,
)
TRADE_IN_RE = re.compile(r"TRADE[- ]?IN", re.I)
TRADE_IN_SECTION_RE = re.compile(
    r"VEHICLE\s+TRADED|TRADE[- ]?IN\s*\(IF\s+ANY\)|ADDITIONAL\s+TRADE[- ]?IN",
    re.I,
)
SIGNATURE_SECTION_RE = re.compile(
    r"SIGNATURE\s+OF\s+(APPLICANT|SELLER|PURCHASER|OWNER|CO-PURCHASER)|"
    r"SIGNATURE\s*\(S\)\s+OF|SUBSCRIBED\s+AND\s+SWO[RM]N",
    re.I,
)
LIENHOLDER_SECTION_RE = re.compile(
    r"LIENHOLDER\s+TO\s+BE\s+RECORDED|PRINTED\s+NAME\s+OF\s+LIENHOLDER|"
    r"FIRST\s+LIEN|SECURED\s+PARTY",
    re.I,
)
LESSEE_SECTION_RE = re.compile(
    r"\bLESSEE\b|LESSOR\s*/\s*LESSEE|LESSEE\s+INFORMATION",
    re.I,
)
PRIMARY_VIN_FIELD_RE = re.compile(
    r"(?:^|\b)1\.\s*VEHICLE\s+IDENTIFICATION\s+NUMBER|"
    r"A\.\s*MAKE\s+OF\s+VEH.*IDENTIFICATION\s+NUMBER|"
    r"YEAR\s+MAKE\s+VEHICLE\s+IDENTIFICATION\s+NUMBER",
    re.I | re.S,
)

MEGA_SECTION_PENALTIES = (
    "TRUTH-IN-LENDING",
    "TRUTH IN LENDING",
    "ODOMETER",
    "FEE COMPUTATION",
    "TITLE FEE",
    "LICENSE PLATE FEE",
    "TAX STATEMENT",
    "LIENHOLDER",
    "FIRST LIEN",
    "SECURED PARTY",
    "SOLD TO",
)


def norm_vin(v: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(v or "")).upper()


def enhance_vin_crop(img: Image.Image) -> Image.Image:
    """Upscale thin VIN strips and apply mild contrast/sharpness before MLLM."""
    out = img.convert("RGB")
    w, h = out.size

    if w < MIN_VIN_CROP_WIDTH:
        ratio = MIN_VIN_CROP_WIDTH / w
        out = out.resize(
            (MIN_VIN_CROP_WIDTH, max(1, int(h * ratio))),
            Image.Resampling.LANCZOS,
        )
        w, h = out.size

    if h < VIN_CROP_PAD_TRIGGER_HEIGHT:
        pad_total = MIN_VIN_CROP_HEIGHT - h
        padded = Image.new("RGB", (w, MIN_VIN_CROP_HEIGHT), (255, 255, 255))
        padded.paste(out, (0, pad_total // 2))
        out = padded

    out = ImageEnhance.Contrast(out).enhance(1.12)
    out = ImageEnhance.Sharpness(out).enhance(1.15)
    return out


def _section_text_lines(section: dict[str, Any]) -> list[str]:
    return [ln.strip() for ln in section.get("text", "").splitlines() if ln.strip()]


def _vin_tokens(text: str, *, min_len: int = 11, ocr: bool = False) -> list[str]:
    pat = VIN_TOKEN_OCR if ocr else VIN_TOKEN
    return [
        norm_vin(m.group(0))
        for m in pat.finditer(text)
        if len(norm_vin(m.group(0))) >= min_len and _looks_like_vin(norm_vin(m.group(0)))
    ]


def _vin_tokens_any(text: str, *, min_len: int = 11) -> list[str]:
    """Strict + OCR-tolerant VIN tokens (deduped), for section scoring."""
    seen: set[str] = set()
    out: list[str] = []
    for ocr in (False, True):
        for tok in _vin_tokens(text, min_len=min_len, ocr=ocr):
            if tok not in seen:
                seen.add(tok)
                out.append(tok)
    compact = norm_vin(text)
    if len(compact) == 17 and _looks_like_vin(compact) and compact not in seen:
        seen.add(compact)
        out.append(compact)
    return out


def _is_trade_line(text: str) -> bool:
    upper = text.upper()
    return bool(TRADE_IN_RE.search(upper) or ("TRADE" in upper and "SECTION" not in upper))


def _is_trade_in_section(text: str, lines: list[str] | None = None) -> bool:
    upper = text.upper()
    line_count = len(lines) if lines else upper.count("\n") + 1
    if "VEHICLE TRADED" in upper:
        return True
    if re.search(r"36\.\s*TRADE[- ]?IN", upper):
        return True
    if re.search(r"ADDITIONAL\s+TRADE[- ]?IN", upper):
        return True
    if lines and re.search(r"TRADE[- ]?IN\s+YEAR", lines[0], re.I):
        return True
    # Compact trade-in blocks (TX/MI) — not full-page DOR owner sections that mention trade-in.
    if (
        line_count <= 28
        and TRADE_IN_RE.search(upper)
        and re.search(r"YEAR\s+MAKE\s+VEHICLE\s+IDENTIFICATION\s+NUMBER", upper)
    ):
        return True
    return False


def _is_signature_section(text: str, line_count: int) -> bool:
    if line_count > 22:
        return False
    return bool(SIGNATURE_SECTION_RE.search(text))


def _is_lienholder_section(text: str) -> bool:
    return bool(LIENHOLDER_SECTION_RE.search(text))


def _is_lessee_section(text: str, lines: list[str] | None = None) -> bool:
    if not LESSEE_SECTION_RE.search(text):
        return False
    if lines and any(_vin_value_len(ln) >= 15 for ln in lines if not _is_trade_line(ln)):
        return False
    return True


def _is_primary_vehicle_vin_section(text: str, lines: list[str]) -> bool:
    upper = text.upper()
    # Trade-in blocks reuse "Year Make Vehicle Identification Number" labels — not subject VIN.
    if lines and re.search(r"TRADE[- ]?IN\s+YEAR", lines[0], re.I):
        return False
    if (
        TRADE_IN_SECTION_RE.search(text)
        or re.search(r"36\.\s*TRADE[- ]?IN", upper)
        or "VEHICLE TRADED" in upper
    ):
        return False
    if PRIMARY_VIN_FIELD_RE.search(text):
        return True
    if "SECTION B" in text.upper() and "VEHICLE" in text.upper():
        return True
    early = lines[: min(8, len(lines))]
    for ln in early:
        if _is_trade_line(ln) or _is_false_vin_header(ln):
            continue
        if _is_primary_vin_label(ln):
            return True
        if re.search(r"1\.\s*VEHICLE\s+IDENTIFICATION", ln, re.I):
            return True
    return False


def _looks_like_vin(compact: str) -> bool:
    """Heuristic: printed VINs mix letters and digits; fee/header words do not."""
    if not (11 <= len(compact) <= 20):
        return False
    digits = sum(c.isdigit() for c in compact)
    letters = sum(c.isalpha() for c in compact)
    return digits >= 4 and letters >= 4


def _is_false_vin_header(text: str) -> bool:
    return bool(_FALSE_VIN_LABEL.search(text))


def _is_primary_vin_label(text: str) -> bool:
    """Printed VIN field label — not fee/inspection headers or odometer blocks."""
    if _is_false_vin_header(text):
        return False
    upper = text.upper()
    if re.search(r"VEHICLE\s+IDENTIFICATION", upper):
        return True
    return bool(
        re.search(r"IDENTIFICATION\s+NUMBER", upper) and re.search(r"\bVIN\b", upper)
    )


def _is_vin_label_line(text: str) -> bool:
    if _is_false_vin_header(text):
        return False
    if _is_primary_vin_label(text):
        return True
    return bool(re.search(r"\bVIN\b", text, re.I))


def _vin_value_len(text: str) -> int:
    """Best VIN-like token length on a line (0 when absent)."""
    best = 0
    for ocr in (False, True):
        toks = _vin_tokens(text, ocr=ocr)
        if toks:
            best = max(best, max(len(t) for t in toks))
    compact = norm_vin(text)
    if _looks_like_vin(compact):
        best = max(best, len(compact))
    return best


def _primary_vehicle_markers(text: str) -> bool:
    upper = text.upper()
    return bool(
        re.search(
            r"SECTION B.*VEHICLE|VEHICLE IDENTIFICATION|VEHICLE INFORMATION",
            upper,
        )
    )


def _mega_penalty(text: str, line_count: int, lines: list[str] | None = None) -> float:
    upper = text.upper()
    hits = sum(1 for kw in MEGA_SECTION_PENALTIES if kw in upper)
    if hits == 0:
        return 0.0
    if _is_trade_in_section(text, lines) or _is_signature_section(text, line_count):
        return 5.0 * min(hits, 4)
    if line_count <= 8 and _primary_vehicle_markers(text):
        return -2.0 * min(hits, 2)
    if line_count <= 15 and _primary_vehicle_markers(text):
        return -3.0 * min(hits, 3)
    return -5.0 * min(hits, 4)


def score_vin_section(section: dict[str, Any], all_sections: list[dict[str, Any]] | None = None) -> float:
    text = section.get("text", "")
    upper = text.upper()
    lines = _section_text_lines(section)
    line_count = section.get("line_count") or len(lines)
    score = 0.0

    if "VEHICLE IDENTIFICATION" in upper or "VEHICLE INFORMATION" in upper:
        score += 6
    elif re.search(r"\bVIN\b", upper):
        score += 4

    if "SECTION B" in upper and "VEHICLE" in upper:
        score += 8
    if lines and re.search(r"SECTION\s+B[- ].*VEHICLE", lines[0], re.I):
        score += 10

    if _is_primary_vehicle_vin_section(text, lines):
        score += 12
    if _is_trade_in_section(text, lines):
        score -= 25
    if _is_lienholder_section(text):
        score -= 30
    if _is_lessee_section(text, lines):
        score -= 22
    if _is_signature_section(text, line_count):
        score -= 18

    for i, ln in enumerate(lines[: max(4, (len(lines) + 2) // 3)]):
        if VIN_LABEL.search(ln) and not _is_trade_line(ln):
            score += 4 - min(i, 3)
            break

    has_17 = False
    for ln in lines:
        if _is_trade_line(ln):
            continue
        if any(len(t) == 17 for t in _vin_tokens_any(ln)):
            has_17 = True
            break
    if not has_17:
        for i, ln in enumerate(lines):
            if VIN_LABEL.search(ln) and i > max(2, int(len(lines) * 0.55)):
                score -= 4
                break

    score -= _mega_penalty(text, line_count, lines)

    primary_vin = _is_primary_vehicle_vin_section(text, lines)
    if line_count > 30:
        score -= 8 if primary_vin else 15
    elif line_count > 22:
        score -= 4 if primary_vin else 8
    elif line_count <= 6 and re.search(r"\bVIN\b|VEHICLE IDENTIFICATION", upper):
        score += 3

    tokens = _vin_tokens_any(text)
    non_trade_tokens = []
    for ln in lines:
        if _is_trade_line(ln):
            continue
        non_trade_tokens.extend(_vin_tokens_any(ln))
    tokens_17 = [t for t in non_trade_tokens if len(t) == 17]
    if tokens_17:
        score += 7
        score += max(0, 2 - abs(17 - len(tokens_17[0])))
    elif non_trade_tokens:
        score += 4
        score += max(0, 2 - abs(17 - len(non_trade_tokens[0])))
    elif tokens:
        score += 2

    trade_lines = sum(1 for ln in lines if _is_trade_line(ln))
    if trade_lines and not non_trade_tokens and not any(VIN_LABEL.search(ln) for ln in lines if not _is_trade_line(ln)):
        score -= 6
    elif trade_lines and non_trade_tokens and all_sections:
        for other in all_sections:
            if other is section:
                continue
            if _primary_vehicle_markers(other.get("text", "")) and _vin_tokens_any(
                other.get("text", "")
            ):
                score -= 3
                break

    return score


def pick_vin_section(sections_data: dict[str, Any]) -> dict[str, Any] | None:
    sections = sections_data["sections"]
    ranked = sorted(sections, key=lambda s: score_vin_section(s, sections), reverse=True)
    if not ranked or score_vin_section(ranked[0], sections) <= 0:
        return None
    return ranked[0]


def best_vin_section_score(sections_data: dict[str, Any]) -> float:
    """Highest VIN section score on a page (0 when no viable section)."""
    sections = sections_data.get("sections") or []
    if not sections:
        return 0.0
    return max(score_vin_section(s, sections) for s in sections)


@lru_cache(maxsize=32)
def _ocr_lines_for_vision(vision_path: str, image_width: int) -> tuple[Line, ...]:
    vision = load_vision(Path(vision_path))
    words = load_words(vision)
    return tuple(words_to_lines(words, page_width=image_width, full_width=True))


def _section_ocr_lines(
    section: dict[str, Any],
    sections_data: dict[str, Any],
    image: Image.Image,
) -> list[tuple[int, str, Box]]:
    vision_path = sections_data.get("vision")
    if not vision_path or not Path(vision_path).exists():
        return []
    indices = set(section.get("line_indices") or [])
    if not indices:
        return []
    ocr_lines = _ocr_lines_for_vision(str(Path(vision_path).resolve()), image.width)
    return [(ln.index, ln.text, ln.content_box) for ln in ocr_lines if ln.index in indices]


def _bounds_dict(box: Box, fallback: dict[str, float], img_h: int) -> dict[str, float]:
    return {
        "min_x": fallback["min_x"],
        "min_y": max(0, box.min_y),
        "max_x": fallback["max_x"],
        "max_y": min(img_h, box.max_y),
    }


def _ocr_compact(text: str) -> str:
    cleaned = text.upper().replace("×", "X").replace("Х", "X")
    return re.sub(r"[^A-Z0-9]", "", cleaned)


def _vin_substring_match(norm: str, compact: str, *, min_len: int = 10) -> bool:
    if not norm or not compact:
        return False
    if norm in compact:
        return True
    if len(norm) >= 8 and norm[-8:] in compact:
        return True
    variants = {
        norm,
        norm.replace("0", "O"),
        norm.replace("O", "0"),
        norm.replace("1", "I"),
        norm.replace("I", "1"),
        norm.replace("B", "8"),
        norm.replace("8", "B"),
    }
    for variant in variants:
        if variant in compact:
            return True
        if len(variant) >= 8 and variant[-8:] in compact:
            return True
        for n in range(17, min_len - 1, -1):
            for i in range(len(variant) - n + 1):
                if variant[i : i + n] in compact:
                    return True
    return False


def _union_bounds(boxes: list[Box], fallback: dict[str, float], img_h: int) -> dict[str, float]:
    if not boxes:
        return fallback
    acc = boxes[0]
    for box in boxes[1:]:
        acc = acc.union(box)
    return _bounds_dict(acc, fallback, img_h)


def _pick_vin_line_indices(lines: list[str]) -> list[int]:
    """Return line indices for the VIN label row and/or printed value row.

    Excludes fee/inspection header lines (ONLY VIN, VIN FEE, VIN INSPECTION).
    Requires VEHICLE IDENTIFICATION NUMBER label and/or a VIN-value row.
    """
    primary_label_idxs: list[int] = []
    token_17_idxs: list[int] = []
    token_other_idxs: list[tuple[int, int]] = []

    for i, text in enumerate(lines):
        if _is_trade_line(text) or _is_false_vin_header(text):
            continue
        if _is_primary_vin_label(text):
            primary_label_idxs.append(i)
        value_len = _vin_value_len(text)
        if value_len == 17:
            token_17_idxs.append(i)
        elif value_len >= 11:
            token_other_idxs.append((i, value_len))

    def _nearby_label(vin_idx: int) -> int | None:
        candidates = [li for li in primary_label_idxs if li <= vin_idx and vin_idx - li <= 4]
        return max(candidates) if candidates else None

    def _vin_row_after(label_idx: int) -> int | None:
        for j in range(label_idx + 1, min(label_idx + 5, len(lines))):
            if _is_trade_line(lines[j]) or _is_false_vin_header(lines[j]):
                continue
            if _vin_value_len(lines[j]) >= 15:
                return j
        return None

    chosen: list[int] = []

    if primary_label_idxs:
        label_idx = min(primary_label_idxs)
        vin_idx = _vin_row_after(label_idx)
        if vin_idx is None and token_17_idxs:
            near = [i for i in token_17_idxs if abs(i - label_idx) <= 6]
            vin_idx = min(near, key=lambda i: abs(i - label_idx)) if near else None
        if vin_idx is None and token_other_idxs:
            near = [i for i, _ in token_other_idxs if abs(i - label_idx) <= 6]
            vin_idx = min(near, key=lambda i: abs(i - label_idx)) if near else None
        chosen.append(label_idx)
        if vin_idx is not None:
            chosen.append(vin_idx)
            if vin_idx + 1 < len(lines) and not _is_trade_line(lines[vin_idx + 1]):
                nxt = lines[vin_idx + 1]
                if not _is_false_vin_header(nxt):
                    # Stop before the next numbered form row (e.g. TX 130-U field 8).
                    if not re.match(r"\s*\d+\.", nxt) and _vin_value_len(nxt) >= 11:
                        chosen.append(vin_idx + 1)
        else:
            for j in range(label_idx + 1, min(label_idx + 3, len(lines))):
                if not _is_trade_line(lines[j]) and not _is_false_vin_header(lines[j]):
                    chosen.append(j)
        return sorted(set(chosen))

    if token_17_idxs:
        vin_idx = token_17_idxs[0]
        label_idx = _nearby_label(vin_idx)
        if label_idx is not None:
            chosen.append(label_idx)
        chosen.append(vin_idx)
        if vin_idx + 1 < len(lines) and not _is_trade_line(lines[vin_idx + 1]):
            nxt = lines[vin_idx + 1]
            if not _is_false_vin_header(nxt):
                if not re.match(r"\s*\d+\.", nxt) and _vin_value_len(nxt) >= 11:
                    chosen.append(vin_idx + 1)
        return sorted(set(chosen))

    if token_other_idxs:
        vin_idx = max(token_other_idxs, key=lambda t: (t[1], -t[0]))[0]
        label_idx = _nearby_label(vin_idx)
        if label_idx is not None:
            chosen.append(label_idx)
        chosen.append(vin_idx)
        return sorted(set(chosen))

    return []


def _pick_vin_boxes(section_line_data: list[tuple[int, str, Box]]) -> list[Box]:
    lines = [text for _, text, _ in section_line_data]
    chosen_local = _pick_vin_line_indices(lines)
    return [section_line_data[i][2] for i in chosen_local]


def tight_vin_bounds(
    section: dict[str, Any],
    image: Image.Image,
    sections_data: dict[str, Any] | None = None,
) -> dict[str, float]:
    b = section["bounds"]
    if sections_data:
        section_line_data = _section_ocr_lines(section, sections_data, image)
        if section_line_data:
            section_line_data.sort(key=lambda x: x[0])
            boxes = _pick_vin_boxes(section_line_data)
            if boxes:
                return _union_bounds(boxes, b, image.height)

    lines = _section_text_lines(section)
    if not lines:
        return b
    chosen = _pick_vin_line_indices(lines)
    if not chosen:
        return b

    h = b["max_y"] - b["min_y"]
    n = len(lines)
    line_h = max(40, h / max(n, 1))
    y0 = b["min_y"] + min(chosen) * line_h - 20
    y1 = b["min_y"] + (max(chosen) + 1) * line_h + 40
    return {
        "min_x": b["min_x"],
        "min_y": max(0, y0),
        "max_x": b["max_x"],
        "max_y": min(image.height, y1),
    }


def vin_visible_in_bounds(
    full_vin: str,
    bounds: dict[str, float],
    sections_data: dict[str, Any],
    image: Image.Image,
) -> bool:
    """True when OCR within bounds contains the full-page VIN (or a 14+ char substring)."""
    norm = norm_vin(full_vin)
    if not norm:
        return False
    vision_path = sections_data.get("vision")
    if not vision_path or not Path(vision_path).exists():
        return False
    ocr_lines = _ocr_lines_for_vision(str(Path(vision_path).resolve()), image.width)
    x0, y0, x1, y1 = bounds["min_x"], bounds["min_y"], bounds["max_x"], bounds["max_y"]
    in_bounds = [
        ln
        for ln in ocr_lines
        if ln.content_box.max_y >= y0 - 8
        and ln.content_box.min_y <= y1 + 8
        and ln.content_box.max_x >= x0 - 8
        and ln.content_box.min_x <= x1 + 8
    ]
    compact = _ocr_compact("".join(ln.text for ln in in_bounds))
    if _vin_substring_match(norm, compact):
        return True
    for ln in in_bounds:
        if _is_trade_line(ln.text):
            continue
        if any(len(t) >= 15 for t in _vin_tokens(ln.text, ocr=True)):
            return True
    return False


# --- WA-577 ticket-CSV crop scoring (3-class: crop | null | both_wrong) ---

_VIN17_STRICT = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")


def _ocr_norm_compact(text: str) -> str:
    """Strip spaces/punct and casefold for GT-in-OCR substring checks."""
    return re.sub(r"[^A-Za-z0-9]", "", str(text or "")).casefold()


def gt_in_ocr_text(ground_truth: str | None, *texts: str | None) -> bool:
    """True when normalized GT appears as substring in OCR (strict, no fuzzy variants)."""
    gt = _ocr_norm_compact(norm_vin(ground_truth))
    if not gt or len(gt) < 11:
        return False
    blob = _ocr_norm_compact("".join(t for t in texts if t))
    if not blob:
        return False
    if gt in blob:
        return True
    if len(gt) >= 15:
        for n in range(17, 14, -1):
            for i in range(len(gt) - n + 1):
                if gt[i : i + n] in blob:
                    return True
    return False


def _vin_edit_dist(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


def vin_matches_printed(extracted: str | None, printed: str | None, *, max_edits: int = 2) -> bool:
    """True when extracted VIN matches printed OCR VIN (exact or minor OCR edits)."""
    ext, prt = norm_vin(extracted), norm_vin(printed)
    if not ext or not prt:
        return False
    if ext == prt:
        return True
    if len(ext) == len(prt) == 17 and _vin_edit_dist(ext, prt) <= max_edits:
        return True
    if len(ext) >= 15 and len(prt) >= 15 and _vin_edit_dist(ext, prt) <= max_edits + 1:
        return True
    return False


def gt_visible_on_page(ground_truth: str | None, ocr_text: str | None, *, max_edits: int = 4) -> bool:
    """True when GT appears in OCR (strict substring or fuzzy token match)."""
    if gt_in_ocr_text(ground_truth, ocr_text):
        return True
    gt_n = norm_vin(ground_truth)
    if not gt_n or not ocr_text:
        return False
    for tok in _vin_tokens_any(ocr_text):
        if len(tok) >= 15 and _vin_edit_dist(tok, gt_n) <= max_edits:
            return True
    return False


def infer_printed_vin(
    full: str | None,
    crop: str | None,
    sections: list[dict[str, Any]] | None,
    picked_section: int | None,
    ocr_text: str | None = None,
) -> str | None:
    """Best guess at the VIN printed on the document from OCR (not extraction)."""
    ocr_tokens: list[str] = []
    if sections is not None and picked_section is not None:
        picked = next((s for s in sections if s.get("index") == picked_section), None)
        if picked:
            ocr_tokens.extend(_vin_tokens_any(picked.get("text", "")))
    if ocr_text:
        ocr_tokens.extend(_vin_tokens_any(ocr_text))

    seen: set[str] = set()
    uniq: list[str] = []
    for tok in ocr_tokens:
        if tok not in seen and len(tok) >= 15:
            seen.add(tok)
            uniq.append(tok)

    if not uniq:
        return None

    full_n, crop_n = norm_vin(full), norm_vin(crop)
    ref = crop_n or full_n
    if ref:
        return min(uniq, key=lambda c: _vin_edit_dist(c, ref))
    prefer_17 = [t for t in uniq if len(t) == 17]
    return prefer_17[0] if prefer_17 else uniq[0]


def classify_from_gt_audit(
    row: dict[str, Any],
    *,
    gt_in_ocr: bool,
    gt_visible_on_page: bool,
    printed_vin: str | None,
) -> tuple[str, str, bool, str]:
    """Return (class, bucket, extraction_correct_vs_doc, notes) from GT/OCR audit."""
    full = row.get("full_page_vin")
    crop = row.get("crop_vin_enhanced")
    gt = row.get("ground_truth")
    full_n, crop_n, gt_n = norm_vin(full), norm_vin(crop), norm_vin(gt)
    ocr_printed = printed_vin

    if crop is None:
        return "null", row.get("result_bucket") or row.get("bucket") or "B1", False, ""

    # full/prod null but crop read a valid printed VIN — crop win, not both_wrong
    if not full_n and crop_n and VIN17.match(crop_n) and ocr_printed and vin_matches_printed(crop, ocr_printed):
        note = "crop correct vs printed; full null"
        if gt_n and gt_n != crop_n:
            if not gt_visible_on_page:
                return "crop", "B2", True, note
            return (
                "both_wrong",
                row.get("result_bucket") or row.get("bucket") or "B1",
                False,
                f"GT {gt_n} visible on page; crop {crop_n} does not match",
            )
        return "crop", row.get("result_bucket") or row.get("bucket") or "B1", True, note

    matches_gt = bool(gt_n and (crop_n == gt_n or full_n == gt_n))
    matches_printed = bool(
        ocr_printed
        and (vin_matches_printed(full, ocr_printed) or vin_matches_printed(crop, ocr_printed))
    )

    if matches_gt:
        bucket = row.get("result_bucket") or row.get("bucket") or "B1"
        return "crop", bucket, False, "crop matches GT"

    # GT visible on page but extraction missed it → true extraction error
    if gt_n and gt_visible_on_page and not matches_gt:
        return (
            "both_wrong",
            row.get("result_bucket") or row.get("bucket") or "B1",
            False,
            f"GT {gt_n} visible on page; extraction does not match printed VIN",
        )

    if matches_printed and ocr_printed:
        note = f"extraction matches printed VIN {ocr_printed}"
        if gt_n and gt_n != norm_vin(ocr_printed):
            return "crop", "B2", True, f"{note}; GT {gt_n} not on doc — B2"
        return "crop", row.get("result_bucket") or row.get("bucket") or "B1", True, note

    if full_n and crop_n and full_n == crop_n and not gt_visible_on_page:
        note = f"full==crop {full_n}; GT not visible on page"
        if gt_n:
            note += f" (GT {gt_n})"
        return "crop", "B2", True, f"{note} — B2 source disagree"

    if not gt_n and full_n and crop_n and full_n == crop_n and ocr_printed and matches_printed:
        return "crop", row.get("result_bucket") or row.get("bucket") or "B1", True, "GT null; extraction matches printed VIN"

    return (
        "both_wrong",
        row.get("result_bucket") or row.get("bucket") or "B1",
        False,
        "extraction wrong vs printed VIN on document",
    )


def extraction_correct_vs_doc(row: dict[str, Any]) -> bool:
    """True when full/crop agree and correctly read what is printed on the page.

    Ticket GT may still differ (B2 source disagreement: printed doc VIN != contract ref).
    Set ``extraction_correct_vs_doc`` on the result row after visual audit.
    """
    flag = row.get("extraction_correct_vs_doc")
    if flag is True:
        return True
    if flag is False:
        return False
    return False


def to_three_class(row: dict[str, Any]) -> str | None:
    """Scored 3-class label: crop | null | both_wrong (None = error/excluded).

    Scoring rules:
    - **crop** — extraction succeeded: matches ticket GT or printed doc
      (``extraction_correct_vs_doc``).
    - **null** — crop returned null.
    - **both_wrong** — wrong vs what is ON the document (not B2 GT mismatch).

    ``both_wrong`` is NOT used when full==crop matches printed OCR but not ticket GT.
    """
    if row.get("error"):
        return None
    if extraction_correct_vs_doc(row) or row.get("crop_matches_truth") or row.get("full_matches_truth"):
        return "crop"
    if row.get("crop_vin_enhanced") is None:
        return "null"
    if row.get("classification") == "both_wrong" and not extraction_correct_vs_doc(row):
        return "both_wrong"
    if row.get("class") in ("crop", "null", "both_wrong"):
        return row["class"]
    cls = row.get("classification") or ""
    if cls == "crop_null":
        return "null"
    if cls == "both_wrong":
        return "both_wrong"
    if cls in ("crop_fixes_full", "crop_wins", "both_ok"):
        return "crop"
    return "both_wrong"


def summarize_three_class(results: list[dict[str, Any]]) -> dict[str, int]:
    from collections import Counter

    scored = [r for r in results if to_three_class(r)]
    counts = Counter(to_three_class(r) for r in scored)
    return {
        "crop": counts.get("crop", 0),
        "null": counts.get("null", 0),
        "both_wrong": counts.get("both_wrong", 0),
        "scored": len(scored),
        "errors_excluded": len(results) - len(scored),
    }
