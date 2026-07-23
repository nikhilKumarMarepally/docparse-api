from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from app.extract.gemini import GeminiExtractor, get_extractor
from app.extract.section_gate import get_section_gate, gate_min_confidence, section_gate_enabled
from app.ocr import run_ocr
from app.paths import ensure_script_path
from app.section_crop import crop_section_image, enhance_section_crop
from app.text_weight import (
    annotate_word_styles,
    bold_text_in_section,
    section_has_bold,
    styles_to_dicts,
    value_is_bold,
)

logger = logging.getLogger(__name__)

ensure_script_path()

from ocr_line_to_sections import (  # noqa: E402
    _make_section,
    lines_to_sections_hv_combined,
)
from ocr_word_to_line_boxes import load_words, words_to_lines  # noqa: E402
from section_layout_breaks import lines_to_sections_human  # noqa: E402
from section_preprocess import annotate_preprocess  # noqa: E402
from section_table_layout import classify_section_layout  # noqa: E402


def _section_label(text: str, index: int) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())[:80]
    if "vin" in norm and ("make" in norm or "year" in norm):
        return "vin_vehicle"
    if "signature" in norm:
        return "signature"
    if any(k in norm for k in ("privacy", "disclosure", "federal law", "state law")):
        return "privacy_boilerplate"
    if "lien" in norm or "holder" in norm:
        return "lienholder"
    if "owner" in norm or "applicant" in norm:
        return "owner_applicant"
    return f"section_{index}"


def _merge_fields(target: dict[str, Any], source: dict[str, Any], warnings: list[str], *, page: int, section: int) -> None:
    for key, value in source.items():
        if key in target and target[key] != value:
            warnings.append(
                f"page {page} section {section}: field '{key}' collision "
                f"({target[key]!r} -> {value!r}); keeping later value"
            )
        target[key] = value


def draw_filter_overlay(
    image_path: Path,
    sections: list[dict[str, Any]],
    out_path: Path,
    *,
    overlay_title: str | None = None,
    overlay_mode: str = "preprocess",
) -> None:
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    if overlay_title:
        draw.rectangle([(0, 0), (img.width, 28)], fill=(20, 20, 20, 230))
        draw.text((8, 6), overlay_title, fill=(255, 255, 255), font=title_font)

    img_w, img_h = img.size
    for section in sections:
        bounds = section.get("bounds") or {}
        x0 = max(0, min(int(bounds.get("min_x", 0)), img_w - 1))
        y0 = max(0, min(int(bounds.get("min_y", 0)), img_h - 1))
        x1 = max(x0, min(int(bounds.get("max_x", 0)), img_w))
        y1 = max(y0, min(int(bounds.get("max_y", 0)), img_h))
        prep = section.get("preprocess") or {}
        gate = section.get("content_gate") or {}
        if overlay_mode == "gate" and gate:
            kept = bool(gate.get("extractable", True))
            conf = gate.get("confidence")
            if conf is not None and isinstance(conf, (int, float)):
                kept = kept and float(conf) >= gate_min_confidence()
        else:
            kept = bool(prep.get("kept", True))
        color = (34, 160, 80) if kept else (220, 50, 50)
        draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=4)
        idx = section.get("index", 0)
        if overlay_mode == "gate" and gate:
            conf = gate.get("confidence")
            conf_s = f" {float(conf):.2f}" if conf is not None else ""
            label = f"S{idx} {'KEEP' if kept else 'SKIP'}{conf_s}"
        else:
            label = f"S{idx}"
        label_w = max(48, 8 * len(label))
        label_top = max(0, y0 - 18)
        label_bottom = max(label_top + 1, y0)
        draw.rectangle([(x0, label_top), (x0 + label_w, label_bottom)], fill=(*color, 220))
        draw.text((x0 + 4, label_top + 2), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, format="PNG")


def process_page(
    page_index: int,
    page_png: Path,
    page_dir: Path,
    extractor: GeminiExtractor,
    *,
    on_step: Callable[[str], None] | None = None,
    skip_llm: bool = False,
) -> dict[str, Any]:
    def step(msg: str) -> None:
        if on_step:
            on_step(msg)

    step("ocr")
    vision_path = page_dir / "vision.json"
    vision = run_ocr(page_png, vision_path)

    step("sections")
    words = load_words(vision)
    with Image.open(page_png) as img:
        page_width = img.width
        page_rgb = img.convert("RGB")

    use_human_layout = bool((vision.get("image_blocks") or {}).get("with_polygons"))
    fw_lines = words_to_lines(words, page_width=page_width, full_width=True)
    split_lines = words_to_lines(
        words,
        page_width=page_width,
        full_width=False,
        split_columns=True,
    )

    if use_human_layout:
        human_chunks, _section_meta = lines_to_sections_human(
            split_lines,
            vision=vision,
            page_width=float(page_width),
            image_rgb=page_rgb,
            min_gap_px=8.0,
        )
        sections_obj = [
            _make_section(i, line_group, None, 6.0) for i, (line_group, _, _) in enumerate(human_chunks)
        ]
        lines = split_lines
    else:
        sections_obj, gap_stats, _column_meta = lines_to_sections_hv_combined(
            split_lines,
            page_width=float(page_width),
            full_width_lines=fw_lines,
            min_gap_px=18.0,
        )
        lines = fw_lines

    raw_sections = [
        s.to_dict(layout=classify_section_layout(s.lines).to_dict()) for s in sections_obj
    ]
    sections_path = page_dir / "sections.json"
    sections_path.write_text(json.dumps({"sections": raw_sections}, indent=2))

    step("filter")
    _no_fasttext = Path("/nonexistent/fasttext.bin")
    annotated: list[dict[str, Any]] = [
        annotate_preprocess(
            s,
            document_type="generic",
            boilerplate_model=_no_fasttext,
            field_model=_no_fasttext,
        )
        for s in raw_sections
    ]

    crops_dir = page_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    page_fields: dict[str, Any] = {}
    page_field_styles: dict[str, bool | None] = {}
    warnings: list[str] = []
    section_rows: list[dict[str, Any]] = []

    step("extract")
    use_gate = section_gate_enabled()
    gate_classifier = get_section_gate() if use_gate else None

    def _layout_kind(section: dict[str, Any]) -> str | None:
        kind = section.get("layout_kind")
        if isinstance(kind, str) and kind:
            return kind
        layout = section.get("layout")
        if isinstance(layout, dict):
            nested = layout.get("layout_kind")
            if isinstance(nested, str) and nested:
                return nested
        return None

    def _prepare_section(section: dict[str, Any]) -> dict[str, Any]:
        idx = int(section.get("index", 0))
        prep = section.get("preprocess") or {}
        preprocess_kept = bool(prep.get("kept", True))
        bounds = section.get("bounds") or {}
        ocr_text = section.get("text") or ""
        label = _section_label(ocr_text, idx)

        crop_path = crops_dir / f"s{idx}.png"
        crop_img = crop_section_image(page_png, bounds)
        crop_img.save(crop_path, format="PNG")

        section_words = []
        for line_idx in section.get("line_indices") or []:
            if 0 <= int(line_idx) < len(lines):
                section_words.extend(lines[int(line_idx)].words)

        word_styles = annotate_word_styles(page_rgb, section_words)

        gate_result = None
        if gate_classifier is not None:
            gate_result = gate_classifier.classify(
                ocr_text,
                crop_img=crop_img,
                layout_kind=_layout_kind(section),
            )
            section["content_gate"] = gate_result.to_dict()

        extractable = (
            gate_result.passes_threshold(gate_min_confidence())
            if gate_result is not None
            else True
        )
        # Preprocess red / gate SKIP do not block Gemini — only skip_llm does.
        run_extract = not skip_llm
        # Only pure line-item grids use the table prompt. section_table (2-col
        # address blocks, totals bands) must stay on schema-free extraction —
        # otherwise S2/S4 falsely become line_items tables.
        kind = _layout_kind(section)
        table_band = bool(section.get("table_band")) or kind == "table"

        fields: dict[str, Any] | None = None
        if run_extract:
            enhanced = enhance_section_crop(crop_img)
            # Fresh extractor per worker — Gemini client is not shared across threads.
            local_extractor = GeminiExtractor(model=extractor.model)
            fields = local_extractor.extract_section(
                enhanced,
                ocr_text,
                section_words=section_words,
                bounds=bounds,
                table_band=table_band,
            )

        if not run_extract and gate_result is not None and not extractable:
            filter_reason = "content_gate:skip"
        elif not preprocess_kept:
            filter_reason = prep.get("reason")
        else:
            filter_reason = None

        return {
            "index": idx,
            "label": label,
            "bounds": bounds,
            "kept": True,  # never hide red preprocess boxes from results
            "preprocess_kept": preprocess_kept,
            "layout_kind": kind,
            "table_band": table_band,
            "gate_passes": extractable if gate_result is not None else None,
            "content_gate": gate_result.to_dict() if gate_result else None,
            "filter_reason": filter_reason,
            "ocr_preview": ocr_text[:400],
            "has_bold": section_has_bold(word_styles),
            "bold_tokens": bold_text_in_section(word_styles),
            "word_styles": styles_to_dicts(word_styles),
            "fields": fields,
            "word_styles_obj": word_styles,
            "crop_path": str(crop_path.relative_to(page_dir.parent.parent)),
        }

    # Parallel Gemini calls — sequential section extracts were the main latency.
    max_workers = min(6, max(1, len(annotated)))
    prepared: list[dict[str, Any] | None] = [None] * len(annotated)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_prepare_section, section): i
            for i, section in enumerate(annotated)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                prepared[i] = fut.result()
            except Exception:
                logger.exception("section extract failed index=%s", i)
                section = annotated[i]
                idx = int(section.get("index", i))
                prepared[i] = {
                    "index": idx,
                    "label": _section_label(section.get("text") or "", idx),
                    "bounds": section.get("bounds") or {},
                    "kept": True,
                    "preprocess_kept": bool((section.get("preprocess") or {}).get("kept", True)),
                    "layout_kind": _layout_kind(section),
                    "table_band": False,
                    "gate_passes": None,
                    "content_gate": None,
                    "filter_reason": "extract_error",
                    "ocr_preview": (section.get("text") or "")[:400],
                    "has_bold": False,
                    "bold_tokens": [],
                    "word_styles": [],
                    "fields": None,
                    "word_styles_obj": [],
                    "crop_path": str((crops_dir / f"s{idx}.png").relative_to(page_dir.parent.parent)),
                }

    for row in prepared:
        assert row is not None
        fields = row.get("fields")
        field_styles: dict[str, bool | None] = {}
        word_styles = row.pop("word_styles_obj", [])
        if not fields:
            if fields is not None:
                warnings.append(
                    f"page {page_index} section {row['index']}: low_signal (0 fields extracted)"
                )
        else:
            for key, value in fields.items():
                if key in ("line_items", "table_columns"):
                    continue
                if isinstance(value, str):
                    field_styles[key] = value_is_bold(value, word_styles)
            _merge_fields(page_fields, fields, warnings, page=page_index, section=int(row["index"]))
            for key, bold in field_styles.items():
                if bold is not None:
                    page_field_styles[key] = bold
        row["field_styles"] = field_styles or None
        section_rows.append(row)

    step("overlay")
    overlay_path = page_dir / "overlay.png"
    draw_filter_overlay(
        page_png,
        annotated,
        overlay_path,
        overlay_title="Sections — green preprocess kept / red filtered (all sections extracted)",
        overlay_mode="preprocess",
    )

    return {
        "page_index": page_index,
        "sections": section_rows,
        "merged_fields": page_fields,
        "field_styles": page_field_styles or None,
        "warnings": warnings,
        "overlay_path": str(overlay_path.relative_to(page_dir.parent.parent)),
    }


def run_pipeline(
    job_id: str,
    job_dir: Path,
    page_pngs: list[Path],
    *,
    filename: str,
    on_step: Callable[[str], None] | None = None,
    skip_llm: bool = False,
) -> dict[str, Any]:
    extractor = get_extractor()
    merged: dict[str, Any] = {}
    all_warnings: list[str] = []
    pages_out: list[dict[str, Any]] = []

    for page_index, page_png in enumerate(page_pngs):
        page_dir = job_dir / f"page_{page_index:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        def page_step(msg: str) -> None:
            if on_step:
                on_step(f"page_{page_index}:{msg}")

        page_result = process_page(
            page_index,
            page_png,
            page_dir,
            extractor,  # type: ignore[arg-type]
            on_step=page_step,
            skip_llm=skip_llm,
        )
        all_warnings.extend(page_result.pop("warnings", []))
        _merge_fields(merged, page_result.get("merged_fields") or {}, all_warnings, page=page_index, section=-1)
        pages_out.append(page_result)

    return {
        "job_id": job_id,
        "status": "completed",
        "source": {"filename": filename, "pages": len(page_pngs)},
        "pages": pages_out,
        "merged_fields": merged,
        "warnings": all_warnings,
    }
