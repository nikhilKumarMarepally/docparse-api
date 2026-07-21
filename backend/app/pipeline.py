from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from app.extract.gemini import GeminiExtractor, get_extractor
from app.ocr import run_ocr
from app.paths import ensure_script_path
from app.text_weight import (
    annotate_word_styles,
    bold_text_in_section,
    section_has_bold,
    styles_to_dicts,
    value_is_bold,
)

ensure_script_path()

from ocr_line_to_sections import (  # noqa: E402
    _make_section,
    lines_to_sections_hv_combined,
)
from ocr_word_to_line_boxes import load_words, words_to_lines  # noqa: E402
from section_content_gemini_gallery import crop_section_image, enhance_section_crop  # noqa: E402
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
) -> None:
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except OSError:
        font = ImageFont.load_default()

    for section in sections:
        bounds = section.get("bounds") or {}
        x0 = int(bounds.get("min_x", 0))
        y0 = int(bounds.get("min_y", 0))
        x1 = int(bounds.get("max_x", 0))
        y1 = int(bounds.get("max_y", 0))
        prep = section.get("preprocess") or {}
        kept = bool(prep.get("kept", True))
        color = (34, 160, 80) if kept else (220, 50, 50)
        draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=4)
        label = f"S{section.get('index', 0)}"
        draw.rectangle([(x0, max(0, y0 - 18)), (x0 + 48, y0)], fill=(*color, 220))
        draw.text((x0 + 4, max(0, y0 - 16)), label, fill=(255, 255, 255), font=font)

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
    for section in annotated:
        idx = int(section.get("index", 0))
        prep = section.get("preprocess") or {}
        kept = bool(prep.get("kept", True))
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
        field_styles: dict[str, bool | None] = {}

        fields: dict[str, Any] | None = None
        if kept and not skip_llm:
            enhanced = enhance_section_crop(crop_img)
            fields = extractor.extract_section(
                enhanced,
                ocr_text,
                section_words=section_words,
                bounds=bounds,
                table_band=bool(
                    section.get("table_band")
                    or section.get("layout_kind") in ("table", "section_table")
                ),
            )
            if not fields:
                warnings.append(f"page {page_index} section {idx}: low_signal (0 fields extracted)")
            else:
                for key, value in fields.items():
                    if key in ("line_items", "table_columns"):
                        continue
                    if isinstance(value, str):
                        field_styles[key] = value_is_bold(value, word_styles)
                _merge_fields(page_fields, fields, warnings, page=page_index, section=idx)
                for key, bold in field_styles.items():
                    if bold is not None:
                        page_field_styles[key] = bold

        section_rows.append(
            {
                "index": idx,
                "label": label,
                "bounds": bounds,
                "kept": kept,
                "filter_reason": None if kept else prep.get("reason"),
                "ocr_preview": ocr_text[:400],
                "has_bold": section_has_bold(word_styles),
                "bold_tokens": bold_text_in_section(word_styles),
                "word_styles": styles_to_dicts(word_styles),
                "fields": fields,
                "field_styles": field_styles or None,
                "crop_path": str(crop_path.relative_to(page_dir.parent.parent)),
            }
        )

    step("overlay")
    overlay_path = page_dir / "overlay.png"
    draw_filter_overlay(page_png, annotated, overlay_path)

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
