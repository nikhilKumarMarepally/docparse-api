"""Convert production pipeline section bounds to YOLO-compatible export format."""

from __future__ import annotations

from typing import Any


def section_to_yolo_entry(
    section: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    class_name: str = "section",
) -> dict[str, Any]:
    bounds = section.get("bounds") or {}
    x1 = float(bounds.get("min_x", 0))
    y1 = float(bounds.get("min_y", 0))
    x2 = float(bounds.get("max_x", 0))
    y2 = float(bounds.get("max_y", 0))

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    iw = max(1, image_width)
    ih = max(1, image_height)

    return {
        "class": class_name,
        "bbox": [x1, y1, x2, y2],
        "bounds": {"min_x": x1, "min_y": y1, "max_x": x2, "max_y": y2},
        "bbox_yolo": [
            (x1 + x2) / 2.0 / iw,
            (y1 + y2) / 2.0 / ih,
            w / iw,
            h / ih,
        ],
        "confidence": 1.0,
        "index": int(section.get("index", 0)),
        "label": section.get("label") or f"section_{int(section.get('index', 0))}",
    }


def entry_to_detection(entry: dict[str, Any]) -> dict[str, Any]:
    bounds = entry.get("bounds")
    if not bounds and entry.get("bbox"):
        x1, y1, x2, y2 = entry["bbox"]
        bounds = {"min_x": x1, "min_y": y1, "max_x": x2, "max_y": y2}
    return {
        "index": int(entry.get("index", 0)),
        "class": entry.get("class") or "section",
        "confidence": float(entry.get("confidence", 1.0)),
        "bounds": bounds or {},
        "label": entry.get("label") or entry.get("class") or "section",
        "bbox_yolo": entry.get("bbox_yolo"),
    }


def entries_to_detections(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry_to_detection(entry) for entry in entries if entry.get("bounds") or entry.get("bbox")]


def build_page_yolo_bounds(
    sections: list[dict[str, Any]],
    *,
    job_id: str,
    page_index: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    with_bounds = [s for s in sections if (s.get("bounds") or {})]
    entries = [
        section_to_yolo_entry(section, image_width, image_height)
        for section in with_bounds
    ]
    entries.sort(key=lambda item: item["index"])
    return {
        "job_id": job_id,
        "page_index": page_index,
        "image_width": image_width,
        "image_height": image_height,
        "bbox_format": "xyxy_pixels",
        "yolo_format": "normalized_cxcywh",
        "source": "production_pipeline",
        "sections": entries,
        "detections": entries_to_detections(entries),
    }
