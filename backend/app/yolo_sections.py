"""Run section-bound detection for the YOLO sections API."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.paths import ensure_script_path
from app.yolo_bounds import build_page_yolo_bounds, entries_to_detections

ensure_script_path()

from form_box_detection import FormBox, detect_high_confidence_sections  # noqa: E402


def yolo_model_path() -> Path | None:
    raw = (os.environ.get("YOLO_SECTION_MODEL") or os.environ.get("DOC_EXTRACT_YOLO_MODEL") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def _form_box_to_detection(index: int, box: FormBox, image_width: int, image_height: int) -> dict[str, Any]:
    iw = max(1, image_width)
    ih = max(1, image_height)
    x1 = float(box.x)
    y1 = float(box.y)
    x2 = float(box.x2)
    y2 = float(box.y2)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return {
        "index": index,
        "class": "section",
        "confidence": float(box.confidence),
        "bounds": {"min_x": x1, "min_y": y1, "max_x": x2, "max_y": y2},
        "bbox": [x1, y1, x2, y2],
        "bbox_yolo": [
            (x1 + x2) / 2.0 / iw,
            (y1 + y2) / 2.0 / ih,
            w / iw,
            h / ih,
        ],
        "label": f"section_{index}",
    }


def detect_opencv_sections(image_bgr: np.ndarray, *, min_confidence: float = 0.9) -> list[dict[str, Any]]:
    height, width = image_bgr.shape[:2]
    boxes = detect_high_confidence_sections(image_bgr, min_confidence=min_confidence)
    return [
        _form_box_to_detection(i, box, width, height)
        for i, box in enumerate(boxes)
    ]


def detect_yolo_sections(image_bgr: np.ndarray, model_path: Path) -> list[dict[str, Any]] | None:
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    height, width = image_bgr.shape[:2]
    model = YOLO(str(model_path))
    results = model.predict(image_bgr, verbose=False)
    detections: list[dict[str, Any]] = []
    for result in results:
        names = result.names or {}
        boxes = result.boxes
        if boxes is None:
            continue
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            cls_id = int(box.cls[0]) if box.cls is not None else 0
            class_name = str(names.get(cls_id, "section"))
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            detections.append(
                {
                    "index": i,
                    "class": class_name,
                    "confidence": conf,
                    "bounds": {"min_x": x1, "min_y": y1, "max_x": x2, "max_y": y2},
                    "bbox": [x1, y1, x2, y2],
                    "bbox_yolo": [
                        (x1 + x2) / 2.0 / max(1, width),
                        (y1 + y2) / 2.0 / max(1, height),
                        w / max(1, width),
                        h / max(1, height),
                    ],
                    "label": class_name,
                }
            )
    detections.sort(key=lambda item: (item["bounds"]["min_y"], item["bounds"]["min_x"]))
    for idx, item in enumerate(detections):
        item["index"] = idx
    return detections


def detect_sections_from_image_bytes(data: bytes) -> dict[str, Any]:
    with Image.open(io.BytesIO(data)) as img:
        rgb = img.convert("RGB")
        image_width, image_height = rgb.size

    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    model_path = yolo_model_path()
    source = "opencv"
    detections: list[dict[str, Any]]

    if model_path is not None:
        yolo_detections = detect_yolo_sections(bgr, model_path)
        if yolo_detections is not None:
            detections = yolo_detections
            source = "yolo"
        else:
            detections = detect_opencv_sections(bgr)
            source = "opencv_fallback"
    else:
        detections = detect_opencv_sections(bgr)

    return {
        "image_width": image_width,
        "image_height": image_height,
        "bbox_format": "xyxy_pixels",
        "yolo_format": "normalized_cxcywh",
        "source": source,
        "model_path": str(model_path) if model_path and source == "yolo" else None,
        "sections": detections,
        "detections": entries_to_detections(detections),
    }


def detect_sections_from_job(
    *,
    job_id: str,
    page_index: int,
    sections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    payload = build_page_yolo_bounds(
        sections,
        job_id=job_id,
        page_index=page_index,
        image_width=image_width,
        image_height=image_height,
    )
    payload["detections"] = entries_to_detections(payload.get("sections") or [])
    return payload
