from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
SUPPORTED_PDF_SUFFIXES = {".pdf"}


def normalize_image(src: Path, dest: Path, *, max_width: int = 1800) -> Path:
    """Convert an image file to PNG, optionally downscaling wide pages."""
    with Image.open(src) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        if w > max_width:
            ratio = max_width / w
            rgb = rgb.resize((max_width, max(1, int(h * ratio))), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(dest, format="PNG")
    return dest


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            return doc.page_count
    except Exception:
        return 1


def _rasterize_pdf_pdftoppm(pdf_path: Path, out_dir: Path, page: int) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("pdftoppm") is None:
        return None
    stem = out_dir / f"page_{page:04d}"
    out_png = out_dir / f"page_{page:04d}.png"
    cmd = [
        "pdftoppm",
        "-singlefile",
        "-png",
        "-r",
        "300",
        "-f",
        str(page),
        "-l",
        str(page),
        str(pdf_path),
        str(stem),
    ]
    subprocess.run(cmd, check=False, capture_output=True)
    return out_png if out_png.exists() else None


def _rasterize_pdf_pymupdf(pdf_path: Path, out_dir: Path, page: int) -> Path:
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"page_{page:04d}.png"
    with fitz.open(pdf_path) as doc:
        pix = doc.load_page(page - 1).get_pixmap(dpi=300)
        pix.save(str(out_png))
    return out_png


def rasterize_pdf(pdf_path: Path, out_dir: Path) -> list[Path]:
    """Rasterize each PDF page to PNG (pdftoppm preferred, pymupdf fallback)."""
    page_count = _pdf_page_count(pdf_path)
    pages: list[Path] = []
    for page in range(1, page_count + 1):
        png = _rasterize_pdf_pdftoppm(pdf_path, out_dir, page)
        if png is None:
            png = _rasterize_pdf_pymupdf(pdf_path, out_dir, page)
        pages.append(normalize_image(png, out_dir / f"page_{page - 1:03d}.png"))
    return pages


def ingest_upload(upload_path: Path, job_dir: Path) -> list[Path]:
    """Return normalized page PNG paths for an uploaded PDF or image."""
    suffix = upload_path.suffix.lower()
    pages_dir = job_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    if suffix in SUPPORTED_PDF_SUFFIXES:
        raw_pages = rasterize_pdf(upload_path, pages_dir / "raw")
        return raw_pages

    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        dest = pages_dir / "page_000.png"
        return [normalize_image(upload_path, dest)]

    raise ValueError(f"Unsupported file type: {suffix}")
