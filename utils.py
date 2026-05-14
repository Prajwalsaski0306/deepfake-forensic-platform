"""
utils.py — Image utility helpers for DeepFake Forensics Lab.
Handles validation, resizing, format conversion, and metadata extraction.
NO Streamlit imports. Pure computation only.
"""

import io
import numpy as np
from PIL import Image, ImageFile

# ── Constants ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024          # 5 MB
MAX_IMAGE_WIDTH: int = 640                           # pixels
ALLOWED_FORMATS: tuple[str, ...] = ("JPEG", "JPG", "PNG")
ALLOWED_MIME_TYPES: tuple[str, ...] = ("image/jpeg", "image/png")

ImageFile.LOAD_TRUNCATED_IMAGES = True


def validate_upload(file_bytes: bytes, filename: str) -> dict:
    """
    Validate an uploaded file for format and size constraints.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename as a string.

    Returns:
        dict with keys: 'valid' (bool), 'error' (str | None).
    """
    # Size check
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return {
            "valid": False,
            "error": "[ SIZE LIMIT ] — Max 5MB. Compress image first.",
        }

    # Format check via extension
    ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else ""
    if ext not in ("JPG", "JPEG", "PNG"):
        return {
            "valid": False,
            "error": "[ FORMAT ERROR ] — Only JPG/PNG accepted.",
        }

    # Integrity check — try to open with Pillow
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()          # raises if corrupted
    except Exception:
        return {
            "valid": False,
            "error": "[ READ ERROR ] — File appears corrupted.",
        }

    return {"valid": True, "error": None}


def load_image(file_bytes: bytes) -> Image.Image:
    """
    Load raw bytes into a PIL Image and convert to RGB.

    Args:
        file_bytes: Raw bytes of the image file.

    Returns:
        PIL Image in RGB mode.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img


def resize_image(img: Image.Image, max_width: int = MAX_IMAGE_WIDTH) -> Image.Image:
    """
    Resize image so width ≤ max_width, maintaining aspect ratio.

    Args:
        img:       PIL Image to resize.
        max_width: Maximum allowed width in pixels.

    Returns:
        Resized PIL Image (or original if already within bounds).
    """
    w, h = img.size
    if w <= max_width:
        return img
    ratio = max_width / w
    new_size = (max_width, int(h * ratio))
    return img.resize(new_size, Image.LANCZOS)


def get_image_metadata(file_bytes: bytes, filename: str) -> dict:
    """
    Extract metadata from an image file.

    Args:
        file_bytes: Raw bytes of the image.
        filename:   Original filename string.

    Returns:
        dict with keys: filename, resolution, file_size, color_mode, format.
    """
    img = Image.open(io.BytesIO(file_bytes))
    w, h = img.size
    size_kb = len(file_bytes) / 1024
    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"

    return {
        "filename": filename,
        "resolution": f"{w} × {h}",
        "file_size": size_str,
        "color_mode": img.mode,
        "format": img.format if img.format else filename.rsplit(".", 1)[-1].upper(),
    }


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to NumPy uint8 array (H, W, C) in RGB order.

    Args:
        img: PIL Image.

    Returns:
        NumPy ndarray.
    """
    return np.array(img.convert("RGB"), dtype=np.uint8)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert NumPy array (H, W, C) to PIL Image.

    Args:
        arr: NumPy ndarray in uint8 range [0, 255].

    Returns:
        PIL Image in RGB mode.
    """
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """
    Serialize a PIL Image to bytes in the specified format.

    Args:
        img: PIL Image to serialize.
        fmt: Output format string (e.g. 'PNG', 'JPEG').

    Returns:
        bytes of the encoded image.
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def generate_scan_id() -> str:
    """
    Generate a pseudo-random forensics scan identifier.

    Returns:
        A string like 'SCAN-7F3A2C91'.
    """
    import random
    hex_chars = "0123456789ABCDEF"
    uid = "".join(random.choices(hex_chars, k=8))
    return f"SCAN-{uid}"
