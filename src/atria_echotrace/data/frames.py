"""Frame image loading and clinician upload handling.

``load_frame`` is a port of the notebook's loader (notebook_as_py.txt L284-298),
which opened the PNG and converted to RGB. Uploads are the production form of the
HITL cell's base64 file input (L1285-1291): the browser posts the image file and the
server validates it before use.
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ..config import display_path
from ..logging_setup import get_logger

logger = get_logger("data.frames")

#: Upload ids are server-generated hex; this also gates path construction.
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{16}$")

#: Formats accepted for clinician-supplied frames. Echo frames arrive as stills.
ALLOWED_UPLOAD_FORMATS: frozenset[str] = frozenset({"PNG", "JPEG", "BMP", "TIFF"})


class UploadError(ValueError):
    """Raised when an uploaded image is unusable or exceeds configured limits."""


def load_frame(path: Path) -> Image.Image:
    """Load an image file as RGB.

    Port of the notebook's ``load_frame`` (L284-298), which raised
    ``FileNotFoundError`` for a missing frame and converted non-RGB modes.

    Raises:
        FileNotFoundError: if the file does not exist.
        UploadError: if the file is not a decodable image.
    """
    if not path.is_file():
        # Project-relative: load failures surface to HTTP clients.
        raise FileNotFoundError(f"Frame not found: {display_path(path)}")
    try:
        with Image.open(path) as img:
            img.load()
            return img.convert("RGB") if img.mode != "RGB" else img.copy()
    except UnidentifiedImageError as exc:
        raise UploadError(f"Not a decodable image: {path.name}") from exc


def save_upload(
    data: bytes,
    uploads_dir: Path,
    max_bytes: int,
    max_pixels: int,
) -> tuple[str, Image.Image, Path]:
    """Validate and store a clinician-supplied frame.

    Validation order matters: the byte cap is checked before decoding so an
    oversized payload is rejected without being expanded in memory, and the pixel
    cap is checked from the header before ``load()`` to refuse decompression-bomb
    dimensions.

    Args:
        data: Raw uploaded bytes.
        uploads_dir: Directory to write into (created if absent).
        max_bytes: Reject payloads larger than this.
        max_pixels: Reject images with more than this many pixels.

    Returns:
        ``(upload_id, rgb_image, saved_png_path)``.

    Raises:
        UploadError: on empty, oversized, undecodable or unsupported input.
    """
    if not data:
        raise UploadError("Uploaded file is empty.")
    if len(data) > max_bytes:
        raise UploadError(
            f"Uploaded file is {len(data) / 1e6:.1f} MB, which exceeds the "
            f"{max_bytes / 1e6:.1f} MB limit."
        )

    import io

    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = (probe.format or "").upper()
            width, height = probe.size
            if fmt not in ALLOWED_UPLOAD_FORMATS:
                raise UploadError(
                    f"Unsupported image format {fmt or 'unknown'}. "
                    f"Expected one of {', '.join(sorted(ALLOWED_UPLOAD_FORMATS))}."
                )
            if width <= 0 or height <= 0:
                raise UploadError("Uploaded image has zero size.")
            if width * height > max_pixels:
                raise UploadError(
                    f"Uploaded image is {width}x{height} "
                    f"({width * height / 1e6:.1f} MP), exceeding the "
                    f"{max_pixels / 1e6:.1f} MP limit."
                )
            probe.load()
            image = probe.convert("RGB") if probe.mode != "RGB" else probe.copy()
    except UnidentifiedImageError as exc:
        raise UploadError("Uploaded file could not be decoded as an image.") from exc
    except OSError as exc:
        raise UploadError(f"Uploaded image is truncated or corrupt: {exc}") from exc

    upload_id = secrets.token_hex(8)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target = uploads_dir / f"{upload_id}.png"
    # Re-encoded as PNG: normalises the format and drops any source metadata,
    # which is also the safer choice for potentially identifiable EXIF content.
    image.save(target, format="PNG")
    logger.info("Stored upload %s (%dx%d) at %s", upload_id, image.width, image.height, target)
    return upload_id, image, target


def upload_path(upload_id: str, uploads_dir: Path) -> Path:
    """Resolve a stored upload by id.

    Raises:
        UploadError: if the id is malformed (guards path construction).
        FileNotFoundError: if no such upload exists.
    """
    if not _UPLOAD_ID_RE.match(upload_id):
        raise UploadError(f"Malformed upload id: {upload_id!r}")
    path = (uploads_dir / f"{upload_id}.png").resolve()
    if not path.is_relative_to(uploads_dir.resolve()):
        raise UploadError(f"Upload id escapes the uploads directory: {upload_id!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Unknown upload id: {upload_id}")
    return path
