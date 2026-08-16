"""Polygon geometry, coordinate conversion, model-output parsing and overlap metrics.

Faithful port of the notebook's evaluation utilities
(notebook_as_py.txt L919-990: ``parse_polygon_from_response``, ``polygon_to_mask``,
``compute_dice``, ``compute_iou``) and the HITL cell's ``parse_polygon`` (L1222-1238).

Coordinate conventions used throughout ATRIA EchoTrace
------------------------------------------------------
* **Normalised** polygons are ``[[y, x], ...]`` with both axes scaled to
  ``[0, NORM_SCALE]``. This is the model's input/output format and the on-disk
  format in ``tracings.json``.
* **Pixel** polygons are ``[[x, y], ...]`` in image pixels. This is what canvases,
  PIL and matplotlib expect.

The two published adapters emit *different* polygon conventions, because their
training data did (measured across all 50 sample frames — see RESEARCH.md §0.5):
CAMUS traces are explicitly closed (``p[0] == p[-1]``) and wind one way, EchoNet
traces are open and wind the other. Nothing here may therefore assume a point
count, closure, or winding direction.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence

import numpy as np

from .structures import NORM_SCALE, resolve_structure

Point = Sequence[float]
Polygon = Sequence[Point]

# Matches the ```json ... ``` fence the prompt instructs the model to emit.
_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
# Locates the polygon key when the fence is absent or the JSON is truncated.
_POLYGON_KEY_RE = re.compile(r'"polygon_2d"\s*:\s*')
# MedGemma emits a thinking trace before this sentinel; the notebook strips it.
_THINKING_SENTINEL = "<unused95>"


def _extract_balanced_list(text: str, start: int) -> str | None:
    """Return the bracket-balanced substring beginning at ``text[start] == '['``."""
    if start >= len(text) or text[start] != "[":
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


# --------------------------------------------------------------------------- #
# Model output parsing
# --------------------------------------------------------------------------- #
def parse_polygon(response: str, target_structure: str = "LV") -> list[list[float]] | None:
    """Extract the target structure's normalised polygon from a model response.

    Port of the notebook's ``parse_polygon_from_response`` (L924-950), plus the
    bare-key regex fallback added by the author's Space.

    Args:
        response: Raw decoded model text.
        target_structure: ``"LV"`` or ``"LA"``; selects which labelled object to take.

    Returns:
        The polygon as ``[[y, x], ...]``, or ``None`` when nothing parseable is found.
    """
    if not response:
        return None

    struct = resolve_structure(target_structure)
    target_label = struct["label"].lower().replace("_", " ")

    if _THINKING_SENTINEL in response:
        response = response.split(_THINKING_SENTINEL, 1)[1].lstrip()

    match = _JSON_FENCE_RE.search(response)
    if match:
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            # Prefer an object whose label matches the requested structure.
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                label = str(obj.get("label", "")).lower().replace("_", " ")
                if target_label in label:
                    return obj.get("polygon_2d")
            # Fallback from the notebook: a lone object is assumed to be the target.
            if len(data) == 1 and isinstance(data[0], dict) and "polygon_2d" in data[0]:
                return data[0]["polygon_2d"]

    # Fence absent, unparseable, or truncated mid-generation: recover the first
    # bracket-balanced value following a "polygon_2d" key.
    #
    # The author's Space uses the regex '"polygon_2d":\s*(\[[^\]]+\]\])' here, but
    # `[^\]]+` cannot cross the inner `]` of a coordinate pair, so it never matches a
    # polygon with more than one vertex. A balanced scan is used instead, which
    # handles the real multi-vertex case the rule was meant to cover.
    for match in _POLYGON_KEY_RE.finditer(response):
        candidate = _extract_balanced_list(response, match.end())
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed

    return None


def sanitize_polygon(
    polygon: Any,
    norm_scale: int = NORM_SCALE,
) -> list[list[int]] | None:
    """Validate and clamp a parsed polygon into well-formed normalised integer points.

    A generative model can emit malformed coordinates, so parsed output is never
    trusted directly. Points that are not numeric pairs are dropped; surviving
    coordinates are clamped to ``[0, norm_scale]``.

    Returns:
        Cleaned ``[[y, x], ...]``, or ``None`` if fewer than 3 valid points remain
        (the minimum for an area-bearing polygon).
    """
    if not isinstance(polygon, (list, tuple)):
        return None

    cleaned: list[list[int]] = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            y = float(point[0])
            x = float(point[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(y) and math.isfinite(x)):
            continue
        cleaned.append(
            [
                int(round(min(max(y, 0.0), float(norm_scale)))),
                int(round(min(max(x, 0.0), float(norm_scale)))),
            ]
        )

    return cleaned if len(cleaned) >= 3 else None


# --------------------------------------------------------------------------- #
# Coordinate conversion
# --------------------------------------------------------------------------- #
def denormalize_polygon(
    polygon: Polygon,
    image_h: int,
    image_w: int,
    norm_scale: int = NORM_SCALE,
) -> list[list[float]]:
    """Convert normalised ``[[y, x], ...]`` to pixel ``[[x, y], ...]``.

    Mirrors the notebook's inline conversion
    ``(p[1] / NORM_SCALE * img_w, p[0] / NORM_SCALE * img_h)`` (L320, L1096, L1246).
    """
    if not polygon or image_h <= 0 or image_w <= 0:
        return []
    scale = float(norm_scale)
    return [
        [round(float(p[1]) / scale * image_w, 2), round(float(p[0]) / scale * image_h, 2)]
        for p in polygon
    ]


def normalize_polygon(
    polygon: Polygon,
    image_h: int,
    image_w: int,
    norm_scale: int = NORM_SCALE,
) -> list[list[int]]:
    """Convert pixel ``[[x, y], ...]`` to normalised integer ``[[y, x], ...]``.

    Mirrors the notebook UI's ``getNormPoly`` (L1550-1553), which rounds to integers.
    """
    if not polygon or image_h <= 0 or image_w <= 0:
        return []
    scale = float(norm_scale)
    out: list[list[int]] = []
    for p in polygon:
        y_norm = int(round(float(p[1]) / image_h * scale))
        x_norm = int(round(float(p[0]) / image_w * scale))
        out.append(
            [
                min(max(y_norm, 0), norm_scale),
                min(max(x_norm, 0), norm_scale),
            ]
        )
    return out


# --------------------------------------------------------------------------- #
# Area and length
# --------------------------------------------------------------------------- #
def shoelace_area(polygon: Polygon) -> float:
    """Absolute polygon area via the shoelace formula.

    The result is invariant to winding direction and to axis order (swapping the
    two coordinates negates the signed area, which ``abs`` removes), so this works
    for both ``[y, x]`` and ``[x, y]`` input provided the order is consistent
    within one polygon. Wrap-around (``i -> (i + 1) % n``) closes open contours
    implicitly and contributes zero for a duplicated closing vertex — both cases
    occur in the real data (RESEARCH.md §0.5).

    Returns:
        Area in squared units of the input coordinates; ``0.0`` for < 3 points.
    """
    if not polygon or len(polygon) < 3:
        return 0.0
    n = len(polygon)
    total = 0.0
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        total += float(a[0]) * float(b[1]) - float(b[0]) * float(a[1])
    return abs(total) / 2.0


def polygon_perimeter(polygon: Polygon) -> float:
    """Closed-contour perimeter in units of the input coordinates."""
    if not polygon or len(polygon) < 2:
        return 0.0
    n = len(polygon)
    return float(
        sum(
            math.dist(
                (float(polygon[i][0]), float(polygon[i][1])),
                (float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])),
            )
            for i in range(n)
        )
    )


# --------------------------------------------------------------------------- #
# Mask rasterisation and overlap metrics (notebook L953-987)
# --------------------------------------------------------------------------- #
def polygon_to_mask(polygon: Polygon, image_h: int, image_w: int) -> np.ndarray:
    """Rasterise a normalised polygon to a binary mask at native resolution.

    Verbatim port of the notebook's ``polygon_to_mask`` (L953-971).
    """
    import skimage.draw

    if not polygon or len(polygon) < 3:
        return np.zeros((image_h, image_w), dtype=np.float32)

    x_coords = np.array([float(p[1]) / NORM_SCALE * image_w for p in polygon])
    y_coords = np.array([float(p[0]) / NORM_SCALE * image_h for p in polygon])

    rr, cc = skimage.draw.polygon(
        y_coords.astype(int),
        x_coords.astype(int),
        (image_h, image_w),
    )

    mask = np.zeros((image_h, image_w), dtype=np.float32)
    mask[rr, cc] = 1
    return mask


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient between two binary masks (notebook L974-980)."""
    intersection = float(np.sum(pred_mask * gt_mask))
    union = float(np.sum(pred_mask) + np.sum(gt_mask))
    if union == 0:
        return 1.0 if float(np.sum(pred_mask)) == 0 else 0.0
    return 2.0 * intersection / union


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection-over-union between two binary masks (notebook L983-987)."""
    intersection = float(np.sum(pred_mask * gt_mask))
    union = float(np.sum(np.logical_or(pred_mask, gt_mask)))
    return intersection / union if union > 0 else 1.0


def polygon_dice(
    polygon_a: Polygon,
    polygon_b: Polygon,
    image_h: int,
    image_w: int,
) -> float:
    """Dice between two normalised polygons, rasterised at the given resolution."""
    return compute_dice(
        polygon_to_mask(polygon_a, image_h, image_w),
        polygon_to_mask(polygon_b, image_h, image_w),
    )


def polygon_iou(
    polygon_a: Polygon,
    polygon_b: Polygon,
    image_h: int,
    image_w: int,
) -> float:
    """IoU between two normalised polygons, rasterised at the given resolution."""
    return compute_iou(
        polygon_to_mask(polygon_a, image_h, image_w),
        polygon_to_mask(polygon_b, image_h, image_w),
    )
