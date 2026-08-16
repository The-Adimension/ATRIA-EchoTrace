"""Polygon overlay rendering with PIL.

Port of the notebook's overlay helpers: ``visualize_polygon_on_image``'s drawing
block (notebook_as_py.txt L316-325) and the HITL cell's ``draw_polygon_on_image``
(L1240-1248), plus the crisper edge stroking used by the author's Space.

All functions take **normalised** ``[[y, x], ...]`` polygons, matching the on-disk
and model formats. The previous implementation guessed whether input was normalised
or pixel space by comparing magnitudes against the image size, which silently
mis-renders whenever a pixel coordinate happens to fall below the normalisation
scale; requiring one explicit convention removes that failure mode entirely.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from ..domain.geometry import Polygon
from ..domain.structures import NORM_SCALE

RGBA = tuple[int, int, int, int]

#: Translucent fill alpha used by the notebook and the Space.
FILL_ALPHA = 60


def to_pixel_points(
    polygon: Polygon,
    image_w: int,
    image_h: int,
    norm_scale: int = NORM_SCALE,
) -> list[tuple[float, float]]:
    """Convert a normalised polygon to PIL ``(x, y)`` tuples.

    Mirrors the notebook's ``(p[1] / NORM_SCALE * w, p[0] / NORM_SCALE * h)``.
    """
    scale = float(norm_scale)
    return [
        (float(p[1]) / scale * image_w, float(p[0]) / scale * image_h) for p in polygon
    ]


def draw_polygon_on_image(
    image: Image.Image,
    polygon: Polygon,
    color: RGBA,
    line_width: int = 2,
    show_vertices: bool = False,
    vertex_radius: int = 3,
    norm_scale: int = NORM_SCALE,
) -> Image.Image:
    """Return a copy of ``image`` with ``polygon`` drawn over it.

    Args:
        image: Base frame.
        polygon: Normalised ``[[y, x], ...]``. Fewer than 3 points draws nothing.
        color: RGBA outline colour; the fill reuses the RGB at ``FILL_ALPHA``.
        line_width: Edge stroke width in pixels.
        show_vertices: Draw a marker at each vertex (the notebook does this for
            ground-truth figures, but not in the 4-panel comparison).
        vertex_radius: Marker radius when ``show_vertices`` is set.

    Returns:
        A new RGB image; the input is never mutated.
    """
    base = image.convert("RGB")
    if not polygon or len(polygon) < 3:
        return base

    points = to_pixel_points(polygon, base.width, base.height, norm_scale)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    outline = color[:3] + (255,)
    draw.polygon(points, outline=outline, fill=color[:3] + (FILL_ALPHA,))

    # Re-stroke each edge: PIL's polygon outline is always 1px wide.
    for index in range(len(points)):
        draw.line(
            [points[index], points[(index + 1) % len(points)]],
            fill=outline,
            width=line_width,
        )

    if show_vertices:
        for x, y in points:
            draw.ellipse(
                [x - vertex_radius, y - vertex_radius, x + vertex_radius, y + vertex_radius],
                fill=outline,
                outline=(255, 255, 255, 255),
            )

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def draw_polygons_on_image(
    image: Image.Image,
    layers: list[tuple[Polygon, RGBA]],
    line_width: int = 2,
    norm_scale: int = NORM_SCALE,
) -> Image.Image:
    """Composite several polygons in order, last on top.

    Used for the 4-panel overlay pane (model beneath clinician revision).
    """
    result = image.convert("RGB")
    for polygon, color in layers:
        result = draw_polygon_on_image(
            result, polygon, color, line_width=line_width, norm_scale=norm_scale
        )
    return result
