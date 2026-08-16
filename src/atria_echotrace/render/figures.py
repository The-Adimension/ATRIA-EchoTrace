"""Publication-ready comparison figures.

Ports of the notebook's three matplotlib figures:

* :func:`ground_truth_figure` — 2-panel "Original | polygon overlay"
  (``visualize_polygon_on_image``, notebook_as_py.txt L301-332)
* :func:`prediction_figure` — 3-panel "Original | Ground Truth | Prediction (Dice)"
  (``visualize_prediction_echocardiographic_frame``, L1090-1130)
* :func:`four_panel_figure` — 4-panel "Original | Model | User | Overlay"
  (``save_polygon_backend``, L1346-1370)

The dark palette matches the author's deployed Space (facecolor ``#0F172A``) so
exported figures are visually consistent with previously published ATRIA outputs.
Matplotlib is pinned to the ``Agg`` backend for headless server rendering.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Must precede pyplot import; server has no display.
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from ..domain.geometry import Polygon  # noqa: E402
from ..domain.structures import (  # noqa: E402
    COLOR_GROUND_TRUTH,
    COLOR_MODEL,
    COLOR_USER,
)
from .overlays import draw_polygon_on_image, draw_polygons_on_image  # noqa: E402

_BACKGROUND = "#0F172A"
_FOREGROUND = "#F8FAFC"
_DPI = 150


def _render(fig) -> bytes:
    """Serialise a figure to PNG bytes and close it."""
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=_DPI,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _panels(
    images: list[Image.Image],
    titles: list[str],
    title_colors: list[str],
    suptitle: str | None,
    panel_width: float = 4.6,
) -> bytes:
    """Lay out a single row of image panels with coloured titles."""
    count = len(images)
    fig, axes = plt.subplots(
        1,
        count,
        figsize=(panel_width * count, panel_width * 1.15),
        facecolor=_BACKGROUND,
    )
    if count == 1:
        axes = [axes]
    for axis, image, title, color in zip(axes, images, titles, title_colors):
        axis.imshow(image)
        axis.set_title(title, color=color, fontsize=11, fontweight="bold", pad=8)
        axis.axis("off")
    if suptitle:
        fig.suptitle(suptitle, color=_FOREGROUND, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _render(fig)


def ground_truth_figure(
    image: Image.Image,
    polygon: Polygon,
    structure_short: str,
    title: str = "",
) -> bytes:
    """2-panel ground-truth overlay (notebook L301-332).

    The notebook drew the polygon in green with red vertex markers; both are kept.
    """
    overlay = draw_polygon_on_image(
        image, polygon, (0, 255, 0, 255), show_vertices=True, vertex_radius=2
    )
    return _panels(
        images=[image.convert("RGB"), overlay],
        titles=["Original Frame", f"{structure_short} ({len(polygon or [])} pts)"],
        title_colors=[_FOREGROUND, "#22C55E"],
        suptitle=title or None,
    )


def prediction_figure(
    image: Image.Image,
    predicted: Polygon,
    ground_truth: Polygon,
    structure_short: str,
    title: str = "",
    dice: float | None = None,
) -> bytes:
    """3-panel prediction-vs-ground-truth figure (notebook L1090-1130)."""
    gt_panel = draw_polygon_on_image(image, ground_truth, (0, 255, 0, 255))
    pred_panel = draw_polygon_on_image(image, predicted, (255, 0, 0, 255))
    dice_suffix = f" (Dice: {dice:.3f})" if dice is not None else ""
    return _panels(
        images=[image.convert("RGB"), gt_panel, pred_panel],
        titles=[
            "Original Image",
            f"Ground Truth ({structure_short})",
            f"Prediction ({structure_short}){dice_suffix}",
        ],
        title_colors=[_FOREGROUND, "#22C55E", "#EF4444"],
        suptitle=title or None,
    )


def four_panel_figure(
    image: Image.Image,
    model_polygon: Polygon,
    user_polygon: Polygon,
    phase_label: str = "Frame",
    case_label: str = "",
) -> bytes:
    """4-panel HITL comparison figure (notebook L1346-1370).

    Panels: Original | Model Prediction (red) | User Revision (green) |
    Overlay (both, model beneath). This is the figure the notebook's
    ``save_polygon_backend`` wrote as ``tracing_vis_img{i}.png``.
    """
    model_panel = draw_polygon_on_image(image, model_polygon, COLOR_MODEL)
    user_panel = draw_polygon_on_image(image, user_polygon, COLOR_USER)
    overlay_panel = draw_polygons_on_image(
        image, [(model_polygon, COLOR_MODEL), (user_polygon, COLOR_USER)]
    )
    suptitle = (
        f"ATRIA EchoTrace — {case_label} ({phase_label})" if case_label else "ATRIA EchoTrace"
    )
    return _panels(
        images=[image.convert("RGB"), model_panel, user_panel, overlay_panel],
        titles=[
            f"Original {phase_label}",
            f"Model Prediction ({len(model_polygon or [])} pts)",
            f"User Revision ({len(user_polygon or [])} pts)",
            "Overlay (Red=Model, Green=User)",
        ],
        title_colors=[_FOREGROUND, "#EF4444", "#22C55E", "#38BDF8"],
        suptitle=suptitle,
    )


def comparison_with_ground_truth_figure(
    image: Image.Image,
    model_polygon: Polygon,
    user_polygon: Polygon,
    ground_truth: Polygon,
    phase_label: str = "Frame",
    case_label: str = "",
) -> bytes:
    """4-panel figure that substitutes ground truth for the plain original.

    Offered for dataset cases, where a reference trace exists and is more
    informative than a second copy of the unannotated frame.
    """
    gt_panel = draw_polygon_on_image(image, ground_truth, COLOR_GROUND_TRUTH)
    model_panel = draw_polygon_on_image(image, model_polygon, COLOR_MODEL)
    user_panel = draw_polygon_on_image(image, user_polygon, COLOR_USER)
    overlay_panel = draw_polygons_on_image(
        image,
        [
            (ground_truth, COLOR_GROUND_TRUTH),
            (model_polygon, COLOR_MODEL),
            (user_polygon, COLOR_USER),
        ],
    )
    suptitle = (
        f"ATRIA EchoTrace — {case_label} ({phase_label})" if case_label else "ATRIA EchoTrace"
    )
    return _panels(
        images=[gt_panel, model_panel, user_panel, overlay_panel],
        titles=[
            f"Ground Truth ({len(ground_truth or [])} pts)",
            f"Model Prediction ({len(model_polygon or [])} pts)",
            f"User Revision ({len(user_polygon or [])} pts)",
            "Overlay (Blue=GT, Red=Model, Green=User)",
        ],
        title_colors=["#38BDF8", "#EF4444", "#22C55E", _FOREGROUND],
        suptitle=suptitle,
    )


def write_png(data: bytes, path: Path) -> Path:
    """Write PNG bytes to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
