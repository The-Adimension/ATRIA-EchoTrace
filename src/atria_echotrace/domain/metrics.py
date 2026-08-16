"""Clinical chamber metrics: areas, perimeters and Fractional Area Change.

Provenance: the author's deployed Space computes EDA / ESA / FAC% / perimeters
(``model_engine.calculate_anatomical_metrics``); this module reproduces those
quantities. The notebook itself produces polygons only, and its disclaimer II
explicitly frames derived metrics ("ejection fraction, cardiac volume...") as
research-only outputs.

One deliberate divergence from both prior implementations, for clinical safety
(RESEARCH.md §3.2): **physical areas are reported only when pixel spacing is
actually known.** EchoNet-Dynamic publishes no pixel spacing and carries a
``spacing = 1.0`` sentinel in ``tracings.json``; multiplying by it yields a
confident-looking cm² number that is not a measurement. Such cases return
``area_cm2 = None`` with ``source = "unknown"`` so the UI can say "not calibrated".
FAC% is a dimensionless ratio and stays valid regardless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .geometry import Polygon, denormalize_polygon, polygon_perimeter, shoelace_area

CalibrationSource = Literal["dataset", "user", "unknown"]

#: mm² per cm².
_MM2_PER_CM2 = 100.0
#: mm per cm.
_MM_PER_CM = 10.0


@dataclass(frozen=True)
class Calibration:
    """Pixel spacing in mm/px, together with where it came from.

    ``source="unknown"`` means the dataset provides no real calibration; physical
    units are then withheld rather than fabricated.
    """

    spacing_h: float | None = None
    spacing_w: float | None = None
    source: CalibrationSource = "unknown"

    @property
    def is_known(self) -> bool:
        """True when both spacings are present and strictly positive."""
        return (
            self.source != "unknown"
            and self.spacing_h is not None
            and self.spacing_w is not None
            and self.spacing_h > 0
            and self.spacing_w > 0
        )

    @classmethod
    def unknown(cls) -> "Calibration":
        return cls(None, None, "unknown")

    @classmethod
    def from_user(cls, spacing_h: float, spacing_w: float) -> "Calibration":
        """Clinician-supplied spacing (mm/px).

        Raises:
            ValueError: if either spacing is not strictly positive.
        """
        if spacing_h <= 0 or spacing_w <= 0:
            raise ValueError(
                f"pixel spacing must be > 0 mm/px, got h={spacing_h}, w={spacing_w}"
            )
        return cls(float(spacing_h), float(spacing_w), "user")


@dataclass(frozen=True)
class FrameMetrics:
    """Geometry of one traced frame."""

    vertices: int
    area_px: float
    perimeter_px: float
    area_cm2: float | None
    perimeter_cm: float | None
    calibration_source: CalibrationSource

    def as_dict(self) -> dict[str, object]:
        return {
            "vertices": self.vertices,
            "area_px": round(self.area_px, 2),
            "perimeter_px": round(self.perimeter_px, 2),
            "area_cm2": None if self.area_cm2 is None else round(self.area_cm2, 3),
            "perimeter_cm": None if self.perimeter_cm is None else round(self.perimeter_cm, 3),
            "calibration_source": self.calibration_source,
        }


@dataclass(frozen=True)
class ChamberMetrics:
    """Paired ED/ES metrics plus Fractional Area Change."""

    ed: FrameMetrics
    es: FrameMetrics
    fac_percent: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "ed": self.ed.as_dict(),
            "es": self.es.as_dict(),
            "fac_percent": None if self.fac_percent is None else round(self.fac_percent, 2),
            "calibration_source": self.ed.calibration_source,
        }


def frame_metrics(
    polygon: Polygon,
    image_h: int,
    image_w: int,
    calibration: Calibration | None = None,
) -> FrameMetrics:
    """Compute area and perimeter for one normalised polygon.

    Args:
        polygon: Normalised ``[[y, x], ...]`` in ``[0, NORM_SCALE]``.
        image_h: Frame height in pixels.
        image_w: Frame width in pixels.
        calibration: Pixel spacing; when unknown, physical units are omitted.

    Returns:
        A :class:`FrameMetrics`. An empty or degenerate polygon yields zero area
        and perimeter rather than raising, so the UI can render a partially traced
        frame.
    """
    calibration = calibration or Calibration.unknown()
    pixel_polygon = denormalize_polygon(polygon, image_h, image_w)

    area_px = shoelace_area(pixel_polygon)
    perimeter_px = polygon_perimeter(pixel_polygon)

    area_cm2: float | None = None
    perimeter_cm: float | None = None

    if calibration.is_known and pixel_polygon:
        sh = float(calibration.spacing_h)  # type: ignore[arg-type]
        sw = float(calibration.spacing_w)  # type: ignore[arg-type]
        # An x-step scales by spacing_w and a y-step by spacing_h, so the area
        # scale factor is exactly their product even for anisotropic pixels.
        area_cm2 = area_px * sh * sw / _MM2_PER_CM2
        # Perimeter cannot be scaled by a single factor under anisotropy, so it is
        # re-measured on physically scaled coordinates.
        physical = [[p[0] * sw, p[1] * sh] for p in pixel_polygon]
        perimeter_cm = polygon_perimeter(physical) / _MM_PER_CM

    return FrameMetrics(
        vertices=len(polygon) if polygon else 0,
        area_px=area_px,
        perimeter_px=perimeter_px,
        area_cm2=area_cm2,
        perimeter_cm=perimeter_cm,
        calibration_source=calibration.source,
    )


def fractional_area_change(ed_area: float, es_area: float) -> float | None:
    """Fractional Area Change as a percentage.

    ``FAC = (EDA - ESA) / EDA * 100``, matching the author's Space. Dimensionless,
    so it is valid computed from px² and needs no calibration.

    Returns:
        FAC in percent, or ``None`` when ED area is not positive (no valid ED
        trace, so the ratio is undefined — reported as unknown rather than 0.0,
        which would read as "no contraction").
    """
    if ed_area <= 0:
        return None
    return (ed_area - es_area) / ed_area * 100.0


def chamber_metrics(
    ed_polygon: Polygon,
    es_polygon: Polygon,
    image_h: int,
    image_w: int,
    calibration: Calibration | None = None,
    es_image_h: int | None = None,
    es_image_w: int | None = None,
) -> ChamberMetrics:
    """Compute paired ED/ES metrics and FAC%.

    ED and ES frames of one case normally share dimensions, but ``es_image_h`` /
    ``es_image_w`` allow differing sizes (possible for user-uploaded frames).
    """
    ed = frame_metrics(ed_polygon, image_h, image_w, calibration)
    es = frame_metrics(
        es_polygon,
        es_image_h if es_image_h is not None else image_h,
        es_image_w if es_image_w is not None else image_w,
        calibration,
    )

    # FAC requires *both* phases to be traced. Without that guard a half-finished
    # study reports "FAC 100%" — arithmetically true for an ES area of zero, and
    # clinically nonsense. Undefined is reported as undefined.
    both_traced = ed.vertices >= 3 and es.vertices >= 3
    fac = fractional_area_change(ed.area_px, es.area_px) if both_traced else None
    return ChamberMetrics(ed=ed, es=es, fac_percent=fac)
