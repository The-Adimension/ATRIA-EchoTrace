"""Clinical metric endpoints: chamber areas, perimeters and Fractional Area Change.

Replaces the previous ``/api/clinical/calculate-fac``. The response now carries an
explicit calibration state, and ``area_cm2`` is ``null`` when pixel spacing is
unknown, rather than reporting a number derived from a placeholder spacing
(RESEARCH.md §3.2).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from ..data.dataset import DatasetRepository
from ..domain.metrics import Calibration, chamber_metrics, frame_metrics
from .deps import repository

router = APIRouter(prefix="/api/clinical", tags=["clinical"])

Polygon = list[list[float]]


class MetricsRequest(BaseModel):
    """Metrics for one traced ED/ES pair.

    Polygons are normalised ``[[y, x], ...]`` in ``[0, norm_scale]`` — the same
    format the model emits and the canvas returns.
    """

    ed_polygon: Polygon = Field(default_factory=list)
    es_polygon: Polygon = Field(default_factory=list)
    image_h: int = Field(gt=0, le=20000)
    image_w: int = Field(gt=0, le=20000)
    es_image_h: int | None = Field(default=None, gt=0, le=20000)
    es_image_w: int | None = Field(default=None, gt=0, le=20000)
    #: Optional case reference; when given, dataset calibration is used by default.
    case_key: str | None = None
    #: Clinician-supplied pixel spacing in mm/px. Overrides dataset calibration.
    spacing_h: float | None = Field(default=None, gt=0, le=100)
    spacing_w: float | None = Field(default=None, gt=0, le=100)

    @model_validator(mode="after")
    def _both_spacings(self) -> "MetricsRequest":
        if (self.spacing_h is None) != (self.spacing_w is None):
            raise ValueError(
                "spacing_h and spacing_w must be supplied together, or both omitted."
            )
        return self


class SingleFrameMetricsRequest(BaseModel):
    """Metrics for a single traced frame (live updates while editing)."""

    polygon: Polygon = Field(default_factory=list)
    image_h: int = Field(gt=0, le=20000)
    image_w: int = Field(gt=0, le=20000)
    case_key: str | None = None
    spacing_h: float | None = Field(default=None, gt=0, le=100)
    spacing_w: float | None = Field(default=None, gt=0, le=100)


def _resolve_calibration(
    repo: DatasetRepository,
    case_key: str | None,
    spacing_h: float | None,
    spacing_w: float | None,
) -> Calibration:
    """Prefer explicit clinician spacing, then the dataset's, else unknown."""
    if spacing_h is not None and spacing_w is not None:
        try:
            return Calibration.from_user(spacing_h, spacing_w)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
    if case_key:
        try:
            return repo.get_case(case_key).calibration
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
    return Calibration.unknown()


@router.post("/metrics")
def compute_metrics(
    request: MetricsRequest,
    repo: DatasetRepository = Depends(repository),
) -> dict[str, Any]:
    """Compute ED/ES areas, perimeters and FAC%.

    FAC% is dimensionless and therefore returned even for uncalibrated frames;
    physical areas are withheld unless spacing is known.
    """
    calibration = _resolve_calibration(
        repo, request.case_key, request.spacing_h, request.spacing_w
    )
    metrics = chamber_metrics(
        ed_polygon=request.ed_polygon,
        es_polygon=request.es_polygon,
        image_h=request.image_h,
        image_w=request.image_w,
        calibration=calibration,
        es_image_h=request.es_image_h,
        es_image_w=request.es_image_w,
    )
    result = metrics.as_dict()
    result["calibration"] = {
        "spacing_h": calibration.spacing_h,
        "spacing_w": calibration.spacing_w,
        "source": calibration.source,
        "note": (
            None
            if calibration.is_known
            else "Pixel spacing is unavailable for this frame, so physical areas "
            "(cm²) are not reported. FAC % is a ratio and remains valid."
        ),
    }
    return result


@router.post("/frame-metrics")
def compute_frame_metrics(
    request: SingleFrameMetricsRequest,
    repo: DatasetRepository = Depends(repository),
) -> dict[str, Any]:
    """Compute area and perimeter for a single frame being edited."""
    calibration = _resolve_calibration(
        repo, request.case_key, request.spacing_h, request.spacing_w
    )
    metrics = frame_metrics(
        polygon=request.polygon,
        image_h=request.image_h,
        image_w=request.image_w,
        calibration=calibration,
    )
    return metrics.as_dict()
