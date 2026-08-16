"""Clinician revision persistence and export endpoints.

Production form of the notebook's ``save_polygon_backend`` (notebook_as_py.txt
L1318-1375): it received the two edited polygons, wrote JSON per frame and rendered a
4-panel figure per frame. This adds durable per-revision directories, coordinate CSVs,
a metrics summary and a ZIP bundle (RESEARCH.md §0.3).
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from ..config import Settings
from ..data.dataset import DatasetRepository
from ..data.frames import UploadError, load_frame, upload_path
from ..domain.geometry import sanitize_polygon
from ..domain.metrics import Calibration, chamber_metrics
from ..domain.structures import resolve_structure
from ..export.package import (
    PhaseTracing,
    list_revisions,
    load_revision,
    new_revision_id,
    write_revision_bundle,
    zip_bytes,
)
from .deps import get_settings, repository

router = APIRouter(prefix="/api/revisions", tags=["revisions"])


class PhaseInput(BaseModel):
    """One traced phase submitted by the clinician."""

    instant: Literal["ED", "ES"]
    stem: str | None = Field(default=None, description="Dataset frame stem")
    upload_id: str | None = Field(default=None, description="Uploaded frame id")
    model_polygon: list[list[float]] = Field(default_factory=list)
    user_polygon: list[list[float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "PhaseInput":
        if bool(self.stem) == bool(self.upload_id):
            raise ValueError("Provide exactly one of 'stem' or 'upload_id'.")
        return self


class RevisionRequest(BaseModel):
    """A complete revision: one or two phases plus provenance."""

    phases: list[PhaseInput] = Field(min_length=1, max_length=2)
    case_key: str | None = None
    #: Apical view for uploaded frames, which carry no such metadata. Dataset frames
    #: take their own view from the repository and ignore this.
    view: str | None = None
    target_structure: str = "LV"
    adapter: str | None = None
    prompt_variant: str | None = None
    notes: str = Field(default="", max_length=20000)
    spacing_h: float | None = Field(default=None, gt=0, le=100)
    spacing_w: float | None = Field(default=None, gt=0, le=100)

    @model_validator(mode="after")
    def _unique_instants(self) -> "RevisionRequest":
        instants = [phase.instant for phase in self.phases]
        if len(set(instants)) != len(instants):
            raise ValueError("Each phase must have a distinct instant (ED, ES).")
        if not any(phase.user_polygon for phase in self.phases):
            raise ValueError("At least one phase must carry a revised polygon.")
        if (self.spacing_h is None) != (self.spacing_w is None):
            raise ValueError("spacing_h and spacing_w must be supplied together.")
        return self


def _resolve_phase_image(
    phase: PhaseInput,
    repo: DatasetRepository,
    settings: Settings,
) -> Image.Image:
    """Load the phase's frame image.

    A missing frame is a hard 404. The previous implementation silently fell back to
    the first PNG in the directory, which rendered a different patient's anatomy into
    the exported report.
    """
    if phase.stem:
        try:
            path = repo.frame_path(phase.stem)
        except LookupError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return load_frame(path)

    assert phase.upload_id is not None  # guaranteed by PhaseInput validation
    try:
        path = upload_path(phase.upload_id, settings.uploads_dir)
    except UploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return load_frame(path)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_revision(
    request: RevisionRequest,
    repo: DatasetRepository = Depends(repository),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Persist a revision and generate every export artefact."""
    try:
        structure = resolve_structure(request.target_structure)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    # Calibration: explicit clinician spacing wins, then the case's own.
    calibration = Calibration.unknown()
    integrity_flags: list[str] = []
    if request.spacing_h is not None and request.spacing_w is not None:
        calibration = Calibration.from_user(request.spacing_h, request.spacing_w)
    elif request.case_key:
        try:
            case = repo.get_case(request.case_key)
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc
        calibration = case.calibration
        integrity_flags = case.integrity_flags()

    phases: list[PhaseTracing] = []
    # Apical view per phase. Recorded because a corpus exported from these revisions
    # needs it to rebuild the training prompt (`atria export-corpus`): dataset frames
    # know their own view, uploads only know what the clinician declared.
    views: dict[str, str] = {}
    for phase in sorted(request.phases, key=lambda p: p.instant != "ED"):
        image = _resolve_phase_image(phase, repo, settings)
        ground_truth: list[list[int]] | None = None
        if phase.stem:
            try:
                frame = repo.get_frame(phase.stem)
                ground_truth = frame.polygon(request.target_structure)
                if frame.view:
                    views[phase.instant] = frame.view
            except LookupError:
                ground_truth = None
        elif request.view:
            views[phase.instant] = request.view.upper()
        phases.append(
            PhaseTracing(
                instant=phase.instant,
                image=image,
                model_polygon=sanitize_polygon(phase.model_polygon) or [],
                user_polygon=sanitize_polygon(phase.user_polygon) or [],
                ground_truth_polygon=ground_truth,
                stem=phase.stem or f"upload:{phase.upload_id}",
            )
        )

    by_instant = {phase.instant: phase for phase in phases}
    ed = by_instant.get("ED")
    es = by_instant.get("ES")
    reference = ed or es
    assert reference is not None  # phases is non-empty
    metrics = chamber_metrics(
        ed_polygon=ed.user_polygon if ed else [],
        es_polygon=es.user_polygon if es else [],
        image_h=reference.image.size[1],
        image_w=reference.image.size[0],
        calibration=calibration,
        es_image_h=es.image.size[1] if es else None,
        es_image_w=es.image.size[0] if es else None,
    )

    revision_id = new_revision_id()
    settings.ensure_output_dirs()
    record = write_revision_bundle(
        revision_id=revision_id,
        output_root=settings.revisions_dir,
        phases=phases,
        metrics=metrics,
        case_label=request.case_key or "uploaded-frames",
        provenance={
            "case_key": request.case_key,
            "target_structure": request.target_structure.upper(),
            "structure_label": structure["label"],
            "adapter": request.adapter,
            "prompt_variant": request.prompt_variant,
            "base_model_id": settings.base_model_id,
            "calibration_source": calibration.source,
            "spacing_h": calibration.spacing_h,
            "spacing_w": calibration.spacing_w,
            "dataset_integrity_flags": integrity_flags,
            "views": views,
        },
        notes=request.notes,
    )
    return record


@router.get("")
def get_revisions(
    limit: int = Query(200, ge=1, le=1000),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """List persisted revisions, newest first."""
    revisions = list_revisions(settings.revisions_dir, limit=limit)
    return {"count": len(revisions), "revisions": revisions}


@router.get("/{revision_id}")
def get_revision(
    revision_id: str,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Fetch one persisted revision record."""
    try:
        return load_revision(revision_id, settings.revisions_dir)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{revision_id}/export.zip")
def download_revision(
    revision_id: str,
    settings: Settings = Depends(get_settings),
) -> Response:
    """Download the revision's full export bundle."""
    try:
        payload = zip_bytes(revision_id, settings.revisions_dir)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="ATRIA_EchoTrace_{revision_id}.zip"'
        },
    )


@router.get("/{revision_id}/files/{filename}")
def download_revision_file(
    revision_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> Response:
    """Serve one artefact from a revision (used for inline figure previews)."""
    from ..export.package import revision_dir

    try:
        directory = revision_dir(revision_id, settings.revisions_dir)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    # Reject any path component: only flat artefact names inside the revision dir.
    if "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid artefact name."
        )
    target = (directory / filename).resolve()
    if not target.is_relative_to(directory) or not target.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No artefact {filename!r} in revision {revision_id}.",
        )

    media_types = {
        ".png": "image/png",
        ".json": "application/json",
        ".csv": "text/csv",
        ".zip": "application/zip",
    }
    return Response(
        content=target.read_bytes(),
        media_type=media_types.get(target.suffix.lower(), "application/octet-stream"),
    )
