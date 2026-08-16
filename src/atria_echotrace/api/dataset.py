"""Dataset browsing endpoints.

Production form of the notebook's dataset exploration cells: the case/frame listings
replace the printed distributions (notebook_as_py.txt L273-280), and the ground-truth
overlay endpoint replaces ``visualize_polygon_on_image`` (L301-369).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status

from ..config import Settings
from ..data.dataset import DatasetRepository
from ..data.frames import UploadError, load_frame, save_upload, upload_path
from ..domain.structures import resolve_structure
from ..render.figures import ground_truth_figure
from .deps import get_settings, repository

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

#: Frames and figures are immutable for a given dataset, so allow client caching.
_CACHE_CONTROL = "public, max-age=3600"


@router.get("/cases")
def list_cases(
    source: str | None = Query(None, description="Filter by dataset, e.g. camus or echonet"),
    view: str | None = Query(None, description="Filter by view, 2CH or 4CH"),
    complete_pairs_only: bool = Query(False, description="Only cases having both ED and ES"),
    repo: DatasetRepository = Depends(repository),
) -> dict[str, Any]:
    """List cases with metadata for the case browser.

    Cases are keyed by ``case_key`` (``"<case_id>_<view>"``) because one patient may
    be imaged in several views.
    """
    cases = repo.list_cases(source=source, view=view, complete_pairs_only=complete_pairs_only)
    return {
        "count": len(cases),
        "sources": sorted({c.source for c in repo.cases.values()}),
        "views": sorted({c.view for c in repo.cases.values() if c.view}),
        "cases": [case.summary() for case in cases],
    }


@router.get("/cases/{case_key}")
def get_case(
    case_key: str,
    repo: DatasetRepository = Depends(repository),
) -> dict[str, Any]:
    """Full case detail, including ground-truth polygons for both structures."""
    try:
        return repo.get_case(case_key).detail()
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/frames/{stem}.png")
def get_frame_image(
    stem: str,
    repo: DatasetRepository = Depends(repository),
) -> Response:
    """Serve one frame's PNG bytes.

    Only stems present in ``tracings.json`` resolve, so a crafted stem cannot reach
    arbitrary files.
    """
    try:
        path = repo.frame_path(stem)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(
        content=path.read_bytes(),
        media_type="image/png",
        headers={"Cache-Control": _CACHE_CONTROL},
    )


@router.get("/frames/{stem}/ground-truth.png")
def get_ground_truth_figure(
    stem: str,
    target_structure: str = Query("LV", description="LV or LA"),
    repo: DatasetRepository = Depends(repository),
) -> Response:
    """2-panel ground-truth overlay figure (notebook L301-332)."""
    try:
        frame = repo.get_frame(stem)
        path = repo.frame_path(stem)
        struct = resolve_structure(target_structure)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    polygon = frame.polygon(target_structure)
    if not polygon:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Frame {stem!r} has no {target_structure.upper()} ground truth. "
                f"{frame.source} frames provide "
                f"{'LV only' if frame.source == 'echonet' else 'LV and LA'}."
            ),
        )

    png = ground_truth_figure(
        image=load_frame(path),
        polygon=polygon,
        structure_short=struct["short"],
        title=f"{stem} ({target_structure.upper()}, raw pts: {frame.lv_points_raw} -> {len(polygon)})",
    )
    return Response(
        content=png, media_type="image/png", headers={"Cache-Control": _CACHE_CONTROL}
    )


@router.post("/uploads", status_code=status.HTTP_201_CREATED)
async def create_upload(
    file: UploadFile = File(..., description="Echocardiographic frame (PNG/JPEG/BMP/TIFF)"),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Accept a clinician-supplied frame.

    Production form of the notebook HITL cell's base64 file input (L1285-1291).
    Uploads are re-encoded to PNG, which also strips source metadata such as EXIF —
    appropriate given the de-identification obligation in disclaimer IV.
    """
    settings.ensure_output_dirs()
    payload = await file.read()
    try:
        upload_id, image, path = save_upload(
            data=payload,
            uploads_dir=settings.uploads_dir,
            max_bytes=settings.max_upload_bytes,
            max_pixels=settings.max_upload_pixels,
        )
    except UploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return {
        "upload_id": upload_id,
        "image_w": image.width,
        "image_h": image.height,
        "filename": file.filename,
        "bytes": len(payload),
        "image_url": f"/api/dataset/uploads/{upload_id}.png",
        "calibration_source": "unknown",
        "note": (
            "Uploaded frames carry no pixel-spacing metadata, so physical areas are "
            "not reported until a spacing is supplied."
        ),
    }


@router.get("/uploads/{upload_id}.png")
def get_upload_image(
    upload_id: str,
    settings: Settings = Depends(get_settings),
) -> Response:
    """Serve a previously uploaded frame."""
    try:
        path = upload_path(upload_id, settings.uploads_dir)
    except UploadError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(
        content=path.read_bytes(),
        media_type="image/png",
        headers={"Cache-Control": _CACHE_CONTROL},
    )


@router.get("/splits/{split}")
def frames_in_split(
    split: str,
    repo: DatasetRepository = Depends(repository),
) -> dict[str, Any]:
    """Frames belonging to a split (notebook ``prepare_*_samples``, L449-528)."""
    frames = repo.frames_in_split(split)
    return {
        "split": split,
        "count": len(frames),
        "stems": [f.stem for f in frames],
    }
