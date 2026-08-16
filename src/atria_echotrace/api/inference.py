"""Model lifecycle and contour prediction endpoints.

Production form of the notebook HITL cell's two backend callbacks: ``load_model``
(L1257-1282) becomes explicit lifecycle endpoints, and ``process_image_backend``
(L1284-1316) becomes ``POST /api/inference/predict``.

Model lifecycle and prediction share one module because they share one resource; the
routes keep separate ``/api/model`` and ``/api/inference`` paths.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from ..config import Settings
from ..data.dataset import DatasetRepository
from ..data.frames import UploadError, load_frame, upload_path
from ..domain.geometry import polygon_dice, polygon_iou
from .deps import get_engine, get_settings, repository, require_ml

router = APIRouter(tags=["inference"])


class LoadRequest(BaseModel):
    """Which adapter to load."""

    adapter: str | None = Field(
        default=None,
        description=(
            "Registry key (base, camus, echonet), a Hugging Face repo id, or a local "
            "checkpoint directory. Defaults to the configured adapter."
        ),
    )


class PredictRequest(BaseModel):
    """One frame to trace."""

    stem: str | None = Field(default=None, description="Dataset frame stem")
    upload_id: str | None = Field(default=None, description="Uploaded frame id")
    target_structure: Literal["LV", "LA"] = "LV"
    view: str | None = Field(default=None, description="2CH or 4CH; inferred for dataset frames")
    instant: str | None = Field(default=None, description="ED or ES; inferred for dataset frames")
    prompt_variant: Literal["training", "generic"] | None = Field(
        default=None,
        description=(
            "Force a prompt template. Defaults to 'training' (the template the "
            "adapters were fine-tuned with) when view and instant are known."
        ),
    )
    max_new_tokens: int | None = Field(default=None, ge=16, le=4096)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "PredictRequest":
        if bool(self.stem) == bool(self.upload_id):
            raise ValueError("Provide exactly one of 'stem' or 'upload_id'.")
        return self


@router.get("/api/model/status")
def model_status(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Poll the model lifecycle. Answers without the AI tier installed."""
    from .deps import ml_available

    if not ml_available():
        return {
            "state": "unavailable",
            "progress": 0.0,
            "message": 'AI tier not installed. Install with: pip install -e ".[ai]"',
            "error": None,
            "adapter": None,
        }
    return get_engine().status()


@router.post("/api/model/load", status_code=status.HTTP_202_ACCEPTED)
def load_model(
    request: LoadRequest,
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_ml),
) -> dict[str, Any]:
    """Begin loading weights in the background.

    Returns 202 immediately; poll ``GET /api/model/status`` for progress. Loading
    several gigabytes cannot complete within a request, and must not block startup.
    """
    from ..ml.engine import AdapterError

    engine = get_engine()
    try:
        return engine.load_async(request.adapter or settings.default_adapter)
    except AdapterError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/api/model/unload")
def unload_model(_: None = Depends(require_ml)) -> dict[str, Any]:
    """Release the model and free device memory (notebook L853-862)."""
    return get_engine().unload()


@router.post("/api/inference/predict")
def predict(
    request: PredictRequest,
    repo: DatasetRepository = Depends(repository),
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_ml),
) -> dict[str, Any]:
    """Predict a contour for one frame.

    Declared ``def`` rather than ``async def`` on purpose: Starlette then runs it in
    its threadpool, so a multi-second generation does not block the event loop
    (RESEARCH.md §2.4).
    """
    from ..ml.engine import ModelNotReady

    view = request.view
    instant = request.instant
    ground_truth = None
    image_h = image_w = None

    if request.stem:
        try:
            frame = repo.get_frame(request.stem)
            path = repo.frame_path(request.stem)
        except LookupError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        # Dataset frames know their own view/instant, so the training prompt applies.
        view = view or frame.view
        instant = instant or frame.instant
        ground_truth = frame.polygon(request.target_structure)
        image_h, image_w = frame.image_h, frame.image_w
        image = load_frame(path)
    else:
        assert request.upload_id is not None
        try:
            path = upload_path(request.upload_id, settings.uploads_dir)
        except UploadError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        image = load_frame(path)
        image_w, image_h = image.size

    engine = get_engine()
    try:
        result = engine.predict(
            image=image,
            target_structure=request.target_structure,
            view=view,
            instant=instant,
            prompt_variant=request.prompt_variant,
            max_new_tokens=request.max_new_tokens,
        )
    except ModelNotReady as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        # The model ran but produced nothing parseable: upstream failure, not a bug here.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    result["image_h"] = image_h
    result["image_w"] = image_w
    result["stem"] = request.stem
    result["upload_id"] = request.upload_id

    # Agreement with the reference trace, when one exists. Cheap and directly
    # comparable to the notebook's evaluation metrics.
    if ground_truth and image_h and image_w:
        result["ground_truth_polygon"] = ground_truth
        result["agreement"] = {
            "dice": round(polygon_dice(result["polygon"], ground_truth, image_h, image_w), 4),
            "iou": round(polygon_iou(result["polygon"], ground_truth, image_h, image_w), 4),
            "reference": "dataset ground truth",
        }
    return result
