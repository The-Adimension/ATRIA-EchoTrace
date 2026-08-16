"""Evaluation endpoints: Dice / IoU / parse-rate over a dataset split.

Production form of the notebook's evaluation cells (notebook_as_py.txt L992-1169).
Runs execute in the background and are polled, because a full split takes minutes on a
GPU and hours on CPU.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..config import Settings
from ..data.dataset import DatasetRepository
from .deps import get_engine, get_settings, repository, require_ml

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


class EvaluationRequest(BaseModel):
    """Which frames to evaluate."""

    split: str | None = Field(default="test", description="Dataset split; null for all")
    source: str | None = Field(default=None, description="Filter by camus or echonet")
    target_structure: Literal["LV", "LA"] = "LV"
    prompt_variant: Literal["training", "generic"] | None = None
    max_samples: int | None = Field(
        default=50,
        ge=1,
        le=5000,
        description="Cap on frames evaluated; the notebook used EVAL_MAX_SAMPLES=50.",
    )


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def start_run(
    request: EvaluationRequest,
    repo: DatasetRepository = Depends(repository),
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_ml),
) -> dict[str, Any]:
    """Start an evaluation run. Returns 202; poll ``GET /api/evaluation/runs/{id}``."""
    from ..ml.evaluate import get_runner, select_frames

    engine = get_engine()
    if not engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Model is not ready (state={engine.status()['state']}). "
                "POST /api/model/load first."
            ),
        )

    frames = select_frames(
        repo=repo,
        target_structure=request.target_structure,
        split=request.split,
        source=request.source,
        max_samples=request.max_samples,
    )
    settings.ensure_output_dirs()
    try:
        run = get_runner().start(
            engine=engine,
            repo=repo,
            frames=frames,
            target_structure=request.target_structure,
            prompt_variant=request.prompt_variant,
            split=request.split,
            source=request.source,
            max_samples=request.max_samples,
            output_dir=settings.evaluations_dir,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return {
        "run_id": run.run_id,
        "state": run.state,
        "selected_frames": len(frames),
        "message": run.message,
        "poll_url": f"/api/evaluation/runs/{run.run_id}",
    }


@router.get("/runs")
def get_runs(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """List persisted runs, plus the live one if any.

    Answers without the AI tier so the UI can show history on a review-only install.
    """
    from ..ml.evaluate import list_runs

    current: dict[str, Any] | None = None
    from .deps import ml_available

    if ml_available():
        from ..ml.evaluate import get_runner

        current = get_runner().current()

    runs = list_runs(settings.evaluations_dir)
    return {"count": len(runs), "current": current, "runs": runs}


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    include_polygons: bool = Query(False, description="Include predicted polygons"),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Fetch one run. A still-running run is served from memory."""
    from ..ml.evaluate import load_run

    from .deps import ml_available

    if ml_available():
        from ..ml.evaluate import get_runner

        live = get_runner().live_run(run_id)
        if live is not None:
            return live.as_dict(include_polygons=include_polygons)

    try:
        data = load_run(run_id, settings.evaluations_dir)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if not include_polygons:
        for result in data.get("results", []):
            result.pop("predicted_polygon", None)
    return data
