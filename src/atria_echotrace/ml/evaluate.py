"""Test-set evaluation: parse rate, Dice and IoU, with best/worst ranking.

Production form of the notebook's evaluation cells: the sequential inference loop and
its summary (notebook_as_py.txt L992-1083) and the best/worst analysis (L1085-1169).
The notebook's ``EVAL_MAX_SAMPLES`` cap (L1000) becomes the ``max_samples`` argument.

Evaluation runs for minutes to hours, so it executes on a background thread with
pollable progress and is persisted to ``outputs/evaluations/<run_id>.json``.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..data.dataset import DatasetRepository, Frame
from ..data.frames import load_frame
from ..domain.geometry import polygon_dice, polygon_iou
from ..domain.structures import resolve_structure
from ..logging_setup import get_logger
from .engine import InferenceEngine
from .prompts import PromptVariant

logger = get_logger("ml.evaluate")

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class FrameResult:
    """Per-frame evaluation outcome."""

    stem: str
    source: str
    view: str
    instant: str
    parsed: bool
    vertices: int | None = None
    dice: float | None = None
    iou: float | None = None
    seconds: float | None = None
    error: str | None = None
    predicted_polygon: list[list[int]] | None = None


@dataclass
class EvaluationRun:
    """A complete evaluation run: configuration, per-frame results and summary."""

    run_id: str
    split: str | None
    source: str | None
    target_structure: str
    adapter: dict[str, Any] | None
    prompt_variant: str | None
    device: str | None
    max_samples: int | None
    state: str = "running"
    started_utc: str = ""
    finished_utc: str | None = None
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    results: list[FrameResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def compute_summary(self) -> dict[str, Any]:
        """Aggregate parse rate, Dice and IoU (notebook L1069-1083).

        Only frames that both parsed *and* had a reference trace contribute to the
        overlap statistics, matching the notebook, which appended scores solely for
        successfully parsed predictions.
        """
        total = len(self.results)
        dice_scores = [r.dice for r in self.results if r.dice is not None]
        iou_scores = [r.iou for r in self.results if r.iou is not None]
        parsed = sum(1 for r in self.results if r.parsed)

        def mean_std(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"mean": None, "std": None, "min": None, "max": None}
            return {
                "mean": round(statistics.fmean(values), 4),
                # pstdev matches numpy's default population std used by the notebook.
                "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

        return {
            "total_samples": total,
            "parsed": parsed,
            "parse_rate_percent": round(parsed / total * 100, 1) if total else 0.0,
            "scored_samples": len(dice_scores),
            "dice": mean_std(dice_scores),
            "iou": mean_std(iou_scores),
        }

    def ranked(self, best: int = 5) -> dict[str, list[dict[str, Any]]]:
        """Best and worst predictions by Dice (notebook L1133-1146)."""
        scored = sorted(
            (r for r in self.results if r.dice is not None),
            key=lambda r: r.dice,  # type: ignore[arg-type,return-value]
            reverse=True,
        )
        compact = [
            {"stem": r.stem, "dice": r.dice, "iou": r.iou, "vertices": r.vertices}
            for r in scored
        ]
        return {"best": compact[:best], "worst": compact[-best:][::-1]}

    def as_dict(self, include_polygons: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_polygons:
            for result in data["results"]:
                result.pop("predicted_polygon", None)
        data["ranked"] = self.ranked()
        return data


def select_frames(
    repo: DatasetRepository,
    target_structure: str = "LV",
    split: str | None = None,
    source: str | None = None,
    max_samples: int | None = None,
) -> list[Frame]:
    """Choose evaluation frames.

    Mirrors ``prepare_echocardiographic_frame_samples`` (notebook L449-500): filter by
    split, require the target polygon and require the PNG to exist, then cap the count.
    """
    frames = [
        frame
        for frame in repo.frames.values()
        if (split is None or (frame.split or "").lower() == split.lower())
        and (source is None or frame.source.lower() == source.lower())
        and frame.polygon(target_structure)
        and (repo.frames_dir / f"{frame.stem}.png").is_file()
    ]
    frames.sort(key=lambda f: f.stem)
    return frames[:max_samples] if max_samples else frames


def evaluate_frames(
    engine: InferenceEngine,
    repo: DatasetRepository,
    frames: list[Frame],
    run: EvaluationRun,
    target_structure: str = "LV",
    prompt_variant: PromptVariant | None = None,
    progress: ProgressCallback | None = None,
) -> EvaluationRun:
    """Run inference over ``frames`` and score against ground truth.

    A per-frame failure is recorded and the loop continues, as the notebook did
    (L1035-1037), so one bad generation cannot abort a long run.
    """
    total = len(frames)
    for index, frame in enumerate(frames, start=1):
        message = f"{index}/{total} {frame.stem}"
        run.progress = (index - 1) / total if total else 0.0
        run.message = message
        if progress is not None:
            progress(index, total, message)

        reference = frame.polygon(target_structure)
        try:
            image = load_frame(repo.frame_path(frame.stem))
            outcome = engine.predict(
                image=image,
                target_structure=target_structure,
                view=frame.view,
                instant=frame.instant,
                prompt_variant=prompt_variant,
            )
        except Exception as exc:  # noqa: BLE001 - recorded per frame, run continues
            logger.warning("Evaluation failed for %s: %s", frame.stem, exc)
            run.results.append(
                FrameResult(
                    stem=frame.stem,
                    source=frame.source,
                    view=frame.view,
                    instant=frame.instant,
                    parsed=False,
                    error=str(exc)[:500],
                )
            )
            continue

        polygon = outcome["polygon"]
        result = FrameResult(
            stem=frame.stem,
            source=frame.source,
            view=frame.view,
            instant=frame.instant,
            parsed=True,
            vertices=len(polygon),
            seconds=outcome["inference_seconds"],
            predicted_polygon=polygon,
        )
        if reference:
            result.dice = round(
                polygon_dice(polygon, reference, frame.image_h, frame.image_w), 4
            )
            result.iou = round(
                polygon_iou(polygon, reference, frame.image_h, frame.image_w), 4
            )
        run.results.append(result)

    run.progress = 1.0
    run.summary = run.compute_summary()
    return run


def new_run_id() -> str:
    return f"eval_{int(time.time())}"


def save_run(run: EvaluationRun, output_dir: Path) -> Path:
    """Persist a run to ``outputs/evaluations/<run_id>.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run.run_id}.json"
    path.write_text(json.dumps(run.as_dict(), indent=2), encoding="utf-8")
    return path


def save_ranked_figures(
    run: EvaluationRun,
    repo: DatasetRepository,
    output_dir: Path,
    count: int = 3,
) -> list[Path]:
    """Render the best and worst predictions as 3-panel figures.

    The notebook's qualitative review step (L1149-1169): after the aggregate numbers, it
    plotted the three best and three worst predictions, because a mean Dice hides the
    systematic failures that looking at the extremes reveals.

    Returns the figures written; empty when nothing was scored.
    """
    from ..render.figures import prediction_figure, write_png

    ranked = run.ranked(best=count)
    by_stem = {result.stem: result for result in run.results}
    structure_short = resolve_structure(run.target_structure)["short"]
    written: list[Path] = []

    for band in ("best", "worst"):
        for position, entry in enumerate(ranked[band][:count], start=1):
            result = by_stem.get(entry["stem"])
            frame = repo.frames.get(entry["stem"])
            if result is None or frame is None or not result.predicted_polygon:
                continue
            reference = frame.polygon(run.target_structure)
            if not reference:
                continue
            png = prediction_figure(
                image=load_frame(repo.frame_path(frame.stem)),
                predicted=result.predicted_polygon,
                ground_truth=reference,
                structure_short=structure_short,
                title=f"{band.upper()}: {frame.stem}",
                dice=result.dice,
            )
            written.append(
                write_png(png, output_dir / f"{run.run_id}_{band}{position}_{frame.stem}.png")
            )
    return written


def load_run(run_id: str, output_dir: Path) -> dict[str, Any]:
    """Load a persisted run.

    Raises:
        ValueError: if ``run_id`` is malformed.
        FileNotFoundError: if no such run exists.
    """
    if not run_id.startswith("eval_") or not run_id[5:].isdigit():
        raise ValueError(f"Malformed evaluation run id: {run_id!r}")
    path = output_dir / f"{run_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown evaluation run: {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_runs(output_dir: Path) -> list[dict[str, Any]]:
    """Summaries of persisted runs, newest first."""
    if not output_dir.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("eval_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries.append(
            {
                "run_id": data.get("run_id", path.stem),
                "state": data.get("state"),
                "split": data.get("split"),
                "source": data.get("source"),
                "target_structure": data.get("target_structure"),
                "adapter": (data.get("adapter") or {}).get("id"),
                "device": data.get("device"),
                "started_utc": data.get("started_utc"),
                "finished_utc": data.get("finished_utc"),
                "summary": data.get("summary", {}),
            }
        )
    return entries


class EvaluationJobRunner:
    """Runs one evaluation at a time on a background thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: EvaluationRun | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._current is not None and self._current.state == "running"

    def live_run(self, run_id: str) -> EvaluationRun | None:
        """Return the in-memory run object if ``run_id`` is currently executing."""
        with self._lock:
            if (
                self._current is not None
                and self._current.run_id == run_id
                and self._current.state == "running"
            ):
                return self._current
            return None

    def current(self) -> dict[str, Any] | None:
        with self._lock:
            if self._current is None:
                return None
            return {
                "run_id": self._current.run_id,
                "state": self._current.state,
                "progress": round(self._current.progress, 3),
                "message": self._current.message,
                "error": self._current.error,
                "completed": len(self._current.results),
            }

    def start(
        self,
        engine: InferenceEngine,
        repo: DatasetRepository,
        frames: list[Frame],
        target_structure: str,
        prompt_variant: PromptVariant | None,
        split: str | None,
        source: str | None,
        max_samples: int | None,
        output_dir: Path,
    ) -> EvaluationRun:
        """Start a run.

        Raises:
            RuntimeError: if a run is already in progress, or no frames were selected.
        """
        if self.is_running:
            raise RuntimeError(
                "An evaluation run is already in progress. Wait for it to finish."
            )
        if not frames:
            raise RuntimeError(
                "No frames matched the selection (split/source/structure filters)."
            )

        run = EvaluationRun(
            run_id=new_run_id(),
            split=split,
            source=source,
            target_structure=target_structure.upper(),
            adapter=engine.status().get("adapter"),
            prompt_variant=prompt_variant,
            device=engine.status().get("device"),
            max_samples=max_samples,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            message=f"Starting evaluation of {len(frames)} frames",
        )
        with self._lock:
            self._current = run

        def worker() -> None:
            try:
                evaluate_frames(
                    engine=engine,
                    repo=repo,
                    frames=frames,
                    run=run,
                    target_structure=target_structure,
                    prompt_variant=prompt_variant,
                )
                run.state = "completed"
                run.message = (
                    f"Completed: parse rate {run.summary['parse_rate_percent']}%, "
                    f"mean Dice {run.summary['dice']['mean']}"
                )
            except Exception as exc:  # noqa: BLE001 - surfaced through the run record
                logger.exception("Evaluation run failed")
                run.state = "error"
                run.error = str(exc)
                run.message = "Evaluation failed."
            finally:
                run.finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                if not run.summary:
                    run.summary = run.compute_summary()
                save_run(run, output_dir)
                logger.info("Evaluation %s finished: %s", run.run_id, run.message)

        thread = threading.Thread(target=worker, name=f"atria-eval-{run.run_id}", daemon=True)
        self._thread = thread
        thread.start()
        return run


_runner = EvaluationJobRunner()


def get_runner() -> EvaluationJobRunner:
    """Process-wide evaluation runner."""
    return _runner
