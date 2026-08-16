"""Raw dataset ingestion — thin wrappers over the real preprocessing scripts.

The notebook's pipeline begins with a preprocessing step the notebook itself does not
contain: its dataset cells consume ``frames/`` + ``tracings.json`` + ``metadata.csv``
"already generated locally" by an external ``preprocess_camus.py``
(notebook_as_py.txt L108-135).

Those scripts were subsequently supplied, and they are the authority: re-running them
reproduces the shipped training corpus byte-for-byte. They are vendored unmodified under
:mod:`.reference` and simply called from here, rather than reimplemented — an earlier
reimplementation was geometrically close (Dice 0.994) but read the wrong container
format, invented splits, and missed the EF key entirely (RESEARCH.md §8.4).

This module therefore contains only what the CLI needs on top: locating the scripts,
reporting results, and turning their failure modes into actionable errors.

Output contract (what the application consumes):

    <output>/frames/<stem>.png
    <output>/tracings.json     {stem: {view, instant, image_h, image_w, spacing_h,
                                       spacing_w, lv_polygon, la_polygon, split,
                                       source, ef, lv_points_raw, la_points_raw}}
    <output>/metadata.csv      key,patient_id,view,instant,split,image_h,image_w,ef,
                               age,sex,image_quality,has_lv,has_la,lv_points,la_points[,source]
    <output>/preprocessing_log.txt
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...logging_setup import get_logger

logger = get_logger("data.ingest")


class IngestError(RuntimeError):
    """Raised when a raw dataset cannot be imported."""


@dataclass
class IngestResult:
    """Summary of an ingest run, read back from the artefacts it produced."""

    output_dir: Path
    source: str
    n_frames: int
    n_cases: int
    n_points: int
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "source": self.source,
            "n_frames": self.n_frames,
            "n_cases": self.n_cases,
            "n_points": self.n_points,
            "warnings": self.warnings,
        }


def summarise_output(output_dir: Path, source: str, n_points: int) -> IngestResult:
    """Build an :class:`IngestResult` from a completed run's ``tracings.json``.

    Raises:
        IngestError: if the run produced no tracings.
    """
    tracings_path = Path(output_dir) / "tracings.json"
    if not tracings_path.is_file():
        raise IngestError(
            f"Preprocessing produced no tracings.json in {output_dir}. "
            f"See {Path(output_dir) / 'preprocessing_log.txt'} for the cause."
        )
    tracings: dict[str, dict[str, Any]] = json.loads(
        tracings_path.read_text(encoding="utf-8")
    )
    if not tracings:
        raise IngestError(
            f"Preprocessing produced an empty tracings.json in {output_dir}. "
            f"See {Path(output_dir) / 'preprocessing_log.txt'} for the skipped items."
        )

    cases = {
        (entry.get("patient_id"), entry.get("view")) for entry in tracings.values()
    }
    return IngestResult(
        output_dir=Path(output_dir),
        source=source,
        n_frames=len(tracings),
        n_cases=len(cases),
        n_points=n_points,
    )


def require(module: str, package: str, dataset: str):
    """Import a dependency of the reference scripts, or explain how to get it.

    Raises:
        IngestError: with the exact install command when the dependency is absent.
    """
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - depends on the [ingest] extra
        raise IngestError(
            f"Ingesting {dataset} needs {package}, which is not installed. "
            f'Install the ingest extra: pip install -e ".[ingest]"'
        ) from exc
