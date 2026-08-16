"""Invoke the vendored reference preprocessors.

Each function here does three things and nothing more: validate that the source layout
is the one the reference script expects, call the script, and summarise what it wrote.
All contour extraction, resampling, normalisation, split assignment and metadata
construction belong to the scripts in :mod:`.reference` and are not duplicated.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Iterator

from ...logging_setup import get_logger
from . import IngestError, IngestResult, require, summarise_output

logger = get_logger("data.ingest.run")


def _reference_logger() -> logging.Logger:
    """A logger for ``merge_datasets``, which takes one as an argument."""
    return logging.getLogger("atria_echotrace.data.ingest.reference")


@contextlib.contextmanager
def _reference_logging() -> Iterator[None]:
    """Let the reference scripts own root logging for the duration of a run.

    Each script calls ``logging.basicConfig(handlers=[FileHandler(...), StreamHandler()])``
    to write its ``preprocessing_log.txt``. ``basicConfig`` does nothing when root
    already has handlers — which it does, because the CLI configures logging at
    startup. The net effect was an empty log file (the ``FileHandler`` truncates the
    file when constructed as an argument, then is discarded).

    Detaching our handlers for the duration lets the scripts configure themselves
    exactly as they do standalone; the originals are restored afterwards.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers = []
    try:
        yield
    finally:
        for handler in root.handlers:
            # Close the file the reference script opened so the log is flushed.
            with contextlib.suppress(Exception):
                handler.close()
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def ingest_camus(
    source_dir: Path,
    output_dir: Path,
    n_points: int = 30,
    lv_label: int = 1,
    la_label: int = 3,
) -> IngestResult:
    """Run the reference CAMUS preprocessor.

    Args:
        source_dir: The ``CAMUS_public`` root — the directory containing
            ``database_nifti/`` and ``database_split/``.
        output_dir: Destination for ``frames/``, ``tracings.json``, ``metadata.csv``.
        n_points: Vertices per contour (30 in the training corpus).
        lv_label: GT mask label for the LV cavity.
        la_label: GT mask label for the left atrium.

    Returns:
        An :class:`IngestResult`.

    Raises:
        IngestError: if the source layout is wrong, SimpleITK is missing, or nothing
            was produced.
    """
    source_dir = Path(source_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    require("SimpleITK", "SimpleITK", "CAMUS")

    if not source_dir.is_dir():
        raise IngestError(f"CAMUS source directory not found: {source_dir}")

    nifti_dir = source_dir / "database_nifti"
    if not nifti_dir.is_dir():
        present = sorted(p.name for p in source_dir.iterdir())[:8]
        raise IngestError(
            f"Expected {nifti_dir} to exist. --source must point at the CAMUS_public "
            f"root, which contains database_nifti/ and database_split/. "
            f"That directory holds: {present}"
        )
    if not (source_dir / "database_split").is_dir():
        logger.warning(
            "No database_split/ under %s; every patient will be recorded with "
            "split='unknown'. The official CAMUS splits live in that directory.",
            source_dir,
        )

    from .reference import preprocess_camus as reference

    output_dir.mkdir(parents=True, exist_ok=True)
    with _reference_logging():
        reference.preprocess_camus(
            camus_root=str(source_dir),
            output_dir=str(output_dir),
            num_points=n_points,
            lv_label=lv_label,
            la_label=la_label,
        )

    result = summarise_output(output_dir, "camus", n_points)
    result.warnings.append(
        "Splits come from the official database_split/subgroup_*.txt lists."
    )
    return result


def ingest_echonet(
    source_dir: Path,
    output_dir: Path,
    n_points: int = 30,
    max_videos: int | None = None,
    target_size: int | None = 224,
) -> IngestResult:
    """Run the reference EchoNet-Dynamic preprocessor.

    Args:
        source_dir: Directory containing ``Videos/``, ``FileList.csv`` and
            ``VolumeTracings.csv``.
        output_dir: Destination.
        n_points: Vertices per contour.
        max_videos: Process only the first N videos.
        target_size: Lanczos-upscale frames to this square size. 224 reproduces the
            training corpus; pass ``None`` for native 112x112.

    Returns:
        An :class:`IngestResult`.

    Raises:
        IngestError: if the layout is wrong, OpenCV is missing, or nothing was produced.
    """
    source_dir = Path(source_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    require("cv2", "opencv-python-headless", "EchoNet-Dynamic")

    missing = [
        name
        for name in ("Videos", "FileList.csv", "VolumeTracings.csv")
        if not (source_dir / name).exists()
    ]
    if missing:
        raise IngestError(
            f"EchoNet source at {source_dir} is missing {', '.join(missing)}. "
            "--source must point at the EchoNet-Dynamic root."
        )

    from .reference import preprocess_echonet as reference

    output_dir.mkdir(parents=True, exist_ok=True)
    with _reference_logging():
        reference.preprocess_echonet(
            echonet_root=str(source_dir),
            output_dir=str(output_dir),
            num_points=n_points,
            max_videos=max_videos,
            target_size=target_size,
        )

    result = summarise_output(output_dir, "echonet", n_points)
    result.warnings.append(
        "EchoNet publishes no pixel spacing; spacing is written as the 1.0 sentinel and "
        "the application withholds physical areas (cm²) for these frames."
    )
    result.warnings.append(
        "ED/ES come from the esf/edf columns of FileList.csv. In the supplied data those "
        "columns are inverted relative to their names, so ~99% of cases end up with "
        "transposed instant labels — reproduced here deliberately, because the adapters "
        "were trained on exactly this. See RESEARCH.md §8.3."
    )
    return result


def merge_unified(
    camus_processed: Path,
    echonet_processed: Path,
    output_dir: Path,
    n_points: int = 30,
) -> IngestResult:
    """Merge the two processed datasets into the unified training corpus.

    This is the step that produced ``unified_processed`` — 22 048 frames — which is what
    the adapters were fine-tuned on.

    Raises:
        IngestError: if either input is missing its ``tracings.json``.
    """
    camus_processed = Path(camus_processed).expanduser()
    echonet_processed = Path(echonet_processed).expanduser()
    output_dir = Path(output_dir).expanduser()

    for label, directory in (("CAMUS", camus_processed), ("EchoNet", echonet_processed)):
        if not (directory / "tracings.json").is_file():
            raise IngestError(
                f"{label} processed directory {directory} has no tracings.json. "
                f"Run `atria ingest {label.lower()}` first."
            )

    from .reference import preprocess_echonet as reference

    output_dir.mkdir(parents=True, exist_ok=True)
    reference.merge_datasets(
        camus_processed=str(camus_processed),
        echonet_processed=str(echonet_processed),
        unified_dir=str(output_dir),
        log=_reference_logger(),
    )

    result = summarise_output(output_dir, "unified", n_points)
    result.warnings.append(
        "Frames from both datasets are copied in, so the directory is self-contained."
    )
    return result
