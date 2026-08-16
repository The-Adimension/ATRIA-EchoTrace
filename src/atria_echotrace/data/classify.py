"""Derive classification-task datasets from the processed three-artefact corpus.

The contour pipeline consumes ``frames/`` + ``tracings.json`` + ``metadata.csv``. The same
material also carries clinical class labels, which this module turns into either of two
independent products for each task:

* **metadata** — one mapping file, ``mapping.csv``, linking every PNG to its class.
  No image is copied.
* **dirs** — a classic ``ImageFolder`` layout: one subdirectory per class holding the
  PNGs themselves.

Both write under ``datasets/classified_datasets/<task>/`` and can coexist there.

Every label is **copied** from the dataset metadata, never invented. The one derived
quantity is the EF bin, which is pure arithmetic on the published EF value.

Label authority:

* **CAMUS** — ``database_nifti/patient####/Info_{view}.cfg``. ``ImageQuality`` is a
  property of the acquisition window, not the patient: it differs between 2CH and 4CH in
  208 of 500 patients, so labels are keyed on ``(patient, view)``.
* **EchoNet** — the authentic ``FileList.csv`` (stock 9-column schema). ``EF`` is its only
  clinical class dimension; it publishes no sex, age or image-quality data.

Paths resolve from :data:`~atria_echotrace.config.PROJECT_ROOT`, which is found by marker
files rather than hard-coded, so a fresh clone works with no edits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from ..config import PROJECT_ROOT, display_path
from ..logging_setup import configure, get_logger

logger = get_logger("data.classify")

Row = Mapping[str, str]
Mode = Literal["metadata", "dirs"]

#: Everything below is repo-root relative — never an absolute user path.
DATASETS_DIR = PROJECT_ROOT / "datasets"
PROCESSED_ROOT = DATASETS_DIR / "processed_datasets"
CLASSIFIED_ROOT = DATASETS_DIR / "classified_datasets"
ORIGINALS_ROOT = DATASETS_DIR / "original_datasets_and_repos"
CAMUS_NIFTI = ORIGINALS_ROOT / "camus_public" / "database_nifti"
ECHONET_FILELIST = ORIGINALS_ROOT / "echonet_dynamic" / "FileList.csv"

#: Width of one ejection-fraction bin, in EF points.
EF_BIN_WIDTH = 5


class ClassificationError(RuntimeError):
    """Raised when a classification dataset cannot be derived."""


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def ef_5pct_bin(value: str | float | None) -> str | None:
    """Bin an ejection fraction into a fixed 5-point band: ``0_5`` … ``95_100``.

    The top bin is closed so that EF = 100 lands in ``95_100`` rather than opening a
    ``100_105`` bin that no guideline recognises.
    """
    if value is None or value == "":
        return None
    try:
        ef = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= ef <= 100.0:
        return None
    low = min(int(ef // EF_BIN_WIDTH) * EF_BIN_WIDTH, 100 - EF_BIN_WIDTH)
    return f"{low}_{low + EF_BIN_WIDTH}"


def slugify(label: str) -> str:
    """Directory-safe form of a class label; the verbatim label stays in classes.json."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    if not slug:
        raise ClassificationError(f"Label {label!r} slugifies to an empty string.")
    return slug


@dataclass(frozen=True)
class TaskSpec:
    """One classification task: which rows it covers and how each is labelled."""

    name: str
    source: str
    description: str
    origin: str
    label_of: Callable[[Row, Row | None], str | None]


@dataclass
class ClassificationResult:
    """What a run produced, read back from what was actually written."""

    task: str
    mode: str
    output_dir: Path
    n_frames: int
    class_counts: dict[str, int]
    skipped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "mode": self.mode,
            "output_dir": display_path(self.output_dir),
            "n_frames": self.n_frames,
            "class_counts": self.class_counts,
            "skipped": self.skipped,
            "warnings": self.warnings,
        }


TASKS: dict[str, TaskSpec] = {
    "camus_quality": TaskSpec(
        name="camus_quality",
        source="camus",
        description="CAMUS acquisition image quality, per view (Good/Medium/Poor).",
        origin="CAMUS Info_{view}.cfg -> ImageQuality",
        label_of=lambda row, origin: (
            (origin or {}).get("ImageQuality") or row.get("image_quality") or None
        ),
    ),
    "camus_ef_5pct": TaskSpec(
        name="camus_ef_5pct",
        source="camus",
        description="CAMUS ejection fraction in fixed 5-point bins (0_5 … 95_100).",
        origin="CAMUS Info_{view}.cfg -> EF",
        label_of=lambda row, origin: ef_5pct_bin((origin or {}).get("EF") or row.get("ef")),
    ),
    "echonet_ef_5pct": TaskSpec(
        name="echonet_ef_5pct",
        source="echonet",
        description="EchoNet ejection fraction in fixed 5-point bins (0_5 … 95_100).",
        origin="EchoNet FileList.csv -> EF",
        label_of=lambda row, origin: ef_5pct_bin((origin or {}).get("EF") or row.get("ef")),
    ),
}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def read_camus_cfgs(nifti_dir: Path = CAMUS_NIFTI) -> dict[tuple[str, str], dict[str, str]]:
    """Parse every ``Info_{view}.cfg`` into ``{(patient, view): fields}``."""
    out: dict[tuple[str, str], dict[str, str]] = {}
    if not Path(nifti_dir).is_dir():
        return out
    for patient_dir in sorted(p for p in Path(nifti_dir).iterdir() if p.is_dir()):
        for view in ("2CH", "4CH"):
            cfg = patient_dir / f"Info_{view}.cfg"
            if not cfg.is_file():
                continue
            fields: dict[str, str] = {}
            for line in cfg.read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            out[(patient_dir.name, view)] = fields
    return out


def read_echonet_filelist(filelist: Path = ECHONET_FILELIST) -> dict[str, dict[str, str]]:
    """Parse ``FileList.csv`` into ``{FileName: row}``."""
    if not Path(filelist).is_file():
        return {}
    with Path(filelist).open(encoding="utf-8", newline="") as handle:
        return {row["FileName"]: dict(row) for row in csv.DictReader(handle)}


def load_origins(task: TaskSpec) -> dict[Any, Row]:
    """Original label records for a task, keyed by :func:`origin_key`."""
    return read_camus_cfgs() if task.source == "camus" else read_echonet_filelist()


def origin_key(row: Row) -> tuple[str, str] | str:
    """The key that joins a processed row to its original label record."""
    if row.get("source") == "camus":
        return (row["patient_id"], row["view"])
    return row["patient_id"].removeprefix("echonet_")


def resolve_processed_dir(source: str, override: Path | None = None) -> Path:
    """Find the processed corpus holding ``source``, searching known layouts.

    Any of ``camus_processed/``, ``echonet_processed/`` or either nesting of
    ``unified_processed/`` is accepted, so the output of ``atria ingest`` works unchanged.

    Raises:
        ClassificationError: naming every location searched, when none qualifies.
    """
    if override is not None:
        chosen = Path(override)
        if not (chosen / "metadata.csv").is_file():
            raise ClassificationError(f"No metadata.csv in {display_path(chosen)}.")
        return chosen

    candidates = [
        PROCESSED_ROOT / f"{source}_processed",
        PROCESSED_ROOT / "unified_processed" / "unified_processed",
        PROCESSED_ROOT / "unified_processed",
    ]
    for candidate in candidates:
        if not (candidate / "metadata.csv").is_file() or not (candidate / "frames").is_dir():
            continue
        with (candidate / "metadata.csv").open(encoding="utf-8", newline="") as handle:
            if any(row.get("source") == source for row in csv.DictReader(handle)):
                return candidate

    searched = "\n  ".join(display_path(c) or str(c) for c in candidates)
    raise ClassificationError(
        f"No processed corpus containing {source!r} frames was found. Searched:\n  "
        f"{searched}\nRun `atria ingest {source} --source <raw> --output "
        f"datasets/processed_datasets/{source}_processed` first, or pass --dataset-dir."
    )


def read_metadata(dataset_dir: Path) -> list[dict[str, str]]:
    """Read the processed ``metadata.csv``; it defines which frames exist."""
    path = Path(dataset_dir) / "metadata.csv"
    if not path.is_file():
        raise ClassificationError(
            f"No metadata.csv in {display_path(dataset_dir)}. Expected a processed corpus "
            "directory containing frames/, tracings.json and metadata.csv."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ClassificationError(f"{display_path(path)} is empty.")
    return rows


# --------------------------------------------------------------------------- #
# labelling
# --------------------------------------------------------------------------- #
def label_rows(
    task: TaskSpec, dataset_dir: Path, origins: Mapping[Any, Row]
) -> tuple[list[tuple[dict[str, str], str]], Counter, list[str]]:
    """Verify the corpus, then pair every usable row with its class label.

    Raises:
        ClassificationError: if frames are missing, or nothing could be labelled.
    """
    rows = read_metadata(dataset_dir)
    frames_dir = Path(dataset_dir) / "frames"
    if not frames_dir.is_dir():
        raise ClassificationError(f"No frames/ directory in {display_path(dataset_dir)}.")

    subset = [r for r in rows if r.get("source") == task.source]
    if not subset:
        found = sorted({r.get("source", "") for r in rows})
        raise ClassificationError(
            f"No rows with source={task.source!r} in "
            f"{display_path(Path(dataset_dir) / 'metadata.csv')} (found: {found})."
        )

    on_disk = {p.stem for p in frames_dir.glob("*.png")}
    missing = {r["key"] for r in subset} - on_disk
    if missing:
        raise ClassificationError(
            f"{len(missing)} of {len(subset)} {task.source} frames named in metadata.csv "
            f"are absent from {display_path(frames_dir)} (e.g. {sorted(missing)[:3]}). "
            "Re-run `atria ingest` before deriving classes."
        )

    warnings: list[str] = []
    if origins:
        unjoined = {k for k in (origin_key(r) for r in subset) if k not in origins}
        if unjoined:
            warnings.append(
                f"{len(unjoined)} record(s) have no original metadata and fall back to "
                "the processed corpus."
            )
        extra = len(origins) - (len({origin_key(r) for r in subset}) - len(unjoined))
        if extra > 0:
            warnings.append(
                f"{extra} record(s) exist in the original metadata but were never "
                "processed into the corpus; they are absent from the output."
            )

    assignments: list[tuple[dict[str, str], str]] = []
    skipped: Counter = Counter()
    for row in subset:
        label = task.label_of(row, origins.get(origin_key(row)))
        if not label:
            skipped["no_label"] += 1
            continue
        assignments.append((row, label))

    if not assignments:
        raise ClassificationError(
            f"Task {task.name!r} labelled none of {len(subset)} {task.source} rows "
            f"(skipped: {dict(skipped)}). Check that {task.origin} is available."
        )
    return assignments, skipped, warnings


def _sort_key(label: str) -> tuple:
    """Order EF bins numerically (5_10 before 10_15), everything else alphabetically."""
    match = re.fullmatch(r"(\d+)_(\d+)", label)
    return (0, int(match.group(1))) if match else (1, label)


def _write_classes_json(
    out_dir: Path, task: TaskSpec, mode: str, counts: Counter,
    slugs: dict[str, str], skipped: Counter, n: int,
) -> None:
    (out_dir / "classes.json").write_text(
        json.dumps(
            {
                "task": task.name,
                "mode": mode,
                "description": task.description,
                "source": task.source,
                "label_origin": task.origin,
                "n_frames": n,
                "ef_bin_width": EF_BIN_WIDTH if "ef" in task.name else None,
                "classes": {
                    slugs[label]: {"label": label, "n_frames": counts[slugs[label]]}
                    for label in sorted(slugs, key=_sort_key)
                },
                "skipped": dict(skipped),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# products
# --------------------------------------------------------------------------- #
def build_metadata_mapping(
    task: TaskSpec,
    dataset_dir: Path | None = None,
    output_root: Path = CLASSIFIED_ROOT,
    *,
    origins: Mapping[Any, Row] | None = None,
    dry_run: bool = False,
) -> ClassificationResult:
    """Collection A — write ``mapping.csv`` linking every PNG to its class. Copies nothing."""
    dataset_dir = resolve_processed_dir(task.source, dataset_dir)
    origins = origins if origins is not None else load_origins(task)
    assignments, skipped, warnings = label_rows(task, dataset_dir, origins)

    slugs = {label: slugify(label) for label in {label for _, label in assignments}}
    counts = Counter(slugs[label] for _, label in assignments)
    out_dir = Path(output_root) / task.name

    if dry_run:
        return ClassificationResult(
            task.name, "metadata", out_dir, len(assignments),
            {k: counts[k] for k in sorted(counts, key=_sort_key)},
            dict(skipped), [*warnings, "dry run — nothing written"],
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(dataset_dir) / "frames"
    with (out_dir / "mapping.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["png", "key", "class", "class_label", "split", "source",
                        "patient_id", "view", "instant"],
        )
        writer.writeheader()
        for row, label in assignments:
            writer.writerow(
                {
                    # Repo-root relative, so the mapping is portable between machines.
                    "png": (frames_dir / f"{row['key']}.png")
                    .relative_to(PROJECT_ROOT).as_posix(),
                    "key": row["key"],
                    "class": slugs[label],
                    "class_label": label,
                    "split": row.get("split", ""),
                    "source": row.get("source", ""),
                    "patient_id": row.get("patient_id", ""),
                    "view": row.get("view", ""),
                    "instant": row.get("instant", ""),
                }
            )
    _write_classes_json(out_dir, task, "metadata", counts, slugs, skipped, len(assignments))
    logger.info("%s [metadata]: %d frames -> %s", task.name, len(assignments),
                display_path(out_dir))
    return ClassificationResult(
        task.name, "metadata", out_dir, len(assignments),
        {k: counts[k] for k in sorted(counts, key=_sort_key)}, dict(skipped), warnings,
    )


def build_class_dirs(
    task: TaskSpec,
    dataset_dir: Path | None = None,
    output_root: Path = CLASSIFIED_ROOT,
    *,
    origins: Mapping[Any, Row] | None = None,
    link: bool = False,
    dry_run: bool = False,
) -> ClassificationResult:
    """Collection B — materialise one directory per class (``ImageFolder`` layout)."""
    dataset_dir = resolve_processed_dir(task.source, dataset_dir)
    origins = origins if origins is not None else load_origins(task)
    assignments, skipped, warnings = label_rows(task, dataset_dir, origins)

    slugs = {label: slugify(label) for label in {label for _, label in assignments}}
    counts = Counter(slugs[label] for _, label in assignments)
    out_dir = Path(output_root) / task.name

    if dry_run:
        return ClassificationResult(
            task.name, "dirs", out_dir, len(assignments),
            {k: counts[k] for k in sorted(counts, key=_sort_key)},
            dict(skipped), [*warnings, "dry run — nothing written"],
        )

    frames_dir = Path(dataset_dir) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for row, label in assignments:
        target_dir = out_dir / slugs[label]
        target_dir.mkdir(parents=True, exist_ok=True)
        _place(frames_dir / f"{row['key']}.png", target_dir / f"{row['key']}.png", link=link)
    _write_classes_json(out_dir, task, "dirs", counts, slugs, skipped, len(assignments))
    logger.info("%s [dirs]: %d frames over %d classes -> %s", task.name, len(assignments),
                len(counts), display_path(out_dir))
    return ClassificationResult(
        task.name, "dirs", out_dir, len(assignments),
        {k: counts[k] for k in sorted(counts, key=_sort_key)}, dict(skipped), warnings,
    )


def _place(src: Path, dst: Path, *, link: bool) -> None:
    """Hard-link or copy one frame, replacing any previous run's file."""
    if dst.exists():
        dst.unlink()
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:  # different volume, or a filesystem without hard links
            pass
    shutil.copy2(src, dst)


BUILDERS: dict[str, Callable[..., ClassificationResult]] = {
    "metadata": build_metadata_mapping,
    "dirs": build_class_dirs,
}


# --------------------------------------------------------------------------- #
# shared CLI, used by the standalone scripts and by `atria classify-*`
# --------------------------------------------------------------------------- #
def build_parser(task: TaskSpec, mode: Mode, prog: str | None = None) -> argparse.ArgumentParser:
    """Argument surface shared by every classification entry point."""
    product = (
        "a single mapping.csv linking every PNG to its class (no images copied)"
        if mode == "metadata"
        else "one directory per class holding the PNGs (ImageFolder layout)"
    )
    parser = argparse.ArgumentParser(
        prog=prog,
        description=f"{task.description} Writes {product}.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Labels come from {task.origin}; none are invented.\n"
            f"Output: datasets/classified_datasets/{task.name}/\n"
            "All paths resolve from the repository root — nothing to edit after cloning."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Processed corpus. Default: auto-detected under "
                             "datasets/processed_datasets/.")
    parser.add_argument("--output-root", type=Path, default=CLASSIFIED_ROOT,
                        help="Default: datasets/classified_datasets/.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the class distribution without writing anything.")
    if mode == "dirs":
        parser.add_argument("--link", action="store_true",
                            help="Hard-link frames instead of copying (saves disk).")
    parser.add_argument("--log-level", default="ERROR", help="DEBUG, INFO, WARNING, ERROR.")
    return parser


def report(result: ClassificationResult) -> None:
    """Print a run summary using project-relative paths only."""
    print(f"\n{result.task}  [{result.mode}]  ({result.n_frames} frames, "
          f"{len(result.class_counts)} classes)")
    print(f"  -> {display_path(result.output_dir)}")
    for name, count in result.class_counts.items():
        print(f"     {name:<12} {count:6}")
    if result.skipped:
        print(f"  skipped: {result.skipped}")
    for warning in result.warnings:
        print(f"  note: {warning}")


def run_cli(task_name: str, mode: Mode, argv: list[str] | None = None,
            prog: str | None = None) -> int:
    """Entry point for one (task, mode) pair. Returns a process exit code."""
    task = TASKS[task_name]
    args = build_parser(task, mode, prog).parse_args(argv)
    configure(args.log_level)
    kwargs: dict[str, Any] = {"dry_run": args.dry_run}
    if mode == "dirs":
        kwargs["link"] = args.link
    try:
        result = BUILDERS[mode](task, args.dataset_dir, args.output_root, **kwargs)
    except ClassificationError as exc:
        print(f"{task_name} [{mode}]: FAILED — {exc}", file=sys.stderr)
        return 1
    report(result)
    return 0
