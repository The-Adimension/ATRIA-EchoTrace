"""Revision persistence and export bundle generation.

Production form of the notebook's ``save_polygon_backend``
(notebook_as_py.txt L1318-1375), which wrote one JSON of model+user polygons and one
4-panel PNG per frame. The file set is extended to the contract used by the author's
deployed Space (``visualization.export_tracing_package``): per-phase JSON, per-phase
coordinate CSVs, a metrics CSV, the 4-panel PNGs, a clinical summary JSON, and a ZIP
bundling all of it.

Each revision is written to its own directory so an export is atomic and
self-describing:

    outputs/revisions/<revision_id>/
        revision.json                  full record (polygons, metrics, provenance)
        tracing_data_img0.json         ED  — notebook-compatible shape
        tracing_data_img1.json         ES  — notebook-compatible shape
        tracing_coordinates_ed.csv
        tracing_coordinates_es.csv
        clinical_metrics_summary.csv
        tracing_vis_img0.png           ED 4-panel figure
        tracing_vis_img1.png           ES 4-panel figure
        echotrace_clinical_summary.json
        ATRIA_EchoTrace_<revision_id>.zip
"""

from __future__ import annotations

import csv
import io
import json
import re
import secrets
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ..domain.geometry import Polygon
from ..domain.metrics import ChamberMetrics
from ..domain.structures import NORM_SCALE
from ..logging_setup import get_logger
from ..render.figures import four_panel_figure, write_png

logger = get_logger("export")

#: Revision ids are server-generated; this pattern also gates path construction.
REVISION_ID_RE = re.compile(r"^rev_\d{10,}_[0-9a-f]{6}$")

_PHASE_LABELS = {"ED": "End-Diastole (ED)", "ES": "End-Systole (ES)"}
#: The notebook indexed its two editors 0 (ED) and 1 (ES); preserved for compatibility.
_PHASE_INDEX = {"ED": 0, "ES": 1}


def new_revision_id() -> str:
    """Generate a sortable, unique revision id (notebook used time + uuid4 hex)."""
    return f"rev_{int(time.time())}_{secrets.token_hex(3)}"


@dataclass
class PhaseTracing:
    """One traced cardiac phase within a revision."""

    instant: str
    image: Image.Image
    model_polygon: list[list[int]] = field(default_factory=list)
    user_polygon: list[list[int]] = field(default_factory=list)
    ground_truth_polygon: list[list[int]] | None = None
    stem: str | None = None

    @property
    def index(self) -> int:
        return _PHASE_INDEX.get(self.instant.upper(), 0)

    @property
    def label(self) -> str:
        return _PHASE_LABELS.get(self.instant.upper(), self.instant)


def _coordinate_rows(
    polygon: Polygon,
    image_w: int,
    image_h: int,
    norm_scale: int = NORM_SCALE,
) -> list[list[Any]]:
    """Build CSV rows pairing normalised and pixel coordinates."""
    rows: list[list[Any]] = []
    for index, point in enumerate(polygon or []):
        y_norm, x_norm = point[0], point[1]
        rows.append(
            [
                index,
                y_norm,
                x_norm,
                round(float(y_norm) / norm_scale * image_h, 2),
                round(float(x_norm) / norm_scale * image_w, 2),
            ]
        )
    return rows


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _flatten_metrics(metrics: ChamberMetrics) -> list[list[Any]]:
    """Flatten metrics into ``Metric_Name, Value`` rows for the summary CSV."""
    data = metrics.as_dict()
    rows: list[list[Any]] = []
    for phase in ("ed", "es"):
        phase_data = data[phase]
        assert isinstance(phase_data, dict)
        for name, value in phase_data.items():
            rows.append([f"{phase.upper()}_{name}", "" if value is None else value])
    rows.append(["FAC_percent", "" if data["fac_percent"] is None else data["fac_percent"]])
    rows.append(["calibration_source", data["calibration_source"]])
    return rows


def write_revision_bundle(
    revision_id: str,
    output_root: Path,
    phases: list[PhaseTracing],
    metrics: ChamberMetrics,
    case_label: str,
    provenance: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    """Persist a clinician revision and generate every export artefact.

    Args:
        revision_id: Id from :func:`new_revision_id`.
        output_root: ``settings.revisions_dir``.
        phases: One entry per traced phase; ED and ES for a normal case.
        metrics: Paired metrics for the revised polygons.
        case_label: Human-readable case identifier for figure titles.
        provenance: Recorded context (case key, adapter, prompt variant, structure,
            device, model id, calibration source, integrity flags).
        notes: Free-text clinician notes.

    Returns:
        The revision record, including a ``files`` map of artefact name to
        repository-relative path and the download URL for the ZIP.

    Raises:
        ValueError: if ``revision_id`` is malformed or no phases are supplied.
    """
    if not REVISION_ID_RE.match(revision_id):
        raise ValueError(f"Malformed revision id: {revision_id!r}")
    if not phases:
        raise ValueError("A revision must contain at least one traced phase.")

    directory = output_root / revision_id
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    for phase in phases:
        width, height = phase.image.size
        index = phase.index

        # 1. Per-phase JSON — same keys the notebook wrote, plus context.
        phase_json = {
            "phase": phase.label,
            "instant": phase.instant,
            "image_index": index,
            "stem": phase.stem,
            "image_w": width,
            "image_h": height,
            "norm_scale": NORM_SCALE,
            "model_polygon_2d": phase.model_polygon or [],
            "user_polygon_2d": phase.user_polygon or [],
            "ground_truth_polygon_2d": phase.ground_truth_polygon,
        }
        name = f"tracing_data_img{index}.json"
        (directory / name).write_text(json.dumps(phase_json, indent=4), encoding="utf-8")
        files[name] = name

        # 2. Per-phase coordinate CSV of the clinician's polygon.
        csv_name = f"tracing_coordinates_{phase.instant.lower()}.csv"
        _write_csv(
            directory / csv_name,
            ["Vertex_Index", "Y_Normalized", "X_Normalized", "Y_Pixel", "X_Pixel"],
            _coordinate_rows(phase.user_polygon, width, height),
        )
        files[csv_name] = csv_name

        # 3. 4-panel comparison figure.
        png_name = f"tracing_vis_img{index}.png"
        write_png(
            four_panel_figure(
                image=phase.image,
                model_polygon=phase.model_polygon or [],
                user_polygon=phase.user_polygon or [],
                phase_label=phase.instant,
                case_label=case_label,
            ),
            directory / png_name,
        )
        files[png_name] = png_name

    # 4. Metrics summary CSV.
    _write_csv(
        directory / "clinical_metrics_summary.csv",
        ["Metric_Name", "Value"],
        _flatten_metrics(metrics),
    )
    files["clinical_metrics_summary.csv"] = "clinical_metrics_summary.csv"

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # `timestamp_utc` and the revision id are both second-granularity, so two saves in
    # the same second are indistinguishable by either. `export_corpus` needs a strict
    # order to decide which revision of a frame supersedes which, so record the raw
    # clock too. Human-facing fields are unchanged.
    created_unix = time.time()

    # 5. Clinical summary JSON (shape follows the author's Space).
    summary = {
        "project": "ATRIA EchoTrace — MedGemma 1.5 contour tracing workspace",
        "github": "https://github.com/The-Adimension/ATRIA-EchoTrace",
        "huggingface_org": "https://huggingface.co/The-Adimension",
        "revision_id": revision_id,
        "case": case_label,
        "timestamp_utc": timestamp,
        "provenance": provenance,
        "metrics": metrics.as_dict(),
        "images": {
            phase.instant: {
                "vertices_model": len(phase.model_polygon or []),
                "vertices_user": len(phase.user_polygon or []),
                "image_w": phase.image.size[0],
                "image_h": phase.image.size[1],
            }
            for phase in phases
        },
        "disclaimer": (
            "Research use only. Not a medical device. All AI-generated contours are "
            "preliminary proposals requiring review by qualified personnel."
        ),
    }
    (directory / "echotrace_clinical_summary.json").write_text(
        json.dumps(summary, indent=4), encoding="utf-8"
    )
    files["echotrace_clinical_summary.json"] = "echotrace_clinical_summary.json"

    # 6. Full revision record — the canonical machine-readable output. It is listed in
    # `files` alongside the rest so the UI offers it for download too.
    zip_name = f"ATRIA_EchoTrace_{revision_id}.zip"
    files["revision.json"] = "revision.json"
    files[zip_name] = zip_name

    record: dict[str, Any] = {
        "revision_id": revision_id,
        "case": case_label,
        "timestamp_utc": timestamp,
        "created_unix": created_unix,
        "notes": notes,
        "provenance": provenance,
        "metrics": metrics.as_dict(),
        "norm_scale": NORM_SCALE,
        "phases": {
            phase.instant: {
                "stem": phase.stem,
                "image_w": phase.image.size[0],
                "image_h": phase.image.size[1],
                "model_polygon_2d": phase.model_polygon or [],
                "user_polygon_2d": phase.user_polygon or [],
                "ground_truth_polygon_2d": phase.ground_truth_polygon,
            }
            for phase in phases
        },
        "files": files,
    }
    record["download_url"] = f"/api/revisions/{revision_id}/export.zip"
    (directory / "revision.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    # 7. ZIP bundle of every artefact except the archive itself.
    with zipfile.ZipFile(directory / zip_name, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            if name == zip_name:
                continue
            archive.write(directory / name, arcname=name)

    logger.info(
        "Wrote revision %s for %s (%d phases, %d artefacts)",
        revision_id,
        case_label,
        len(phases),
        len(record["files"]),
    )
    return record


def revision_dir(revision_id: str, output_root: Path) -> Path:
    """Resolve a revision directory by id.

    Raises:
        ValueError: if the id is malformed or escapes ``output_root``.
        FileNotFoundError: if no such revision exists.
    """
    if not REVISION_ID_RE.match(revision_id):
        raise ValueError(f"Malformed revision id: {revision_id!r}")
    directory = (output_root / revision_id).resolve()
    if not directory.is_relative_to(output_root.resolve()):
        raise ValueError(f"Revision id escapes the output directory: {revision_id!r}")
    if not directory.is_dir():
        raise FileNotFoundError(f"Unknown revision id: {revision_id}")
    return directory


def load_revision(revision_id: str, output_root: Path) -> dict[str, Any]:
    """Load a persisted revision record.

    Raises:
        ValueError / FileNotFoundError: as :func:`revision_dir`.
    """
    path = revision_dir(revision_id, output_root) / "revision.json"
    if not path.is_file():
        raise FileNotFoundError(f"Revision {revision_id} has no revision.json")
    return json.loads(path.read_text(encoding="utf-8"))


def list_revisions(output_root: Path, limit: int = 200) -> list[dict[str, Any]]:
    """List persisted revisions, newest first, with compact summaries."""
    if not output_root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for directory in sorted(output_root.iterdir(), reverse=True):
        if not directory.is_dir() or not REVISION_ID_RE.match(directory.name):
            continue
        record_path = directory / "revision.json"
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable revision %s: %s", directory.name, exc)
            continue
        metrics = record.get("metrics") or {}
        entries.append(
            {
                "revision_id": record.get("revision_id", directory.name),
                "case": record.get("case"),
                "timestamp_utc": record.get("timestamp_utc"),
                "fac_percent": metrics.get("fac_percent"),
                "calibration_source": metrics.get("calibration_source"),
                "notes": record.get("notes", ""),
                "provenance": record.get("provenance", {}),
                "download_url": f"/api/revisions/{record.get('revision_id', directory.name)}/export.zip",
            }
        )
        if len(entries) >= limit:
            break
    return entries


def zip_bytes(revision_id: str, output_root: Path) -> bytes:
    """Read a revision's ZIP bundle, rebuilding it if it is missing."""
    directory = revision_dir(revision_id, output_root)
    archive_path = directory / f"ATRIA_EchoTrace_{revision_id}.zip"
    if archive_path.is_file():
        return archive_path.read_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(directory.iterdir()):
            if item.is_file() and item.suffix != ".zip":
                archive.write(item, arcname=item.name)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
#  Ground-truth evolution loop: revisions -> trainable corpus                  #
# --------------------------------------------------------------------------- #
#
# Notebook-readme.txt states the intent: "Revised data feeds directly back into
# iterative model improvement or serves as enhanced ground truth" (DEITY *Ethics*:
# "Ground-truth evolution loop"). The notebook itself never closed that loop — it wrote
# JSON and PNG to /content and stopped. This emits the *existing* three-artifact
# contract, so `DatasetRepository` and `atria train` consume it unchanged:
#
#     <output_dir>/frames/<stem>.png
#     <output_dir>/tracings.json
#     <output_dir>/metadata.csv
#
# The clinician's `user_polygon` becomes the ground truth. `model_polygon` is carried
# only as provenance, never as a label.

#: Uploaded phases are recorded as ``upload:<id>``; corpus stems must be filenames.
_UPLOAD_STEM_RE = re.compile(r"^upload:([0-9a-f]+)$")


def _corpus_frame_source(
    stem: str, dataset_dir: Path | None, uploads_dir: Path
) -> tuple[str, Path | None, str]:
    """Resolve one revision phase to ``(corpus_stem, image_path, source)``."""
    upload = _UPLOAD_STEM_RE.match(stem)
    if upload:
        return f"upload_{upload.group(1)}", uploads_dir / f"{upload.group(1)}.png", "upload"
    path = dataset_dir / "frames" / f"{stem}.png" if dataset_dir else None
    return stem, path, "echonet" if stem.startswith("echonet") else "camus"


def _source_views(dataset_dir: Path | None) -> dict[str, str]:
    """Map dataset stem -> apical view, read straight from the source contract.

    Revisions record their own view, but ones saved before that field existed do not.
    For dataset-backed frames the view is *recoverable* rather than guessable, so it is
    recovered here; uploads have no such source and are skipped instead.
    """
    if not dataset_dir:
        return {}
    path = dataset_dir / "tracings.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        stem: str(entry["view"])
        for stem, entry in (raw.items() if isinstance(raw, dict) else [])
        if isinstance(entry, dict) and entry.get("view")
    }


def export_corpus(
    output_dir: Path,
    output_root: Path,
    uploads_dir: Path,
    dataset_dir: Path | None = None,
    revision_ids: list[str] | None = None,
    split: str = "train",
) -> dict[str, Any]:
    """Materialise revised contours as a dataset that ``atria train`` accepts.

    Args:
        output_dir: Destination for ``frames/``, ``tracings.json``, ``metadata.csv``.
        output_root: The revisions root (``outputs/revisions``).
        uploads_dir: Where uploaded frames live, for upload-backed phases.
        dataset_dir: Source dataset, for dataset-backed phases.
        revision_ids: Restrict to these revisions; ``None`` exports every one.
        split: Split label written for every frame.

    Returns:
        A summary with ``frames``, ``revisions``, ``skipped`` and ``dir``.

    Raises:
        ValueError: if no revision yielded a usable frame.
    """
    wanted = set(revision_ids or [])
    records = [
        load_revision(entry["revision_id"], output_root)
        for entry in list_revisions(output_root, limit=10_000)
        if not wanted or entry["revision_id"] in wanted
    ]
    missing = wanted - {record["revision_id"] for record in records}
    if missing:
        raise ValueError(f"No such revision(s): {', '.join(sorted(missing))}")

    source_views = _source_views(dataset_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    tracings: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    used_revisions: set[str] = set()

    # Oldest first, so a later revision of the same frame supersedes an earlier one.
    #
    # The key must be a *total* order. Sorting on `timestamp_utc` alone is not: it has
    # second granularity, and on a tie Python's stable sort preserves the input order,
    # which `list_revisions()` supplies newest-first — so two saves in the same second
    # let the OLDER polygon overwrite the newer one and land in the corpus as ground
    # truth. `created_unix` is the raw clock; `revision_id` breaks any remaining tie
    # deterministically. Records written before `created_unix` existed sort as 0.0 and
    # therefore stay behind every newer record, which is the correct fallback.
    for record in sorted(
        records,
        key=lambda r: (float(r.get("created_unix") or 0.0), str(r.get("revision_id", ""))),
    ):
        provenance = record.get("provenance") or {}
        structure = str(provenance.get("target_structure") or "LV").upper()
        poly_key = "la_polygon" if structure == "LA" else "lv_polygon"
        views = provenance.get("views") or {}
        for instant, phase in (record.get("phases") or {}).items():
            polygon = phase.get("user_polygon_2d") or []
            if len(polygon) < 3:
                continue
            raw_stem = str(phase.get("stem") or "")
            stem, image_path, source = _corpus_frame_source(raw_stem, dataset_dir, uploads_dir)
            if not image_path or not image_path.is_file():
                skipped.append(f"{record['revision_id']}/{instant}: frame image unavailable")
                continue
            view = str(views.get(instant) or source_views.get(raw_stem) or "")
            if not view:
                skipped.append(f"{record['revision_id']}/{instant}: no apical view recorded")
                continue
            (frames_dir / f"{stem}.png").write_bytes(image_path.read_bytes())
            tracings[stem] = {
                "source": source,
                "view": view,
                "instant": instant,
                "image_h": int(phase.get("image_h") or 0),
                "image_w": int(phase.get("image_w") or 0),
                "spacing_h": provenance.get("spacing_h"),
                "spacing_w": provenance.get("spacing_w"),
                poly_key: [[int(p[0]), int(p[1])] for p in polygon],
                "split": split,
                # Provenance — deliberately *not* ground truth.
                "revised_from": record["revision_id"],
                "model_polygon_2d": phase.get("model_polygon_2d") or [],
                "calibration_source": provenance.get("calibration_source"),
                "dataset_integrity_flags": provenance.get("dataset_integrity_flags") or [],
            }
            used_revisions.add(record["revision_id"])

    if not tracings:
        raise ValueError(
            "No revision produced a usable frame. A frame needs a revised polygon of at "
            "least 3 vertices, a recorded apical view, and its source image still on disk."
            + (f" Skipped: {'; '.join(skipped)}" if skipped else "")
        )

    (output_dir / "tracings.json").write_text(json.dumps(tracings, indent=2), encoding="utf-8")
    _write_csv(
        output_dir / "metadata.csv",
        ["key", "split", "source", "view", "instant", "revised_from"],
        [
            [stem, entry["split"], entry["source"], entry["view"], entry["instant"], entry["revised_from"]]
            for stem, entry in sorted(tracings.items())
        ],
    )
    logger.info(
        "Exported %d frame(s) from %d revision(s) to %s",
        len(tracings),
        len(used_revisions),
        output_dir,
    )
    return {
        "dir": output_dir,
        "frames": len(tracings),
        "revisions": sorted(used_revisions),
        "skipped": skipped,
        "split": split,
    }
