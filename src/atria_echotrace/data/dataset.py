"""Dataset repository for the three-artifact contract.

Production form of the notebook's dataset cells:
``load_echocardiographic_frame_data`` (notebook_as_py.txt L254-280), the per-split
sanity check (L371-389), and the distribution summaries printed at L273-280.

The contract, as produced by the upstream preprocessor and shipped in
``sample-dataset/``:

* ``frames/<stem>.png``   — one RGB frame per stem, variable resolution
* ``tracings.json``       — ``{stem: {..., lv_polygon, la_polygon, ...}}``
* ``metadata.csv``        — optional; one row per stem with ``key`` and ``split``
* ``manifest.json``       — optional; provenance, checksums, case pairing

``tracings.json`` is authoritative. ``manifest.json`` is used for case pairing when
present and is otherwise derived from the stems.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable

from ..config import display_path
from ..domain.geometry import denormalize_polygon, shoelace_area
from ..domain.metrics import Calibration
from ..logging_setup import get_logger

logger = get_logger("data")

#: Datasets known not to publish pixel spacing. EchoNet-Dynamic distributes
#: 112x112 frames with no calibration metadata, so its ``spacing`` fields in
#: ``tracings.json`` are placeholders (RESEARCH.md §0.5).
DATASETS_WITHOUT_SPACING: frozenset[str] = frozenset({"echonet"})

#: An exact 1.0 mm/px in *both* axes is the preprocessor's "unknown" placeholder;
#: real probe geometry gives values like CAMUS's 0.308 mm/px.
_UNCALIBRATED_SENTINEL = 1.0


class DatasetError(RuntimeError):
    """Raised when the dataset directory does not satisfy the contract."""


class RecordNotFound(LookupError):
    """Raised when a requested frame or case does not exist.

    A plain ``KeyError`` would be idiomatic, but ``str(KeyError("msg"))`` returns
    ``"'msg'"`` — the repr, complete with quotes — which then leaks into API error
    bodies. This carries a clean message instead.
    """


def _resolve_calibration(source: str, spacing_h: Any, spacing_w: Any) -> Calibration:
    """Decide whether a frame carries real pixel spacing.

    Withholding calibration is the safe default: a wrong spacing silently
    fabricates physical measurements (RESEARCH.md §3.2).
    """
    if source.lower() in DATASETS_WITHOUT_SPACING:
        return Calibration.unknown()
    try:
        h = float(spacing_h)
        w = float(spacing_w)
    except (TypeError, ValueError):
        return Calibration.unknown()
    if h <= 0 or w <= 0:
        return Calibration.unknown()
    if h == _UNCALIBRATED_SENTINEL and w == _UNCALIBRATED_SENTINEL:
        return Calibration.unknown()
    return Calibration(spacing_h=h, spacing_w=w, source="dataset")


@dataclass(frozen=True)
class Frame:
    """One echocardiographic frame and its ground-truth tracings."""

    stem: str
    case_id: str
    source: str
    view: str
    instant: str
    image_h: int
    image_w: int
    calibration: Calibration
    lv_polygon: list[list[int]] | None
    la_polygon: list[list[int]] | None
    split: str | None
    ef: float | None
    lv_points_raw: int | None
    la_points_raw: int | None

    @property
    def has_lv(self) -> bool:
        return bool(self.lv_polygon) and len(self.lv_polygon or []) >= 3

    @property
    def has_la(self) -> bool:
        return bool(self.la_polygon) and len(self.la_polygon or []) >= 3

    def polygon(self, target_structure: str) -> list[list[int]] | None:
        """Ground-truth polygon for ``"LV"`` or ``"LA"``."""
        return self.lv_polygon if target_structure.upper() == "LV" else self.la_polygon

    def as_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "case_id": self.case_id,
            "source": self.source,
            "view": self.view,
            "instant": self.instant,
            "image_h": self.image_h,
            "image_w": self.image_w,
            "spacing_h": self.calibration.spacing_h,
            "spacing_w": self.calibration.spacing_w,
            "calibration_source": self.calibration.source,
            "has_lv": self.has_lv,
            "has_la": self.has_la,
            "lv_polygon": self.lv_polygon,
            "la_polygon": self.la_polygon,
            "lv_polygon_px": denormalize_polygon(self.lv_polygon or [], self.image_h, self.image_w),
            "la_polygon_px": denormalize_polygon(self.la_polygon or [], self.image_h, self.image_w),
            "split": self.split,
            "ef": self.ef,
            "lv_points_raw": self.lv_points_raw,
            "la_points_raw": self.la_points_raw,
            "image_url": f"/api/dataset/frames/{self.stem}.png",
        }


def case_key_of(case_id: str, view: str) -> str:
    """Addressable key for a case.

    A patient can be imaged in more than one view (``patient0047`` appears in both
    2CH and 4CH in ``sample-dataset``), so ``case_id`` alone is not unique — keying
    on it silently discards frames. The manifest agrees: it lists 25 cases across 24
    distinct patient ids.
    """
    return f"{case_id}_{view}" if view else case_id


@dataclass(frozen=True)
class Case:
    """A patient/acquisition grouping the ED and ES frames of one view."""

    case_id: str
    source: str
    view: str
    frames: dict[str, Frame] = field(default_factory=dict)

    @property
    def case_key(self) -> str:
        """Unique key combining patient id and view."""
        return case_key_of(self.case_id, self.view)

    @property
    def ed(self) -> Frame | None:
        return self.frames.get("ED")

    @property
    def es(self) -> Frame | None:
        return self.frames.get("ES")

    @property
    def calibration(self) -> Calibration:
        for instant in ("ED", "ES"):
            frame = self.frames.get(instant)
            if frame is not None:
                return frame.calibration
        return Calibration.unknown()

    @property
    def ef(self) -> float | None:
        for frame in self.frames.values():
            if frame.ef is not None:
                return frame.ef
        return None

    @property
    def has_la(self) -> bool:
        return any(f.has_la for f in self.frames.values())

    def integrity_flags(self) -> list[str]:
        """Data-integrity warnings for this case's ground truth.

        ``es_area_exceeds_ed`` fires when the frame labelled end-systole encloses a
        larger area than the one labelled end-diastole. That is physiologically
        impossible for a contracting ventricle, so it indicates the ED/ES labels are
        transposed for this case. It is reported, never silently corrected: the
        labels feed the inference prompt's ``instant_name``, so quietly swapping them
        here would desynchronise the application from the convention the adapters
        were trained under (see RESEARCH.md §0.4-0.5).
        """
        flags: list[str] = []
        ed, es = self.ed, self.es
        if ed is not None and es is not None and ed.has_lv and es.has_lv:
            ed_area = shoelace_area(
                denormalize_polygon(ed.lv_polygon or [], ed.image_h, ed.image_w)
            )
            es_area = shoelace_area(
                denormalize_polygon(es.lv_polygon or [], es.image_h, es.image_w)
            )
            if ed_area > 0 and es_area >= ed_area:
                flags.append("es_area_exceeds_ed")
        return flags

    def summary(self) -> dict[str, Any]:
        """Compact record for the case-browser list."""
        reference = self.ed or self.es
        return {
            "case_key": self.case_key,
            "case_id": self.case_id,
            "source": self.source,
            "view": self.view,
            "instants": sorted(self.frames),
            "complete_pair": self.ed is not None and self.es is not None,
            "image_h": reference.image_h if reference else None,
            "image_w": reference.image_w if reference else None,
            "spacing_h": self.calibration.spacing_h,
            "spacing_w": self.calibration.spacing_w,
            "calibration_source": self.calibration.source,
            "has_la": self.has_la,
            "ef": self.ef,
            "integrity_flags": self.integrity_flags(),
            "frames": {
                instant: {
                    "stem": frame.stem,
                    "image_url": f"/api/dataset/frames/{frame.stem}.png",
                }
                for instant, frame in sorted(self.frames.items())
            },
        }

    def detail(self) -> dict[str, Any]:
        """Full record including ground-truth polygons for both structures."""
        return {
            **self.summary(),
            "frames": {instant: frame.as_dict() for instant, frame in sorted(self.frames.items())},
        }


@dataclass
class ValidationReport:
    """Outcome of validating a dataset directory (notebook cells L371-389, L273-280)."""

    dataset_dir: Path
    n_tracings: int = 0
    n_frames_present: int = 0
    missing_pngs: list[str] = field(default_factory=list)
    frames_without_lv: list[str] = field(default_factory=list)
    n_cases: int = 0
    incomplete_cases: list[str] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)
    view_counts: dict[str, int] = field(default_factory=dict)
    instant_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    lv_point_counts: dict[int, int] = field(default_factory=dict)
    la_point_counts: dict[int, int] = field(default_factory=dict)
    uncalibrated_sources: list[str] = field(default_factory=list)
    #: Cases whose ES trace encloses at least as much area as their ED trace.
    instant_area_anomalies: list[dict[str, Any]] = field(default_factory=list)
    has_metadata_csv: bool = False
    has_manifest: bool = False

    @property
    def ok(self) -> bool:
        return self.n_tracings > 0 and not self.missing_pngs

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dataset_dir": str(self.dataset_dir),
            "n_tracings": self.n_tracings,
            "n_frames_present": self.n_frames_present,
            "missing_pngs": self.missing_pngs,
            "frames_without_lv": self.frames_without_lv,
            "n_cases": self.n_cases,
            "incomplete_cases": self.incomplete_cases,
            "source_counts": self.source_counts,
            "view_counts": self.view_counts,
            "instant_counts": self.instant_counts,
            "split_counts": self.split_counts,
            "lv_point_counts": {str(k): v for k, v in sorted(self.lv_point_counts.items())},
            "la_point_counts": {str(k): v for k, v in sorted(self.la_point_counts.items())},
            "uncalibrated_sources": self.uncalibrated_sources,
            "instant_area_anomalies": self.instant_area_anomalies,
            "has_metadata_csv": self.has_metadata_csv,
            "has_manifest": self.has_manifest,
        }


class DatasetRepository:
    """Loads and serves the frames/tracings/metadata contract."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.frames_dir = self.dataset_dir / "frames"
        self.tracings_path = self.dataset_dir / "tracings.json"
        self.manifest_path = self.dataset_dir / "manifest.json"
        self.metadata_path = self.dataset_dir / "metadata.csv"

    # ---------------------------------------------------------------- loading
    @cached_property
    def manifest(self) -> dict[str, Any]:
        """Optional provenance manifest; empty when absent."""
        if not self.manifest_path.is_file():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable manifest %s: %s", self.manifest_path, exc)
            return {}

    @cached_property
    def _splits_from_metadata(self) -> dict[str, str]:
        """Map stem -> split from ``metadata.csv`` when it exists.

        The notebook read this with pandas and drove split selection from it
        (L265, L461-462). ``tracings.json`` also carries ``split``; the CSV wins
        when both are present, matching the notebook's precedence.
        """
        if not self.metadata_path.is_file():
            return {}
        try:
            import pandas as pd

            frame = pd.read_csv(self.metadata_path)
        except Exception as exc:  # pragma: no cover - depends on user-supplied file
            logger.warning("Ignoring unreadable metadata.csv %s: %s", self.metadata_path, exc)
            return {}
        if "key" not in frame.columns or "split" not in frame.columns:
            logger.warning(
                "metadata.csv lacks required 'key'/'split' columns; found %s",
                list(frame.columns),
            )
            return {}
        return {
            str(row.key): str(row.split)
            for row in frame.itertuples(index=False)
            if isinstance(getattr(row, "key", None), str) or getattr(row, "key", None) is not None
        }

    @cached_property
    def frames(self) -> dict[str, Frame]:
        """All frames keyed by stem.

        Raises:
            DatasetError: if ``tracings.json`` is missing or malformed.
        """
        if not self.tracings_path.is_file():
            raise DatasetError(
                f"tracings.json not found at {self.tracings_path}. "
                "Point ATRIA_DATASET_DIR at a directory containing "
                "frames/, tracings.json (and optionally metadata.csv, manifest.json)."
            )
        try:
            raw = json.loads(self.tracings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Could not read {self.tracings_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise DatasetError(
                f"{self.tracings_path} must contain a JSON object keyed by frame stem, "
                f"got {type(raw).__name__}."
            )

        case_by_stem = self._case_ids_from_manifest()
        splits = self._splits_from_metadata

        frames: dict[str, Frame] = {}
        for stem, entry in raw.items():
            if not isinstance(entry, dict):
                logger.warning("Skipping tracings entry %r: expected an object", stem)
                continue
            source = str(entry.get("source") or ("echonet" if stem.startswith("echonet") else "camus"))
            view = str(entry.get("view") or "")
            instant = str(entry.get("instant") or "")
            frames[stem] = Frame(
                stem=stem,
                case_id=case_by_stem.get(stem) or self._derive_case_id(stem, view, instant),
                source=source,
                view=view,
                instant=instant,
                image_h=int(entry.get("image_h") or 0),
                image_w=int(entry.get("image_w") or 0),
                calibration=_resolve_calibration(
                    source, entry.get("spacing_h"), entry.get("spacing_w")
                ),
                lv_polygon=entry.get("lv_polygon") or None,
                la_polygon=entry.get("la_polygon") or None,
                split=splits.get(stem) or (str(entry["split"]) if entry.get("split") else None),
                ef=float(entry["ef"]) if entry.get("ef") is not None else None,
                lv_points_raw=entry.get("lv_points_raw"),
                la_points_raw=entry.get("la_points_raw"),
            )
        logger.info("Loaded %d tracings from %s", len(frames), self.tracings_path)
        return frames

    def _case_ids_from_manifest(self) -> dict[str, str]:
        """Stem -> case_id, taken from the manifest when it provides one."""
        mapping: dict[str, str] = {}
        for item in self.manifest.get("frames", []) or []:
            if isinstance(item, dict) and item.get("stem") and item.get("case_id"):
                mapping[str(item["stem"])] = str(item["case_id"])
        return mapping

    @staticmethod
    def _derive_case_id(stem: str, view: str, instant: str) -> str:
        """Strip the trailing ``_<view>_<instant>`` to recover the case id.

        Only an exact suffix is removed. The previous implementation used
        ``stem.replace("_ED", "")``, which also mangles ids containing those
        letters elsewhere.
        """
        for suffix in (f"_{view}_{instant}", f"_{instant}"):
            if suffix != "_" and suffix and stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    @cached_property
    def cases(self) -> dict[str, Case]:
        """All cases keyed by ``case_key`` (patient id + view), grouping ED/ES frames."""
        grouped: dict[str, Case] = {}
        for frame in self.frames.values():
            key = case_key_of(frame.case_id, frame.view)
            case = grouped.get(key)
            if case is None:
                case = Case(case_id=frame.case_id, source=frame.source, view=frame.view)
                grouped[key] = case
            existing = case.frames.get(frame.instant)
            if existing is not None:
                # Two frames claiming the same case/view/instant means the dataset
                # violates the contract; dropping one silently would hide it.
                logger.warning(
                    "Duplicate %s frame for case %s: %r and %r. Keeping %r.",
                    frame.instant,
                    key,
                    existing.stem,
                    frame.stem,
                    existing.stem,
                )
                continue
            case.frames[frame.instant] = frame
        return dict(sorted(grouped.items()))

    # ---------------------------------------------------------------- queries
    def get_frame(self, stem: str) -> Frame:
        """Look up one frame by exact stem.

        Raises:
            RecordNotFound: if the stem is unknown. Deliberately exact — the previous
                implementation substring-matched and could return another
                patient's frame.
        """
        try:
            return self.frames[stem]
        except KeyError as exc:
            raise RecordNotFound(f"Unknown frame stem: {stem!r}") from exc

    def get_case(self, case_key: str) -> Case:
        """Look up one case by ``case_key`` (``"<case_id>_<view>"``).

        A bare ``case_id`` is also accepted when exactly one view exists for that
        patient, so single-view datasets need not know about the composite key.

        Raises:
            RecordNotFound: if the key is unknown, or ambiguous across multiple views.
        """
        case = self.cases.get(case_key)
        if case is not None:
            return case
        matches = [c for c in self.cases.values() if c.case_id == case_key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RecordNotFound(
                f"Case id {case_key!r} is ambiguous across views "
                f"{sorted(c.view for c in matches)}; use one of "
                f"{sorted(c.case_key for c in matches)}."
            )
        raise RecordNotFound(f"Unknown case: {case_key!r}")

    def list_cases(
        self,
        source: str | None = None,
        view: str | None = None,
        complete_pairs_only: bool = False,
    ) -> list[Case]:
        """Filtered case list for the browser."""
        result: Iterable[Case] = self.cases.values()
        if source:
            result = (c for c in result if c.source.lower() == source.lower())
        if view:
            result = (c for c in result if c.view.upper() == view.upper())
        if complete_pairs_only:
            result = (c for c in result if c.ed is not None and c.es is not None)
        return list(result)

    def frames_in_split(self, split: str) -> list[Frame]:
        """Frames belonging to ``split`` (notebook L375, L461-462)."""
        return [f for f in self.frames.values() if (f.split or "").lower() == split.lower()]

    def frame_path(self, stem: str) -> Path:
        """Filesystem path of a known frame's PNG.

        Resolving through :meth:`get_frame` first means only stems present in
        ``tracings.json`` are ever addressable, which also forecloses path
        traversal via crafted stems.

        Raises:
            RecordNotFound: unknown stem, or a stem that escapes the frames directory.
            FileNotFoundError: known stem whose PNG is missing.
        """
        frame = self.get_frame(stem)
        path = (self.frames_dir / f"{frame.stem}.png").resolve()
        frames_root = self.frames_dir.resolve()
        if not path.is_relative_to(frames_root):
            raise RecordNotFound(f"Frame stem escapes the frames directory: {stem!r}")
        if not path.is_file():
            # Project-relative: this message reaches HTTP clients as a 404 body.
            raise FileNotFoundError(
                f"Frame image missing on disk: {display_path(path)}"
            )
        return path

    # ------------------------------------------------------------- validation
    def validate(self) -> ValidationReport:
        """Check the dataset against the contract and summarise its composition."""
        report = ValidationReport(dataset_dir=self.dataset_dir)
        report.has_manifest = self.manifest_path.is_file()
        report.has_metadata_csv = self.metadata_path.is_file()

        frames = self.frames
        report.n_tracings = len(frames)

        uncalibrated: set[str] = set()
        for stem, frame in frames.items():
            if (self.frames_dir / f"{stem}.png").is_file():
                report.n_frames_present += 1
            else:
                report.missing_pngs.append(stem)

            if not frame.has_lv:
                report.frames_without_lv.append(stem)

            report.source_counts[frame.source] = report.source_counts.get(frame.source, 0) + 1
            report.view_counts[frame.view] = report.view_counts.get(frame.view, 0) + 1
            report.instant_counts[frame.instant] = report.instant_counts.get(frame.instant, 0) + 1
            split = frame.split or "unassigned"
            report.split_counts[split] = report.split_counts.get(split, 0) + 1

            if frame.lv_polygon:
                n = len(frame.lv_polygon)
                report.lv_point_counts[n] = report.lv_point_counts.get(n, 0) + 1
            if frame.la_polygon:
                n = len(frame.la_polygon)
                report.la_point_counts[n] = report.la_point_counts.get(n, 0) + 1

            if not frame.calibration.is_known:
                uncalibrated.add(frame.source)

        report.uncalibrated_sources = sorted(uncalibrated)
        cases = self.cases
        report.n_cases = len(cases)
        report.incomplete_cases = [
            c.case_key for c in cases.values() if c.ed is None or c.es is None
        ]

        for case in cases.values():
            if "es_area_exceeds_ed" not in case.integrity_flags():
                continue
            ed, es = case.ed, case.es
            assert ed is not None and es is not None  # implied by the flag
            report.instant_area_anomalies.append(
                {
                    "case_key": case.case_key,
                    "source": case.source,
                    "ed_area_px": round(
                        shoelace_area(
                            denormalize_polygon(ed.lv_polygon or [], ed.image_h, ed.image_w)
                        ),
                        1,
                    ),
                    "es_area_px": round(
                        shoelace_area(
                            denormalize_polygon(es.lv_polygon or [], es.image_h, es.image_w)
                        ),
                        1,
                    ),
                    "ef": case.ef,
                }
            )
        if report.instant_area_anomalies:
            logger.warning(
                "%d case(s) have an ES trace at least as large as their ED trace, "
                "which suggests transposed ED/ES labels: %s",
                len(report.instant_area_anomalies),
                ", ".join(a["case_key"] for a in report.instant_area_anomalies[:5]),
            )
        return report
