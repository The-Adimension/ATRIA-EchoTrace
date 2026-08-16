"""Ingest: the vendored reference preprocessors and the wrappers around them.

The strongest assertion available is *provenance*: run the vendored script over the real
raw data and require its output to equal the shipped training corpus exactly. Those
tests are skipped when `datasets/` is absent, so a clean checkout still runs green.

Everything else here checks the wrapper contract — layout validation, dependency
reporting, result summarisation — which needs neither the raw data nor the extras.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atria_echotrace.data.ingest import IngestError, IngestResult, summarise_output

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets"
CAMUS_ROOT = DATASETS / "original_datasets_and_repos" / "camus_public"
ECHONET_ROOT = DATASETS / "original_datasets_and_repos" / "echonet_dynamic"
UNIFIED = DATASETS / "processed_datasets" / "unified_processed" / "unified_processed"
VENDORED = REPO / "src" / "atria_echotrace" / "data" / "ingest" / "reference"


# --------------------------------------------------------------- vendored files
def test_reference_scripts_are_vendored() -> None:
    """`atria ingest` depends on these being present in the package."""
    assert (VENDORED / "preprocess_camus.py").is_file()
    assert (VENDORED / "preprocess_echonet.py").is_file()
    assert (VENDORED / "README.md").is_file()


def test_vendored_scripts_are_byte_identical_to_the_supplied_originals() -> None:
    """Any drift here silently breaks reproduction of the training corpus."""
    originals = DATASETS / "data_processing_scripts"
    if not originals.is_dir():
        pytest.skip("datasets/data_processing_scripts not present")
    for name in ("preprocess_camus.py", "preprocess_echonet.py"):
        assert (VENDORED / name).read_bytes() == (originals / name).read_bytes(), (
            f"{name} has diverged from the supplied original"
        )


# ------------------------------------------------------------------ provenance
@pytest.mark.ingest
def test_reference_camus_reproduces_the_shipped_corpus() -> None:
    """The vendored CAMUS extractor must reproduce the training data exactly.

    This is the assertion that justifies vendoring rather than reimplementing: the
    polygon this code produces from the raw mask is *identical* to the one the adapters
    were fine-tuned against, not merely close to it.
    """
    pytest.importorskip("SimpleITK", reason="ingest extra not installed")
    if not CAMUS_ROOT.is_dir() or not (UNIFIED / "tracings.json").is_file():
        pytest.skip("raw CAMUS data or unified corpus not present")

    import sys

    import numpy as np
    import SimpleITK as sitk

    sys.path.insert(0, str(VENDORED))
    from preprocess_camus import extract_contour_polygon  # type: ignore[import-not-found]

    shipped = json.loads((UNIFIED / "tracings.json").read_text(encoding="utf-8"))

    checked = 0
    for patient in ("patient0001", "patient0002"):
        for view in ("2CH", "4CH"):
            for instant in ("ED", "ES"):
                key = f"{patient}_{view}_{instant}"
                gt_path = CAMUS_ROOT / "database_nifti" / patient / f"{key}_gt.nii.gz"
                if key not in shipped or not gt_path.is_file():
                    continue
                mask = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))))

                lv, lv_raw = extract_contour_polygon((mask == 1).astype(np.uint8), 30)
                la, _ = extract_contour_polygon((mask == 3).astype(np.uint8), 30)

                assert lv == shipped[key]["lv_polygon"], f"{key} LV polygon differs"
                assert la == shipped[key]["la_polygon"], f"{key} LA polygon differs"
                assert lv_raw == shipped[key]["lv_points_raw"], f"{key} raw count differs"
                checked += 1

    assert checked >= 4, f"expected several frames to compare, checked {checked}"


@pytest.mark.ingest
def test_reference_camus_polygons_carry_the_documented_conventions() -> None:
    """Index subsampling duplicates the closing vertex and truncates coordinates."""
    pytest.importorskip("SimpleITK", reason="ingest extra not installed")
    if not CAMUS_ROOT.is_dir():
        pytest.skip("raw CAMUS data not present")

    import sys

    import numpy as np
    import SimpleITK as sitk

    sys.path.insert(0, str(VENDORED))
    from preprocess_camus import extract_contour_polygon  # type: ignore[import-not-found]

    gt = CAMUS_ROOT / "database_nifti" / "patient0001" / "patient0001_4CH_ED_gt.nii.gz"
    mask = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(gt))))
    polygon, raw = extract_contour_polygon((mask == 1).astype(np.uint8), 30)

    assert len(polygon) == 30
    assert raw > 500, "a CAMUS LV boundary should have hundreds of raw contour points"
    # np.linspace includes both endpoints of a closed ring (RESEARCH.md §8.2).
    assert polygon[0] == polygon[-1]
    for y, x in polygon:
        assert isinstance(y, int) and isinstance(x, int)
        assert 0 <= y <= 1000 and 0 <= x <= 1000


@pytest.mark.ingest
def test_reference_echonet_polygon_matches_the_shipped_corpus() -> None:
    """The chord→polygon construction must reproduce the training data exactly."""
    pytest.importorskip("cv2", reason="ingest extra not installed")
    if not (ECHONET_ROOT / "VolumeTracings.csv").is_file():
        pytest.skip("raw EchoNet data not present")
    if not (UNIFIED / "tracings.json").is_file():
        pytest.skip("unified corpus not present")

    import csv
    import sys
    from collections import defaultdict

    import numpy as np

    sys.path.insert(0, str(VENDORED))
    from preprocess_echonet import build_lv_polygon  # type: ignore[import-not-found]

    shipped = json.loads((UNIFIED / "tracings.json").read_text(encoding="utf-8"))
    wanted = [k for k in shipped if k.startswith("echonet_")][:2]
    if not wanted:
        pytest.skip("no EchoNet entries in the unified corpus")

    videos = {k.split("_")[1] for k in wanted}
    trace: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    with (ECHONET_ROOT / "VolumeTracings.csv").open() as handle:
        next(handle)
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 6 or parts[0].removesuffix(".avi") not in videos:
                continue
            trace[parts[0].removesuffix(".avi")][int(float(parts[5]))].append(
                tuple(map(float, parts[1:5]))
            )

    rows = {r["FileName"]: r for r in csv.DictReader((ECHONET_ROOT / "FileList.csv").open())}

    checked = 0
    for key in wanted:
        entry = shipped[key]
        video = key.split("_")[1]
        meta = rows.get(video)
        if meta is None or video not in trace:
            continue
        frame_idx = int(float(meta["esf" if entry["instant"] == "ES" else "edf"]))
        if frame_idx not in trace[video]:
            continue
        polygon, raw = build_lv_polygon(
            np.array(trace[video][frame_idx]),
            int(meta["FrameHeight"]),
            int(meta["FrameWidth"]),
            30,
        )
        assert polygon == entry["lv_polygon"], f"{key} polygon differs"
        assert raw == entry["lv_points_raw"]
        checked += 1

    assert checked >= 1, "expected at least one EchoNet frame to compare"


# --------------------------------------------------------------- run wrapper
@pytest.mark.ingest
def test_camus_wrapper_reproduces_the_corpus_and_writes_its_log(tmp_path: Path) -> None:
    """Full wrapper path: real NIfTI in, corpus-identical artefacts + a real log out."""
    pytest.importorskip("SimpleITK", reason="ingest extra not installed")
    if not CAMUS_ROOT.is_dir() or not (UNIFIED / "tracings.json").is_file():
        pytest.skip("raw CAMUS data or unified corpus not present")

    import shutil

    from atria_echotrace.data.dataset import DatasetRepository
    from atria_echotrace.data.ingest.run import ingest_camus

    # A miniature CAMUS root: one patient per official split.
    source = tmp_path / "camus_mini"
    (source / "database_nifti").mkdir(parents=True)
    shutil.copytree(CAMUS_ROOT / "database_split", source / "database_split")
    wanted = [p for p in ("patient0001", "patient0027", "patient0200")
              if (CAMUS_ROOT / "database_nifti" / p).is_dir()]
    if not wanted:
        pytest.skip("expected CAMUS patients not present")
    for patient in wanted:
        shutil.copytree(
            CAMUS_ROOT / "database_nifti" / patient, source / "database_nifti" / patient
        )

    output = tmp_path / "out"
    result = ingest_camus(source, output, n_points=30)

    assert result.n_frames == 4 * len(wanted)
    assert result.source == "camus"

    # The reference script's own log must actually be written. It calls
    # logging.basicConfig, which no-ops when root already has handlers.
    log = output / "preprocessing_log.txt"
    assert log.is_file() and log.stat().st_size > 0, "preprocessing_log.txt is empty"
    assert "PREPROCESSING COMPLETE" in log.read_text(encoding="utf-8")

    # Output is identical to the corpus the adapters were trained on.
    shipped = json.loads((UNIFIED / "tracings.json").read_text(encoding="utf-8"))
    fresh = json.loads((output / "tracings.json").read_text(encoding="utf-8"))
    for key, entry in fresh.items():
        assert key in shipped
        for field in ("lv_polygon", "la_polygon", "split", "spacing_h", "spacing_w"):
            assert entry[field] == shipped[key][field], f"{key}.{field} differs"

    # Official CAMUS splits, not an invented partition.
    assert {e["split"] for e in fresh.values()} <= {"train", "val", "test"}

    # And the application can load the result.
    report = DatasetRepository(output).validate()
    assert report.ok
    assert report.lv_point_counts == {30: result.n_frames}


# ------------------------------------------------------------ wrapper contract
def test_camus_wrapper_rejects_a_missing_source(tmp_path: Path) -> None:
    from atria_echotrace.data.ingest.run import ingest_camus

    pytest.importorskip("SimpleITK", reason="ingest extra not installed")
    with pytest.raises(IngestError, match="not found"):
        ingest_camus(tmp_path / "nope", tmp_path / "out")


def test_camus_wrapper_requires_the_nifti_layout(tmp_path: Path) -> None:
    """--source must be the CAMUS_public root, not a patient directory."""
    from atria_echotrace.data.ingest.run import ingest_camus

    pytest.importorskip("SimpleITK", reason="ingest extra not installed")
    (tmp_path / "patient0001").mkdir()
    with pytest.raises(IngestError, match="database_nifti"):
        ingest_camus(tmp_path, tmp_path / "out")


def test_echonet_wrapper_names_every_missing_file(tmp_path: Path) -> None:
    from atria_echotrace.data.ingest.run import ingest_echonet

    pytest.importorskip("cv2", reason="ingest extra not installed")
    with pytest.raises(IngestError) as excinfo:
        ingest_echonet(tmp_path, tmp_path / "out")
    message = str(excinfo.value)
    for expected in ("Videos", "FileList.csv", "VolumeTracings.csv"):
        assert expected in message


def test_unified_wrapper_requires_processed_inputs(tmp_path: Path) -> None:
    from atria_echotrace.data.ingest.run import merge_unified

    (tmp_path / "camus").mkdir()
    (tmp_path / "echonet").mkdir()
    with pytest.raises(IngestError, match="no tracings.json"):
        merge_unified(tmp_path / "camus", tmp_path / "echonet", tmp_path / "out")


# ------------------------------------------------------------- summarisation
def test_summarise_counts_frames_and_cases(tmp_path: Path) -> None:
    tracings = {
        "p1_4CH_ED": {"patient_id": "p1", "view": "4CH"},
        "p1_4CH_ES": {"patient_id": "p1", "view": "4CH"},
        "p1_2CH_ED": {"patient_id": "p1", "view": "2CH"},
        "p2_4CH_ED": {"patient_id": "p2", "view": "4CH"},
    }
    (tmp_path / "tracings.json").write_text(json.dumps(tracings), encoding="utf-8")

    result = summarise_output(tmp_path, "camus", 30)
    assert isinstance(result, IngestResult)
    assert result.n_frames == 4
    # Cases are (patient, view) pairs: p1/4CH, p1/2CH, p2/4CH.
    assert result.n_cases == 3
    assert result.as_dict()["source"] == "camus"


def test_summarise_reports_a_run_that_produced_nothing(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no tracings.json"):
        summarise_output(tmp_path, "camus", 30)

    (tmp_path / "tracings.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IngestError, match="empty tracings.json"):
        summarise_output(tmp_path, "camus", 30)


def test_missing_dependency_names_the_install_command() -> None:
    from atria_echotrace.data.ingest import require

    with pytest.raises(IngestError, match=r'pip install -e "\.\[ingest\]"'):
        require("a_module_that_does_not_exist", "some-package", "CAMUS")
