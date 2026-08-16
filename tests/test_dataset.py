"""Dataset repository behaviour against the real bundled sample dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atria_echotrace.data.dataset import (
    DatasetError,
    DatasetRepository,
    RecordNotFound,
    case_key_of,
)


# ------------------------------------------------------------------- loading
def test_loads_all_sample_frames(dataset_repo) -> None:
    assert len(dataset_repo.frames) == 50


def test_case_count_matches_the_manifest(dataset_repo) -> None:
    """Cases are keyed by patient *and* view; the manifest is the cross-check.

    ``patient0047`` appears in both 2CH and 4CH, so keying on patient id alone
    silently discards two frames and reports 24 cases instead of 25.
    """
    manifest = json.loads((dataset_repo.dataset_dir / "manifest.json").read_text())
    assert len(dataset_repo.cases) == manifest["n_cases"] == 25
    assert len(dataset_repo.frames) == manifest["n_frames"] == 50


def test_multi_view_patient_keeps_both_views_intact(dataset_repo) -> None:
    keys = [key for key in dataset_repo.cases if key.startswith("patient0047")]
    assert sorted(keys) == ["patient0047_2CH", "patient0047_4CH"]
    for key in keys:
        case = dataset_repo.cases[key]
        assert sorted(case.frames) == ["ED", "ES"]
        for instant, frame in case.frames.items():
            assert frame.view in key
            assert frame.stem.endswith(f"{frame.view}_{instant}")


def test_every_frame_has_its_png(dataset_repo) -> None:
    for stem in dataset_repo.frames:
        assert dataset_repo.frame_path(stem).is_file()


def test_all_lv_polygons_have_thirty_bounded_points(dataset_repo) -> None:
    for stem, frame in dataset_repo.frames.items():
        assert frame.lv_polygon is not None, stem
        assert len(frame.lv_polygon) == 30, stem
        for y, x in frame.lv_polygon:
            assert 0 <= y <= 1000, stem
            assert 0 <= x <= 1000, stem


def test_la_polygons_present_for_camus_absent_for_echonet(dataset_repo) -> None:
    for frame in dataset_repo.frames.values():
        if frame.source == "camus":
            assert frame.has_la, frame.stem
            assert len(frame.la_polygon or []) == 30
        else:
            assert not frame.has_la, frame.stem


def test_polygon_accessor_selects_structure(camus_case) -> None:
    frame = camus_case.ed
    assert frame.polygon("LV") == frame.lv_polygon
    assert frame.polygon("LA") == frame.la_polygon
    assert frame.polygon("lv") == frame.lv_polygon


# --------------------------------------------------------------- calibration
def test_camus_frames_are_calibrated(dataset_repo) -> None:
    for frame in dataset_repo.frames.values():
        if frame.source == "camus":
            assert frame.calibration.is_known, frame.stem
            assert frame.calibration.source == "dataset"
            assert 0.05 < (frame.calibration.spacing_h or 0) < 2.0


def test_echonet_frames_are_uncalibrated(dataset_repo) -> None:
    """A 1.0 mm/px sentinel must not be mistaken for a real calibration."""
    for frame in dataset_repo.frames.values():
        if frame.source == "echonet":
            assert not frame.calibration.is_known, frame.stem
            assert frame.calibration.source == "unknown"


# ----------------------------------------------------------------- integrity
def test_echonet_cases_are_flagged_for_transposed_instants(dataset_repo) -> None:
    """All nine EchoNet cases have an ES trace larger than their ED trace.

    Physiologically impossible, so the repository flags it rather than correcting it.
    """
    flagged = [c.case_key for c in dataset_repo.cases.values() if c.integrity_flags()]
    assert len(flagged) == 9
    assert all(key.startswith("echonet_") for key in flagged)
    for case in dataset_repo.cases.values():
        if case.source == "camus":
            assert case.integrity_flags() == [], case.case_key


def test_validation_report_summarises_the_sample_dataset(dataset_repo) -> None:
    report = dataset_repo.validate()
    assert report.ok
    assert report.n_tracings == 50
    assert report.n_frames_present == 50
    assert report.missing_pngs == []
    assert report.n_cases == 25
    assert report.incomplete_cases == []
    assert report.source_counts == {"camus": 32, "echonet": 18}
    assert report.view_counts == {"4CH": 34, "2CH": 16}
    assert report.instant_counts == {"ED": 25, "ES": 25}
    assert report.lv_point_counts == {30: 50}
    assert report.uncalibrated_sources == ["echonet"]
    assert len(report.instant_area_anomalies) == 9


# -------------------------------------------------------------------- lookup
def test_unknown_frame_raises_record_not_found(dataset_repo) -> None:
    with pytest.raises(RecordNotFound, match="Unknown frame stem"):
        dataset_repo.get_frame("no_such_frame")


def test_unknown_case_raises_record_not_found(dataset_repo) -> None:
    with pytest.raises(RecordNotFound, match="Unknown case"):
        dataset_repo.get_case("no_such_case")


def test_record_not_found_message_has_no_repr_quoting(dataset_repo) -> None:
    """str(KeyError) would wrap the message in quotes and leak into API bodies."""
    try:
        dataset_repo.get_frame("missing")
    except RecordNotFound as exc:
        assert str(exc).startswith("Unknown frame stem")


def test_ambiguous_bare_case_id_is_rejected_with_guidance(dataset_repo) -> None:
    with pytest.raises(RecordNotFound, match="ambiguous across views"):
        dataset_repo.get_case("patient0047")


def test_unambiguous_bare_case_id_resolves(dataset_repo) -> None:
    case = dataset_repo.get_case("patient0258")
    assert case.case_key == "patient0258_4CH"


def test_lookup_is_exact_not_substring(dataset_repo) -> None:
    """Substring matching could return a different patient's frame."""
    with pytest.raises(RecordNotFound):
        dataset_repo.get_frame("patient0258_4CH_E")
    with pytest.raises(RecordNotFound):
        dataset_repo.get_frame("atient0258_4CH_ED")


@pytest.mark.parametrize(
    "hostile",
    ["../../etc/passwd", "..\\..\\windows\\system32", "/etc/shadow", "patient0258/../../x"],
)
def test_frame_path_rejects_traversal_attempts(dataset_repo, hostile: str) -> None:
    """Only stems present in tracings.json are addressable, which forecloses traversal."""
    with pytest.raises(RecordNotFound):
        dataset_repo.frame_path(hostile)


def test_case_key_helper() -> None:
    assert case_key_of("patient0047", "2CH") == "patient0047_2CH"
    assert case_key_of("solo", "") == "solo"


# --------------------------------------------------------------- filters etc.
def test_list_cases_filters_by_source_and_view(dataset_repo) -> None:
    camus = dataset_repo.list_cases(source="camus")
    assert camus and all(c.source == "camus" for c in camus)
    two_chamber = dataset_repo.list_cases(view="2CH")
    assert two_chamber and all(c.view == "2CH" for c in two_chamber)
    assert dataset_repo.list_cases(source="camus", view="2CH")
    assert dataset_repo.list_cases(source="nonexistent") == []


def test_list_cases_complete_pairs_only(dataset_repo) -> None:
    cases = dataset_repo.list_cases(complete_pairs_only=True)
    assert len(cases) == 25
    assert all(c.ed and c.es for c in cases)


def test_frames_in_split(dataset_repo) -> None:
    assert len(dataset_repo.frames_in_split("test")) == 50
    assert dataset_repo.frames_in_split("train") == []


def test_case_detail_serialises_polygons(camus_case) -> None:
    detail = camus_case.detail()
    assert detail["case_key"] == camus_case.case_key
    ed = detail["frames"]["ED"]
    assert len(ed["lv_polygon"]) == 30
    assert len(ed["lv_polygon_px"]) == 30
    # Pixel coordinates must fall inside the frame.
    for x, y in ed["lv_polygon_px"]:
        assert 0 <= x <= ed["image_w"]
        assert 0 <= y <= ed["image_h"]


# ------------------------------------------------------------ error handling
def test_missing_tracings_raises_dataset_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="tracings.json not found"):
        DatasetRepository(tmp_path).frames


def test_malformed_tracings_raises_dataset_error(tmp_path: Path) -> None:
    (tmp_path / "tracings.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(DatasetError, match="must contain a JSON object"):
        DatasetRepository(tmp_path).frames


def test_invalid_json_raises_dataset_error(tmp_path: Path) -> None:
    (tmp_path / "tracings.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="Could not read"):
        DatasetRepository(tmp_path).frames


# ------------------------------------------------- the real training corpus
UNIFIED = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "processed_datasets"
    / "unified_processed"
    / "unified_processed"
)


def _unified_repo() -> DatasetRepository:
    if not (UNIFIED / "tracings.json").is_file():
        pytest.skip("the real unified_processed corpus is not present")
    return DatasetRepository(UNIFIED)


def test_full_training_corpus_loads_and_validates() -> None:
    """The data layer must handle the actual 22k-frame corpus, not just 50 frames."""
    report = _unified_repo().validate()

    assert report.ok
    assert report.n_tracings == 22048
    assert report.n_frames_present == 22048
    assert report.missing_pngs == []
    assert report.n_cases == 11024
    assert report.incomplete_cases == []
    assert report.source_counts == {"camus": 2000, "echonet": 20048}
    assert report.view_counts == {"4CH": 21048, "2CH": 1000}
    assert report.instant_counts == {"ED": 11024, "ES": 11024}
    # The real splits: CAMUS from database_split/, EchoNet from FileList.csv.
    assert report.split_counts == {"train": 16520, "val": 2776, "test": 2752}
    # Every polygon in the training data is exactly 30 points.
    assert report.lv_point_counts == {30: 22048}
    assert report.la_point_counts == {30: 2000}
    assert report.uncalibrated_sources == ["echonet"]


def test_transposition_is_confined_to_echonet_at_full_scale() -> None:
    """Quantifies RESEARCH.md §8.3 over the whole corpus rather than 9 sample cases."""
    repo = _unified_repo()
    flagged = [c for c in repo.cases.values() if c.integrity_flags()]
    by_source: dict[str, int] = {}
    for case in flagged:
        by_source[case.source] = by_source.get(case.source, 0) + 1

    assert by_source.get("camus", 0) == 0, "CAMUS instants are correctly labelled"
    echonet_cases = sum(1 for c in repo.cases.values() if c.source == "echonet")
    assert by_source["echonet"] / echonet_cases > 0.95, (
        "≈99% of EchoNet cases carry transposed ED/ES labels; the adapters were "
        "trained this way, so the application surfaces it rather than correcting it"
    )


def test_sample_dataset_is_an_exact_subset_of_the_training_corpus(dataset_repo) -> None:
    """Provenance: the bundled 50 frames are real held-out training-corpus frames."""
    unified = _unified_repo()
    for stem, frame in dataset_repo.frames.items():
        assert stem in unified.frames, f"{stem} is not in the training corpus"
        reference = unified.frames[stem]
        assert frame.lv_polygon == reference.lv_polygon
        assert frame.la_polygon == reference.la_polygon
        assert (frame.image_h, frame.image_w) == (reference.image_h, reference.image_w)
        # All bundled frames come from the held-out test split.
        assert reference.split == "test"


def test_metadata_csv_split_overrides_tracings(tmp_path: Path) -> None:
    """metadata.csv is authoritative for splits, as in the notebook."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    tracings = {
        "p1_4CH_ED": {
            "patient_id": "p1",
            "view": "4CH",
            "instant": "ED",
            "image_h": 10,
            "image_w": 10,
            "spacing_h": 0.3,
            "spacing_w": 0.3,
            "lv_polygon": [[0, 0], [0, 500], [500, 500]],
            "la_polygon": None,
            "split": "test",
            "source": "camus",
        }
    }
    (tmp_path / "tracings.json").write_text(json.dumps(tracings), encoding="utf-8")
    (tmp_path / "metadata.csv").write_text("key,split\np1_4CH_ED,train\n", encoding="utf-8")

    repo = DatasetRepository(tmp_path)
    assert repo.get_frame("p1_4CH_ED").split == "train"
