"""The ground-truth evolution loop: revisions -> trainable corpus.

Notebook-readme.txt states the intent ("Revised data feeds directly back into iterative
model improvement or serves as enhanced ground truth"); the notebook never implemented
it. These tests assert the exported corpus is the same three-artifact contract the
vendored preprocessors emit, so ``DatasetRepository`` and ``atria train`` accept it
unchanged, and that the clinician's polygon — not the model's — is the label.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atria_echotrace.data.dataset import DatasetRepository
from atria_echotrace.export.package import export_corpus


def _nudge(polygon: list[list[int]], delta: int = 7) -> list[list[int]]:
    """A recognisably different polygon, standing in for a clinician's edit."""
    return [[min(1000, y + delta), x] for y, x in polygon]


def _save_revision(client, case_key: str = "patient0258_4CH", notes: str = "") -> dict:
    detail = client.get(f"/api/dataset/cases/{case_key}").json()
    payload = {
        "case_key": case_key,
        "target_structure": "LV",
        "adapter": "camus",
        "notes": notes,
        "phases": [
            {
                "instant": instant,
                "stem": detail["frames"][instant]["stem"],
                "model_polygon": detail["frames"][instant]["lv_polygon"],
                "user_polygon": _nudge(detail["frames"][instant]["lv_polygon"]),
            }
            for instant in ("ED", "ES")
        ],
    }
    response = client.post("/api/revisions", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _export(settings, out: Path, **kwargs) -> dict:
    return export_corpus(
        output_dir=out,
        output_root=settings.revisions_dir,
        uploads_dir=settings.uploads_dir,
        dataset_dir=settings.dataset_dir,
        **kwargs,
    )


def test_revision_round_trips_through_the_corpus_contract(client, settings, tmp_path) -> None:
    """AC1/AC2/AC5: export the three artefacts, reload them, compare vertex for vertex."""
    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    expected_ed = _nudge(detail["frames"]["ED"]["lv_polygon"])
    record = _save_revision(client)

    out = tmp_path / "corpus"
    summary = _export(settings, out)

    # AC1: exactly the contract the preprocessors emit.
    assert (out / "tracings.json").is_file()
    assert (out / "metadata.csv").is_file()
    assert (out / "frames" / "patient0258_4CH_ED.png").is_file()
    assert summary["frames"] == 2
    assert summary["revisions"] == [record["revision_id"]]

    # AC2: loads unmodified through the production repository.
    reloaded = DatasetRepository(out)
    frames = reloaded.frames
    assert set(frames) == {"patient0258_4CH_ED", "patient0258_4CH_ES"}

    frame = frames["patient0258_4CH_ED"]
    assert frame.view == "4CH"
    assert frame.instant == "ED"
    assert frame.split == "train"

    # AC5: the clinician's polygon survives vertex for vertex, and it is *not* the model's.
    assert frame.lv_polygon == expected_ed
    assert frame.lv_polygon != detail["frames"]["ED"]["lv_polygon"]

    # AC2 continued: the corpus is trainable — sample preparation accepts it.
    from atria_echotrace.ml.datasets import prepare_samples

    samples = prepare_samples(reloaded, split="train", target_structure="LV")
    assert len(samples) == 2
    assert {s["view"] for s in samples} == {"4CH"}
    assert {s["instant"] for s in samples} == {"ED", "ES"}


def test_model_polygon_is_provenance_not_ground_truth(client, settings, tmp_path) -> None:
    """AC3/AC4: model_polygon is retained but never becomes the label."""
    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    record = _save_revision(client)
    out = tmp_path / "corpus"
    _export(settings, out)

    tracings = json.loads((out / "tracings.json").read_text(encoding="utf-8"))
    entry = tracings["patient0258_4CH_ED"]

    # The label is the revision.
    assert entry["lv_polygon"] == _nudge(detail["frames"]["ED"]["lv_polygon"])
    # The model's proposal is kept, under a key no loader treats as a polygon.
    assert entry["model_polygon_2d"] == detail["frames"]["ED"]["lv_polygon"]
    assert entry["lv_polygon"] != entry["model_polygon_2d"]
    # AC4: provenance points back at the revision and carries integrity flags through.
    assert entry["revised_from"] == record["revision_id"]
    assert "dataset_integrity_flags" in entry
    assert entry["calibration_source"] == "dataset"


def test_integrity_flags_carry_through_for_a_transposed_case(client, settings, tmp_path) -> None:
    """AC4: an ED/ES-transposed source case stays flagged in the exported corpus."""
    cases = client.get("/api/dataset/cases").json()["cases"]
    flagged = next((c for c in cases if c.get("integrity_flags")), None)
    if flagged is None:  # pragma: no cover - the sample dataset ships flagged cases
        pytest.skip("no flagged case in the sample dataset")
    _save_revision(client, case_key=flagged["case_key"])

    out = tmp_path / "corpus"
    _export(settings, out)
    tracings = json.loads((out / "tracings.json").read_text(encoding="utf-8"))
    assert any(
        "es_area_exceeds_ed" in (entry.get("dataset_integrity_flags") or [])
        for entry in tracings.values()
    )


def test_uploaded_frame_exports_with_unknown_calibration(client, settings, tmp_path) -> None:
    """AC6: an upload-backed revision exports, carrying calibration_source unknown."""
    source = settings.dataset_dir / "frames" / "patient0258_4CH_ED.png"
    upload = client.post(
        "/api/dataset/uploads",
        files={"file": ("mine.png", source.read_bytes(), "image/png")},
    )
    assert upload.status_code == 201, upload.text
    upload_id = upload.json()["upload_id"]

    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    response = client.post(
        "/api/revisions",
        json={
            "target_structure": "LV",
            "view": "2CH",  # declared by the clinician; uploads carry no view
            "phases": [
                {
                    "instant": "ED",
                    "upload_id": upload_id,
                    "model_polygon": [],
                    "user_polygon": _nudge(detail["frames"]["ED"]["lv_polygon"]),
                }
            ],
        },
    )
    assert response.status_code == 201, response.text

    out = tmp_path / "corpus"
    summary = _export(settings, out)
    assert summary["frames"] == 1

    tracings = json.loads((out / "tracings.json").read_text(encoding="utf-8"))
    stem, entry = next(iter(tracings.items()))
    assert stem.startswith("upload_")
    assert entry["source"] == "upload"
    assert entry["view"] == "2CH"
    assert entry["calibration_source"] == "unknown"
    assert entry["spacing_h"] is None
    assert (out / "frames" / f"{stem}.png").is_file()

    # Still a valid corpus.
    assert DatasetRepository(out).frames[stem].view == "2CH"


def test_later_revision_supersedes_an_earlier_one(client, settings, tmp_path) -> None:
    """Two revisions of the same frame collapse to one, newest winning.

    Note this test was intermittently green for the wrong reason before the ordering fix:
    it only failed when both saves happened to land in the same wall-clock second. The
    deterministic version of this guarantee is
    :func:`test_same_second_saves_still_supersede_correctly`.
    """
    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    _save_revision(client, notes="first")
    second = client.post(
        "/api/revisions",
        json={
            "case_key": "patient0258_4CH",
            "target_structure": "LV",
            "notes": "second",
            "phases": [
                {
                    "instant": "ED",
                    "stem": detail["frames"]["ED"]["stem"],
                    "model_polygon": [],
                    "user_polygon": _nudge(detail["frames"]["ED"]["lv_polygon"], delta=21),
                }
            ],
        },
    ).json()

    out = tmp_path / "corpus"
    _export(settings, out)
    tracings = json.loads((out / "tracings.json").read_text(encoding="utf-8"))
    assert tracings["patient0258_4CH_ED"]["revised_from"] == second["revision_id"]
    assert tracings["patient0258_4CH_ED"]["lv_polygon"] == _nudge(
        detail["frames"]["ED"]["lv_polygon"], delta=21
    )


def test_same_second_saves_still_supersede_correctly(client, settings, tmp_path) -> None:
    """The newer revision wins even when both are saved within one second.

    `timestamp_utc` and the revision id are both second-granularity, so two saves in the
    same second are indistinguishable by either. Sorting on the timestamp alone left the
    order to a stable sort over `list_revisions()`, which yields newest-first — so the
    OLDER polygon overwrote the newer one and became the exported ground truth. This
    forces the collision instead of waiting for it.
    """
    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    stem = detail["frames"]["ED"]["stem"]
    reference = detail["frames"]["ED"]["lv_polygon"]

    def save(delta: int) -> dict:
        response = client.post(
            "/api/revisions",
            json={
                "case_key": "patient0258_4CH",
                "target_structure": "LV",
                "phases": [
                    {
                        "instant": "ED",
                        "stem": stem,
                        "model_polygon": [],
                        "user_polygon": _nudge(reference, delta=delta),
                    }
                ],
            },
        )
        assert response.status_code == 201, response.text
        return response.json()

    first = save(9)
    second = save(23)

    # Force the exact collision: identical second-granularity timestamps, so the only
    # usable ordering signal is `created_unix`.
    stamps = []
    for record_id in (first["revision_id"], second["revision_id"]):
        path = settings.revisions_dir / record_id / "revision.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["timestamp_utc"] = "2026-01-01T00:00:00Z"
        stamps.append(record["created_unix"])
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    assert stamps[1] > stamps[0], "created_unix must be strictly monotonic between saves"

    out = tmp_path / "corpus"
    _export(settings, out)
    tracings = json.loads((out / "tracings.json").read_text(encoding="utf-8"))
    entry = tracings[stem]
    assert entry["revised_from"] == second["revision_id"]
    assert entry["lv_polygon"] == _nudge(reference, delta=23)


def test_records_without_created_unix_still_export(client, settings, tmp_path) -> None:
    """Revisions written before `created_unix` existed must not break the export.

    They sort as 0.0, i.e. behind every newer record — the safe fallback, since a record
    that predates the field is by definition older than one that carries it.
    """
    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    stem = detail["frames"]["ED"]["stem"]
    reference = detail["frames"]["ED"]["lv_polygon"]

    legacy = _save_revision(client)
    legacy_path = settings.revisions_dir / legacy["revision_id"] / "revision.json"
    record = json.loads(legacy_path.read_text(encoding="utf-8"))
    del record["created_unix"]  # as an older release wrote it
    legacy_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    out = tmp_path / "corpus"
    summary = _export(settings, out)
    assert summary["frames"] == 2
    tracings = json.loads((out / "tracings.json").read_text(encoding="utf-8"))
    assert tracings[stem]["lv_polygon"] == _nudge(reference)

    # And a newer record still beats the legacy one for the same frame.
    newer = client.post(
        "/api/revisions",
        json={
            "case_key": "patient0258_4CH",
            "target_structure": "LV",
            "phases": [
                {
                    "instant": "ED",
                    "stem": stem,
                    "model_polygon": [],
                    "user_polygon": _nudge(reference, delta=31),
                }
            ],
        },
    ).json()
    out2 = tmp_path / "corpus2"
    _export(settings, out2)
    tracings2 = json.loads((out2 / "tracings.json").read_text(encoding="utf-8"))
    assert tracings2[stem]["revised_from"] == newer["revision_id"]


def test_selecting_revisions_and_split(client, settings, tmp_path) -> None:
    """--revision restricts the export; --split labels it."""
    first = _save_revision(client, notes="first")
    other = next(
        c["case_key"]
        for c in client.get("/api/dataset/cases").json()["cases"]
        if c["case_key"] != "patient0258_4CH" and set(c["instants"]) == {"ED", "ES"}
    )
    _save_revision(client, case_key=other, notes="second")

    out = tmp_path / "corpus"
    summary = _export(settings, out, revision_ids=[first["revision_id"]], split="val")
    assert summary["revisions"] == [first["revision_id"]]
    assert summary["split"] == "val"
    assert all(f.split == "val" for f in DatasetRepository(out).frames.values())

    with pytest.raises(ValueError, match="No such revision"):
        _export(settings, tmp_path / "other", revision_ids=["rev_1111111111_abcdef"])


def test_export_with_no_revisions_fails_clearly(settings, tmp_path) -> None:
    with pytest.raises(ValueError, match="No revision produced a usable frame"):
        _export(settings, tmp_path / "empty")


def test_ranked_figures_are_rendered(settings, tmp_path) -> None:
    """Notebook phase 21: `atria evaluate` renders its best/worst predictions.

    Built from a synthetic run so no model weights are needed — the figure writer is
    what is under test, not inference.
    """
    from atria_echotrace.data.dataset import DatasetRepository
    from atria_echotrace.ml.evaluate import EvaluationRun, FrameResult, save_ranked_figures

    repo = DatasetRepository(settings.dataset_dir)
    scored = [
        frame
        for frame in sorted(repo.frames.values(), key=lambda f: f.stem)
        if frame.lv_polygon
    ][:4]
    assert len(scored) == 4

    run = EvaluationRun(
        run_id="eval_test",
        split=None,
        source=None,
        target_structure="LV",
        adapter=None,
        prompt_variant=None,
        device="cpu",
        max_samples=None,
    )
    for rank, frame in enumerate(scored):
        run.results.append(
            FrameResult(
                stem=frame.stem,
                source=frame.source,
                view=frame.view,
                instant=frame.instant,
                parsed=True,
                vertices=len(frame.lv_polygon or []),
                dice=0.9 - rank * 0.2,
                iou=0.8 - rank * 0.2,
                predicted_polygon=frame.lv_polygon,
            )
        )

    written = save_ranked_figures(run, repo, tmp_path, count=2)
    names = sorted(p.name for p in written)
    assert len(written) == 4, names
    assert sum("best" in n for n in names) == 2
    assert sum("worst" in n for n in names) == 2
    for path in written:
        assert path.is_file() and path.stat().st_size > 0
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # count=0 disables it, and an unscored run writes nothing.
    assert save_ranked_figures(run, repo, tmp_path, count=0) == []
    empty = EvaluationRun(
        run_id="eval_empty",
        split=None,
        source=None,
        target_structure="LV",
        adapter=None,
        prompt_variant=None,
        device="cpu",
        max_samples=None,
    )
    assert save_ranked_figures(empty, repo, tmp_path, count=3) == []
