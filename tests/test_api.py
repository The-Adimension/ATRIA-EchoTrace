"""HTTP API behaviour, exercised against the real sample dataset."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from PIL import Image


# ------------------------------------------------------------------- meta
def test_health_answers_without_dependencies(client) -> None:
    response = client.get("/api/meta/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_capabilities_reports_tiers_and_adapters(client) -> None:
    payload = client.get("/api/meta/capabilities").json()
    assert payload["tiers"]["review"] is True
    assert "ai" in payload["tiers"]
    assert [a["id"] for a in payload["adapters"]] == ["base", "camus", "echonet"]
    assert payload["base_model_id"] == "google/medgemma-1.5-4b-it"
    assert payload["dataset"]["n_frames"] == 50
    assert payload["dataset"]["n_cases"] == 25
    assert payload["norm_scale"] == 1000


def test_capabilities_reports_local_adapter_availability(client) -> None:
    """A checkpoint on disk needs no token, and the UI must be able to say so."""
    from atria_echotrace.config import local_adapter_dir

    for entry in client.get("/api/meta/capabilities").json()["adapters"]:
        assert "available_locally" in entry
        expected = local_adapter_dir(entry["id"]) is not None
        assert entry["available_locally"] is expected


def test_local_adapter_dir_requires_an_adapter_config(tmp_path) -> None:
    from atria_echotrace.config import local_adapter_dir

    root = tmp_path
    (root / "adapters" / "atria-echotrace-camus").mkdir(parents=True)
    # A directory without adapter_config.json is not a usable checkpoint.
    assert local_adapter_dir("camus", root=root) is None

    (root / "adapters" / "atria-echotrace-camus" / "adapter_config.json").write_text("{}")
    resolved = local_adapter_dir("camus", root=root)
    assert resolved is not None and resolved.is_dir()
    assert local_adapter_dir("echonet", root=root) is None


def test_capabilities_reports_where_weights_come_from(client) -> None:
    """The UI explains model provenance from this, so it must always be present."""
    weights = client.get("/api/meta/capabilities").json()["weights"]
    assert weights["base"]["source"] in {"local", "cache", "hub"}
    assert weights["base"]["detail"]
    assert {a["id"] for a in weights["adapters"]} == {"camus", "echonet"}
    # Folder names only — enough to know what to create, without disclosing where
    # this installation lives on disk.
    assert weights["models_dir"] == "models/"
    assert weights["adapters_dir"] == "adapters/"
    for entry in [weights["base"], *weights["adapters"]]:
        path = entry["path"]
        assert path is None or not path.startswith(("/", "C:", "\\")), path


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/meta/health",
        "/api/meta/capabilities",
        "/api/meta/dataset-report",
        "/api/dataset/cases",
        "/api/dataset/cases/patient0258_4CH",
        "/api/model/status",
        "/api/evaluation/runs",
        "/api/revisions",
        # Error bodies are a classic disclosure route.
        "/api/dataset/frames/does-not-exist.png",
        "/api/dataset/cases/does-not-exist",
    ],
)
def test_no_endpoint_discloses_host_filesystem_paths(client, endpoint: str) -> None:
    """Absolute paths must never leave the process.

    They reveal the operating-system user name and the machine's directory layout to
    anything that can reach the API, and they end up in screenshots and bug reports.
    Paths in responses are project-relative instead.
    """
    import getpass
    import os

    body = client.get(endpoint).text
    for marker in (getpass.getuser(), os.path.expanduser("~"), os.getcwd()):
        if marker:
            assert marker not in body, f"{endpoint} leaked {marker!r}"


def test_display_path_keeps_project_paths_relative_and_hides_the_rest(tmp_path) -> None:
    from pathlib import Path

    from atria_echotrace.config import display_path

    root = tmp_path / "project"
    (root / "models" / "medgemma").mkdir(parents=True)

    assert display_path(root / "models" / "medgemma", root=root) == "models/medgemma"
    assert display_path(root, root=root) == "."
    # Anything outside the project is reduced to a recognisable tail only.
    outside = display_path(Path.home() / ".cache" / "huggingface" / "hub", root=root)
    assert outside == ".../huggingface/hub"
    assert str(Path.home()) not in outside
    assert display_path(None) is None


def test_capabilities_never_leaks_the_token(client) -> None:
    body = client.get("/api/meta/capabilities").text
    assert "has_token" in body
    assert "hf_" not in body


def test_disclaimers_include_mandatory_oversight(client) -> None:
    payload = client.get("/api/meta/disclaimers").json()
    titles = [d["title"] for d in payload["disclaimers"]]
    assert len(titles) == 7
    assert any("MANDATORY HUMAN OVERSIGHT" in t for t in titles)
    assert [p["pillar"] for p in payload["deity"]] == [
        "DATA",
        "ETHICS",
        "INFORMATICS",
        "TECHNOLOGY",
        "YOU",
    ]
    assert len(payload["citations"]) == 5


def test_dataset_report_endpoint(client) -> None:
    payload = client.get("/api/meta/dataset-report").json()
    assert payload["ok"] is True
    assert len(payload["instant_area_anomalies"]) == 9


# ---------------------------------------------------------------- dataset
def test_list_cases(client) -> None:
    payload = client.get("/api/dataset/cases").json()
    assert payload["count"] == 25
    assert payload["sources"] == ["camus", "echonet"]
    assert sorted(payload["views"]) == ["2CH", "4CH"]
    first = payload["cases"][0]
    assert {"case_key", "case_id", "source", "view", "instants", "frames"} <= set(first)


@pytest.mark.parametrize(
    ("query", "expected_source"),
    [("?source=camus", "camus"), ("?source=echonet", "echonet")],
)
def test_list_cases_source_filter(client, query: str, expected_source: str) -> None:
    payload = client.get(f"/api/dataset/cases{query}").json()
    assert payload["count"] > 0
    assert all(c["source"] == expected_source for c in payload["cases"])


def test_list_cases_view_filter(client) -> None:
    payload = client.get("/api/dataset/cases?view=2CH").json()
    assert all(c["view"] == "2CH" for c in payload["cases"])


def test_case_detail_includes_ground_truth(client) -> None:
    payload = client.get("/api/dataset/cases/patient0258_4CH").json()
    assert payload["case_key"] == "patient0258_4CH"
    assert sorted(payload["frames"]) == ["ED", "ES"]
    assert len(payload["frames"]["ED"]["lv_polygon"]) == 30
    assert payload["calibration_source"] == "dataset"


def test_case_detail_flags_transposed_instants(client) -> None:
    payload = client.get("/api/dataset/cases/echonet_0X171FD888481D524D_4CH").json()
    assert payload["integrity_flags"] == ["es_area_exceeds_ed"]
    assert payload["calibration_source"] == "unknown"


def test_unknown_case_returns_problem_document(client) -> None:
    response = client.get("/api/dataset/cases/does_not_exist")
    assert response.status_code == 404
    body = response.json()
    assert body["status"] == 404
    assert body["title"] == "Not found"
    # No repr quoting from str(KeyError).
    assert body["detail"].startswith("Unknown case")


def test_ambiguous_case_id_returns_404_with_guidance(client) -> None:
    response = client.get("/api/dataset/cases/patient0047")
    assert response.status_code == 404
    assert "ambiguous across views" in response.json()["detail"]


def test_frame_png_is_served(client) -> None:
    response = client.get("/api/dataset/frames/patient0258_4CH_ED.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == (669, 552)


def test_unknown_frame_png_returns_404(client) -> None:
    assert client.get("/api/dataset/frames/nope.png").status_code == 404


def test_ground_truth_figure_is_rendered(client) -> None:
    response = client.get("/api/dataset/frames/patient0258_4CH_ED/ground-truth.png")
    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.content)).width > 400


def test_ground_truth_figure_for_la_on_camus(client) -> None:
    response = client.get(
        "/api/dataset/frames/patient0258_4CH_ED/ground-truth.png?target_structure=LA"
    )
    assert response.status_code == 200


def test_ground_truth_figure_for_la_on_echonet_explains_absence(client) -> None:
    response = client.get(
        "/api/dataset/frames/echonet_0X171FD888481D524D_4CH_ED/ground-truth.png?target_structure=LA"
    )
    assert response.status_code == 404
    assert "no LA ground truth" in response.json()["detail"]


def test_invalid_structure_is_rejected(client) -> None:
    response = client.get(
        "/api/dataset/frames/patient0258_4CH_ED/ground-truth.png?target_structure=RV"
    )
    assert response.status_code == 400
    assert "Unknown target structure" in response.json()["detail"]


def test_splits_endpoint(client) -> None:
    payload = client.get("/api/dataset/splits/test").json()
    assert payload["count"] == 50


# ---------------------------------------------------------------- uploads
def _png_bytes(width: int = 64, height: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (width, height), color=90).save(buffer, format="PNG")
    return buffer.getvalue()


def test_upload_round_trip(client) -> None:
    response = client.post(
        "/api/dataset/uploads", files={"file": ("frame.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["image_w"] == 64
    assert payload["calibration_source"] == "unknown"

    fetched = client.get(payload["image_url"])
    assert fetched.status_code == 200
    assert Image.open(io.BytesIO(fetched.content)).size == (64, 64)


def test_upload_rejects_non_image(client) -> None:
    response = client.post(
        "/api/dataset/uploads", files={"file": ("x.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 422


def test_upload_rejects_empty_file(client) -> None:
    response = client.post("/api/dataset/uploads", files={"file": ("x.png", b"", "image/png")})
    assert response.status_code == 422


def test_upload_rejects_oversized_file(client, settings) -> None:
    settings.max_upload_bytes = 512
    response = client.post(
        "/api/dataset/uploads", files={"file": ("big.png", _png_bytes(400, 400), "image/png")}
    )
    assert response.status_code == 422
    assert "exceeds" in response.json()["detail"]


def test_malformed_upload_id_is_rejected(client) -> None:
    assert client.get("/api/dataset/uploads/..%2F..%2Fsecret.png").status_code in (400, 404)


# ---------------------------------------------------------------- clinical
def test_metrics_for_calibrated_case(client) -> None:
    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    payload = client.post(
        "/api/clinical/metrics",
        json={
            "ed_polygon": detail["frames"]["ED"]["lv_polygon"],
            "es_polygon": detail["frames"]["ES"]["lv_polygon"],
            "image_h": detail["frames"]["ED"]["image_h"],
            "image_w": detail["frames"]["ED"]["image_w"],
            "case_key": "patient0258_4CH",
        },
    ).json()
    assert payload["ed"]["area_cm2"] is not None
    assert payload["fac_percent"] > 0
    assert payload["calibration"]["source"] == "dataset"
    assert payload["calibration"]["note"] is None


def test_metrics_withholds_cm2_for_uncalibrated_case(client) -> None:
    detail = client.get("/api/dataset/cases/echonet_0X171FD888481D524D_4CH").json()
    payload = client.post(
        "/api/clinical/metrics",
        json={
            "ed_polygon": detail["frames"]["ED"]["lv_polygon"],
            "es_polygon": detail["frames"]["ES"]["lv_polygon"],
            "image_h": 224,
            "image_w": 224,
            "case_key": "echonet_0X171FD888481D524D_4CH",
        },
    ).json()
    assert payload["ed"]["area_cm2"] is None
    assert payload["ed"]["area_px"] > 0
    assert payload["calibration"]["source"] == "unknown"
    assert "not reported" in payload["calibration"]["note"]


def test_metrics_accepts_user_spacing_override(client) -> None:
    payload = client.post(
        "/api/clinical/metrics",
        json={
            "ed_polygon": [[250, 250], [250, 750], [750, 750], [750, 250]],
            "es_polygon": [[375, 375], [375, 625], [625, 625], [625, 375]],
            "image_h": 100,
            "image_w": 100,
            "case_key": "echonet_0X171FD888481D524D_4CH",
            "spacing_h": 1.0,
            "spacing_w": 1.0,
        },
    ).json()
    assert payload["calibration"]["source"] == "user"
    assert payload["ed"]["area_cm2"] == pytest.approx(25.0)
    assert payload["fac_percent"] == pytest.approx(75.0)


def test_metrics_requires_both_spacings(client) -> None:
    response = client.post(
        "/api/clinical/metrics",
        json={
            "ed_polygon": [[0, 0], [0, 500], [500, 500]],
            "es_polygon": [],
            "image_h": 100,
            "image_w": 100,
            "spacing_h": 0.3,
        },
    )
    assert response.status_code == 422


def test_frame_metrics_endpoint(client) -> None:
    payload = client.post(
        "/api/clinical/frame-metrics",
        json={
            "polygon": [[250, 250], [250, 750], [750, 750], [750, 250]],
            "image_h": 100,
            "image_w": 100,
        },
    ).json()
    assert payload["area_px"] == pytest.approx(2500.0)
    assert payload["vertices"] == 4


def test_metrics_rejects_invalid_dimensions(client) -> None:
    response = client.post(
        "/api/clinical/metrics",
        json={"ed_polygon": [], "es_polygon": [], "image_h": 0, "image_w": 100},
    )
    assert response.status_code == 422


# --------------------------------------------------------------- revisions
def _revision_payload(detail: dict, notes: str = "test") -> dict:
    ed = detail["frames"]["ED"]
    es = detail["frames"]["ES"]
    nudge = lambda poly: [[min(1000, y + 5), x] for y, x in poly]  # noqa: E731
    return {
        "case_key": detail["case_key"],
        "target_structure": "LV",
        "adapter": "camus",
        "notes": notes,
        "phases": [
            {
                "instant": "ED",
                "stem": ed["stem"],
                "model_polygon": ed["lv_polygon"],
                "user_polygon": nudge(ed["lv_polygon"]),
            },
            {
                "instant": "ES",
                "stem": es["stem"],
                "model_polygon": es["lv_polygon"],
                "user_polygon": nudge(es["lv_polygon"]),
            },
        ],
    }


def test_revision_round_trip_produces_every_artifact(client, settings) -> None:
    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    response = client.post("/api/revisions", json=_revision_payload(detail, "verification"))
    assert response.status_code == 201
    record = response.json()

    expected = {
        "revision.json",
        "tracing_data_img0.json",
        "tracing_data_img1.json",
        "tracing_coordinates_ed.csv",
        "tracing_coordinates_es.csv",
        "clinical_metrics_summary.csv",
        "tracing_vis_img0.png",
        "tracing_vis_img1.png",
        "echotrace_clinical_summary.json",
    }
    assert expected <= set(record["files"])

    directory = settings.revisions_dir / record["revision_id"]
    for name in record["files"]:
        assert (directory / name).is_file(), name
        assert (directory / name).stat().st_size > 0

    # Provenance is recorded, including the dataset's own integrity flags.
    assert record["provenance"]["case_key"] == "patient0258_4CH"
    assert record["provenance"]["calibration_source"] == "dataset"
    assert record["notes"] == "verification"

    # Correct patient, correct frames.
    assert record["phases"]["ED"]["stem"] == "patient0258_4CH_ED"
    assert record["phases"]["ES"]["stem"] == "patient0258_4CH_ES"


def test_revision_zip_contains_all_artifacts(client) -> None:
    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    record = client.post("/api/revisions", json=_revision_payload(detail)).json()

    response = client.get(f"/api/revisions/{record['revision_id']}/export.zip")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
    assert "revision.json" in names
    assert "tracing_vis_img0.png" in names
    assert len(names) >= 9


def test_revision_listing_and_fetch(client) -> None:
    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    record = client.post("/api/revisions", json=_revision_payload(detail)).json()

    listing = client.get("/api/revisions").json()
    assert listing["count"] >= 1
    assert any(r["revision_id"] == record["revision_id"] for r in listing["revisions"])

    fetched = client.get(f"/api/revisions/{record['revision_id']}").json()
    assert fetched["revision_id"] == record["revision_id"]


def test_revision_artifact_download(client) -> None:
    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    record = client.post("/api/revisions", json=_revision_payload(detail)).json()
    response = client.get(
        f"/api/revisions/{record['revision_id']}/files/tracing_vis_img0.png"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_revision_rejects_unknown_frame_rather_than_substituting(client) -> None:
    """The previous implementation fell back to another patient's frame."""
    response = client.post(
        "/api/revisions",
        json={
            "case_key": "patient0258_4CH",
            "phases": [
                {
                    "instant": "ED",
                    "stem": "not_a_real_frame",
                    "model_polygon": [],
                    "user_polygon": [[1, 1], [1, 500], [500, 500]],
                }
            ],
        },
    )
    assert response.status_code == 404
    assert "Unknown frame stem" in response.json()["detail"]


def test_revision_requires_a_revised_polygon(client) -> None:
    response = client.post(
        "/api/revisions",
        json={
            "case_key": "patient0258_4CH",
            "phases": [
                {
                    "instant": "ED",
                    "stem": "patient0258_4CH_ED",
                    "model_polygon": [[1, 1], [1, 2], [2, 2]],
                    "user_polygon": [],
                }
            ],
        },
    )
    assert response.status_code == 422


def test_revision_rejects_duplicate_instants(client) -> None:
    response = client.post(
        "/api/revisions",
        json={
            "phases": [
                {
                    "instant": "ED",
                    "stem": "patient0258_4CH_ED",
                    "user_polygon": [[1, 1], [1, 500], [500, 500]],
                },
                {
                    "instant": "ED",
                    "stem": "patient0258_4CH_ES",
                    "user_polygon": [[1, 1], [1, 500], [500, 500]],
                },
            ]
        },
    )
    assert response.status_code == 422


def test_revision_phase_requires_exactly_one_source(client) -> None:
    response = client.post(
        "/api/revisions",
        json={
            "phases": [
                {
                    "instant": "ED",
                    "stem": "patient0258_4CH_ED",
                    "upload_id": "0123456789abcdef",
                    "user_polygon": [[1, 1], [1, 500], [500, 500]],
                }
            ]
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize("bad_id", ["../secrets", "rev_bad", "rev_1_x"])
def test_malformed_revision_id_is_rejected(client, bad_id: str) -> None:
    assert client.get(f"/api/revisions/{bad_id}").status_code in (400, 404)


def test_revision_artifact_path_traversal_is_rejected(client) -> None:
    detail = client.get("/api/dataset/cases/patient0258_4CH").json()
    record = client.post("/api/revisions", json=_revision_payload(detail)).json()
    response = client.get(
        f"/api/revisions/{record['revision_id']}/files/..%2F..%2Ftracings.json"
    )
    assert response.status_code in (400, 404)


def test_upload_based_revision(client) -> None:
    upload = client.post(
        "/api/dataset/uploads", files={"file": ("f.png", _png_bytes(120, 120), "image/png")}
    ).json()
    response = client.post(
        "/api/revisions",
        json={
            "target_structure": "LV",
            "phases": [
                {
                    "instant": "ED",
                    "upload_id": upload["upload_id"],
                    "model_polygon": [],
                    "user_polygon": [[200, 200], [200, 800], [800, 800], [800, 200]],
                }
            ],
        },
    )
    assert response.status_code == 201
    record = response.json()
    assert record["case"] == "uploaded-frames"
    assert record["metrics"]["ed"]["area_cm2"] is None


# ------------------------------------------------- model tier when absent
def test_model_status_answers_regardless_of_tier(client) -> None:
    payload = client.get("/api/model/status").json()
    assert payload["state"] in {"unavailable", "unloaded", "loading", "ready", "error"}


def test_evaluation_listing_answers_regardless_of_tier(client) -> None:
    payload = client.get("/api/evaluation/runs").json()
    assert "runs" in payload


# --------------------------------------------------------------------- SPA
def test_spa_is_served_at_root(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "ATRIA EchoTrace" in response.text
    assert "importmap" in response.text


@pytest.mark.parametrize(
    "asset",
    [
        "/vendor/preact.module.js",
        "/vendor/hooks.module.js",
        "/vendor/htm.module.js",
        "/js/app.js",
        "/js/api.js",
        "/js/canvas-editor.js",
        "/css/atria.css",
    ],
)
def test_frontend_assets_are_served(client, asset: str) -> None:
    assert client.get(asset).status_code == 200


def test_frontend_references_no_external_origins(client) -> None:
    """The app must run air-gapped: no third-party hosts in the shipped assets."""
    for asset in ("/", "/js/app.js", "/js/api.js", "/js/canvas-editor.js", "/css/atria.css"):
        body = client.get(asset).text
        for forbidden in ("https://esm.sh", "https://unpkg.com", "cdn.", "googleapis"):
            assert forbidden not in body, f"{asset} references {forbidden}"


def test_openapi_schema_is_available(client) -> None:
    schema = client.get("/api/openapi.json").json()
    assert schema["info"]["title"] == "ATRIA EchoTrace"
    assert "/api/inference/predict" in schema["paths"]
