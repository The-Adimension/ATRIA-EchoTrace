"""The evaluation run lifecycle: start, poll, persist.

`POST /api/evaluation/runs` hands work to a background thread and returns 202. That
thread is the only place in the codebase holding shared mutable state across threads
(progress, state transitions, the persisted summary), and it had no coverage: the SPA
never calls it and the earlier tests only exercised the listing endpoint.

The real-weights test is `model`-marked and deliberately tiny (one frame) — it exists to
prove the lifecycle completes and persists, not to measure accuracy.
"""

from __future__ import annotations

import time

import pytest


def _poll_until_terminal(client, run_id: str, timeout_s: float = 600.0) -> dict:
    """Poll the run until it leaves the running state, or fail loudly."""
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        response = client.get(f"/api/evaluation/runs/{run_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        if last.get("state") in {"completed", "error"}:
            return last
        time.sleep(2.0)
    pytest.fail(f"run {run_id} never reached a terminal state; last={last.get('state')}")


# ------------------------------------------------------------------ review tier
def test_starting_a_run_without_the_ai_tier_is_refused(client) -> None:
    """503 with instructions, and no run is created."""
    from atria_echotrace.api.deps import ml_available

    if ml_available():
        pytest.skip("the [ai] extra is installed; this asserts the review-tier path")

    before = client.get("/api/evaluation/runs").json()["count"]
    response = client.post("/api/evaluation/runs", json={"max_samples": 1})
    assert response.status_code == 503
    assert "[ai]" in response.json()["detail"]
    # Refused before anything was scheduled: no dangling thread, no new run.
    assert client.get("/api/evaluation/runs").json()["count"] == before


def test_starting_a_run_without_a_loaded_model_is_refused(client) -> None:
    """409, not a crash and not a run that fails asynchronously."""
    from atria_echotrace.api.deps import ml_available

    if not ml_available():
        pytest.skip("the [ai] extra is not installed")

    status = client.get("/api/model/status").json()
    if status.get("state") == "ready":
        pytest.skip("a model is already loaded in this process")

    before = client.get("/api/evaluation/runs").json()["count"]
    response = client.post("/api/evaluation/runs", json={"max_samples": 1})
    assert response.status_code == 409
    assert "not ready" in response.json()["detail"].lower()
    assert client.get("/api/evaluation/runs").json()["count"] == before


def test_unknown_run_id_is_a_clean_404(client) -> None:
    assert client.get("/api/evaluation/runs/eval_0000000000").status_code == 404


def test_malformed_run_id_is_rejected(client) -> None:
    assert client.get("/api/evaluation/runs/..%2Fescape").status_code in {400, 404}


# ------------------------------------------------------------- real weights
@pytest.mark.model
@pytest.mark.adapter
def test_run_completes_and_persists(client, settings) -> None:
    """One frame, end to end: 202 → polling → completed → persisted with a summary."""
    from atria_echotrace.api.deps import ml_available

    if not ml_available():
        pytest.skip("the [ai] extra is not installed")

    # Loading is asynchronous: 202 Accepted, then poll. 200 is accepted too in case an
    # already-ready engine short-circuits.
    load = client.post("/api/model/load", json={"adapter": "camus"})
    if load.status_code not in {200, 202}:
        pytest.skip(f"adapter unavailable: {load.text[:120]}")

    deadline = time.time() + 600
    while time.time() < deadline:
        if client.get("/api/model/status").json().get("state") in {"ready", "error"}:
            break
        time.sleep(2.0)
    if client.get("/api/model/status").json().get("state") != "ready":
        pytest.skip("model did not become ready")

    started = client.post(
        "/api/evaluation/runs",
        json={"split": "test", "source": "camus", "max_samples": 1},
    )
    assert started.status_code == 202, started.text
    body = started.json()
    assert body["selected_frames"] == 1
    run_id = body["run_id"]

    final = _poll_until_terminal(client, run_id)
    assert final["state"] == "completed", final.get("error")

    summary = final["summary"]
    assert summary["total_samples"] == 1
    assert summary["parse_rate_percent"] == 100.0
    assert summary["dice"]["mean"] is not None
    assert final["progress"] == 1.0

    # Persisted to disk, and visible in the listing.
    assert (settings.evaluations_dir / f"{run_id}.json").is_file()
    listed = client.get("/api/evaluation/runs").json()
    assert any(item["run_id"] == run_id for item in listed["runs"])
