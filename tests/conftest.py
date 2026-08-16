"""Shared test fixtures.

Tests run against the real bundled ``sample-dataset`` (50 frames, 25 cases) and, for
the tests marked ``model``, against real MedGemma weights. Nothing is mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATASET = REPO_ROOT / "sample-dataset"
NOTEBOOK_SOURCE = REPO_ROOT / "notebook_as_py.txt"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def sample_dataset_dir() -> Path:
    if not (SAMPLE_DATASET / "tracings.json").is_file():
        pytest.skip(f"sample-dataset not found at {SAMPLE_DATASET}")
    return SAMPLE_DATASET


@pytest.fixture(scope="session")
def notebook_source() -> str:
    """The original notebook export, used to assert prompt/hyperparameter fidelity."""
    if not NOTEBOOK_SOURCE.is_file():
        pytest.skip(f"notebook source not found at {NOTEBOOK_SOURCE}")
    return NOTEBOOK_SOURCE.read_text(encoding="utf-8")


@pytest.fixture
def dataset_repo(sample_dataset_dir: Path):
    from atria_echotrace.data.dataset import DatasetRepository

    return DatasetRepository(sample_dataset_dir)


@pytest.fixture
def settings(tmp_path: Path, sample_dataset_dir: Path):
    """Settings pointed at the real dataset but a temporary output directory."""
    from atria_echotrace.config import Settings

    value = Settings(dataset_dir=sample_dataset_dir, output_dir=tmp_path / "outputs")
    value.ensure_output_dirs()
    return value


@pytest.fixture
def client(settings):
    """A TestClient whose app uses the temporary output directory."""
    from fastapi.testclient import TestClient

    from atria_echotrace.api import deps
    from atria_echotrace.api.app import create_app

    deps.reset_caches()
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
    deps.reset_caches()


@pytest.fixture
def camus_case(dataset_repo):
    """A calibrated CAMUS case with both ED and ES and an LA reference."""
    for case in dataset_repo.cases.values():
        if case.source == "camus" and case.ed and case.es and case.has_la:
            return case
    pytest.skip("no complete CAMUS case in the sample dataset")


@pytest.fixture
def echonet_case(dataset_repo):
    """An uncalibrated EchoNet case with both ED and ES."""
    for case in dataset_repo.cases.values():
        if case.source == "echonet" and case.ed and case.es:
            return case
    pytest.skip("no complete EchoNet case in the sample dataset")
