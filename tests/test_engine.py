"""Engine logic that needs no model weights.

``atria_echotrace.ml.engine`` imports torch only inside functions, so adapter
resolution and error classification are testable on a review-tier install.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atria_echotrace.ml.engine import (
    AdapterError,
    _is_hub_access_error,
    resolve_adapter,
)


# ------------------------------------------------------------ adapter registry
@pytest.mark.parametrize("value", [None, "", "base"])
def test_base_resolves_to_no_adapter(value) -> None:
    entry = resolve_adapter(value)
    assert entry["id"] == "base"
    assert entry["repo"] is None


def test_registry_ids_resolve() -> None:
    for adapter_id in ("camus", "echonet"):
        entry = resolve_adapter(adapter_id)
        assert entry["id"] == adapter_id
        assert entry["repo"]


def test_local_checkpoint_is_preferred_over_the_gated_repo() -> None:
    """When adapters/ holds a checkpoint, it wins — no token, no network."""
    from atria_echotrace.config import local_adapter_dir

    for adapter_id in ("camus", "echonet"):
        local = local_adapter_dir(adapter_id)
        entry = resolve_adapter(adapter_id)
        if local is None:
            assert entry.get("source") != "local"
            assert entry["repo"].startswith("The-Adimension/")
        else:
            assert entry["source"] == "local"
            assert Path(entry["repo"]) == local


def test_full_repo_id_resolves_to_its_registry_entry() -> None:
    entry = resolve_adapter("The-Adimension/EchoTrace-MedGemma-CAMUS")
    assert entry["id"] == "camus"


def test_arbitrary_repo_id_is_accepted() -> None:
    entry = resolve_adapter("someone/their-adapter")
    assert entry["repo"] == "someone/their-adapter"


def test_local_directory_path_is_accepted(tmp_path: Path) -> None:
    """The notebook's HITL cell took a local ``lora_path``; that still works."""
    entry = resolve_adapter(str(tmp_path))
    assert Path(entry["repo"]) == tmp_path


def test_unknown_adapter_is_rejected_with_guidance() -> None:
    with pytest.raises(AdapterError, match="Unknown adapter"):
        resolve_adapter("not-a-real-adapter")


# ------------------------------------------------------- base-model resolution
def _make_model_dir(path: Path) -> Path:
    """A minimal directory that Transformers would recognise."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type": "gemma3"}', encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "folder",
    ["medgemma-1.5-4b-it", "google--medgemma-1.5-4b-it", "google/medgemma-1.5-4b-it"],
)
def test_local_model_dir_accepts_each_hand_placed_layout(tmp_path: Path, folder: str) -> None:
    """People drop a downloaded model in under any of these names."""
    from atria_echotrace.config import local_model_dir

    expected = _make_model_dir(tmp_path / "models" / folder)
    assert local_model_dir("google/medgemma-1.5-4b-it", root=tmp_path) == expected


def test_local_model_dir_requires_a_config_json(tmp_path: Path) -> None:
    """A folder without config.json is not a usable checkout; ignore it."""
    from atria_echotrace.config import local_model_dir

    (tmp_path / "models" / "medgemma-1.5-4b-it").mkdir(parents=True)
    assert local_model_dir("google/medgemma-1.5-4b-it", root=tmp_path) is None


def test_local_model_dir_absent_returns_none(tmp_path: Path) -> None:
    from atria_echotrace.config import local_model_dir

    assert local_model_dir("google/medgemma-1.5-4b-it", root=tmp_path) is None


def test_base_model_resolves_to_the_local_copy_when_present() -> None:
    """The project's models/ folder must win over the cache and the Hub.

    This is what lets the AI tier run with no token and no network.
    """
    from atria_echotrace.config import Settings, local_model_dir

    settings = Settings()
    reference, source = settings.resolve_base_model()
    local = local_model_dir(settings.base_model_id)
    if local is None:
        # No hand-placed copy here: it must fall back, never silently claim "local".
        assert source in {"cache", "hub"}
        assert reference == settings.base_model_id
    else:
        assert source == "local"
        assert reference == str(local)


def test_weights_report_describes_every_weight() -> None:
    from atria_echotrace.config import Settings

    report = Settings().weights_report()
    assert set(report) >= {"base", "adapters", "models_dir", "adapters_dir", "has_token"}
    assert report["base"]["source"] in {"local", "cache", "hub"}
    assert isinstance(report["base"]["ready"], bool)
    assert report["base"]["detail"]

    ids = {a["id"] for a in report["adapters"]}
    assert ids == {"camus", "echonet"}
    for adapter in report["adapters"]:
        assert adapter["source"] in {"local", "hub"}
        # Every entry must tell the operator what to do, ready or not.
        assert adapter["detail"]


def test_weights_report_never_leaks_the_token() -> None:
    from atria_echotrace.config import Settings

    report = Settings().weights_report()
    assert isinstance(report["has_token"], bool)
    assert "hf_" not in repr(report)


# --------------------------------------------------- hub error classification
@pytest.mark.parametrize(
    "message",
    [
        "401 Client Error. Cannot access gated repo for url ...",
        "403 Forbidden",
        "Unauthorized",
        "Connection error, and we cannot find the requested files in the cache",
        "Read timed out",
        "We couldn't connect to huggingface.co: name resolution failed",
    ],
)
def test_hub_access_errors_are_recognised(message: str) -> None:
    assert _is_hub_access_error(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory",
        "Error while deserializing header: header too small",
        "size mismatch for lm_head.weight",
    ],
)
def test_genuine_failures_are_not_treated_as_hub_errors(message: str) -> None:
    """A corrupt checkpoint or OOM must not be retried as if it were a network blip."""
    assert _is_hub_access_error(RuntimeError(message)) is False


# ------------------------------------------------------------- error messages
def test_gated_message_never_blames_a_local_checkpoint(tmp_path: Path) -> None:
    """A local directory can never be gated; naming it would mislead the operator."""
    from atria_echotrace.config import Settings
    from atria_echotrace.ml.engine import InferenceEngine

    # Force the base model to resolve remotely so it is the only legitimate target.
    engine = InferenceEngine(Settings(base_model_id="some-org/not-on-disk"))
    adapter = {"id": "camus", "repo": "C:/local/checkpoint", "source": "local"}
    text = engine._explain_load_error(RuntimeError("401 gated repo"), adapter)
    assert "some-org/not-on-disk" in text
    assert "C:/local/checkpoint" not in text


def test_gated_message_is_not_emitted_when_everything_is_local(monkeypatch) -> None:
    """With no remote weight involved, "access denied" would misdirect entirely."""
    from atria_echotrace.config import Settings
    from atria_echotrace.ml.engine import InferenceEngine

    import atria_echotrace.config as config_module

    # Settings is a frozen-ish pydantic model, so patch the resolver it calls.
    monkeypatch.setattr(config_module, "local_model_dir", lambda *a, **k: Path("/models/x"))
    engine = InferenceEngine(Settings())
    adapter = {"id": "camus", "repo": "/adapters/x", "source": "local"}
    text = engine._explain_load_error(RuntimeError("401 gated repo"), adapter)
    assert "Access denied" not in text
    assert "resolves to a local folder" in text
    assert "ATRIA_OFFLINE=1" in text


def test_gated_message_names_a_remote_adapter() -> None:
    from atria_echotrace.config import Settings
    from atria_echotrace.ml.engine import InferenceEngine

    engine = InferenceEngine(Settings())
    adapter = {"id": "camus", "repo": "The-Adimension/EchoTrace-MedGemma-CAMUS"}
    text = engine._explain_load_error(RuntimeError("401 gated repo"), adapter)
    assert "The-Adimension/EchoTrace-MedGemma-CAMUS" in text


def test_out_of_memory_message_suggests_cpu() -> None:
    from atria_echotrace.config import Settings
    from atria_echotrace.ml.engine import InferenceEngine

    engine = InferenceEngine(Settings())
    text = engine._explain_load_error(RuntimeError("CUDA out of memory"), {"id": "base"})
    assert "ATRIA_FORCE_CPU=1" in text
