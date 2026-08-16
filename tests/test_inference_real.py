"""End-to-end inference against real MedGemma weights. No mocks.

Run with::

    pytest -m model                      # base model, from the local HF cache
    pytest -m "model and adapter"        # additionally the gated LoRA adapters

These are slow: loading the 4-bit model takes ~30-50 s and a single 1024-token
generation takes seconds on a modern GPU, minutes on an emulated-bf16 card or CPU.
They are excluded from a default ``pytest`` run only by the marker, not by mocking.
"""

from __future__ import annotations

import pytest

from atria_echotrace.config import Settings
from atria_echotrace.domain.geometry import polygon_dice, polygon_iou

pytestmark = pytest.mark.model


def _require_ml() -> None:
    from importlib.util import find_spec

    for name in ("torch", "transformers", "peft"):
        if find_spec(name) is None:
            pytest.skip(f"the [ai] extra is not installed ({name} missing)")


@pytest.fixture(scope="module")
def engine():
    """A real engine with the un-adapted base model loaded from the local cache."""
    _require_ml()
    from atria_echotrace.ml.engine import InferenceEngine

    instance = InferenceEngine(Settings(offline=True))
    try:
        instance.load("base")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"base weights unavailable: {exc}")
    yield instance
    instance.unload()


# --------------------------------------------------------------- device policy
def test_device_policy_never_selects_float16() -> None:
    """float16 makes Gemma-family models emit only padding (RESEARCH.md §2.2)."""
    _require_ml()
    from atria_echotrace.ml.runtime import select_device_policy

    policy = select_device_policy()
    assert policy["compute_dtype_name"] in {"bfloat16", "float32"}
    if policy["device"] == "cuda":
        assert policy["compute_dtype_name"] == "bfloat16"
        assert policy["quantization"] == "nf4"


def test_device_policy_honours_force_cpu() -> None:
    _require_ml()
    from atria_echotrace.ml.runtime import select_device_policy

    policy = select_device_policy(force_cpu=True)
    assert policy["device"] == "cpu"
    assert policy["quantization"] is None


# ------------------------------------------------------------- model lifecycle
def test_model_loads_and_reports_ready(engine) -> None:
    status = engine.status()
    assert status["state"] == "ready"
    assert status["base_model_id"] == "google/medgemma-1.5-4b-it"
    assert status["adapter"]["id"] == "base"
    assert status["load_seconds"] > 0
    if status["device"] == "cuda":
        assert status["compute_dtype"] == "bfloat16"


def test_predict_returns_a_usable_polygon(engine, dataset_repo) -> None:
    """The real model must return a parseable, in-bounds polygon.

    Contour *accuracy* is an adapter property; this asserts the pipeline —
    prompt construction, generation, parsing, sanitising — produces valid geometry.
    """
    from atria_echotrace.data.frames import load_frame

    frame = dataset_repo.get_frame("patient0258_4CH_ED")
    image = load_frame(dataset_repo.frame_path(frame.stem))

    result = engine.predict(image, "LV", view=frame.view, instant=frame.instant)

    assert result["vertices"] >= 3
    assert result["prompt_variant"] == "training"
    assert result["target_structure"] == "LV"
    assert result["inference_seconds"] > 0
    for y, x in result["polygon"]:
        assert 0 <= y <= 1000
        assert 0 <= x <= 1000
        assert isinstance(y, int) and isinstance(x, int)


def test_predict_rejects_unknown_structure(engine, dataset_repo) -> None:
    from atria_echotrace.data.frames import load_frame

    frame = dataset_repo.get_frame("patient0258_4CH_ED")
    image = load_frame(dataset_repo.frame_path(frame.stem))
    with pytest.raises(ValueError, match="Unknown target structure"):
        engine.predict(image, "RV", view="4CH", instant="ED")


def test_predict_before_load_raises(dataset_repo) -> None:
    _require_ml()
    from PIL import Image

    from atria_echotrace.ml.engine import InferenceEngine, ModelNotReady

    idle = InferenceEngine(Settings(offline=True))
    with pytest.raises(ModelNotReady):
        idle.predict(Image.new("RGB", (64, 64)), "LV", view="4CH", instant="ED")


def test_unload_releases_the_model() -> None:
    _require_ml()
    from atria_echotrace.ml.engine import InferenceEngine

    instance = InferenceEngine(Settings(offline=True))
    status = instance.unload()
    assert status["state"] == "unloaded"
    assert instance.is_ready is False


# ------------------------------------------------------- gated LoRA adapters
@pytest.mark.adapter
@pytest.mark.parametrize("adapter_id", ["camus", "echonet"])
def test_adapter_produces_an_accurate_contour(adapter_id: str, dataset_repo) -> None:
    """Each published adapter should trace its own dataset's frames well.

    Resolves the adapter from the local checkpoints under ``adapters/`` when present,
    which needs no token; otherwise it falls back to the gated Hugging Face repo.
    """
    _require_ml()
    from atria_echotrace.data.frames import load_frame
    from atria_echotrace.ml.engine import InferenceEngine

    stem = (
        "patient0258_4CH_ED"
        if adapter_id == "camus"
        else "echonet_0X171FD888481D524D_4CH_ED"
    )
    frame = dataset_repo.get_frame(stem)
    image = load_frame(dataset_repo.frame_path(stem))

    instance = InferenceEngine(Settings())
    try:
        instance.load(adapter_id)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"adapter {adapter_id} unavailable (no local checkpoint, no token): {exc}")

    try:
        result = instance.predict(image, "LV", view=frame.view, instant=frame.instant)
        dice = polygon_dice(
            result["polygon"], frame.lv_polygon, frame.image_h, frame.image_w
        )
        iou = polygon_iou(result["polygon"], frame.lv_polygon, frame.image_h, frame.image_w)
        print(f"\n{adapter_id}: {result['vertices']} vertices, Dice {dice:.4f}, IoU {iou:.4f}")

        assert result["vertices"] >= 3
        # A fine-tuned adapter should overlap the reference substantially. This is a
        # deliberately loose floor: it catches "the model traced something unrelated"
        # without asserting a benchmark number that hardware could shift.
        assert dice > 0.5, f"{adapter_id} Dice {dice:.4f} against ground truth"
    finally:
        instance.unload()
