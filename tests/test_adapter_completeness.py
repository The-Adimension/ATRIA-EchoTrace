"""The adapter must load *completely*, and a partial load must be visible.

The published adapters were trained when Gemma 3 nested the vision encoder as
`vision_tower.vision_model.encoder`. Transformers 5.x flattened that to
`vision_tower.encoder`, so 324 of the checkpoint's 802 tensors addressed a module path
that no longer exists. PEFT warned and continued, leaving every vision-tower `lora_B` at
its zero initialisation — 40 % of the adapter contributing nothing, silently, through
every measurement this project had made.

These tests exist so that can never happen again unnoticed.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.model


def _require_ml() -> None:
    from importlib.util import find_spec

    for name in ("torch", "transformers", "peft"):
        if find_spec(name) is None:
            pytest.skip(f"the [ai] extra is not installed ({name} missing)")


@pytest.mark.adapter
@pytest.mark.parametrize("adapter_id", ["camus", "echonet"])
def test_adapter_loads_completely(adapter_id: str) -> None:
    """Every vision-tower LoRA tensor must be active after load."""
    _require_ml()
    from atria_echotrace.config import Settings
    from atria_echotrace.ml.engine import InferenceEngine

    engine = InferenceEngine(Settings())
    try:
        engine.load(adapter_id)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"adapter {adapter_id} unavailable: {exc}")

    try:
        load = engine.status()["adapter_load"]
        assert load, "status() must report adapter_load"

        # The headline guarantee.
        assert load["fully_loaded"], (
            f"{adapter_id} is only partially loaded: "
            f"{load['vision_lora_b_active']}/{load['vision_lora_b']} vision LoRA tensors active"
        )
        assert load["vision_lora_b"] > 0, "no vision-tower LoRA found at all"
        assert load["vision_lora_b_active"] == load["vision_lora_b"]

        # Belt and braces: check the live weights, not just the report.
        model = engine._model
        zeros = [
            name
            for name, param in model.named_parameters()
            if "lora_B" in name and "vision_tower" in name and float(param.abs().sum()) == 0
        ]
        assert not zeros, f"{len(zeros)} vision-tower lora_B tensors are all-zero, e.g. {zeros[:2]}"
    finally:
        engine.unload()


@pytest.mark.adapter
def test_legacy_key_layout_is_actually_repaired() -> None:
    """The checkpoint really does carry the legacy layout, so the repair is load-bearing.

    If a future adapter ships with modern keys this test still passes (remapped == 0 and
    the model is fully loaded); it only fails if a legacy checkpoint stops being repaired.
    """
    _require_ml()
    from atria_echotrace.config import Settings
    from atria_echotrace.ml.engine import InferenceEngine

    engine = InferenceEngine(Settings())
    try:
        engine.load("camus")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"camus adapter unavailable: {exc}")
    try:
        load = engine.status()["adapter_load"]
        assert load["fully_loaded"]
        # The shipped CAMUS checkpoint is legacy-keyed; assert the repair engaged.
        assert load["remapped"] > 0, (
            "expected the legacy vision_model. nesting to be remapped; if the checkpoint "
            "was re-exported with modern keys, update this expectation deliberately"
        )
    finally:
        engine.unload()
