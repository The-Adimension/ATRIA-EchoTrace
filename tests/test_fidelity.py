"""Fidelity tests: assert the port matches the original notebook, byte for byte.

These tests read ``notebook_as_py.txt`` at run time and compare the extracted literals
with the application's constants. That makes the notebook the authority rather than a
transcription of it, so any drift is caught rather than assumed absent.
"""

from __future__ import annotations

import re

import pytest

from atria_echotrace.domain.structures import (
    INSTANT_NAMES,
    NORM_SCALE,
    STRUCTURE_INFO,
    VIEW_NAMES,
)
from atria_echotrace.ml import train
from atria_echotrace.ml.prompts import (
    PROMPT_TEMPLATE_GENERIC,
    PROMPT_TEMPLATE_TRAINING,
    build_prompt,
    create_ground_truth_response,
)


def _extract_assignments(source: str, name: str) -> list[str]:
    """Extract every ``name = ( ... )`` parenthesised block from the notebook text.

    ``notebook_as_py.txt`` contains shell magics (``! pip install``), so it cannot be
    parsed with ``ast``; the blocks are located textually instead.
    """
    blocks: list[str] = []
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if not re.match(rf"^{re.escape(name)}\s*=\s*\($", line.strip()):
            continue
        collected = [line]
        for following in lines[index + 1 :]:
            collected.append(following)
            if following.rstrip() == ")":
                break
        blocks.append("\n".join(collected))
    return blocks


def _eval_block(block: str, name: str) -> str:
    namespace: dict[str, object] = {}
    exec(block, namespace)  # noqa: S102 - executing an extracted literal, by design
    return namespace[name]  # type: ignore[return-value]


def test_notebook_defines_two_prompt_templates(notebook_source: str) -> None:
    """The notebook contains exactly two PROMPT_TEMPLATE definitions (RESEARCH.md §0.4)."""
    blocks = _extract_assignments(notebook_source, "PROMPT_TEMPLATE")
    assert len(blocks) == 2, (
        f"expected 2 PROMPT_TEMPLATE definitions in the notebook, found {len(blocks)}"
    )


def test_prompt_templates_match_notebook_byte_for_byte(notebook_source: str) -> None:
    """Both shipped templates are exactly the notebook's, character for character."""
    blocks = _extract_assignments(notebook_source, "PROMPT_TEMPLATE")
    extracted = {_eval_block(block, "PROMPT_TEMPLATE") for block in blocks}

    assert PROMPT_TEMPLATE_TRAINING in extracted, (
        "the training prompt template does not match any template in the notebook"
    )
    assert PROMPT_TEMPLATE_GENERIC in extracted, (
        "the generic prompt template does not match any template in the notebook"
    )
    assert extracted == {PROMPT_TEMPLATE_TRAINING, PROMPT_TEMPLATE_GENERIC}


def test_templates_differ_only_in_the_query_line() -> None:
    """The two variants differ solely in whether view and instant are named."""
    assert PROMPT_TEMPLATE_TRAINING != PROMPT_TEMPLATE_GENERIC
    assert "apical {view_name} echocardiogram view at {instant_name}" in PROMPT_TEMPLATE_TRAINING
    assert "apical echocardiogram view" in PROMPT_TEMPLATE_GENERIC
    assert "{view_name}" not in PROMPT_TEMPLATE_GENERIC
    assert "{instant_name}" not in PROMPT_TEMPLATE_GENERIC

    head = "Instructions:\nThe following user query will require outputting polygon"
    assert PROMPT_TEMPLATE_TRAINING.startswith(head)
    assert PROMPT_TEMPLATE_GENERIC.startswith(head)


def test_structure_info_matches_notebook(notebook_source: str) -> None:
    """STRUCTURE_INFO labels and polygon keys match the notebook's config cell."""
    assert '"label": "left_ventricle_endocardium"' in notebook_source
    assert STRUCTURE_INFO["LV"]["label"] == "left_ventricle_endocardium"
    assert STRUCTURE_INFO["LV"]["name"] == "left ventricle endocardial border"
    assert STRUCTURE_INFO["LV"]["poly_key"] == "lv_polygon"
    assert STRUCTURE_INFO["LA"]["label"] == "left_atrium"
    assert STRUCTURE_INFO["LA"]["name"] == "left atrium border"
    assert STRUCTURE_INFO["LA"]["poly_key"] == "la_polygon"

    assert VIEW_NAMES == {"2CH": "2-chamber", "4CH": "4-chamber"}
    assert INSTANT_NAMES == {"ED": "end-diastole", "ES": "end-systole"}


def test_norm_scale_matches_notebook(notebook_source: str) -> None:
    assert "NORM_SCALE = 1000" in notebook_source
    assert NORM_SCALE == 1000


@pytest.mark.parametrize(
    ("constant", "expected"),
    [
        ("lora_alpha", 32),
        ("lora_dropout", 0.05),
        ("r", 32),
    ],
)
def test_lora_hyperparameters_match_notebook(
    notebook_source: str, constant: str, expected: float
) -> None:
    """LoRA hyperparameters match the notebook's Configure LoRA cell.

    These same values appear in the published adapters' ``adapter_config.json``
    (RESEARCH.md §0.2), so this pins the port to both sources at once.
    """
    match = re.search(rf"^{re.escape(constant)} = ([0-9.e-]+)", notebook_source, re.MULTILINE)
    assert match, f"{constant} not found in the notebook"
    assert float(match.group(1)) == expected

    shipped = {
        "lora_alpha": train.LORA_ALPHA,
        "lora_dropout": train.LORA_DROPOUT,
        "r": train.LORA_R,
    }[constant]
    assert float(shipped) == expected


def test_lora_structural_config_matches_notebook(notebook_source: str) -> None:
    assert 'target_modules="all-linear"' in notebook_source
    assert train.LORA_TARGET_MODULES == "all-linear"
    assert train.LORA_MODULES_TO_SAVE == ["lm_head", "embed_tokens"]
    # modules_to_save entries appear on their own lines in the notebook cell.
    assert '"lm_head"' in notebook_source
    assert '"embed_tokens"' in notebook_source


def test_sft_hyperparameters_match_notebook(notebook_source: str) -> None:
    """Batch/accumulation/sequence settings match the notebook's SFTConfig cell."""
    assert "per_device_train_batch_size=16" in notebook_source
    assert train.PER_DEVICE_TRAIN_BATCH_SIZE == 16
    assert "per_device_eval_batch_size=4" in notebook_source
    assert train.PER_DEVICE_EVAL_BATCH_SIZE == 4
    assert "gradient_accumulation_steps=16" in notebook_source
    assert train.GRADIENT_ACCUMULATION_STEPS == 16
    assert "max_length=2048" in notebook_source
    assert train.MAX_SEQUENCE_LENGTH == 2048
    assert "learning_rate = 2e-4" in notebook_source
    assert train.DEFAULT_LEARNING_RATE == 2e-4
    assert "num_train_epochs = 10" in notebook_source
    assert train.DEFAULT_EPOCHS == 10
    # The extra image-token id masked out of the loss.
    assert "labels[labels == 262144] = -100" in notebook_source
    assert train.EXTRA_IMAGE_TOKEN_ID == 262144


def test_ground_truth_response_matches_notebook_format() -> None:
    """The training target string matches ``create_ground_truth_response`` (L416-423)."""
    polygon = [[100, 200], [150, 250], [200, 300]]
    response = create_ground_truth_response(polygon, "LV")
    assert response.startswith(
        "The left ventricle endocardial border traces the inner wall of the heart chamber.\n\n"
    )
    assert "Final Answer: ```json" in response
    assert '"label": "left_ventricle_endocardium"' in response
    assert '"polygon_2d": [[100, 200], [150, 250], [200, 300]]' in response


def test_build_prompt_defaults_to_training_variant_when_view_known() -> None:
    """Inference should match the fine-tuning distribution by default."""
    text, variant = build_prompt("LV", view="4CH", instant="ED")
    assert variant == "training"
    assert "apical 4-chamber echocardiogram view at end-diastole" in text
    assert "left ventricle endocardial border" in text


def test_build_prompt_falls_back_to_generic_without_view() -> None:
    text, variant = build_prompt("LV", view=None, instant=None)
    assert variant == "generic"
    assert "This is an apical echocardiogram view." in text


def test_build_prompt_rejects_training_variant_without_view() -> None:
    with pytest.raises(ValueError, match="names the view and cardiac instant"):
        build_prompt("LV", view=None, instant=None, variant="training")


def test_build_prompt_renders_la_structure() -> None:
    text, _ = build_prompt("LA", view="2CH", instant="ES")
    assert "left atrium border" in text
    assert "left_atrium" in text
    assert "apical 2-chamber echocardiogram view at end-systole" in text
