"""Prompt templates and ground-truth response construction.

The notebook contains **two** prompt templates that differ in one line, and the
difference matters (RESEARCH.md §0.4):

* :data:`PROMPT_TEMPLATE_TRAINING` — used by the fine-tuning and evaluation cells
  (notebook_as_py.txt L398-414). Its query names the view and the cardiac instant.
  **The published adapters were trained with this template.**
* :data:`PROMPT_TEMPLATE_GENERIC` — used by the HITL interface cell (L1205-1220) and
  by the author's deployed Space. It omits view and instant, because a Colab file
  upload has no such metadata. It still *accepts* those keyword arguments, where
  ``str.format`` silently ignores them.

Both are preserved verbatim. The application defaults to the training variant
whenever view and instant are known, so inference matches the distribution the LoRA
weights were optimised for, and falls back to the generic variant otherwise.
"""

from __future__ import annotations

import json
from typing import Literal

from ..domain.structures import INSTANT_NAMES, VIEW_NAMES, resolve_structure

PromptVariant = Literal["training", "generic"]

#: Verbatim from notebook_as_py.txt L398-414 (training + evaluation cells).
PROMPT_TEMPLATE_TRAINING = (
    "Instructions:\n"
    "The following user query will require outputting polygon coordinates for "
    "{structure_name} tracing. "
    "The format of polygon coordinates is [[y0, x0], [y1, x1], ...] where each point is "
    "[y, x] representing the boundary. Always normalize the x and y coordinates to the "
    "range [0, 1000], meaning that a point at 15% of the image width would be associated "
    "with an x coordinate of 150. You MUST output a single parseable json list of objects "
    'enclosed into ```json...``` brackets, for instance ```json[{{"polygon_2d": '
    '[[100, 200], [150, 250]], "label": "{structure_label}"}}]``` is a valid output. '
    "Now answer to the user query.\n\n"
    "Query:\n"
    "This is an apical {view_name} echocardiogram view at {instant_name}. "
    "Trace the {structure_name}. "
    'Output the final answer in the format "Final Answer: X" where X is a JSON list of '
    'objects with "polygon_2d" and "label" keys. Answer:'
)

#: Verbatim from notebook_as_py.txt L1205-1220 (Adapter-Human Exchange Interface cell).
PROMPT_TEMPLATE_GENERIC = (
    "Instructions:\n"
    "The following user query will require outputting polygon coordinates for "
    "{structure_name} tracing. "
    "The format of polygon coordinates is [[y0, x0], [y1, x1], ...] where each point is "
    "[y, x] representing the boundary. Always normalize the x and y coordinates to the "
    "range [0, 1000], meaning that a point at 15% of the image width would be associated "
    "with an x coordinate of 150. You MUST output a single parseable json list of objects "
    'enclosed into ```json...``` brackets, for instance ```json[{{"polygon_2d": '
    '[[100, 200], [150, 250]], "label": "{structure_label}"}}]``` is a valid output. '
    "Now answer to the user query.\n\n"
    "Query:\n"
    "This is an apical echocardiogram view. Trace the {structure_name}. "
    'Output the final answer in the format "Final Answer: X" where X is a JSON list of '
    'objects with "polygon_2d" and "label" keys. Answer:'
)

TEMPLATES: dict[str, str] = {
    "training": PROMPT_TEMPLATE_TRAINING,
    "generic": PROMPT_TEMPLATE_GENERIC,
}


def build_prompt(
    target_structure: str = "LV",
    view: str | None = None,
    instant: str | None = None,
    variant: PromptVariant | None = None,
) -> tuple[str, PromptVariant]:
    """Render the prompt for one inference or training example.

    Args:
        target_structure: ``"LV"`` or ``"LA"``.
        view: ``"2CH"`` or ``"4CH"``; required by the training variant.
        instant: ``"ED"`` or ``"ES"``; required by the training variant.
        variant: Force a template. When ``None``, the training variant is chosen if
            both view and instant are known, otherwise the generic one.

    Returns:
        ``(prompt_text, variant_used)``.

    Raises:
        ValueError: for an unknown structure, an unknown variant, or the training
            variant without a recognised view/instant.
    """
    struct = resolve_structure(target_structure)

    view_key = (view or "").upper()
    instant_key = (instant or "").upper()
    known = view_key in VIEW_NAMES and instant_key in INSTANT_NAMES

    if variant is None:
        variant = "training" if known else "generic"
    if variant not in TEMPLATES:
        raise ValueError(
            f"Unknown prompt variant {variant!r}. Expected one of {sorted(TEMPLATES)}."
        )
    if variant == "training" and not known:
        raise ValueError(
            "The training prompt variant names the view and cardiac instant, so both "
            f"are required. Got view={view!r}, instant={instant!r}. Expected view in "
            f"{sorted(VIEW_NAMES)} and instant in {sorted(INSTANT_NAMES)}; use "
            "variant='generic' when they are unknown."
        )

    text = TEMPLATES[variant].format(
        structure_name=struct["name"],
        structure_label=struct["label"],
        view_name=VIEW_NAMES.get(view_key, ""),
        instant_name=INSTANT_NAMES.get(instant_key, ""),
    )
    return text, variant


def create_ground_truth_response(polygon: list, target_structure: str = "LV") -> str:
    """Build the expected assistant response for a training example.

    Verbatim port of the notebook's ``create_ground_truth_response`` (L416-423).
    """
    struct = resolve_structure(target_structure)
    objects = [{"polygon_2d": polygon, "label": struct["label"]}]
    gt_json = json.dumps(objects)
    return (
        f"The {struct['name']} traces the inner wall of the heart chamber.\n\n"
        f"Final Answer: ```json{gt_json}```"
    )


def build_messages(image, prompt_text: str) -> list[dict]:
    """Build the chat message list for the processor.

    Matches the notebook's structure (L1298) and the MedGemma model card: a single
    user turn carrying the image followed by the text.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
