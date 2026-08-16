"""Anatomical structure configuration.

Faithful port of the notebook's config cell (notebook_as_py.txt L198-219) and the
HITL cell's copy of the same constants (L1194-1202). ``TARGET_STRUCTURE`` is no longer a
module-level flag: the notebook switched structures by editing the constant, whereas the
application selects per request, so the flag becomes a parameter.
"""

from __future__ import annotations

from typing import Final, Literal

#: Polygon coordinates are normalised to [0, NORM_SCALE] (notebook L219).
NORM_SCALE: Final[int] = 1000

TargetStructure = Literal["LV", "LA"]

#: Verbatim from the notebook (L205-210). ``poly_key`` indexes tracings.json.
STRUCTURE_INFO: Final[dict[str, dict[str, str]]] = {
    "LV": {
        "label": "left_ventricle_endocardium",
        "name": "left ventricle endocardial border",
        "short": "LV endocardium",
        "poly_key": "lv_polygon",
        "has_key": "has_lv",
    },
    "LA": {
        "label": "left_atrium",
        "name": "left atrium border",
        "short": "LA",
        "poly_key": "la_polygon",
        "has_key": "has_la",
    },
}

#: View / instant names used in natural-language prompts (notebook L394-395).
VIEW_NAMES: Final[dict[str, str]] = {"2CH": "2-chamber", "4CH": "4-chamber"}
INSTANT_NAMES: Final[dict[str, str]] = {"ED": "end-diastole", "ES": "end-systole"}

#: Overlay colours. Red = model, green = clinician, blue = overlay accent.
#: Taken from the author's deployed Space (config.py STRUCTURE_INFO) so exported
#: figures match previously published ATRIA EchoTrace outputs.
COLOR_MODEL: Final[tuple[int, int, int, int]] = (239, 68, 68, 255)
COLOR_USER: Final[tuple[int, int, int, int]] = (34, 197, 94, 255)
COLOR_GROUND_TRUTH: Final[tuple[int, int, int, int]] = (14, 165, 233, 255)


def resolve_structure(target: str) -> dict[str, str]:
    """Return the STRUCTURE_INFO entry for ``target``.

    Raises:
        ValueError: if ``target`` is not a known structure. The notebook silently
            defaulted unknown values to LV; failing loudly is safer in a clinical tool,
            because a mistyped structure would otherwise trace the wrong anatomy.
    """
    key = target.upper()
    if key not in STRUCTURE_INFO:
        raise ValueError(
            f"Unknown target structure {target!r}. Expected one of {sorted(STRUCTURE_INFO)}."
        )
    return STRUCTURE_INFO[key]
