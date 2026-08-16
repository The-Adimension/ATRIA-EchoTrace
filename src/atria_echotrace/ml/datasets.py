"""Training-sample preparation and Hugging Face dataset construction.

Faithful port of the notebook's dataset cells:
``prepare_echocardiographic_frame_samples`` (notebook_as_py.txt L449-500),
``load_images_for_samples`` (L535-554), ``format_sample_for_training`` (L557-591) and
the ``DatasetDict`` assembly (L606-616).
"""

from __future__ import annotations

from typing import Any

from ..data.dataset import DatasetRepository
from ..data.frames import load_frame
from ..logging_setup import get_logger
from .prompts import PromptVariant, build_prompt, create_ground_truth_response

logger = get_logger("ml.datasets")


def prepare_samples(
    repo: DatasetRepository,
    split: str = "train",
    target_structure: str = "LV",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Collect samples for one split.

    Port of ``prepare_echocardiographic_frame_samples`` (L449-500): keep only frames
    in the split that have a target polygon of >= 3 points *and* an existing PNG, then
    cap the count. Skip counts are logged, as the notebook printed them.
    """
    samples: list[dict[str, Any]] = []
    skipped_no_polygon = 0
    skipped_no_png = 0

    for frame in sorted(repo.frames_in_split(split), key=lambda f: f.stem):
        if max_samples and len(samples) >= max_samples:
            break
        polygon = frame.polygon(target_structure)
        if not polygon or len(polygon) < 3:
            skipped_no_polygon += 1
            continue
        if not (repo.frames_dir / f"{frame.stem}.png").is_file():
            skipped_no_png += 1
            continue
        samples.append(
            {
                "key": frame.stem,
                "view": frame.view,
                "instant": frame.instant,
                "polygon": polygon,
                "image_h": frame.image_h,
                "image_w": frame.image_w,
                "patient_id": frame.case_id,
            }
        )

    logger.info(
        "Split %r: %d samples prepared (target=%s); skipped %d without polygon, %d without PNG",
        split,
        len(samples),
        target_structure.upper(),
        skipped_no_polygon,
        skipped_no_png,
    )
    return samples


def load_images_for_samples(
    repo: DatasetRepository, samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach a decoded RGB image to each sample (port of L535-554).

    Frames that fail to load are dropped with a warning rather than aborting the run,
    matching the notebook's behaviour.
    """
    loaded: list[dict[str, Any]] = []
    failed = 0
    for index, sample in enumerate(samples):
        if index % 200 == 0:
            logger.info("Loading images: %d/%d", index, len(samples))
        try:
            sample["image"] = load_frame(repo.frame_path(sample["key"]))
            loaded.append(sample)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not stop training
            failed += 1
            if failed <= 5:
                logger.warning("Error loading %s: %s", sample["key"], exc)
    logger.info("Loaded %d images, %d failed", len(loaded), failed)
    return loaded


def format_sample_for_training(
    sample: dict[str, Any],
    target_structure: str = "LV",
    prompt_variant: PromptVariant | None = None,
) -> dict[str, Any]:
    """Convert a sample into the conversational format SFTTrainer expects.

    Port of ``format_sample_for_training`` (L557-591). The prompt defaults to the
    training variant, which is what the published adapters were fine-tuned with
    (RESEARCH.md §0.4).
    """
    prompt, _ = build_prompt(
        target_structure=target_structure,
        view=sample["view"],
        instant=sample["instant"],
        variant=prompt_variant or "training",
    )
    response = create_ground_truth_response(sample["polygon"], target_structure)

    return {
        "image": sample["image"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response},
                ],
            },
        ],
        "polygon": sample["polygon"],
        "key": sample["key"],
        "image_h": sample["image_h"],
        "image_w": sample["image_w"],
    }


def build_hf_dataset(
    repo: DatasetRepository,
    target_structure: str = "LV",
    train_max_samples: int | None = 1000,
    val_max_samples: int | None = 200,
    prompt_variant: PromptVariant | None = None,
) -> Any:
    """Build the ``DatasetDict`` used for fine-tuning (port of L594-616).

    Defaults reproduce the notebook's ``TRAIN_MAX_SAMPLES=1000`` and
    ``VAL_MAX_SAMPLES=200`` (L222-223).

    Raises:
        RuntimeError: if the training split ends up empty, which would otherwise
            surface much later as an opaque trainer error.
    """
    from datasets import Dataset, DatasetDict

    train_samples = load_images_for_samples(
        repo, prepare_samples(repo, "train", target_structure, train_max_samples)
    )
    val_samples = load_images_for_samples(
        repo, prepare_samples(repo, "val", target_structure, val_max_samples)
    )

    if not train_samples:
        raise RuntimeError(
            "No training samples found. The dataset must provide frames whose split is "
            "'train' (from metadata.csv or the 'split' field in tracings.json) with a "
            f"{target_structure.upper()} polygon. The bundled sample-dataset contains "
            "only 'test' frames, so training needs the full preprocessed dataset."
        )

    train_data = [
        format_sample_for_training(s, target_structure, prompt_variant) for s in train_samples
    ]
    val_data = [
        format_sample_for_training(s, target_structure, prompt_variant) for s in val_samples
    ]

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
        }
    )
    logger.info("Dataset built (%s): %s", target_structure.upper(), dataset)
    return dataset
