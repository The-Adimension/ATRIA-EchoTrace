"""QLoRA fine-tuning of MedGemma 1.5 for contour tracing.

Faithful port of the notebook's fine-tuning cells:
model load with quantisation (notebook_as_py.txt L618-658), ``LoraConfig`` (L660-686),
``collate_fn`` (L688-744), ``SFTConfig`` (L755-794), ``SFTTrainer`` construction
(L806-823), ``trainer.train()`` (L832-839) and saving (L844-851).

Every hyperparameter below matches the notebook **and** the published
``adapter_config.json`` of the released adapters (RESEARCH.md §0.2), so re-running this
reproduces the released training configuration.

One necessary adaptation: the notebook sets ``bf16=True`` unconditionally, which
requires CUDA compute capability >= 8.0. On older GPUs this module switches to fp16;
on CPU both are disabled. Nothing else deviates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings, settings as default_settings
from ..data.dataset import DatasetRepository
from ..logging_setup import get_logger
from .datasets import build_hf_dataset
from .runtime import configure_torch_allocator, select_device_policy

logger = get_logger("ml.train")

# --- LoRA hyperparameters, verified against the released adapter_config.json ---
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"
LORA_MODULES_TO_SAVE = ["lm_head", "embed_tokens"]

# --- SFT hyperparameters (notebook L760-794) ---
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
MAX_SEQUENCE_LENGTH = 2048

#: Extra image-token id masked out of the loss (notebook L741).
EXTRA_IMAGE_TOKEN_ID = 262144


@dataclass
class TrainingResult:
    """Outcome of a fine-tuning run."""

    output_dir: str
    epochs: float
    train_samples: int
    eval_samples: int
    metrics: dict[str, Any]


def build_peft_config() -> Any:
    """Build the LoRA configuration (notebook L669-680)."""
    from peft import LoraConfig

    return LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        modules_to_save=list(LORA_MODULES_TO_SAVE),
    )


def build_collate_fn(processor: Any):
    """Build the multimodal collator (notebook ``collate_fn``, L690-744).

    Labels are the input ids with padding and image tokens masked to ``-100`` so the
    loss is computed only over real text.
    """

    def collate_fn(examples: list[dict[str, Any]]):
        texts = []
        images = []

        for example in examples:
            image = example["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append([image])
            texts.append(
                processor.apply_chat_template(
                    example["messages"], add_generation_prompt=False, tokenize=False
                ).strip()
            )

        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )

        labels = batch["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id

        # Image-token id lookup differs across tokenizer versions; the notebook
        # wrapped this in a bare try/except.
        try:
            image_token_id = processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map.get("boi_token", "<image>")
            )
        except Exception:  # pragma: no cover - tokenizer-version dependent
            image_token_id = None

        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        labels[labels == EXTRA_IMAGE_TOKEN_ID] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def build_sft_config(
    output_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    use_bf16: bool = True,
    use_fp16: bool = False,
) -> Any:
    """Build the ``SFTConfig`` (notebook L769-794)."""
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=3,
        learning_rate=learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=0.3,
        warmup_steps=0.1,
        lr_scheduler_type="linear",
        push_to_hub=False,
        report_to="tensorboard",
        logging_dir=str(output_dir / "logs"),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
    )


def run_training(
    output_dir: Path,
    repo: DatasetRepository | None = None,
    settings: Settings | None = None,
    target_structure: str = "LV",
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    train_max_samples: int | None = 1000,
    val_max_samples: int | None = 200,
) -> TrainingResult:
    """Fine-tune MedGemma 1.5 with QLoRA and save the adapter.

    Args:
        output_dir: Destination for checkpoints, the adapter and TensorBoard logs.
        repo: Dataset repository; built from settings when omitted.
        settings: Configuration override.
        target_structure: ``"LV"`` or ``"LA"``.
        epochs: Training epochs (notebook default 10).
        learning_rate: Peak learning rate (notebook default 2e-4).
        train_max_samples: Cap on training samples (notebook default 1000).
        val_max_samples: Cap on validation samples (notebook default 200).

    Returns:
        A :class:`TrainingResult`.

    Raises:
        RuntimeError: if CUDA is unavailable. Fine-tuning a 4-B vision-language model
            on CPU is not viable, so this fails immediately with an explanation
            rather than appearing to hang for days.
    """
    import gc

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from trl import SFTTrainer

    settings = settings or default_settings
    repo = repo or DatasetRepository(settings.dataset_dir)

    configure_torch_allocator()
    policy = select_device_policy(force_cpu=settings.force_cpu)
    if policy["device"] != "cuda":
        raise RuntimeError(
            "Fine-tuning requires a CUDA GPU. The notebook's configuration "
            "(batch 16 x grad-accum 16, 4-bit base, LoRA on the vision tower) targets "
            "a 16 GB+ device. Use `atria evaluate` or the inference API on CPU instead."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()
    logger.info(
        "GPU: %s (%s GB), compute dtype %s",
        policy["gpu_name"],
        policy["total_memory_gb"],
        policy["compute_dtype_name"],
    )

    token = settings.token_value()
    model_id = settings.base_model_id

    # Quantisation for training, mirroring the notebook's cell (L641-647). The
    # notebook passed load_in_4bit=False there, keeping bf16 storage while still
    # routing through BitsAndBytes; that is preserved.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=policy["compute_dtype"],
        bnb_4bit_quant_storage=policy["compute_dtype"],
    )

    logger.info("Loading model %s for fine-tuning", model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        dtype=policy["compute_dtype"],
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        token=token,
    )
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    # Right padding for training (notebook L654).
    processor.tokenizer.padding_side = "right"

    data = build_hf_dataset(
        repo=repo,
        target_structure=target_structure,
        train_max_samples=train_max_samples,
        val_max_samples=val_max_samples,
    )

    # Always bf16, never fp16: Gemma-family activations overflow the fp16 range, which
    # corrupts training the same way it corrupts inference (RESEARCH.md §2.2). On a
    # pre-Ampere GPU bf16 is emulated — correct, but slow enough to be impractical.
    args = build_sft_config(
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        use_bf16=True,
        use_fp16=False,
    )
    if not policy["bf16_supported"]:
        logger.warning(
            "This GPU (%s) has no native bfloat16, so training runs on emulated "
            "bf16 and will be extremely slow. float16 is not used as a substitute "
            "because Gemma-family models overflow in fp16. The released adapters "
            "were trained on hardware with native bf16.",
            policy["capability"],
        )

    # Reduced eval set, as the notebook (L819).
    validation = data["validation"]
    eval_dataset = validation.shuffle(seed=42).select(range(min(50, len(validation))))

    gc.collect()
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=eval_dataset,
        peft_config=build_peft_config(),
        processing_class=processor,
        data_collator=build_collate_fn(processor),
    )
    logger.info(
        "Trainer ready: %d train / %d eval samples", len(data["train"]), len(eval_dataset)
    )

    train_output = trainer.train()

    trainer.save_model()
    processor.save_pretrained(str(output_dir))
    logger.info("Adapter and processor saved to %s", output_dir)

    return TrainingResult(
        output_dir=str(output_dir),
        epochs=float(epochs),
        train_samples=len(data["train"]),
        eval_samples=len(eval_dataset),
        metrics=dict(train_output.metrics or {}),
    )
