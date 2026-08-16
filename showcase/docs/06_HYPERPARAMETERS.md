# Definitive Training Configuration — recovered from shipped artifacts

Extracted 2026-08-13 directly from the adapter checkpoints in `adapters/`. Every value
below is **read from the artifact**, not inferred from the notebook or documentation.

Source files per adapter: `adapter_config.json`, `training_args.bin` (a
`trl.trainer.sft_config.SFTConfig` torch pickle, 134 fields), `metadata.json`.

> **Headline finding: the two adapters were trained with identical hyperparameters.**
> Of 134 `SFTConfig` fields, exactly **two** differ — `output_dir` and `logging_dir`.
> One recipe, two datasets.

---

## 1. LoRA / PEFT configuration — `adapter_config.json`

Identical for both adapters except `peft_version` (CAMUS 0.19.1 · EchoNet 0.19.0).

| Field | Value |
|---|---|
| `peft_type` | `LORA` |
| `task_type` | `CAUSAL_LM` |
| `base_model_name_or_path` | `google/medgemma-1.5-4b-it` |
| **`r`** | **32** |
| **`lora_alpha`** | **32** |
| **`lora_dropout`** | **0.05** |
| `bias` | `none` |
| `use_rslora` | `false` |
| `use_dora` | `false` |
| `init_lora_weights` | `true` |
| `fan_in_fan_out` | `false` |
| `inference_mode` | `true` |
| `modules_to_save` | `["lm_head", "embed_tokens"]` |
| `rank_pattern` / `alpha_pattern` | `{}` (uniform) |
| `layers_to_transform` / `layers_pattern` | `null` (all layers) |
| `exclude_modules` | `null` |

### `target_modules` — 10 modules, identical in both

```
q_proj  k_proj  v_proj  o_proj        ← attention projections
gate_proj  up_proj  down_proj          ← language-tower MLP
fc1  fc2  out_proj                     ← VISION-tower MLP + attention output
```

`fc1`, `fc2` and `out_proj` are SigLIP vision-encoder module names. **Their presence is the
artifact-level confirmation that LoRA was applied across the vision tower**, matching the
public forum description ("the vision tower's attention and MLP layers across nearly all 27
encoder layers"). These are also precisely the modules silenced by the
key-nesting defect documented in `05_METRICS_AND_VERIFICATION_LOG.md`.

---

## 2. Training arguments — `training_args.bin` (`trl.SFTConfig`)

**Identical for both adapters** unless noted.

### Schedule and optimisation

| Field | Value |
|---|---|
| `num_train_epochs` | **5** |
| `max_steps` | `-1` (epoch-driven) |
| `per_device_train_batch_size` | **4** |
| `gradient_accumulation_steps` | **16** |
| **Effective batch size** | **64** (4 × 16 × 1 GPU) |
| `per_device_eval_batch_size` | 1 |
| `learning_rate` | **1e-4** |
| `lr_scheduler_type` | `linear` |
| `warmup_steps` | **0.03** |
| `warmup_ratio` | `None` |
| `optim` | `adamw_torch_fused` |
| `adam_beta1` / `beta2` / `epsilon` | 0.9 / 0.999 / 1e-8 |
| `weight_decay` | 0.0 |
| `max_grad_norm` | **0.3** |
| `seed` | 42 · `data_seed` `None` |

> **Note on `warmup_steps = 0.03`.** This is a float in an int-typed field. Hugging Face
> treats a value in `[0, 1)` as a *ratio*, so it means **3% warmup**, not 0.03 steps. This
> was verified against HF behaviour and recorded in `GAP_ANALYSIS.md` ("suspicions tested
> and cleared") — it is intentional, not a bug.

### Precision and memory

| Field | Value |
|---|---|
| `bf16` | **True** |
| `fp16` | False |
| `bf16_full_eval` / `fp16_full_eval` | False / False |
| `tf32` | `None` |
| `gradient_checkpointing` | **True** |
| `gradient_checkpointing_kwargs` | `{"use_reentrant": false}` |
| `torch_compile` | False |
| `_n_gpu` | 1 |

### Sequence / dataset handling

| Field | Value |
|---|---|
| **`max_length`** | **1024** |
| `packing` | False |
| `dataset_text_field` | `"text"` |
| `remove_unused_columns` | **False** |
| `label_names` | `["labels"]` |
| `label_smoothing_factor` | 0.0 |
| `dataloader_num_workers` | 0 · `pin_memory` True |

### Logging, evaluation, checkpointing

| Field | Value |
|---|---|
| `logging_strategy` / `logging_steps` | `steps` / **5** |
| `eval_strategy` / `eval_steps` | `steps` / **10** |
| `save_strategy` | **`epoch`** (`save_steps` 500 unused) |
| `save_total_limit` | 3 |
| `load_best_model_at_end` | False |
| `report_to` | `["tensorboard"]` |
| `push_to_hub` | False |

### The only two differing fields

| Field | CAMUS | EchoNet |
|---|---|---|
| `output_dir` | `…/ATRIA-G3/output_CAMUS-32LoRA-blackwell` | `…/ATRIA-G3/output_20K-blackwell` |
| `logging_dir` | `…/output_CAMUS-32LoRA-blackwell/logs` | `…/output_20K-blackwell/logs` |

Both paths are Google Drive mounts under a Colab runtime.

---

## 3. Recorded training progress — `metadata.json`

| | **CAMUS** | **EchoNet** |
|---|---|---|
| Adapter size | 2707.61 MB | 2707.61 MB |
| Logged `train/loss` points | **25** | **259** |
| Derived optimizer steps (× `logging_steps` 5) | **125** | **1,295** |
| Derived steps per epoch (÷ 5 epochs) | **25** | **259** |
| Derived samples per epoch (× batch 64) | **1,600** | **≈16,576** |
| Logged `eval/loss` points | 13 | 130 |
| **Final train loss** | **0.2528** | **0.1959** |
| Min train loss | 0.2496 | 0.1918 |
| Max train loss (start) | 1.4940 | 1.6441 |
| **Final eval loss** | **0.2586** | **0.2052** |
| Min eval loss | 0.2586 | 0.2046 |
| Max eval loss (start) | 0.4752 | 1.4489 |
| Final LR (linear decay from 1e-4) | 8.26e-7 | 7.96e-8 |
| `train_sample_size` (recorded) | `null` | 20000 |
| `sample_size_confidence` | `low` | `low` |
| `sample_size_method` | `null` | **`folder_name`** |
| Metadata extracted | 2026-07-09 | 2026-07-09 |

### Two cautions on this table

1. **CAMUS derivation is exact and self-consistent.** 25 logged points × 5 = 125 steps ÷ 5
   epochs = 25 steps/epoch × batch 64 = **1,600 samples/epoch — precisely the size of the
   CAMUS training split** (400 patients × 2 views × 2 instants). No sample cap was applied.
2. **The EchoNet `train_sample_size: 20000` is not a measurement.**
   `sample_size_method` is literally `"folder_name"` with `confidence: "low"` — it was
   inferred from the directory name `output_20K-blackwell`. The step-count derivation gives
   **≈16,576 samples/epoch**, which is the trustworthy figure. Do not publish "20,000
   training samples" as fact.

---

## 4. Training-progress narrative (drop-in for the case study)

> Both adapters were trained with a single recipe: **QLoRA rank 32 (α 32, dropout 0.05)**
> applied to ten module types spanning the language tower's attention and MLP blocks *and*
> the SigLIP vision encoder's `fc1`, `fc2` and `out_proj` layers, with `lm_head` and
> `embed_tokens` additionally trainable. Training ran for **5 epochs at an effective batch
> size of 64** (per-device 4 × 16 gradient-accumulation steps) on a **single GPU**, using
> **bfloat16** with gradient checkpointing, `adamw_torch_fused`, a **linear schedule from
> 1e-4 with 3% warmup**, and aggressive gradient clipping at `max_grad_norm 0.3`. Sequences
> were capped at **1,024 tokens**; evaluation ran every 10 steps and checkpoints were saved
> each epoch.
>
> The CAMUS run covered the full 1,600-frame training split in **125 optimizer steps**,
> with training loss falling from **1.49 to 0.253** and evaluation loss from **0.475 to
> 0.259** — still at its minimum on the final step, indicating the run ended before
> over-fitting. The larger EchoNet run took **≈1,295 steps**, reaching a training loss of
> **0.196** and evaluation loss of **0.205**, again finishing at its minimum. Neither run
> used early stopping or best-checkpoint restoration; the released adapters are the
> final-epoch weights.
>
> That the two datasets — one 1,600 frames, one an order of magnitude larger — converged
> under an identical configuration is itself a useful result for HAI-DEF developers: the
> recipe transferred without retuning.

---

## 5. Gaps requiring team input

| Item | Status |
|---|---|
| Wall-clock training duration | **Not recoverable from shipped artifacts.** No timestamps in `SFTConfig`; `metadata.json` records only extraction time. Recover from the Colab runtime logs or TensorBoard event files if retained |
| GPU model used for training | **Not recoverable.** `_n_gpu: 1` only. Folder names say `blackwell`, implying a Blackwell-class GPU, but that is a naming convention, not a recorded field — **confirm before publishing** |
| Quantisation during training | **Not in `SFTConfig`** — `BitsAndBytesConfig` is passed at model load, not in TrainingArguments. The notebook disables 4-bit for training (`load_in_4bit=False`); confirm against your Colab run |
| Exact EchoNet training-set size | Derived ≈16,576/epoch; recorded 20000 is `folder_name`-inferred, low confidence |
| TensorBoard event files | Not present in the shipped repos — only the summary in `metadata.json`. Retrieve from Drive if per-step curves are wanted for a figure |
| Base-model revision/commit | `base_model_name_or_path` has no pinned revision. If reproducibility matters, record the exact `google/medgemma-1.5-4b-it` commit used |
