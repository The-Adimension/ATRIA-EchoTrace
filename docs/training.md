# Training

Every value below was read from the **shipped artifacts** — `adapter_config.json` and the
`trl.SFTConfig` pickle inside `training_args.bin` — not copied from the notebook. Raw data:
[`evidence/training/`](../evidence/training/).

## The configuration

| LoRA / PEFT | |
|---|---|
| Base model | `google/medgemma-1.5-4b-it` |
| Rank `r` / `lora_alpha` / dropout | **32 / 32 / 0.05** |
| Quantisation | **4-bit NF4** + double-quant, bfloat16 compute |
| Target modules (10) | `q_proj k_proj v_proj o_proj` · `gate_proj up_proj down_proj` · **`fc1 fc2 out_proj`** |
| Also trainable | `lm_head`, `embed_tokens` |
| rslora / dora | false / false |

`fc1`, `fc2` and `out_proj` are SigLIP vision-encoder layers. Their presence is the artifact-level
proof that LoRA was applied **across the vision tower**, not only the language head.

| Schedule & precision | |
|---|---|
| Epochs / effective batch | **5** / **64** (4 per-device × 16 accum × 1 GPU) |
| Learning rate | **1e-4**, linear decay, **3% warmup** |
| Optimiser | `adamw_torch_fused` · β 0.9/0.999 · weight decay 0.0 |
| Gradient clipping | `max_grad_norm` **0.3** |
| Precision | **bf16** + gradient checkpointing |
| `max_length` / seed | 1024 / 42 |

**On `warmup_steps = 0.03`** — a float in an int-typed field. HuggingFace treats any value in
`[0, 1)` as a *ratio*, so it means 3% warmup, not 0.03 steps. Verified against HF behaviour;
intentional, not a bug.

## Six runs, two datasets

| Run | Dataset | Frames | LR | Epochs | Batch | Eval loss | Token acc. | Elapsed |
|---|---|---|---|---|---|---|---|---|
| 1 | mixed baseline | 4,512 | 2e-4 | 3 | 2 | 0.193 | 0.921 | 5.7 h |
| 2 | CAMUS | 1,600 | 1e-5 | 3 | 2 | 0.752 | 0.790 | 2.0 h |
| 3 | CAMUS | 1,504 | 2e-4 | 8 | 2 | 0.390 | 0.840 | 5.1 h |
| 4 | EchoNet | 16,541 | 1e-4 | 3 | 2 | 0.218 | 0.910 | — |
| **5** | **EchoNet** | **16,576** | **1e-4** | **5** | **4** | **0.205** | **0.916** | **21.3 h** |
| **6** | **CAMUS** | **1,600** | **1e-4** | **5** | **4** | **0.259** | **0.895** | **2.1 h** |

Runs 5 and 6 are the published adapters. Elapsed is HuggingFace's recorded `train_runtime`; Run 4
terminated without writing a final summary, so its runtime is left blank rather than estimated.

### The recipe was found, not guessed

Runs 2, 3 and 6 share the identical 1,600-frame CAMUS split, isolating one variable. 1e-5 never
leaves the plateau. 2e-4 over eight epochs learns but stalls at 0.390. **1e-4 over five epochs
reaches 0.259 in 125 steps** — better than eight epochs achieved in 376, in under half the time.

### It transferred without retuning

Runs 5 and 6 differ only in the data. Of 134 `SFTConfig` fields, **exactly two differ** — the
output and logging directories. One corpus is 1,600 frames in 125 optimizer steps; the other is
16,576 frames in ~1,295 steps. Both converge, both finish **at** their minimum eval loss, and
neither used early stopping or best-checkpoint restoration — the released weights are the final
epoch.

### What it costs

The shipped CAMUS adapter: **128 minutes** on one GPU. EchoNet: 21.3 hours. Throughput roughly
doubled between the early runs and the final two (0.66 → ~1.06 samples/s) when per-device batch
went from 2 to 4. The five runs that logged a runtime total **36 hours** — the entire search,
dead ends included.

## Do not publish

The EchoNet `train_sample_size: 20000` recorded in metadata is `folder_name`-inferred with
confidence `low`. The step-count derivation gives **≈16,576**, which is the trustworthy figure.
