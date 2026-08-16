# evidence/ — the data behind every published number

Small, tracked, canonical. If a figure appears in the README, the docs or the showcase, its source
is here. Nothing in this directory is generated at read time; it is the recorded output of the
runs that actually happened.

```
benchmark/    the 200-frame held-out evaluation
training/     all six fine-tuning runs
adapters/     the released LoRA configurations and recorded training summaries
```

## benchmark/

| File | What it is |
|---|---|
| `pointwise.csv` | Per-frame point-to-curve distance — **the primary metric** — plus in-bounds and self-intersection flags |
| `per_frame.csv` | Per-frame Dice / IoU / MAD / HD95 / NSD against the expert mask |
| `scores.json` | Full scoring output, including per-patient EF |
| `overlays_manifest.csv` | Per-frame class, point count, distance and overlay path |
| `raw_predictions.jsonl` | The model's verbatim responses, one per line, flushed as produced |

`raw_predictions.jsonl` is the ground truth for everything else in this directory: the CSVs are
derived from it, so a disputed number can be traced back to the exact generation.

Regenerate with `tools/benchmark/`. See [`../docs/evaluation.md`](../docs/evaluation.md).

## training/

| File | What it is |
|---|---|
| `all_runs_metrics.csv` | Every logged scalar from all six runs — step, tag, wall time, value |
| `finetuning_runs_overview.csv` / `.json` | One row per run: LR, epochs, batch, final losses, token accuracy |
| `all_runs_metadata_and_configs.json` | Recovered configuration per run |
| `Run_*__hyperparameters.json` | Per-run hyperparameters and sample size, as recorded |

Extracted from the TensorBoard event files by `tools/training/extract_tb_logs.py`. The event files
themselves ship in [`../showcase/ft_runs/`](../showcase/ft_runs/).

See [`../docs/training.md`](../docs/training.md).

## adapters/

`adapter_config.json` and `training_metadata.json` for both released adapters, read from the
shipped checkpoints. This is the artifact-level proof of the LoRA configuration — rank 32, ten
target modules including the SigLIP vision tower's `fc1`, `fc2` and `out_proj`.

The weights themselves are **not** here. They are gated on Hugging Face; see
[`../docs/reproducing.md`](../docs/reproducing.md).

## A note on trust

Two figures in this directory are recorded but **not** measurements, and are documented as such:

- EchoNet `train_sample_size: 20000` is `folder_name`-inferred with confidence `low`. The
  step-count derivation gives ≈16,576, which is the figure to use.
- The GPU model is not a recorded field anywhere. Folder names say "blackwell"; that is a naming
  convention, not evidence.
