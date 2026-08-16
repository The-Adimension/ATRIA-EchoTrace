# Adapting to your own data

The pipeline does not assume CAMUS or EchoNet. It assumes the **three-artifact contract**. Conform
to that and every downstream stage works unchanged.

On a CAMUS-sized corpus this is **an afternoon**, not a programme: the shipped CAMUS adapter took
**128 minutes** on a single GPU.

---

## 1. Bring your frames into the contract

```
<your-dataset>/frames/<stem>.png      one RGB frame per stem, any resolution
<your-dataset>/tracings.json          {stem: {...}}
<your-dataset>/metadata.csv           optional: key,split,…
```

Each `tracings.json` entry needs at minimum `image_h`, `image_w` and `lv_polygon` as `[[y, x]]`
normalised to `[0, 1000]`. Supply `spacing_h` / `spacing_w` in mm/px if you have them — use `1.0`
to mean *unknown*, and the platform will withhold physical areas rather than invent them.

The two vendored preprocessors in `datasets/data_processing_scripts/` are the reference
implementation. They are a working template for a third source, not pseudocode.

```bash
atria --dataset-dir /path/to/your/data ingest
```

## 2. Look at what you have before you train

```bash
atria --dataset-dir /path/to/your/data classify
```

This is the stage that exposes labelling problems. It is how the per-view quality grading in CAMUS
and the transposed ED/ES labels in EchoNet were found. Run it and read the output before spending
GPU time.

## 3. Fine-tune

```bash
atria --dataset-dir /path/to/your/data train --output-dir ./my-adapter
```

The published recipe is the default, and it is a **finding rather than a default**: on the
identical CAMUS split, lr 1e-5 plateaus at eval 0.752, lr 2e-4 over eight epochs stalls at 0.390,
and **lr 1e-4 over five epochs reaches 0.259 in 125 steps** — a better result in a third of the
time. See [training.md](training.md) and `evidence/training/`.

| | |
|---|---|
| LoRA | r 32 · α 32 · dropout 0.05 · 10 target modules incl. the SigLIP vision tower |
| Quantisation | 4-bit NF4 + double-quant, bfloat16 compute |
| Schedule | 5 epochs · effective batch 64 (4 × 16 × 1 GPU) · linear 1e-4 · 3% warmup |
| Precision | **bf16** + gradient checkpointing · `max_grad_norm` 0.3 · `max_length` 1024 |

**bfloat16 is not optional** for Gemma-family models. float16 produces `<pad>`-only output. If your
card lacks native bf16 that is a hardware constraint to plan around, not a flag to flip.

That recipe transferred across a 10× difference in corpus size without retuning — of 134
`SFTConfig` fields, exactly two differ between the two published adapters (the output paths). If
it needed retuning for your data, that would be a real finding; we would like to hear about it.

## 4. Benchmark against your own held-out split

```bash
atria --dataset-dir /path/to/your/data evaluate --adapter ./my-adapter
```

The harness writes per-frame JSONL as it goes, so an interrupt costs one frame, and it renders
every prediction for visual inspection.

**Report point-to-curve distance, not only Dice.** The model emits coordinates; region overlap
measures a downstream rasterisation the model never performed. Match the metric to the
representation — see [evaluation.md](evaluation.md).

Render the overlays and *look at them*. The 12.5% self-intersection rate and the confidently
misplaced worst case in our own results were both found by looking, not by aggregating.

## 5. Close the loop

```bash
atria --dataset-dir /path/to/your/data serve          # clinician corrects vertices
atria --dataset-dir /path/to/your/data export-corpus  # corrections → training contract
```

The saved revision records both polygons — the model's proposal and the human's correction — and
`export-corpus` re-emits them in the same three-artifact contract training consumes. Your
clinicians' corrections become your next training set.

---

## One honest caveat

`atria train` **ports** the notebook recipe. It is not a byte-exact replay of the Colab runs that
produced the published adapters, and we do not claim it reproduces those weights. It reproduces the
*method*. If you need the exact published artifacts, take them from Hugging Face.
