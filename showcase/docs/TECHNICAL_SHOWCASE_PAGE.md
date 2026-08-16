# Teaching MedGemma to Trace the Heart: QLoRA fine-tuning for 30-point polygon coordinates

**How The Adimension adapted MedGemma 1.5 4B to emit cardiac contours as editable JSON
coordinates — with a 200-frame benchmark, a 200-image evaluation gallery, and one silent
PEFT failure mode every vision-tower fine-tuner should know about.**

> Built with [Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations).
> **Research use only — not a cleared or approved medical device.**

---

## TL;DR for developers

| | |
|---|---|
| **Task** | Left-ventricular endocardial border as an ordered **30-point polygon in JSON**, not a segmentation mask |
| **Base model** | `google/medgemma-1.5-4b-it` |
| **Method** | QLoRA — **r 32, α 32, dropout 0.05**, 10 target modules **including the SigLIP vision tower**, effective batch 64, bf16, 5 epochs |
| **Hardware** | Single GPU. 4-bit NF4 at inference; runs on a consumer card |
| **Benchmark** | Official CAMUS test split — **50 patients, 200 frames, never trained on** |
| **Output validity** | **200/200 parsed**, 200/200 in bounds |
| **Geometric accuracy** | **Median point-to-curve 4.98 mm** (IQR 3.37–7.52) |
| **Segmentation accuracy** | **Dice 0.744 ± 0.135**, IoU 0.609 ± 0.156 (n = 200, vs the original expert mask) |
| **Reusable warning** | **324 of 802 adapter tensors loaded into nothing, silently** — see §6 |

---

## 1. Why coordinates instead of a mask

Tracing the endocardial border is the most repeated manual act in echocardiography. Every
ejection fraction and every volume starts with a clinician drawing a line around a moving,
noisy, partly-visible chamber.

The standard ML answer is semantic segmentation: predict a mask. For a *clinical* workflow
that answer has a specific, practical weakness — **a mask is hard to correct.** A
cardiologist who disagrees with one section of a boundary cannot nudge it; they can only
repaint it. And the disagreement is thrown away, because there is no compact structure that
records "the model said here, the human said there."

Reframing the output as **an ordered list of 30 boundary points** changes what the system
can do:

```json
[{"polygon_2d": [[184, 496], [205, 490], [225, 486], ...], "label": "left_ventricle_endocardium"}]
```

Coordinates are normalised to a `[0, 1000]` grid, so the contract is
**resolution-independent** — a 112×112 EchoNet frame and a 843×512 CAMUS frame use the same
output space, and no upload is ever resized.

That single change gives you three things a mask does not:

1. **Editable output.** Vertices are draggable. Correction is a gesture, not a repaint.
2. **A diffable record.** Model polygon and human polygon are the same data type, so
   disagreement is measurable per-point.
3. **A training signal.** The corrected polygon *is* a label in the original training
   format. Corrections flow back into the corpus with no conversion step.

This is what makes a vision-language model the right tool rather than a heavier one. A
segmentation network has no mechanism to emit JSON. MedGemma does — and being open-weight,
it can be adapted where the data already lives.

---

## 2. End-to-end pipeline

```
raw CAMUS NIfTI / EchoNet AVI
        │  vendored preprocessors (byte-identical to the originals)
        ▼
frames/*.png + tracings.json + metadata.csv        ← the three-artifact contract
        │
        ├──▶ classification sets (quality, EF bins)   — ImageFolder or mapping.csv
        │
        ├──▶ QLoRA fine-tune (MedGemma 1.5 4B)
        │            │
        │            ▼
        │      LoRA adapter ──▶ evaluate ──▶ 200-frame benchmark + overlay gallery
        │                              │
        ▼                              ▼
   trace & revise  ◀──────────────  clinician corrects vertices
        │
        ▼
   export-corpus ──▶ back to the same three-artifact contract ──▶ retrain
```

Every stage is independently enterable from a CLI (`atria ingest`, `classify`, `train`,
`evaluate`, `export-corpus`, `publish-adapter`) or the browser workstation. The loop at the
bottom is the point: **the clinician's correction becomes the next generation's ground
truth**, in the same format training already consumes.

The preprocessing scripts are **vendored byte-identically** and re-run to reproduce the
training corpus exactly — 22,048 frames / 11,024 cases. Provenance is executable, not
asserted.

---

## 3. The QLoRA configuration — recovered from the shipped checkpoints

These values are read directly from `adapter_config.json` and `training_args.bin`
(a `trl.SFTConfig`, 134 fields) in the released adapters — not from a notebook default.

**Both published adapters were trained with an identical recipe.** Of 134 `SFTConfig`
fields, exactly **two** differ: `output_dir` and `logging_dir`. One configuration, two
datasets an order of magnitude apart in size.

### LoRA

| Field | Value |
|---|---|
| `r` | **32** |
| `lora_alpha` | **32** |
| `lora_dropout` | 0.05 |
| `bias` | `none` |
| `use_rslora` / `use_dora` | false / false |
| `task_type` | `CAUSAL_LM` |
| `modules_to_save` | `lm_head`, `embed_tokens` |

### `target_modules` — 10 modules, and this is the interesting part

```
q_proj  k_proj  v_proj  o_proj      ← attention projections
gate_proj  up_proj  down_proj        ← language-tower MLP
fc1  fc2  out_proj                   ← SigLIP VISION-tower MLP + attention output
```

`fc1`, `fc2` and `out_proj` are vision-encoder modules. **Adapting the visual tower — not
just the language head — is what lets the model learn what an endocardial border looks
like** rather than only how to format an answer. It is also precisely what the failure in
§6 silently switched off.

### Training arguments

| Field | Value | | Field | Value |
|---|---|---|---|---|
| `num_train_epochs` | 5 | | `bf16` | **True** |
| `per_device_train_batch_size` | 4 | | `fp16` | False |
| `gradient_accumulation_steps` | 16 | | `gradient_checkpointing` | True (`use_reentrant: false`) |
| **effective batch** | **64** | | `max_length` | **1024** |
| `learning_rate` | **1e-4** | | `packing` | False |
| `lr_scheduler_type` | linear | | `logging_steps` | 5 |
| `warmup_steps` | **0.03** → 3 % ratio | | `eval_steps` | 10 |
| `optim` | `adamw_torch_fused` | | `save_strategy` | epoch (limit 3) |
| `max_grad_norm` | **0.3** | | `seed` | 42 |

> **`warmup_steps = 0.03` is deliberate, not a typo.** Hugging Face treats a value in
> `[0, 1)` in that field as a *ratio*, so it means 3 % warmup. Worth knowing before you
> "fix" it in your own config.

### What the loss curves recorded

| | CAMUS | EchoNet |
|---|---|---|
| Logged train points (`logging_steps` 5) | 25 → **125 optimizer steps** | 259 → **≈1,295 steps** |
| Samples/epoch (derived, batch 64) | **1,600** — exactly the CAMUS train split | ≈16,576 |
| Train loss | 1.494 → **0.253** | 1.644 → **0.196** |
| Eval loss | 0.475 → **0.259** | 1.449 → **0.205** |

Both runs finished **at their minimum eval loss**, with no early stopping and no
best-checkpoint restoration — the released weights are the final epoch. Two datasets an
order of magnitude apart converged under one configuration without retuning.

**bfloat16 is mandatory for Gemma-family models.** On a pre-Ampere card, substituting
float16 makes MedGemma emit nothing but `<pad>` tokens. Measured, not theorised — keep bf16
even where it is emulated.

---

## 4. Evaluation — and the gallery is the primary evidence

**Test set:** the official CAMUS `subgroup_testing` split — **50 patients, 200 frames**,
selected by the dataset authors' own split file, never seen in training. Quality mix:
**94 Good · 76 Medium · 30 Poor**. Inference config: 4-bit NF4 + double-quant, bf16
compute, training-matched prompt (view and cardiac instant declared).

### 4.1 The 200-image evaluation gallery

Every one of the 200 predictions was rendered at native resolution and reviewed —
**reference in green, prediction in red, predicted vertices marked** — with a caption
carrying the frame identity, point count and distance.

📁 **`outputs/benchmark/full200/overlays/index.html`** — a filterable gallery (best / worst /
2CH / 4CH / ED / ES / Good / Medium / Poor), sorted best-first, with each PNG named by its
own score so the directory sorts by quality.

| | Frame | Score | File |
|---|---|---|---|
| **Best** | `patient0219_4CH_ED` (Good) | **1.20 mm** | `001.203mm__patient0219_4CH_ED.png` |
| **Typical** | `patient0220_4CH_ED` (Good) | **4.98 mm** | `004.977mm__patient0220_4CH_ED.png` |
| **Worst** | `patient0266_2CH_ED` (Poor) | **18.59 mm** | `018.586mm__patient0266_2CH_ED.png` |

**Open the worst one.** It is the most informative image in the set: the prediction is a
**plausible, well-formed ventricle traced in the wrong place** — correct point count, no
self-intersection, displaced by roughly a chamber width. The model did not produce noise; it
confidently traced the wrong structure. No scalar metric expresses that, and it is exactly
the failure mode a reviewing clinician must be able to catch. **This is the argument for
human-in-the-loop, made visually.**

### 4.2 Primary metric — point-to-curve distance

The primary question for a coordinate-generating model is *"do the predicted points land on
the expert tracing?"* That is point-to-curve distance: the shortest Euclidean distance from
each predicted point to the expert **polyline** (piecewise-linear, edge projection with the
parameter clipped to `[0,1]`) — *not* distance to the nearest vertex, and *not* rasterised
mask-boundary distance.

| Metric | Value |
|---|---|
| Outputs parsed to a valid polygon | **200 / 200** |
| Coordinates in bounds | **200 / 200** |
| **Median point-to-curve distance** | **4.98 mm** |
| Mean | 5.97 mm |
| IQR | **3.37 – 7.52 mm** |
| Best / worst frame | 1.20 / 18.59 mm |

**Stratified (per-frame medians):**

| Stratum | Median | n | | Stratum | Median | n |
|---|---|---|---|---|---|---|
| 4CH | **4.76 mm** | 100 | | Good | **4.61 mm** | 94 |
| 2CH | **5.45 mm** | 100 | | Medium | **6.40 mm** | 76 |
| ED | 4.74 mm | 100 | | Poor | **6.79 mm** | 30 |
| ES | 5.49 mm | 100 | | | | |

Accuracy degrades **monotonically with acquisition quality** — 4.61 → 6.40 → 6.79 mm. That
is the behaviour a clinician expects from a system that is genuinely reading the image, and
it is stronger evidence of a sane model than any single headline number.

### 4.3 Standard segmentation metrics (n = 200)

For comparability with the segmentation literature, the predicted polygons were also
rasterised and scored with the conventional battery. **Two reference standards are reported
separately and never merged:**

**(a) vs the original expert mask** from `_gt.nii.gz` — the clinical ground truth:

| Metric | Mean ± SD | Median | Range |
|---|---|---|---|
| **Dice** | **0.744 ± 0.135** | 0.777 | 0.263 – 0.940 |
| **IoU** | **0.609 ± 0.156** | 0.635 | 0.151 – 0.886 |
| MAD | 6.22 ± 3.08 mm | 5.42 | 1.50 – 16.80 |
| HD95 | 13.11 ± 5.88 mm | 11.71 | 4.00 – 34.03 |
| NSD @ 2 mm | 0.227 ± 0.157 | 0.202 | 0.000 – 0.670 |

**(b) vs the 30-point training polygon** — what the model was actually trained to reproduce:

| Metric | Mean ± SD | Median |
|---|---|---|
| Dice | 0.743 ± 0.137 | 0.773 |
| MAD | 6.24 ± 3.10 mm | 5.45 |
| HD95 | 13.11 ± 5.91 mm | 11.85 |

Dice stratified by view: **4CH 0.765 ± 0.113 · 2CH 0.723 ± 0.152**. By instant:
**ED 0.779 · ES 0.710**. By quality: Good 0.782 · Medium 0.701 · Poor 0.735 — note this is
**not** monotonic, unlike the point-to-curve figures; the Poor subgroup is small (n = 30)
and its Dice sits above Medium. We report it as measured rather than smoothing it.

### 4.4 The representation is not the bottleneck

Before attributing error to the model, we measured the **discretisation ceiling** — how much
accuracy is lost purely by representing an expert mask as 30 index-uniform points. This is
reference-versus-reference and costs no inference:

| Ceiling (200 frames) | Value |
|---|---|
| Dice | **0.9905** |
| MAD | **0.214 mm** |
| HD95 | 0.542 mm |

The 30-point representation is **near-lossless**. A Dice of 0.744 is therefore a property of
the model, not of the output format — and there is substantial headroom before the
representation constrains anything.

### 4.5 Why we do not lead with Dice

The single-frame figures that appear in our earlier materials — **Dice 0.8924** (CAMUS) and
**0.7026** (EchoNet) — are **n = 1 smoke values**, and the CAMUS one is a 4-chamber frame,
the best-performing view. They are printed observations from a regression test, not
accuracy. **The defensible figures are the n = 200 numbers in §4.3.**

More fundamentally: region overlap measures a *downstream construction*. The adapter emits
coordinates; converting them to a mask and computing Dice inserts a rasterisation step the
model never performed. We report both because the field expects Dice — but point-to-curve
is the metric that matches the model's actual output.

---

## 5. Reproducing the evaluation

```bash
# the harness that produced everything above
outputs/benchmark/run200.py            # inference over the 200-frame split (resumable)
outputs/benchmark/rescore_pointwise.py # primary: output validity, point count, point-to-curve
outputs/benchmark/score200.py          # secondary: Dice / IoU / MAD / HD95 / NSD
outputs/benchmark/make_overlays.py     # the 200-image gallery (CPU only, no inference)

# how the test split, ceiling and gates were built
outputs/benchmark/provenance/          # 11 scripts, kept for traceability
```

Per-frame data ships alongside: `pointwise.csv` (point-to-curve), `per_frame.csv`
(segmentation battery), `overlays/manifest.csv` (per-frame class, point count, distance),
`raw_predictions.jsonl` (every raw model response, so any metric can be recomputed without
re-running the GPU).

---

## 6. A silent failure mode worth your attention

**This is the part most likely to save another HAI-DEF developer a week.**

The published adapters were trained when Gemma 3 nested the vision encoder as
`vision_tower.vision_model.encoder`. **Transformers 5.x flattened that path to
`vision_tower.encoder`.**

Consequence: **324 of the checkpoint's 802 tensors addressed a module path that no longer
existed.** PEFT emits a *non-fatal* warning and continues. Every vision-tower `lora_B`
therefore stayed at its **zero initialisation** — and a LoRA update of `B·A` with `B = 0`
contributes exactly nothing.

**The model loaded. It ran. It produced plausible contours.** Roughly **40 % of the
fine-tuning was inert**, and the only signal was a warning in a log most pipelines filter
out.

If you fine-tune a vision tower and later upgrade `transformers`, check for this:

```python
status = engine.status()["adapter_load"]
assert status["fully_loaded"], status          # remapped / vision_lora_b_active / vision_lora_b
```

The fix is a key remap at load time (`engine.repair_legacy_adapter_keys`), and it is guarded
by a regression test that fails if it ever silently returns. **Do not trust "the adapter
loaded."** Verify that every tensor found a home — and never filter warnings out of a
fine-tuning log.

---

## 7. What a developer can reuse tomorrow

| Reusable | Where |
|---|---|
| **Normalised polygon contract** — `[y, x]` in `[0, 1000]`, resolution-independent | `domain/geometry.py` |
| **Structured-output parsing that fails loudly** — fenced JSON, schema check, in-bounds and self-intersection validation | `domain/geometry.py`, `rescore_pointwise.py` |
| **Point-to-curve implementation** — validated against 5 known-answer geometric cases before use on real data | `rescore_pointwise.py` |
| **Adapter-completeness check** — catches the §6 failure | `ml/engine.py`, `tests/test_adapter_completeness.py` |
| **Resumable benchmark harness** — flush-per-frame JSONL; an interrupt costs one frame | `outputs/benchmark/run200.py` |
| **Overlay gallery generator** — CPU-only visual QA from persisted predictions, no GPU replay | `make_overlays.py` |
| **Correction → corpus loop** — revisions re-emitted in the training contract | `export/package.py::export_corpus` |
| **Device policy** — bf16 enforcement + CPU-only-wheel diagnosis (`torch.version.cuda is None` is the real check) | `ml/runtime.py` |

Three transferable lessons, independent of cardiology:

1. **Match the metric to the output.** We first scored a coordinate generator with Dice and
   EF. Both measured downstream constructions. Re-scoring with point-to-curve changed the
   interpretation, not the model.
2. **Render your predictions.** The 12.5 % self-intersection rate and the
   confidently-misplaced worst case were both found by looking, not by aggregating.
3. **A partially-loaded adapter is indistinguishable from a working one** at the output
   level. Assert completeness.

---

## 8. Honest limitations

- **Scope:** left ventricle only. The published adapters were tuned on LV; LA is out of
  distribution.
- **Structural validity:** **25 / 200 (12.5 %) of generated polygons self-intersect** — a
  downstream area calculation would silently mishandle these.
- **Point count:** median **31**, range 29–53. Only 40 % emit exactly 30; **89 % (178/200)
  land on 30 or 31**. The count is not hard-constrained by decoding.
- **View gap:** 2-chamber is consistently weaker than 4-chamber (5.45 vs 4.76 mm;
  Dice 0.723 vs 0.765).
- **Failure mode:** the worst cases are *plausible shapes in the wrong location*, not
  obvious garbage — which is precisely why review is mandatory.
- **Latency:** median **86 s/frame** on a pre-Ampere Quadro RTX 5000 (sm_75, no native
  bf16). This is not a real-time system on that hardware.
- **Scale:** 200 frames on CAMUS. The full 2,752-frame combined split has not been run, and
  no EchoNet benchmark of equivalent rigour exists yet.
- **EchoNet ED/ES labels are transposed in 99 % of cases** — a property of the source data.
  We **flag and do not correct** it, because relabelling at inference would push every
  request outside the adapter's training distribution.
- **`atria train` does not reproduce the published adapters** — it faithfully ports the
  notebook recipe, which differs from the shipped configuration in §3.

---

## 9. Built on the DEITY Principles

ATRIA EchoTrace is the reference implementation of the **DEITY Principles Framework**
(Anwer, *Eur Heart J Imaging Methods Pract*, [10.1093/ehjimp/qyaf038](https://doi.org/10.1093/ehjimp/qyaf038)).
Each principle forced a decision you can check in the code:

- **Data** — original preprocessors vendored byte-identically; the corpus is *reproduced*, not described.
- **Ethics** — known data defects surfaced, never silently repaired. cm² withheld when pixel spacing is unknown rather than computed from a placeholder.
- **Informatics** — the output is structure a human can read, diff and edit.
- **Technology** — 4-bit, single GPU, offline-capable, local-first weights.
- **You** — the clinician's correction is the product, and it re-enters training by design.

---

## 10. Intended use

**ATRIA EchoTrace is research software. It is not a cleared or approved medical device and
must not be used for diagnosis or direct patient care.** Every contour is a *proposal*
requiring independent review by a qualified echocardiographer. Human-in-the-loop review is
architectural, not advisory — the interface exists to be corrected.

---

## Assets

| | |
|---|---|
| **GitHub** | <https://github.com/The-Adimension/ATRIA-EchoTrace> |
| **Colab (no gated weights needed)** | [Open notebook](https://colab.research.google.com/drive/1qofahQ8LztTrB_Us9j1Iyz2aYeS2_2rH?usp=sharing) |
| **CAMUS adapter** | [The-Adimension/EchoTrace-MedGemma-CAMUS](https://huggingface.co/The-Adimension/EchoTrace-MedGemma-CAMUS) · DOI [10.57967/hf/9541](https://doi.org/10.57967/hf/9541) |
| **EchoNet adapter** | [The-Adimension/EchoTrace-MedGemma-EchoNet](https://huggingface.co/The-Adimension/EchoTrace-MedGemma-EchoNet) · DOI [10.57967/hf/9540](https://doi.org/10.57967/hf/9540) |
| **Collection** | [ATRIA-EchoTrace](https://huggingface.co/collections/The-Adimension/atria-echotrace-6a5884b6c17ec3aab49d0875) |
| **Demo Space** | [The-Adimension-ATRIA-EchoTrace](https://huggingface.co/spaces/ShehabAnwer/The-Adimension-ATRIA-EchoTrace) |
| **Evaluation gallery** | `outputs/benchmark/full200/overlays/index.html` (200 overlays) |
| **HAI-DEF forum thread** | [Fine-tuning MedGemma 1.5 for polygon-based contouring](https://discuss.ai.google.dev/t/atria-echotrace-fine-tuning-medgemma-1-5-for-polygon-based-heart-structure-contouring/172907) |
| **DEITY paper** | [10.1093/ehjimp/qyaf038](https://doi.org/10.1093/ehjimp/qyaf038) |
| **Base model** | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) |

Adapters are **gated** to keep research artifacts traceable and ensure users acknowledge
the research-only intended use; access requests are reviewed. Both are Apache-2.0 with DOIs.

**Datasets:** CAMUS (Leclerc et al., *IEEE TMI* 2019) · EchoNet-Dynamic (Ouyang et al.,
*Nature* 2020), used under their respective terms.

---

### Collaboration

The Adimension welcomes echocardiography laboratories willing to contribute corrected
tracings, and HAI-DEF developers working on structured-output adaptation of MedGemma. The
platform, the benchmark harness and the provenance scripts are open.

*Shehab Anwer, MD — Founder, The Adimension*
