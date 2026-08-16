# Locked Metrics Table — publication-ready

Every number re-verified 2026-08-13 by reading the source artifact. **This table supersedes
all earlier metric statements in this package and in the repository documentation.**

Two corrections were found during this lock and are flagged ⚠ below.

---

## TIER 1 — Lead with these

Primary evaluation: **official CAMUS test split, 50 patients, 200 frames, never trained on.**
Config A2 — 4-bit NF4 + double-quant, bfloat16 compute, training-matched prompt.

| Metric | Value | Source file |
|---|---|---|
| Frames evaluated | **200** | `full200/raw_predictions.jsonl` |
| Held-out patients | **50** | `full200/overlays/manifest.csv` |
| **Output parsed to a valid polygon** | **200 / 200 (100%)** | `full200/raw_predictions.jsonl` |
| **Median point-to-curve distance** | **4.98 mm** | `full200/overlays/manifest.csv` |
| Mean point-to-curve distance | 5.97 mm | same |
| Interquartile range | **3.37 – 7.52 mm** | same |
| Best / worst frame | 1.20 / 18.59 mm | same |
| Coordinates in bounds | **200 / 200** | `full200/pointwise.csv` |
| Image-quality mix of the test set | Good 94 · Medium 76 · Poor 30 | `full200/overlays/manifest.csv` |

**Metric definition to publish alongside:** shortest Euclidean distance from each predicted
point to the expert polyline (piecewise-linear, edge projection with `t` clipped to [0,1]) —
*not* point-to-nearest-vertex and *not* rasterised-mask boundary distance. Implementation
validated against five known-answer geometric cases before use.

---

## TIER 2 — Supporting, publish freely

### Stratified point-to-curve (per-frame medians)

| Stratum | Median | n | Source |
|---|---|---|---|
| View — 4CH | **4.76 mm** | 100 | `full200/pointwise.csv` |
| View — 2CH | **5.45 mm** | 100 | same |
| Instant — ED | 4.74 mm | 100 | same |
| Instant — ES | 5.49 mm | 100 | same |
| Quality — Good | **4.61 mm** | 94 | same |
| Quality — Medium | **6.40 mm** | 76 | same |
| Quality — Poor | **6.79 mm** | 30 | same |

> **Publish the quality stratification.** Accuracy degrades monotonically with acquisition
> quality (4.61 → 6.40 → 6.79 mm). That is exactly the behaviour a clinician expects, and
> demonstrating it is stronger evidence of a sane model than the headline number alone.

### Structural validity of the generated polygons

| Metric | Value | Source |
|---|---|---|
| Point count — median | ⚠ **31** | `full200/overlays/manifest.csv` |
| Point count — range | 29 – 53 | same |
| Exactly 30 points | 80 / 200 (40%) | same |
| **30 or 31 points** | **178 / 200 (89%)** | same |
| ⚠ **Self-intersecting polygons** | **25 / 200 (12.5%)** | `full200/pointwise.csv` |

### Corpus and platform

| Metric | Value | Source |
|---|---|---|
| Training corpus | 22,048 frames / 11,024 cases | `unified_processed/` |
| CAMUS / EchoNet split of corpus | 2,000 / 20,048 frames | `metadata.csv` |
| Official CAMUS patient split | 400 / 50 / 50 | `database_split/subgroup_*.txt` |
| Vendored preprocessors reproduce corpus | byte-identical | `tests/test_ingest.py` |
| EchoNet frame-index derivation | 20,048 / 20,048 (100%) | verified this session |
| Test suite | 282 total; 254 pass / 9 skip (review tier) | `pytest` |
| Inference latency | median **86 s/frame** | `full200/raw_predictions.jsonl` |

### Data findings

| Finding | Value | Source |
|---|---|---|
| CAMUS quality differs 2CH vs 4CH | **208 / 500 patients (41.6%)** | 1,000 `Info_*.cfg` files |
| EchoNet ED/ES transposed | **9,922 / 10,024 (99.0%)** | corpus polygon areas |
| CAMUS ED/ES transposed | 0 / 1,000 (0.0%) | same |

### The adapter-loading defect

| Fact | Value | Source |
|---|---|---|
| Checkpoint tensors | **802** | `adapter_model.safetensors` |
| Addressing a stale module path | **324 (≈40%)** | `engine.repair_legacy_adapter_keys` |
| Cause | `vision_tower.vision_model.encoder` → `vision_tower.encoder` (transformers 5.x) | — |
| Status | detected, repaired, regression-tested | `tests/test_adapter_completeness.py` |

---

## TIER 3 — Use ONLY with the stated caveat

| Number | Mandatory caveat | Source |
|---|---|---|
| **Dice 0.6718 ± 0.1525**, IoU 0.5277 ± 0.1904 | **n = 3**, CAMUS adapter on CAMUS test frames. The most defensible Dice figure available | `outputs/evaluations/eval_1784997417.json` |
| Per-frame: **4CH_ED 0.8865** · 2CH_ED 0.5829 · 2CH_ES 0.5461 | Same n=3 run. **Publish the view gap** — it is the honest detail | same |
| Dice 0.5753 ± 0.1808 | **n = 3, deliberate mismatch** — CAMUS adapter on *EchoNet* frames. Not an accuracy claim | `eval_1785165023.json` |
| Dice 0.8924 / IoU 0.8056 | **n = 1**, single 4CH frame, best-performing view. A *single-frame smoke value* | `tests/test_inference_real.py` |
| Dice 0.7026 / IoU 0.5415 | **n = 1**, EchoNet adapter, same caveat | same |

---

## MUST NOT PUBLISH as headline figures

| Claim | Why |
|---|---|
| "Dice 0.89" / "89% accuracy" | n = 1, cherry-picked view. **The single largest credibility risk in the submission** |
| Any EF / volume accuracy | Out of scope — the adapter emits coordinates, not volumes |
| Any LA (left atrium) result | Published adapters are **LV-only** |
| "20,000 EchoNet training samples" | Recorded value is `folder_name`-inferred, low confidence (see `06_HYPERPARAMETERS.md §3`) |
| Full-split (2,752-frame) benchmark | Never run |
| "Docker verified" | Authored, never executed |
| "`atria train` reproduces the published adapter" | It ports the notebook recipe, which differs |
| "Clinically validated" / "production-ready" | Neither is true — research use only |

---

## ⚠ Two corrections applied during this lock

1. **Point count median is 31, not 30.** Earlier package drafts, `README.md` and
   `DELIVERY.md` state "median 30". The verified distribution is
   `{29:2, 30:80, 31:98, 32:16, 33:1, 39:2, 53:1}` → median **31**; exactly-30 is only
   **40%**. The defensible framing is **"178/200 (89%) land on 30 or 31 points"**, which is
   what the drafts already say and remains correct.

2. **Self-intersection rate (25/200, 12.5%) was not previously surfaced anywhere.** It is a
   genuine structural-validity limitation and belongs in the case study rather than being
   omitted — one polygon in eight crosses itself, which a downstream area calculation would
   silently mis-handle.

**Action:** correct "median 30" → "median 31" in `README.md`, `DELIVERY.md` and
`01_CASE_STUDY_DRAFT.md`, and add the self-intersection figure to the limitations paragraph.

---

## Recommended headline block for the case study

> Evaluated on the official CAMUS test split — **50 patients, 200 frames, never seen in
> training**:
>
> - **200/200** outputs parsed to a valid polygon; **200/200** in bounds
> - **Median point-to-curve distance 4.98 mm** (IQR 3.37–7.52)
> - **178/200 (89%)** polygons within one point of the 30-point target
> - Accuracy tracks acquisition quality as expected: **4.61 mm (Good) → 6.40 (Medium) → 6.79 (Poor)**
> - Known limitations: 2-chamber views are weaker than 4-chamber (5.45 vs 4.76 mm), and
>   **25/200 (12.5%)** of generated polygons self-intersect
