# Metrics Sheet & Verification Log

Every number offered for the showcase, with how it was verified. **Use only numbers from
the "verified" table.** Anything in the "do not publish" table is either unverified or
actively misleading.

---

## 1. Verified — safe to publish

### Primary benchmark — official CAMUS test split, config A2

Verified by reading `outputs/benchmark/full200/overlays/manifest.csv` (2026-08-13).

| Metric | Value |
|---|---|
| Frames evaluated | **200** |
| Patients (held out, never trained on) | **50** |
| Outputs parsed to a valid polygon | **200 / 200** |
| Median point-to-curve distance | **4.98 mm** |
| Mean point-to-curve distance | **5.97 mm** |
| Best frame | **1.20 mm** |
| Worst frame | **18.59 mm** |
| Point count — median | **30** |
| Point count 30–31 | **178 / 200 (89%)** |
| Image-quality mix | Good 94 · Medium 76 · Poor 30 |

*Metric definition:* shortest Euclidean distance from each predicted point to the expert
polyline (piecewise-linear, edge projection with clipping) — not point-to-vertex, not
rasterised-mask boundary. Implementation validated against 5 known-answer geometric cases
before use.

### Corpus and platform

| Metric | Value | Verified by |
|---|---|---|
| Training corpus | **22,048 frames / 11,024 cases** | Frame count read from disk |
| CAMUS portion | 2,000 frames (500 patients × 2 views × 2 instants) | Metadata |
| EchoNet portion | 20,048 frames (10,024 videos × 2) | Metadata |
| Official CAMUS split | 400 / 50 / 50 patients | `database_split/subgroup_*.txt` |
| Test suite | **282 tests** (254 pass / 9 skip, review tier) | `pytest` run this session |
| Classification tasks | 3 — quality (3 classes), CAMUS EF (16 bins), EchoNet EF (19 bins) | Built and counted |

### Data findings

| Finding | Value | Verified by |
|---|---|---|
| CAMUS image quality differs between 2CH and 4CH | **208 / 500 patients (41.6%)** | Parsed all 1,000 `Info_*.cfg` files |
| EchoNet ED/ES transposed | **9,922 / 10,024 (99.0%)** | Recomputed from corpus polygon areas |
| CAMUS ED/ES transposed | **0 / 1,000 (0.0%)** | Same method |
| Vendored preprocessors reproduce the corpus | **byte-identical** | Test suite |
| EchoNet frame-index derivation reproduces corpus | **20,048 / 20,048 (100.00%)** | Rebuilt every polygon |

### The adapter-loading defect

| Fact | Value |
|---|---|
| Checkpoint tensors total | **802** |
| Tensors addressing a stale module path | **324 (≈40%)** |
| Cause | `vision_tower.vision_model.encoder` (Gemma 3) → `vision_tower.encoder` (transformers 5.x) |
| Symptom | Non-fatal PEFT warning; every vision-tower `lora_B` left at zero init; model loads and produces plausible output |
| Status | Detected, repaired at load, regression-tested (`tests/test_adapter_completeness.py`) |

### Environment

Windows 11 · Python 3.14.4 · Quadro RTX 5000 16 GB (sm_75, **no native bf16**) ·
torch 2.13.0+cu126 · transformers 5.14.1 · peft 0.19.1 · bitsandbytes 0.49.2.
Quantisation for the benchmark: **4-bit NF4 + double-quant, bfloat16 compute**, with the
training-matched prompt (view + cardiac instant declared).

---

## 2. Publish only with the stated caveat

| Number | Caveat — always attach |
|---|---|
| **Dice 0.8924 / IoU 0.806** (CAMUS) | **n = 1.** A single 4-chamber frame — the best-performing view. A *smoke value*, not accuracy |
| **Dice 0.7026 / IoU 0.542** (EchoNet) | **n = 1**, same caveat |
| **Dice 0.672 ± 0.153** | n = 3, matched adapter/source. The most defensible Dice figure. Per-view: **4CH 0.887 · 2CH 0.583 / 0.546** |

**The 2CH/4CH gap is the honest headline inside these numbers** and worth stating openly:
on the only matched evidence, the adapter is markedly weaker on 2-chamber views.

---

## 3. Do NOT publish

| Claim | Why not |
|---|---|
| Any EF / volume accuracy | Correctly reframed as out of scope; the adapter emits coordinates, not volumes |
| Any LA (left atrium) result | Published adapters are **LV-only** |
| Epochs / batch / LR from the notebook | The published adapters used **different** values. Extract from the checkpoint first |
| "Production-ready" / "clinically validated" | Neither is true |
| Full-split (2,752-frame) benchmark | Never run |
| Docker verified | Authored, never executed |
| `atria train` reproduces the published adapter | It does not — it ports the notebook recipe |

---

## 4. Verification log — what was actually run

| Claim | Method | Result |
|---|---|---|
| 200-frame benchmark figures | Recomputed from `manifest.csv` | ✅ as tabled |
| 50 held-out patients | Counted unique patient IDs | ✅ 50 |
| Quality mix | Grouped manifest by quality | ✅ 94/76/30 |
| Corpus size | Counted PNGs on disk | ✅ 22,048 |
| Classification tasks | Counted class directories | ✅ 3 / 16 / 19 |
| CAMUS per-view quality divergence | Parsed 1,000 cfg files | ✅ 208/500 |
| EchoNet transposition | Recomputed polygon areas | ✅ 9,922/10,024 |
| Frame-index derivation | Rebuilt all polygons | ✅ 20,048/20,048 |
| Test suite | `pytest -m "not model"` | ✅ 254 passed, 9 skipped |
| Adapter metadata, DOIs, gating | HF Hub API | ✅ |
| Forum post content | Fetched live | ✅ incl. Google staff reply |
| DEITY definitions | Fetched the OUP article | ✅ D/E/I/T/Y confirmed |
| Visilant structure | Fetched the case study | ✅ section-by-section |
| Impact Challenge winners | Fetched Google blog | ✅ |
| GitHub README state | Fetched live | ✅ confirms understatement |

### Not verified — you must confirm

| Item | Issue |
|---|---|
| **HF Space live status** | Fetch returned **HTTP 401** |
| **Colab notebook runs** | Not executed |
| **X / LinkedIn / YouTube assets** | Not publicly indexed in my searches — supply URLs |
| **Published-adapter hyperparameters** | Must be read from the gated checkpoints locally |
| **DEITY citation year** | OUP advance 2025 vs issue 2026 — pick one and be consistent |
| **`atria ingest` full run** | Verified by argument parsing only; `[ingest]` extra not installed here |
