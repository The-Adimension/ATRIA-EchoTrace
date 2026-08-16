# Architecture

A single local process: a FastAPI backend serving both a JSON API and a buildless browser
workstation. Weights resolve locally first, so it runs fully offline.

```
src/atria_echotrace/
├── api/        FastAPI routes, dependency wiring, the stage registry
├── data/       corpus loading, the dataset contract, classification
│   └── ingest/reference/   the VENDORED preprocessors — do not edit
├── domain/     geometry, metrics, revisions — no I/O, no framework
├── ml/         model loading, adapter repair, inference, device policy
├── export/     revisions back into the training contract
├── render/     overlay rendering, CPU-only
└── web/        the workstation — plain JS, no build step
```

## Two tiers

**Review tier** — no weights, no GPU. Loads a corpus, renders frames, lets a clinician draw and
correct contours, exports revisions. This is what `./run.sh` starts.

**AI tier** (`pip install -e ".[ai]"`) — adds local inference and training. Requires the gated
MedGemma weights and a LoRA adapter.

## The stage registry

`api/stages.py` is the single source of truth for the pipeline's seven stages: `preprocess`,
`classify`, `train`, `evaluate`, `trace_revise`, `export_corpus`, `publish`. Each carries a stable
machine id, a command, and a computed readiness state. The UI renders whatever the registry
reports — there are **zero stage titles hardcoded in the frontend**, so the pipeline can be
reordered or extended without touching the web layer.

## Adapter loading, and why it is guarded

`ml/engine.py` repairs legacy adapter keys at load time. The published adapters were trained when
Gemma 3 nested the vision encoder as `vision_tower.vision_model.encoder`; transformers 5.x
flattened it to `vision_tower.encoder`. **324 of 802 checkpoint tensors addressed a path that no
longer existed.** PEFT emits a non-fatal warning and continues, leaving every vision-tower
`lora_B` at its zero initialisation — roughly 40% of the fine-tuning inert while the model still
loaded cleanly and produced plausible contours.

`tests/test_adapter_completeness.py` fails if the bound-tensor count ever drops again. The
transferable lesson: **count the tensors your checkpoint contains, count the ones that actually
bound to a module, and fail loudly on a mismatch.** A non-fatal warning is not a safety net.

## Device policy

`ml/runtime.py` enforces **bfloat16** for Gemma-family models — float16 makes MedGemma emit
nothing but `<pad>` tokens — and diagnoses the most common deployment failure: a CPU-only torch
wheel presenting as "GPU not detected". `atria doctor` reports both.

## The output contract

The model emits a JSON list of `[y, x]` pairs normalised to `[0, 1000]`. The parser fails loudly:
fenced-JSON extraction, schema check, in-bounds validation and a self-intersection test. A
malformed generation is a visible error, never a silently repaired polygon.

## The correction loop

A saved revision records **both** polygons — the model's proposal and the human's correction.
`atria export-corpus` re-emits them in the same three-artifact contract the original corpus uses,
so a clinician's disagreement becomes training data without a conversion step. That is the whole
reason the model emits coordinates rather than a mask.
