# CODE INVENTORY — ATRIA EchoTrace

Read-only inventory. No files were moved, edited, or deleted to produce this document.
Line counts are `Get-Content | Measure-Object -Line` on 2026-08-04. Notebook-phase
references are copied from each module's own docstring (this codebase cites
`notebook_as_py.txt` line ranges directly — see the `### <file>` docstring dump this
document was built from), not inferred.

Status legend: **CORE** = required for the shipped serving/CLI path · **SUPPORT** =
shipped, but only reachable from a non-default stage (ingest/train/publish) or from
benchmark tooling, not from `atria serve` · **TEST** · **TOOLING** = benchmark/eval
scripts outside the package · **SUPERSEDED** = dead, excluded by `.gitignore`.

---

## 1. `src/atria_echotrace/` — the shipped package (~8,000 lines, 39 files)

### 1.1 Top level

| Path | Lines | Role | Notebook phase | Status | Dependency risk |
|---|---|---|---|---|---|
| `__init__.py` | 5 | Package marker, points to RESEARCH.md §1 for the phase map | — | CORE | Package import root — removing breaks everything |
| `__main__.py` | 7 | `python -m atria_echotrace` alias for the `atria` console script | — | CORE | Only `run.cmd`/`run.sh` use this form before the script is on PATH |
| `cli.py` | 512 | 7 subcommands: `serve · doctor · ingest · train · evaluate · export-corpus · publish-adapter` | all stages (entry point) | CORE | Sole entry point for every non-HTTP stage; `atria` console script (`pyproject.toml`) resolves here |
| `config.py` | 259 | `Settings`, `PROJECT_ROOT`, weight resolution (project → HF cache → Hub), **`display_path()`** | replaces notebook `@param`/Colab `userdata.get('HF_TOKEN')` (L112-250, L173-186) | CORE | **`display_path()` is the sole no-absolute-path guarantee** — every client-facing surface routes through it (memory: [[atria-no-absolute-paths-to-clients]]) |
| `logging_setup.py` | 44 | Structured logging; `_reference_logging()` context manager that detaches root handlers so vendored scripts' own `preprocessing_log.txt` isn't silently emptied by `logging.basicConfig` no-op | replaces bare `print` calls throughout the notebook | CORE | Removing `_reference_logging()` silently breaks the vendored preprocessors' own log output (HANDOVER §6 trap) |

### 1.2 `domain/` — scientific core, no I/O

| Path | Lines | Role | Notebook phase | Status | Dependency risk |
|---|---|---|---|---|---|
| `structures.py` | 50 | `STRUCTURE_INFO`, `NORM_SCALE=1000`, view/instant name tables | config cell (L198-219) + HITL cell copy (L1194-1202) | CORE | Imported by geometry, prompts, api — a single source of truth for LV/LA labels and the `[0,1000]` scale |
| `geometry.py` | 266 | Shoelace area, `polygon_to_mask`, Dice/IoU, `parse_polygon`, `sanitize_polygon`, resampling | notebook eval utilities (L919-990) | CORE | **`sanitize_polygon`/`parse_polygon` are the model-output hardening layer** — every prediction (serving, evaluate, and this session's benchmark scripts) passes through here |
| `metrics.py` | 166 | Chamber areas px²/cm² with explicit calibration state, FAC %, perimeter | reproduces the author's deployed Space's `calculate_anatomical_metrics` | CORE | `api/clinical.py` depends on this for every FAC computation |

### 1.3 `data/` — dataset repository, frames, ingest wrappers

| Path | Lines | Role | Notebook phase | Status | Dependency risk |
|---|---|---|---|---|---|
| `dataset.py` | 522 | `DatasetRepository`: 3-artifact contract (`frames/`,`tracings.json`,`metadata.csv`), case listing, validation, keyed by `(case_id, view)` | `load_echocardiographic_frame_data` (L254-280) + split logic | CORE | Keying trap documented in HANDOVER §6 — `case_id` alone silently loses cases present in both 2CH/4CH |
| `frames.py` | 111 | `load_frame`, upload validation, path-traversal guards | notebook frame loader (L284-298) | CORE | Upload hardening surface — reused by this session's `outputs/benchmark/make_overlays.py` for CPU-only rendering |
| `ingest/__init__.py` | 89 | `IngestError`, `IngestResult`, `summarise_output` | — | SUPPORT | Only reachable via `atria ingest` (`[ingest]` extra) |
| `ingest/run.py` | 186 | Thin wrappers that validate source layout, call the vendored scripts, summarise output | — | SUPPORT | Gate between `atria ingest` and the immutable reference scripts below |
| `ingest/reference/__init__.py` | 8 | Marks the vendoring boundary | — | CORE (immutable) | — |
| `ingest/reference/preprocess_camus.py` | 342 | **The user's real CAMUS NIfTI→PNG preprocessor, vendored byte-identically (sha256-verified)** | is the preprocessing step | CORE (immutable) | **NEVER EDIT** ([[vendored-reference-scripts-are-immutable]]). `atria ingest camus` reproduces the training corpus byte-for-byte only because this file is untouched |
| `ingest/reference/preprocess_echonet.py` | 609 | **The user's real EchoNet AVI preprocessor, vendored byte-identically**; contains the `esf`/`edf` column-inversion at line 254 that produces the documented 99% ED/ES transposition | is the preprocessing step | CORE (immutable) | **NEVER EDIT** — the transposition is deliberately surfaced, not fixed, here ([[echonet-ed-es-transposed-deliberately]]) |

### 1.4 `ml/` — inference, training, evaluation

| Path | Lines | Role | Notebook phase | Status | Dependency risk |
|---|---|---|---|---|---|
| `engine.py` | 573 | `InferenceEngine` (load/predict), **`repair_legacy_adapter_keys()`**, `status()["adapter_load"]` | notebook's cached `load_model` (L1257-1282) + inference callback | CORE | **The vision-tower LoRA key-remap fix lives here** (HANDOVER §6: 324/802 tensors addressed a stale module path; this repairs it and reports `fully_loaded`). Regression-tested by `tests/test_adapter_completeness.py` |
| `prompts.py` | 126 | The **two** notebook prompt templates (byte-equal asserted), `build_messages`/`build_prompt` | RESEARCH.md §0.4 — the two variants differ in one line and the difference matters | CORE | Config A2 (this session's primary benchmark config) depends on the `training` variant matching exactly |
| `runtime.py` | 279 | Device/dtype policy, `diagnose_cpu_fallback()` (forced / no-GPU / CPU-only-wheel / driver-mismatch) | notebook hard-coded bf16+NF4 (L139-150, L641-647, L883-888) | CORE | **This is the GPU-misdiagnosis fix from earlier in this session** ([[torch-cpu-wheel-trap]]) — collapsing the 4 states back to "no CUDA device" was the original defect |
| `datasets.py` | 159 | Training-sample construction for `SFTTrainer` | `prepare_echocardiographic_frame_samples` (L449-500) | SUPPORT | Only reachable via `atria train` |
| `train.py` | 265 | QLoRA fine-tuning, faithful port of hyperparameters | fine-tuning cells (L618-686) | SUPPORT | Never executed on this machine (DELIVERY §4) — asserted against notebook + published `adapter_config.json` only |
| `evaluate.py` | 377 | Split evaluation: parse rate, Dice, IoU, best/worst figure ranking | evaluation cells (L992-1169) | SUPPORT | Powers `atria evaluate`; superseded **in framing** for LV contour accuracy by this session's point-to-curve re-score, but still the only code path that produces Dice/IoU/EF-adjacent numbers for the shipped CLI |
| `publish.py` | 154 | LoRA adapter → HF Hub publish, gated by confirmation | HF adapter-transfer cell (L1580-1754) | SUPPORT | Gate exercised, no upload ever performed (DELIVERY §4) |
| `reference/__init__.py` | 1 | Vendoring-boundary marker | — | SUPPORT | — |
| `reference/camus_ef.py` | 199 | **CAMUS group's own Simpson's-biplane EF code, vendored verbatim from `script_camus_ef.ipynb`** | not a notebook-app phase — added this session for benchmark clinical-EF scoring | SUPPORT | **Not imported anywhere in `src/atria_echotrace/` itself** — only `outputs/benchmark/score200.py` and `rescore_pointwise.py` import it. Lives under `src/` (correctly, as a vendored artifact) but is not wired into `atria serve`/`atria evaluate`. Needs the numpy-2 `np.cross` monkeypatch applied by every caller — see §4 below |

### 1.5 `render/`, `export/`

| Path | Lines | Role | Notebook phase | Status | Dependency risk |
|---|---|---|---|---|---|
| `render/overlays.py` | 89 | Polygon-overlay drawing with PIL: `draw_polygons_on_image`, `to_pixel_points` | notebook overlay helpers (L316-325 + HITL `draw_polygon_on_image`) | CORE | **Reused unmodified by this session's `outputs/benchmark/make_overlays.py`** — the only bug found there (vertex-order swap) was in the *caller*, not this function |
| `render/figures.py` | 180 | 2/3/4-panel publication figures | 3 notebook matplotlib figures | CORE | `atria evaluate --figures` and revision export depend on this |
| `export/package.py` | 480 | Revision bundles (PNG/JSON/CSV/ZIP), **`export_corpus()`** | `save_polygon_backend` (L1318-1375) | CORE | Contains the **N1 fix** (`created_unix` monotonic sort key) closed in GAP_ANALYSIS Round 3 — the ground-truth evolution loop's correctness lives here |

### 1.6 `api/` — FastAPI surface

| Path | Lines | Role | Notebook phase | Status | Dependency risk |
|---|---|---|---|---|---|
| `app.py` | 177 | App factory; SPA mount with `_RevalidatingStatics` (forces `Cache-Control: no-cache`) | — | CORE | **Removing `_RevalidatingStatics` silently un-ships every future UI fix** ([[spa-assets-need-cache-control]]) — browsers apply heuristic freshness without it |
| `clinical.py` | 118 | Chamber area / perimeter / FAC endpoints | replaces old `/api/clinical/calculate-fac` | CORE | — |
| `dataset.py` | 166 | Case/frame browsing endpoints | dataset exploration cells (L273-280) | CORE | — |
| `deps.py` | 73 | Process-singleton dataset repository + inference engine, `ingest_available()` | — | CORE | GPU weights live in the engine singleton here; re-instantiating per-request would reload multi-GB weights on every call |
| `evaluation.py` | 109 | `POST /api/evaluation/runs` background-thread lifecycle | evaluation cells (L992-1169) | CORE | Was the N3 gap (untested thread lifecycle) — now covered by `tests/test_evaluation_lifecycle.py` |
| `inference.py` | 157 | Model lifecycle + contour-prediction endpoints | `load_model` + `process_image_backend` callbacks | CORE | The live HTTP path this session's benchmark scripts deliberately bypassed (they call `engine.py` directly, not this HTTP layer, to avoid uvicorn overhead at 200-frame scale) |
| `meta.py` | 245 | Capabilities, disclaimers, DEITY content, Weights panel data | notebook markdown cells (L31-70) | CORE | `GET /api/meta/capabilities` is what the UI's Stages panel and tier-gating read |
| `revisions.py` | 257 | Clinician revision persistence + export endpoints | `save_polygon_backend` (L1318-1375) | CORE | Path-traversal guard on `GET /revisions/{id}/files/{name}` lives here (GAP_ANALYSIS "suspicions cleared") |

### 1.7 `web/` — buildless SPA (13 files, ~3.6k lines JS/CSS per HANDOVER §1)

| Path | Role | Status | Notes |
|---|---|---|---|
| `index.html` | SPA shell | CORE | — |
| `css/atria.css` | Clinical dark theme | CORE | — |
| `js/app.js` | State manager, Stages panel, upload drop slots, revision list | CORE | Largest JS file — holds the Phase B upload entry point and the 5-stage launcher added in GAP_ANALYSIS Step 1/3/4 |
| `js/api.js` | Typed API client | CORE | GAP_ANALYSIS G6 confirmed every endpoint here has a live caller |
| `js/canvas-editor.js` | Interactive vertex editor | CORE | Carries the N5 accessibility fix (`role="application"` + `aria-label`) |
| `vendor/{preact,hooks,htm}.module.js` | Vendored, zero-CDN framework | CORE | Frontend third-party network requests measured at **zero** |
| `fonts/ibm-plex-*.woff2` (5 files) | Vendored web fonts | CORE | Same zero-CDN guarantee |

---

## 2. Launchers and packaging

| Path | Role | Status | Notes |
|---|---|---|---|
| `run.cmd` | Windows click-and-run: creates `.venv` via `uv` (or stdlib `venv` fallback), installs, picks the torch backend (`nvidia-smi` probe), serves | CORE | Verified: "created `.venv`, installed, served HTTP 200" (DELIVERY §2, twice — in-place and on a clean tree) |
| `run.sh` | macOS/Linux equivalent | CORE | Same verification path, not independently re-run this session |
| `docker/Dockerfile` | Container build, `ARG TORCH_INDEX` override | SUPPORT | **Authored, never executed** — Docker isn't installed on this machine (DELIVERY §4) |
| `docker/docker-compose.yml` | Compose stack, `ai-gpu` profile reserving `driver: nvidia` | SUPPORT | Same — unexecuted but inspected-correct (GAP_ANALYSIS "suspicions cleared": Docker already knew about the CPU-wheel trap before the launchers did |
| `pyproject.toml` | `setuptools`, `where=["src"]`, `atria` console script, `[ai]`/`[ingest]`/`[dev]` extras | CORE | **`where=["src"]` is the proof that the top-level `atria_echotrace/` directory is not part of the installable package** — see DIRECTORY_MAP.md §3 |

---

## 3. `tests/` (14 files)

| Path | Exercises |
|---|---|
| `conftest.py` | Shared fixtures |
| `test_fidelity.py` | Prompt templates + LoRA/SFT hyperparameters byte/value-equal to the notebook and published `adapter_config.json` |
| `test_metrics.py` | Shoelace, FAC %, calibration-state areas |
| `test_geometry.py` | `parse_polygon`, `sanitize_polygon`, Dice/IoU, resampling |
| `test_dataset.py` | `DatasetRepository`, `(case_id, view)` keying |
| `test_cli.py` | All 7 subcommands |
| `test_ingest.py` | Vendored preprocessors reproduce the shipped corpus byte-identically (needs `[ingest]`) |
| `test_inference_real.py` | Real-weights (`model`/`adapter` marked) single-frame Dice smoke tests |
| `test_engine.py` | `InferenceEngine` load/predict, non-weight paths |
| `test_api.py` | FastAPI endpoint contracts |
| `test_device_diagnosis.py` | `diagnose_cpu_fallback()` state classification |
| `test_corpus.py` | `export_corpus()`, including the **N1 regression test** (`test_later_revision_supersedes_an_earlier_one`) |
| `test_evaluation_lifecycle.py` | **N3 fix** — 503/409/404/400 paths + a real `model`-marked 202→poll→completed run |
| `test_adapter_completeness.py` | **The vision-tower LoRA repair regression test** — asserts `adapter_load.fully_loaded` and that no vision-tower `lora_B` tensor is all-zero; proven to fail without `repair_legacy_adapter_keys()` |

282 tests total (272 without model weights), per HANDOVER §1 / GAP_ANALYSIS Round 3 outcome — not independently re-run to produce this inventory (re-running was out of this phase's scope).

---

## 4. Benchmark / scoring utilities — `outputs/benchmark/*.py` (in-project, 4 files)

These are **not part of the shipped package** (not under `src/`, not imported by `cli.py` or `api/`) but encode metric logic that does not exist anywhere else and was the deliverable of this session's re-evaluation work.

| Path | Role | Status | Notes |
|---|---|---|---|
| `run200.py` | The 200-frame CAMUS-test-split inference runner, config A2 (4-bit NF4, training-matched prompt). Resumable JSONL, flush-per-frame, Windows sleep-suppression | TOOLING | Produced `outputs/benchmark/full200/raw_predictions.jsonl` — the source-of-truth prediction file every other script here reads |
| `score200.py` | Segmentation/EF-style scorer (Dice/IoU/HD95/EF vs both reference standards) | TOOLING (superseded framing) | **Correct code, superseded task-framing** — the user's correction reclassified the adapter's task as coordinate generation, not segmentation/EF. Kept as provenance of the original (now-secondary) analysis; not deleted since nothing was factually wrong, only mis-weighted |
| `rescore_pointwise.py` | **The corrected re-score**: output-JSON validity, point count, point-to-polyline (piecewise-linear) fidelity in mm, vs both the 30-point training polygon and the dense `_gt.nii.gz` contour | TOOLING (current primary) | Implements the user's exact corrected metric specification; validated against 5 known-answer geometric test cases before trusting on real data |
| `make_overlays.py` | Visual QA gallery: green reference / red prediction / yellow predicted-vertex dots on native-resolution frames + `index.html` + `manifest.csv` | TOOLING (current) | CPU-only, reuses `render/overlays.py` unmodified; the vertex-order bug found and fixed during this session's own work was in this file, not in the reused production function |

`outputs/benchmark/` also holds run data (not code): `camus_test/` (staged official 200-frame test split), `pilot20/`, `pilot20_fixed/` (A/B/C three-way pilot), `full200/` (the 200-frame run + `overlays/`), `logs/`.

---

## 5. Repair / hardening / remap logic — cross-reference

Called out separately since these are exactly the pieces "nothing valuable can be lost":

| Logic | Where | What breaks without it |
|---|---|---|
| Vision-tower LoRA key remap | `ml/engine.py::repair_legacy_adapter_keys` | 40% of every published adapter's contribution (vision-tower `lora_B`) silently stays at zero init — measured, not theoretical |
| No-absolute-path guarantee | `config.py::display_path()` | Host username/path disclosure across all 41 client-facing surfaces scanned in DELIVERY §2 |
| CPU-only-wheel misdiagnosis fix | `ml/runtime.py::diagnose_cpu_fallback()` | Collapses back to "no CUDA device detected" — the exact defect this session started by fixing |
| SPA cache invalidation | `api/app.py::_RevalidatingStatics` | UI upgrades silently invisible behind browser heuristic caching |
| Corpus export precedence (N1) | `export/package.py` (`created_unix` sort key) | Two same-second revisions can export the **older** polygon as ground truth |
| Path-traversal guard | `api/revisions.py` (`GET /revisions/{id}/files/{name}`) | Arbitrary file read via `..`/`/`/`\` in the `name` path parameter |
| Upload/traversal validation | `data/frames.py` | Unsanitised upload paths |
| numpy-2 `np.cross` 2-D shim | Applied at the call site in every script that imports `ml/reference/camus_ef.py` (`score200.py`, `rescore_pointwise.py`) — **not inside the vendored file itself** | Vendored EF code raises on numpy ≥2.0; the shim is intentionally kept out of the vendored file to preserve byte-identity with the CAMUS group's original |

---

## 6. Not inventoried here

External-to-the-project scratch scripts (11 additional `.py` files under a Claude Code session's temp scratchpad, containing unique pilot/calibration/gate logic not copied into `outputs/benchmark/`) are catalogued separately in **EXTERNAL_ARTIFACTS.md** — they are real, unique work product but live outside this repository entirely.
