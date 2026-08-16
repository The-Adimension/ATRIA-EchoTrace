# ANALYSIS — ATRIA EchoTrace

Read-only analysis. No code was changed, nothing was run, no inference performed.
Evidence is cited as `path:symbol` and `notebook_as_py.txt` line ranges. Where the
canonical docs and the code disagree, the code wins; where I am uncertain, I say so.

---

## 1. What the application is

**Product definition.** ATRIA EchoTrace is a single-operator clinical workstation for
tracing the left-ventricle endocardial border on echocardiography frames. A fine-tuned
MedGemma 1.5 4B model proposes a contour as a list of normalised `[y, x]` coordinates; a
human reviews it, corrects it vertex by vertex on a canvas, and saves the correction. Those
corrections are exportable as a trainable corpus, closing a loop back to fine-tuning. It
runs as one local process serving both a JSON API and a browser UI on `127.0.0.1:8000`.

**Intended user and use.** A clinician or imaging researcher working on their own machine,
on their own frames or on CAMUS/EchoNet data. Explicitly **research use only, not a medical
device** — disclaimers are served at `/api/meta/disclaimers` (`api/meta.py:263`) and shown
in the UI. Every AI contour is framed as a proposal requiring human review.

**Runtime shape.**

| Aspect | Detail | Evidence |
|---|---|---|
| Package | `atria-echotrace` 1.0.0, src-layout, 39 Python modules (~8k lines) | `pyproject.toml`, `where = ["src"]` |
| Entry points | `atria` console script · `python -m atria_echotrace` · `run.cmd`/`run.sh` | `pyproject.toml:54-55`, `__main__.py` |
| Process model | One uvicorn process; API endpoints are sync `def` so FastAPI runs them in a threadpool; a `threading.RLock` serialises GPU access | `cli.py:cmd_serve`, `ml/engine.py` |
| Frontend | Buildless Preact + htm SPA served same-origin by the same process — no Node, no bundler | `web/`, `api/app.py` |
| Tier: review | ~250 MB, no torch. Browsing, upload, manual tracing, geometry, FAC, figures, exports, `export-corpus` | base `dependencies` in `pyproject.toml` |
| Tier: `[ai]` | torch/transformers/peft/bitsandbytes/trl — inference, training, evaluation | `pyproject.toml:31-40` |
| Tier: `[ingest]` | SimpleITK + OpenCV — raw NIfTI/AVI preprocessing only | `pyproject.toml:45-48` |
| Tier reporting | `GET /api/meta/capabilities` returns `tiers`; the UI disables what is unavailable and states why | `api/meta.py:195` |

**In scope** (per code): dataset browsing, frame upload, LV (and LA) contour proposal,
interactive revision, clinical area/FAC metrics, revision persistence and export,
revisions→corpus, and CLI stages for ingest/train/evaluate/publish.

**Deliberately out of scope** (per `README.md "Not built"`, `DELIVERY.md §4`, and confirmed
by absence in code): authentication/multi-user/database, PACS/DICOM networking,
video/temporal modelling, RV/valves/strain, training sweeps, active learning, CI/telemetry,
client-side build tooling, versioned corpora, browser-driven training/evaluation/publishing,
and upload resizing.

---

## 2. What it does — capabilities mapped to code

| Capability | User-facing behaviour | Code entry points | Tier |
|---|---|---|---|
| **Dataset browsing** | Filter cases by source/view, thumbnails, badges for uncalibrated data and ED/ES integrity flags, EF shown when published | `api/dataset.py:27,48` → `data/dataset.py:DatasetRepository`; UI `CaseBrowser` (`app.js:1041`) | review |
| **Frame upload** | ED and ES drop slots plus an apical-view selector; images re-encoded to RGB PNG, **never resized** | `api/dataset.py:123` → `data/frames.py`; UI `UploadPanel` (`app.js:977`) | review |
| **Model load / adapter selection** | Explicit load/unload; adapter dropdown marks locally-present checkpoints `·local`; status pill shows weight source | `api/inference.py:63,79,99` → `ml/engine.py:InferenceEngine`; UI `TopBar` (`app.js:656`) | AI |
| **Inference** | "Trace ED + ES" or per-pane "Trace"; returns a polygon plus the prompt variant used | `api/inference.py:105` → `engine.predict`, `ml/prompts.py:build_prompt` | AI |
| **Prompt/structure config** | LV/LA toggle, adapter select, prompt-variant select (`training`/`generic`) exposed in the toolbar | `app.js:1219-1260`; `domain/structures.py`, `ml/prompts.py` | AI |
| **Interactive revision (HITL)** | Dual ED/ES canvases, drag/add/delete vertices, keyboard-operable (Tab/arrows/Shift/Ctrl+Z), 4 panel modes (Original/Model/Revision/Overlay), live Dice-vs-model agreement | `web/js/canvas-editor.js`, `EditorPane` (`app.js:1319`) | review (editing works without AI) |
| **Metrics** | Area px²/cm², perimeter, FAC %; **cm² withheld when spacing is unknown** with an explicit calibration state rather than a fabricated number | `api/clinical.py:88,125` → `domain/metrics.py`; UI `MetricsRail` (`app.js:1535`) | review |
| **Save / export / persistence** | Saves revision bundle: JSON, PNGs, CSVs, 4-panel figure, ZIP; revisions listed in the left rail and reopenable with both polygons restored | `api/revisions.py:111,210,234,255` → `export/package.py`; `render/figures.py` | review |
| **export-corpus** | Revisions → `frames/` + `tracings.json` + `metadata.csv` that `atria train` consumes unchanged; `user_polygon` is the label; newest revision wins; integrity flags carry through | `cli.py:cmd_export_corpus` → `export/package.py:export_corpus` | review |
| **ingest** | Runs the *vendored original* preprocessors over raw CAMUS NIfTI / EchoNet AVI | `cli.py` `p_ingest` → `data/ingest/run.py` → `data/ingest/reference/*.py` | `[ingest]` |
| **train** | QLoRA fine-tune | `cli.py` `p_train` → `ml/train.py` | AI |
| **evaluate** | Parse rate, Dice, IoU over a split; `--figures` renders best/worst 3-panel comparisons | `cli.py` `p_eval` → `ml/evaluate.py:save_ranked_figures`; also `POST /api/evaluation/runs` (`api/evaluation.py:37`) | AI |
| **publish-adapter** | Scans a checkpoint folder, pre-selects inference-relevant files, gated repo by default, requires `--yes` | `cli.py` `p_pub` → `ml/publish.py` | AI |
| **doctor / readiness / weights** | Prints Python, dirs, token presence, **per-weight resolved source**, tier availability, dataset validation, ED/ES integrity warnings, and the CPU-only-torch diagnosis with the exact reinstall command | `cli.py:cmd_doctor`; `config.py:weights_report`, `ml/runtime.py:diagnose_cpu_fallback`; UI Weights modal (`app.js:858`) | review |

---

## 3. How the service is offered

### 3.1 Primary interface — the browser SPA

One FastAPI process serves the API and the SPA together (`api/app.py`), so "click-and-run"
is genuinely one command with no second server and no build step. UI regions:

- **Top bar** — model status pill + load/unload, and three modals: **About** (disclaimers),
  **Weights** (per-weight resolved source and what to place where), **Stages** (the five-stage
  launcher).
- **Left rail** — upload panel (ED/ES slots + view selector), case browser, recent revisions.
- **Centre stage** — toolbar (Structure LV/LA · Adapter · Prompt variant), then dual ED/ES
  editor panes with per-pane panel switching.
- **Right rail** — metrics (areas, perimeter, FAC %), calibration state, integrity notes.

The frontend talks to the backend through a single typed client (`web/js/api.js`, `api`
object) over same-origin `fetch`. Assets are served with `Cache-Control: no-cache`
(`api/app.py::_RevalidatingStatics`) so an upgraded install cannot silently keep running the
previous JavaScript. Preact, htm and the fonts are vendored — measured **zero** third-party
requests.

### 3.2 Secondary interfaces

**CLI** — 7 subcommands, each mapping to a notebook phase (`cli.py:1-13`):
`serve` · `doctor` · `ingest` · `train` · `evaluate` · `export-corpus` · `publish-adapter`.
`--dataset-dir` and `--output-dir` are global flags (`cli.py:428-433`), so
`atria train --dataset-dir ./revised_corpus` — the command the Stages panel prints — is valid.

**HTTP API** — six routers: `/api/dataset`, `/api/model` + `/api/inference`,
`/api/clinical`, `/api/revisions`, `/api/evaluation`, `/api/meta`; OpenAPI docs at
`/api/docs`.

**Launchers** — `run.cmd` / `run.sh` create a venv (`uv` when available, stdlib `venv`
otherwise), install, **select the correct torch build** via an `nvidia-smi` probe
(overridable with `ATRIA_TORCH_BACKEND`), and serve.

### 3.3 Interaction model

- **Local-first weights.** For the base model and each adapter independently: project folder
  (`models/`, `adapters/`) → HF cache → Hub. Only the last needs a token and a network
  (`config.py:weights_report`).
- **Offline-capable.** Verified in `DELIVERY.md §2` with `HF_HUB_OFFLINE=1`, no token, empty
  cache: weights resolve `[local]`, model loads, prediction identical to in-place.
- **GPU needed only for the AI tier.** Everything else — browsing, upload, manual tracing,
  metrics, figures, exports, corpus export — runs on CPU with no weights at all.
- **Stages model (0/A/B/B→A/C)** is real in code, not just docs: `STAGES` in `app.js:758`
  with live per-machine readiness predicates reading `caps.tiers`. Multi-hour or irreversible
  stages (train, evaluate, publish) surface their *command* rather than a button — a
  deliberate refusal to put a GPU job or a public Hub write behind an unauthenticated
  loopback click.

---

## 4. Gap vs the notebook

### A. Fully implemented

| Notebook phase | Production component | Fidelity |
|---|---|---|
| Config cell: `STRUCTURE_INFO`, `NORM_SCALE`, view/instant names (L198-219) | `domain/structures.py` | **=** verbatim; `TARGET_STRUCTURE` becomes a per-request parameter instead of a module constant |
| Data loading `load_echocardiographic_frame_data` (L254-280) | `data/dataset.py:DatasetRepository` | **≈** same 3-artifact contract, plus validation and `(case_id, view)` keying |
| Frame loader (L284-298) | `data/frames.py:load_frame` | **=** |
| Prompt templates — **both** variants (L398-414 training, L1205-1220 generic) | `ml/prompts.py` | **=** byte-equal, asserted by `tests/test_fidelity.py` |
| `create_ground_truth_response` (L416-423) | `ml/prompts.py:create_ground_truth_response` | **=** |
| Sample prep (L449-500), image loading, collator (L690) | `ml/datasets.py`, `ml/train.py:build_collate_fn` | **=** |
| QLoRA fine-tuning (L618-851) | `ml/train.py` | **=** hyperparameters asserted against notebook *and* published `adapter_config.json` |
| `parse_polygon_from_response`, `polygon_to_mask`, Dice, IoU (L919-990) | `domain/geometry.py` | **=** plus `sanitize_polygon` hardening |
| Evaluation loop + summary (L992-1083) | `ml/evaluate.py`, `api/evaluation.py` | **≈** same metrics, plus background-run lifecycle and persistence |
| Best/worst ranking + figures (L1085-1169) | `ml/evaluate.py:ranked`, `save_ranked_figures` | **=** |
| HITL model load / predict / save callbacks (L1257-1375) | `ml/engine.py`, `api/inference.py`, `api/revisions.py`, `export/package.py` | **≈** same contract, hardened |
| Overlay drawing (L316-325, L1240-1248) | `render/overlays.py` | **=** |
| 4-panel comparison figure (L1346-1370) | `render/figures.py` | **=** |
| HITL dual-canvas editor UI (L1381+) | `web/` SPA | **≈** superset — see §4C |
| HF adapter transfer (L1580-1754) | `ml/publish.py`, `atria publish-adapter` | **≈** same scan/select/gate logic, CLI instead of ipywidgets |

### B. Partially implemented

| What exists | What is weaker / missing | Practical impact |
|---|---|---|
| `atria train` faithfully ports the **notebook's** hyperparameters (epochs 10, batch 16, lr 2e-4, `max_length` 2048) | The **published adapters were trained with different values** (5 epochs, batch 4, lr 1e-4, `max_length` 1024 — derived in `BENCHMARK_PLAN.md §1` from the adapter's own `metadata.json`/`training_args.bin`). Running `atria train` will **not** reproduce the shipped adapter | Moderate. Correct as a notebook port, but a user who assumes "train reproduces the artefact" is wrong. **This is documented only in `BENCHMARK_PLAN.md`, not in README/DELIVERY where a user would look.** |
| `atria train` is a faithful port and is unit-asserted | **Never executed** — no training run performed (`DELIVERY.md §4`) | Moderate — the one major path with no runtime evidence |
| LV **and** LA are selectable in the UI and API (`app.js:1221`, `api/inference.py:44`) | The published adapters are **LV-only** (`BENCHMARK_PLAN.md §2`). Selecting LA runs an adapter on anatomy it was never tuned for. The UI disables LA only when the *dataset* lacks an LA reference — it does not warn that the *adapter* is LV-only | Moderate. A user can silently get out-of-distribution LA output that looks like a normal prediction. The notebook had the same switch, but a notebook user would have trained an LA adapter first. |
| Docker packaging authored, inspected-correct, `ai-gpu` profile present | **Never executed** — Docker not installed on this machine | Low for local use; unknown for container deployment |
| Evaluation now has a real 200-frame CAMUS benchmark under the corrected task framing | Only CAMUS LV; the EchoNet adapter still has **n=1** evidence; no full 2 752-frame combined split | Low-moderate — honestly disclosed, and the CAMUS result is the one that matters for the primary adapter |

### C. Not implemented, or deliberately different

| Item | Classification | Reason |
|---|---|---|
| Colab scaffolding: `drive.mount`, `userdata.get('HF_TOKEN')`, `!pip install`, `google.colab.output` callbacks, runtime-restart handling (L79-190, L1172-1190) | **Intentional** — must not be ported. Replaced by `config.py` env settings + venv launchers | Not a gap |
| `TRAIN/VAL/TEST_MAX_SAMPLES` module constants (L221-224) | **Intentional** — became CLI flags (`--train-samples`, `--val-samples`, `--max-samples`) | Not a gap |
| Notebook's `load_model` reusing a global namespace model (L1262-1266) | **Intentional** — replaced by an explicit engine singleton with lifecycle endpoints and a GPU lock | Improvement |
| Notebook saved to a fixed `/content/tracing_results` with 2 files per image (L1323) | **Intentional divergence** — replaced by revision bundles (JSON + PNGs + CSVs + ZIP), listed and reopenable | Improvement |
| Notebook's bare `except:` in `parse_polygon` (L1236) | **Intentional divergence** — replaced by explicit parsing with `sanitize_polygon` | Improvement |
| Notebook HITL prompt hardcodes `VIEW_NAMES["2CH"]` and derives instant from pane index (L1293-1297) | **Intentional divergence** — the app takes the real view from the dataset or the upload selector, and picks the training template when both are known | Improvement (and the basis of the benchmark's config A2) |
| Notebook computes **no** clinical metrics (no area, no FAC, no cm²) | **Deliberate addition**, sourced from the author's deployed Space, not invented | Not a notebook gap |
| Notebook has **no** preprocessing cells — it consumes the 3-artifact contract as given (L190-194) | **Deliberate addition** — `atria ingest` wraps the user's *real* scripts, vendored byte-identically | Improvement |
| Notebook has no revisions→corpus loop | **Deliberate addition** (`export-corpus`) | Improvement |
| EchoNet ED/ES transposition | **Intentional non-fix** — correcting at inference would push every EchoNet request out of its adapter's training distribution; surfaced and flagged instead | Not a gap |
| Per-stage quantisation differences (train off / eval 4-bit no-double-quant / serve 4-bit with) | **Intentional** — preserved as the notebook's own ablation; serving's values are what the measured accuracy was obtained under | Not a gap |

---

## 5. Synthesis

### What the app is good for today

A researcher or clinician can, on one machine and largely offline, browse or upload
echo frames, get an LV contour proposed by a locally-resolved fine-tuned MedGemma adapter,
correct it on a real canvas editor, see honest clinical metrics (with cm² withheld rather
than fabricated when calibration is unknown), save an auditable revision bundle, and turn
accumulated corrections into a training corpus. The review tier does all of that except the
AI proposal in ~250 MB with no GPU. Every notebook phase has a live production component;
nothing is stubbed, and unavailable capability is reported rather than faked.

### Top remaining gaps, ranked by user impact

1. **`atria train` won't reproduce the published adapter, and only `BENCHMARK_PLAN.md` says so.** → One clarifying sentence in README/DELIVERY next to the train instructions.
2. **LA is selectable but the adapters are LV-only.** → One UI note/tooltip on the LA toggle stating the shipped adapters were tuned on LV.
3. **`atria train` has never been executed.** → Unresolvable without a GPU training run; correctly disclosed today.
4. **Docker never executed.** → Correctly disclosed; low priority for a local-first tool.
5. **EchoNet adapter accuracy is still n=1.** → Real but honestly labelled; the CAMUS 200-frame result covers the primary adapter.

### Already better than the notebook as a production system

Explicit engine lifecycle with a GPU lock instead of notebook globals · local-first offline
weight resolution · the vendored real preprocessors (a stage the notebook lacks entirely) ·
clinical metrics with an explicit calibration state · auditable, reopenable revision bundles
· the revisions→corpus loop · the vision-tower LoRA key repair with `fully_loaded` reporting
· CPU-only-torch diagnosis · no-absolute-path guarantee across 41 client surfaces · 282 tests
· and a 200-frame benchmark scored against the adapter's actual task.

### What should NOT be treated as a gap

Missing Colab scaffolding (Drive mount, `userdata`, `!pip`, `google.colab.output`), the
absence of module-level `TARGET_STRUCTURE`/`MAX_SAMPLES` constants, the fixed
`/content/tracing_results` output path, the notebook's bare-`except` parser, the hardcoded
2CH prompt in the HITL cell, the uncorrected EchoNet ED/ES transposition, and the per-stage
quantisation spread. Each is either an environment artifact that must not be ported or a
documented, reasoned divergence.
