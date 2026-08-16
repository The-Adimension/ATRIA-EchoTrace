# DIRECTORY MAP — ATRIA EchoTrace

> **Point-in-time audit snapshot (2026-08-04) — superseded, kept for provenance.**
> It describes the tree *before* the cleanup it recommended was approved and executed.
> `atria_echotrace/`, `.agents/`, `PROJECT.md`, `TEST_INFRA.md` and `TEST_READY.md` are
> no longer at the repository root — they now live under `.superseded/`, and the stale
> 5.02 GB virtual environment was deleted. For the current tree, see
> [README.md](../../README.md). Do not use this file as a map of the repository today.

Read-only. Sizes measured 2026-08-04 via `Get-ChildItem -Recurse | Measure-Object -Sum Length`.
Not a git repository — nothing here has been moved or deleted.

---

## 1. Current tree, top level

| Path | Type | Size | Purpose | Required: prod | Required: dev/test | Sensitivity |
|---|---|---|---|---|---|---|
| `src/atria_echotrace/` | code | ~1 MB | **The shipped application** — see CODE_INVENTORY.md | **yes** | yes | keep |
| `tests/` | test | small | 282 tests (272 without weights) | no | **yes** | keep |
| `docker/` | config | small | Dockerfile + compose, authored/unexecuted | no (launchers cover click-and-run) | no | keep |
| `sample-dataset/` | data | small | 50 frames, byte-exact subset of the training corpus | no | **yes** (real-data tests, demo) | keep |
| `models/` | weights | 8.05 GB | Local MedGemma base weights (gitignored) | yes, if present — else Hub fallback | for `model`-marked tests | keep, do not commit |
| `adapters/` | weights | 5.35 GB | Local LoRA adapters (gitignored) | yes, if present — else Hub fallback | for `adapter`-marked tests | keep, do not commit |
| `datasets/` | data | 12.83 GB | Supplied source material (bulk gitignored; scripts inside are not) | no (not needed to serve) | yes, for `atria ingest` + this session's benchmark | **keep — explicitly protected**, review before touch |
| `outputs/` | outputs | 0.07 GB | Runtime output: revisions, uploads, evaluations, **this session's `benchmark/`** | no (regenerated at runtime) | contains this session's only copy of benchmark results | keep — currently the only record of the 200-frame re-score |
| `.venv/` | env | 1.11 GB | AI-tier dev venv (torch) | no | yes, active dev environment | keep, regenerable |
| `.venv-review/` | env | 0.39 GB | Review-tier dev venv (no torch) | no | yes, active dev environment | keep, regenerable |
| `.pytest_cache/` | temp | ~0 | pytest's own cache | no | no (regenerates) | safe to ignore |
| `code-chat/` | unknown | 575 KB | One file: `Session1.docx`, dated 2026-07-27 — **not referenced by any doc in this project** | no | no | **UNKNOWN — needs your call** (see §4) |
| `atria_echotrace/` (top level, not under `src/`) | superseded | 5.04 GB | Earlier prototype + its own `venv/` (5.02 GB of that is the venv alone) + `revisions/` (90 files, real test artifacts from the old system) | **no — `pyproject.toml`'s `where=["src"]` proves this is not the installed package** | no | **SUPERSEDED per the project's own docs** (HANDOVER §7, README "Superseded prototype", `.gitignore`) |
| `.superseded/ingest/` | superseded | ~0 | `camus.py`, `echonet.py` — the reimplemented preprocessing that the vendored reference scripts replaced | no | no | SUPERSEDED, documented as such |
| `.agents/` | superseded | ~0 | 9 role-named subfolders (`orchestrator`, `worker_m1`, `worker_m2_m3`, `challenger_m4`, `auditor_m4`, `e2e_tester`, `victory_auditor`, `sentinel`, `explorer_discovery`), each with `ORIGINAL_REQUEST.md`/`BRIEFING.md`/`progress.md`/`handoff.md` — process artifacts from an earlier multi-agent build of the prototype | no | no | SUPERSEDED, documented as dead in HANDOVER §7 |
| `PROJECT.md` | superseded docs | small | Describes the OLD prototype's layout (`atria_echotrace/app`, different API schema, synthetic-contour fallback) | no | no | SUPERSEDED — confirmed by direct read: describes `/api/inference/predict` with a `model_id` param and a "precomputed fallback" that doesn't exist in the shipped app |
| `TEST_INFRA.md` | superseded docs | small | Old 4-tier test spec for the prototype's `atria_echotrace/tests/` | no | no | SUPERSEDED — confirmed: references `test_backend_api.py`, which does not exist under `tests/` |
| `TEST_READY.md` | superseded docs | small | Old test suite's readiness report ("59 passed", Python 3.13) — current suite is 282 tests on Python 3.14.4 | no | no | SUPERSEDED |
| `CLAUDE.md` | docs | small | Operating rules for this session | n/a | n/a | keep |
| `README.md` | docs | 45 KB | How to run, architecture, notebook-phase mapping, findings | n/a | n/a | keep |
| `HANDOVER.md` | docs | 22 KB | **Authoritative resume-work prompt** — read first per its own instruction | n/a | n/a | keep |
| `DELIVERY.md` | docs | 16 KB | What's delivered/measured/not-verified | n/a | n/a | keep |
| `RESEARCH.md` | docs | 34 KB | Every architecture decision + evidence, the 28-row notebook map | n/a | n/a | keep |
| `PLAN.md` | docs | 29 KB | Architecture + 3 addenda; **its own file-structure diagram (§3) predates `ml/reference/` and the benchmark work** | n/a | n/a | keep — see REVISION_AUDIT.md for the staleness note |
| `GAP_ANALYSIS.md` | docs | 36 KB | 3-round gap analysis, N1-N5, all closed | n/a | n/a | keep |
| `BENCHMARK_PLAN.md` | docs | 15 KB | This session's CAMUS LV benchmark plan, revision 2 | n/a | n/a | keep |
| `notebook_as_py.txt` | source material | 66 KB | The original notebook exported to Python — the ground truth every module docstring cites by line number | n/a | n/a | keep — primary source |
| `Notebook-readme.txt` | source material | 19 KB | Notebook's own documentation | n/a | n/a | keep |
| `ATRIA-EchoTrace-FT-HF-MedGemma.txt` | source material | 37 KB | Fine-tuning notes | n/a | n/a | keep |
| `.env.example` | config | small | Placeholder template, never a real secret | n/a | n/a | keep |
| `.gitignore` | config | small | Excludes venvs, outputs, weights, bulk datasets, and every path in §2 below | n/a | n/a | keep |
| `pyproject.toml` | config | small | Package definition, `where=["src"]`, extras, console script | **yes** | yes | keep |
| `run.cmd` / `run.sh` | code | small | Click-and-run launchers | **yes** | no | keep |

Not separately tabulated: `datasets/original_datasets_and_repos/echonet_dynamic/` alone contains ~142,000 raw video files — confirmed present, not enumerated further (out of scope; it is bulk source data, gitignored, and explicitly protected by your prohibitions).

---

## 2. What `.gitignore` already says about this tree

The project's own `.gitignore` independently corroborates most of the above without me asserting anything new:

```
/.superseded/
/atria_echotrace/
/.agents/
/PROJECT.md
/TEST_INFRA.md
/TEST_READY.md
```

These six paths are already the project's own declared "dead, kept only because this isn't a git repo" list (`.gitignore` lines 43-56, comment: *"Kept locally because deletion here would be unrecoverable, but excluded so it never reaches another machine and misleads a new reader"*). `code-chat/` is **not** in this list — it was never classified either way, which is why it's UNKNOWN rather than SUPERSEDED below.

---

## 3. Proposed target production tree

This is the tree `HANDOVER.md §1` and `PLAN.md §3` already describe as the delivered product — restated here as a concrete before/after, not a new design:

```
ATRIA-Platform/
├── pyproject.toml  .env.example  .gitignore  run.cmd  run.sh
├── README.md  RESEARCH.md  PLAN.md  DELIVERY.md  HANDOVER.md  GAP_ANALYSIS.md  BENCHMARK_PLAN.md
├── notebook_as_py.txt  Notebook-readme.txt  ATRIA-EchoTrace-FT-HF-MedGemma.txt
├── docker/
├── src/atria_echotrace/            (all of CODE_INVENTORY.md §1)
├── tests/
├── sample-dataset/
├── models/  adapters/               (gitignored, local-only)
├── datasets/                        (gitignored bulk, vendored scripts travel with src/)
├── outputs/                         (gitignored, includes benchmark/ provenance)
└── .venv/  .venv-review/            (gitignored)
```

No path in this target tree is new — it is exactly the current tree minus the seven items flagged SUPERSEDED/UNKNOWN in §1.

---

## 4. Current vs proposed — contrast

| Path | Current | Target | Migration note |
|---|---|---|---|
| `atria_echotrace/` (top-level) | 5.04 GB, present | absent | **ARCHIVE or DELETE** (your call — see CLEANUP_PLAN.md; the project's own docs already call it dead) |
| `.superseded/ingest/` | present, ~0 | absent | ARCHIVE or DELETE |
| `.agents/` | present, ~0 | absent | ARCHIVE or DELETE |
| `PROJECT.md`, `TEST_INFRA.md`, `TEST_READY.md` | present | absent | ARCHIVE or DELETE |
| `code-chat/Session1.docx` | present, 575 KB | undetermined | **UNKNOWN** — not documented anywhere as either live or dead; recommend you open it once before any decision, since I cannot read `.docx` content in this pass |
| `.pytest_cache/` | present, ~0 | absent (regenerates) | safe DELETE, zero risk, zero value kept |
| Everything else in §1 | present | present, unchanged | **KEEP in place** |

Nothing in this table is a new proposal — every MOVE/ARCHIVE/DELETE candidate here is a path the project's own `.gitignore` and `HANDOVER.md §7` already flagged as dead before this audit started. This audit's contribution is confirming each one still matches what the docs claim (done by direct listing, not by trusting the docs blindly — see the `PROJECT.md`/`TEST_INFRA.md` content-verification notes in §1) and quantifying the size (5.04 GB, almost entirely one stale `venv/`).
