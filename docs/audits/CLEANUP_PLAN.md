# CLEANUP PLAN — classification only, nothing executed

**This is not a git repository.** Every DELETE below is unrecoverable — there is no
`git reflog`, no stash, no branch to check out. That fact drives every recommendation:
DELETE is reserved for items with a size/space rationale and genuinely zero unique
content (tool caches, virtual environments, exact duplicates). Everything with any
authored content, however dead, is recommended ARCHIVE instead, per your own stated
preference for archive + `.gitignore` over irreversible deletion.

**Nothing in this document has been executed.** No file has been moved, deleted, or
renamed to produce it.

---

## KEEP — required for the app, tests, docs, or provenance

Everything in `DIRECTORY_MAP.md §1` not listed below: `src/atria_echotrace/`, `tests/`,
`docker/`, `sample-dataset/`, `models/`, `adapters/`, `datasets/` (source material),
`outputs/` (including this session's `benchmark/`), `.venv/`, `.venv-review/`, `run.cmd`,
`run.sh`, `pyproject.toml`, `.gitignore`, `.env.example`, and all ten canonical docs
(`README`, `RESEARCH`, `PLAN`, `DELIVERY`, `HANDOVER`, `GAP_ANALYSIS`, `BENCHMARK_PLAN`,
plus the three notebook-source `.txt` files). No changes recommended to any of these.

---

## ARCHIVE — dead, but keep the content somewhere safe (not deleted)

| Path | Why | Risk if removed entirely | Recoverable? | Affects CODE_INVENTORY.md? |
|---|---|---|---|---|
| `atria_echotrace/app/`, `tests/`, `revisions/` (top-level, excluding `venv/` — ~20 MB, ~150 files) | Superseded prototype source + its historical test-run revision artifacts. Already gitignored and already documented as dead in HANDOVER §7 / README "Superseded prototype". Its specific defects (fabricated cm², invented prompt, synthetic-contour fallback, wrong-patient fallback) are already catalogued in RESEARCH.md §6 | None to the running app — nothing imports it (confirmed: `pyproject.toml` `where=["src"]`). Risk is purely to *history* — if you ever want to see exactly what the pre-port implementation did, this is the only copy | **No — not recoverable once deleted, this is not a git repo** | Not part of CODE_INVENTORY.md — confirmed unused by the package build |
| `.superseded/ingest/{camus.py,echonet.py}` | The reimplemented preprocessing that the vendored reference scripts replaced. Already gitignored, already in a folder named for exactly this purpose | None — already inert, already "archived" by its own location | No | CODE_INVENTORY.md §1.3 references this by name as the thing the vendored scripts replaced |
| `.agents/` (35 files, ~200 KB) | Process artifacts (briefings/handoffs/progress notes) from an earlier multi-agent build of the prototype. Already gitignored | None — no code, purely process notes; lowest priority item in this whole plan | No | Not referenced by CODE_INVENTORY.md |
| `PROJECT.md`, `TEST_INFRA.md`, `TEST_READY.md` | Describe the dead prototype's architecture/API/test suite in convincing enough detail (proper tables, "DONE" milestones, "59 passed") that a future reader could mistake them for current. Already gitignored | Low functional risk, **moderate misleading-a-future-reader risk** — these are the three files in this plan most likely to actually fool someone | No | Not referenced by CODE_INVENTORY.md; REVISION_AUDIT.md §3 flags these by name |
| 10 unique scripts in the `a560eeb4-…` session scratchpad (`stage_testset.py`, `calibrate.py`, `ef_gate.py`, `gate.py`, `pilot.py`, `pilot_fixed.py`, `run_bc.py`, `score_abc.py`, `score_pilot.py`, `prove_remap.py`) | Real, unique benchmark-provenance code with no in-project copy (EXTERNAL_ARTIFACTS.md §1) — this is the *inverse* of the rest of this table: not dead code to retire, but live provenance to rescue from a temp directory before it's cleared | **High** — a temp-directory cleanup (OS or tooling housekeeping) would permanently delete the only record of how `camus_test/`, `ceiling.json`, `ef_gate.json`, and the A/B/C pilot comparison were produced | No — external to the project, no backup | CODE_INVENTORY.md §6 cross-references this exact list |

**Suggested destination if you approve:** a `outputs/benchmark/provenance/` (scratchpad
scripts) and `.superseded/prototype/` (top-level `atria_echotrace/app`+`tests`+`revisions`,
renamed to sit alongside the existing `.superseded/ingest/` for a single dead-material
location) — naming only, not acted on.

---

## DELETE — safe, zero unique content, space reclaimed

| Path | Why | Risk if removed | Recoverable? | Affects CODE_INVENTORY.md? |
|---|---|---|---|---|
| `atria_echotrace\venv\` (top-level, **5.02 GB, 40,943 files**) | A stale Python virtual environment — pure third-party package installs, zero authored content. `PLAN.md` line 128-131 already documented this venv as something to be *moved and reused as `.venv/`*, which happened; this is the abandoned original | **None** — a venv is 100% regenerable by definition, and nothing here is unique | No, but nothing to recover — no venv anywhere contains unique information | Not referenced anywhere |
| `.pytest_cache\` (top-level, trivial) | pytest's own cache, regenerates on the next test run | None | No, but regenerates automatically | Not referenced |
| `a560eeb4-…\scratchpad\run200.py`, `…\scratchpad\__pycache__\` (external, temp) | `run200.py` here is an earlier, superseded copy — the in-project `outputs/benchmark/run200.py` is newer (17:28 vs 13:49) and is what actually produced `full200/raw_predictions.jsonl`. `__pycache__` is compiled bytecode | None | No, but zero unique value either way | Not referenced |
| `%TEMP%\atria_inspect\` (external, empty) | Empty directory, dated six weeks before this project's earliest file — not tied to this work | None (nothing inside it) | N/A — empty | Not referenced |

**Combined space reclaimed if all four rows are deleted: ~5.02 GB**, almost entirely the
one stale venv.

---

## UNKNOWN — needs your explicit decision

| Path | Why it's unresolved | What I'd need to classify it |
|---|---|---|
| `code-chat/Session1.docx` (project root, 575 KB, dated 2026-07-27) | Not referenced by any canonical doc, not gitignored alongside the other dead material, not obviously live or dead from metadata alone. I did not open it — `.docx` content isn't something this pass read | Your read of the file, or you telling me what it is, so it can either join the canonical docs or the archive pile |

---

## Approval package

**Keep as the production app** — everything under `src/atria_echotrace/`, `tests/`,
`docker/`, launchers, `sample-dataset/`, `models/`, `adapters/`, `datasets/` (source
material), `outputs/` (including this session's benchmark work), both venvs, and all ten
canonical docs. Nothing here is a cleanup candidate.

**Archive (not delete)** — the dead prototype's *code* (`atria_echotrace/app`, `tests`,
`revisions` — excluding its venv), `.superseded/ingest/`, `.agents/`, `PROJECT.md`,
`TEST_INFRA.md`, `TEST_READY.md`, and — separately and higher-priority — the 10 unique
benchmark-provenance scripts currently stranded in a Claude Code temp scratchpad.

**May delete inside the project** — `atria_echotrace\venv\` (5.02 GB, zero unique
content) and `.pytest_cache\` (trivial, regenerates).

**May delete outside the project** — the duplicate `run200.py` and `__pycache__` in the
`a560eeb4-…` scratchpad, and the empty, unrelated `%TEMP%\atria_inspect\`.

**Needs your call** — `code-chat/Session1.docx`. Everything else in this plan has
evidence behind its classification; this one file doesn't yet.

**Nothing has been touched.** Tell me which rows to act on — I can do it in one pass once
you've decided, archiving before anything is deleted.
