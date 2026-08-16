# ARTIFACT REPORT — post-session integrity delta

Read-only. No file was created, modified, moved, or deleted to produce this report, and
nothing was executed. Scanned 2026-08-08.

---

## 1. Baseline

The approved cleanup (2026-08-04) and `ANALYSIS.md` assumed this tree shape:

- `src/atria_echotrace/`, `tests/`, `docker/`, `sample-dataset/`, launchers, canonical docs — clean
- `models/`, `adapters/`, `datasets/` — bulk data/weights, gitignored, untouched
- `outputs/` — runtime output plus the benchmark evidence trail
- `.superseded/{prototype,agents,docs,ingest}` — all dead material consolidated
- `outputs/benchmark/provenance/` — 11 rescued scripts
- No stale venv, no empty prototype shells, no unique work stranded in temp

**That baseline still holds.** The four things the cleanup removed have not returned:
`atria_echotrace/` and `.agents/` are gone, both `.pytest_cache/` are gone, and every
session scratchpad under the project's Claude temp tree is now empty.

---

## 2. Delta findings

Six items changed since `ANALYSIS.md` was written. **Three are new user work from today
(08-Aug), which I have not touched and am not proposing to touch.** The two that matter for
report integrity are D1 and D2.

### D1 — `datasets/processed_datasets/camus_processed/` no longer holds an extracted corpus
It now contains **only** `camus_processed.zip` (257.8 MB, dated 22-May). The `frames/`,
`tracings.json` and `metadata.csv` that were presumably extracted there are absent, and the
directory's mtime is **today, 08-Aug 20:11**.

**Impact: none on the app, none on ANALYSIS.md.** Verified rather than assumed:
- `config.py:175` — `dataset_dir` defaults to `PROJECT_ROOT / "sample-dataset"`, not to
  `datasets/`. The shipped app never reads `processed_datasets/`.
- The bulk-corpus test targets `unified_processed/unified_processed/`
  (`tests/test_dataset.py:226-232`) — **which I confirmed is fully intact**: 22,048 frames,
  `tracings.json` and `metadata.csv` both still dated 13-Apr, unchanged. The
  22,048-frame / 11,024-case claim keeps its evidence path.
- The benchmark reads `outputs/benchmark/camus_test/` (self-contained, staged) and
  `datasets/original_datasets_and_repos/camus_public/database_nifti` (unchanged, 25-Jul).

So this is recoverable from the zip if ever needed, and nothing currently depends on it.

### D2 — `datasets/processed_datasets/echonet_processed/metadata.csv` was modified today
2,430,903 bytes, mtime **08-Aug 20:05**, while its sibling `tracings.json` remains 13-Apr.

**I cannot tell whether the content changed or the file was merely re-touched** — no hash
baseline exists for `datasets/` (only `sample-dataset/` carries a `manifest.sha256`).
Impact is bounded: the app doesn't read this path, and the unified corpus has its own
`metadata.csv` (13-Apr, unmodified) which is what the tests validate. Flagging it because
an unexplained modification to a corpus artifact is worth your awareness, not because it
breaks anything I can identify.

### D3–D5 — new classification work in progress (today)
- `datasets/classification_scripts/` — 3 files, **all zero bytes**:
  `camus-ef-apical-4c_2c.txt`, `camus-quality-good_medium_poor.txt`,
  `camus-window-apical-4c_2c - Copy.txt` (08-Aug 20:13)
- `datasets/classified_datasets/` — **completely empty** directory (08-Aug 21:53)

These read as your own new CAMUS-classification effort just getting started, unrelated to
anything this session produced. The zero-byte files may be intentional placeholders or
failed writes — only you can say. **Recommendation: leave alone, UNKNOWN.**

### D6 — session task-transcript debris in temp
`…\Temp\claude\C--Users-…-ATRIA-Platform\{session}\tasks\` holds 27 files / 6.76 MB of
subagent transcripts, 6.69 MB of it from this session (including the two agents that died
on the usage limit). Pure debris — no unique content.

---

## 3. Inventory table

| Path | Status | Incorporated? | Recommendation | Notes |
|---|---|---|---|---|
| `src/atria_echotrace/` (133 files) | CORE | yes | **KEEP** | Unchanged since cleanup |
| `tests/` (40 files) | CORE | yes | **KEEP** | Unchanged |
| `run.cmd`, `run.sh`, `docker/`, `pyproject.toml`, `.env.example` | CORE | yes | **KEEP** | Unchanged |
| `sample-dataset/` | CORE | yes | **KEEP** | Default `dataset_dir`; carries `manifest.sha256` |
| `.claude/{launch.json,settings.local.json}` | SUPPORT | yes (tooling config) | **KEEP** | Project-local dev config, 2 small files |
| `models/`, `adapters/` | CORE (data) | yes | **KEEP** | 13.4 GB gated weights |
| `datasets/original_datasets_and_repos/`, `data_processing_scripts/` | SUPPORT (data) | yes | **KEEP** | Benchmark dense reference reads this |
| `datasets/processed_datasets/unified_processed/` | EVIDENCE | yes (bulk test) | **KEEP** | 22,048 frames — backs the corpus claim |
| `datasets/processed_datasets/camus_processed/` (zip only) | UNKNOWN | no | **UNKNOWN** | D1 — nothing depends on it; zip retained |
| `datasets/processed_datasets/echonet_processed/` | EVIDENCE | partial (source lineage) | **KEEP** | D2 — metadata.csv mtime drift, unverifiable |
| `datasets/classification_scripts/` (3 × 0 bytes) | UNKNOWN | no | **UNKNOWN** | D3 — your new work, empty files |
| `datasets/classified_datasets/` (empty) | UNKNOWN | no | **UNKNOWN** | D4 — your new work, empty dir |
| `outputs/benchmark/{run200,score200,rescore_pointwise,make_overlays}.py` | TOOLING | no (not shipped) | **KEEP** | Produced all benchmark evidence |
| `outputs/benchmark/provenance/` (11 scripts) | TOOLING | no | **KEEP** | The 04-Aug rescue; verified all 11 present |
| `outputs/benchmark/full200/` (4 files + 202 overlays, 28.9 MB) | EVIDENCE | docs-only (cited in DELIVERY/ANALYSIS) | **KEEP** | Backs every headline benchmark number |
| `outputs/benchmark/camus_test/` (200 frames + 5 files, 25 MB) | EVIDENCE | no | **KEEP** | The staged test split + `PROVENANCE.md` |
| `outputs/benchmark/pilot20_fixed/` (4 files) | EVIDENCE | no | **KEEP** | A/B/C comparison; read by `rescore_pointwise.py` |
| `outputs/benchmark/pilot20/` (3 files, 0.08 MB) | DUPLICATE | no | **ARCHIVE or DELETE** | Superseded by `pilot20_fixed/`; not referenced by any current script |
| `outputs/benchmark/logs/` (9 logs, 0.33 MB) | EVIDENCE | no | **KEEP** | Run logs incl. `full200*.log`, `prove_remap.log` |
| `outputs/benchmark/__pycache__/` (2 files) | TEMP | no | **DELETE** | Bytecode, regenerates |
| `outputs/evaluations/` (6 files, 1.78 MB) | EVIDENCE | docs-only | **KEEP** | `eval_1784997417` / `eval_1785165023` are cited by name in DELIVERY §2 |
| `outputs/revisions/` (6 bundles, ~14 MB) | EVIDENCE | no | **KEEP** | Backs the revision→corpus round-trip claim |
| `outputs/uploads/` (11 PNGs, 1.22 MB) | TEMP | no | **UNKNOWN** | Manual UI-test upload debris from 27–28 Jul; at least 3 are byte-identical duplicates (150,774 B) |
| `.superseded/{prototype,agents,docs,ingest}` | SUPERSEDED | no | **KEEP** | Correctly archived 04-Aug |
| `code-chat/Session1.docx` | UNKNOWN | no | **UNKNOWN** | Still unclassified from the prior audit — your call outstanding |
| Canonical docs (12 `.md`/`.txt` at root) | CORE (docs) | yes | **KEEP** | Includes the 5 audit docs + `ANALYSIS.md` |
| `…\Temp\claude\…\{session}\tasks\` (27 files, 6.76 MB) | TEMP | no | **DELETE** | D6 — subagent transcripts |
| All session `scratchpad/` dirs | TEMP | n/a | — | Verified **empty**; the 04-Aug rescue holds |

---

## 4. Bloat / duplication risks

Only genuine risks listed; nothing here is a style preference.

1. **`outputs/benchmark/pilot20/` vs `pilot20_fixed/`** — the only true duplication. `pilot20`
   is the pre-fix pilot; `rescore_pointwise.py`'s `RUNS` dict reads **only** `pilot20_fixed`.
   Small (0.08 MB), but two similarly-named result dirs is exactly the kind of thing that
   misleads a future reader into quoting the wrong run.
2. **`outputs/uploads/`** — 11 test PNGs, with at least three byte-identical at 150,774 B.
   Trivial size; the risk is that they look like user data when they are UI-test debris.
3. **`camus_processed.zip` (257.8 MB)** — largest single reclaimable item, but it is the
   *only* remaining copy of that extracted corpus, so deleting it is a one-way door. I would
   not touch it.
4. **Not a risk:** the 28.9 MB of overlay PNGs. They are the visual QA evidence `ANALYSIS.md`
   and `DELIVERY.md` both cite, and regenerating them requires the persisted predictions
   plus a rendering pass. Keep.

---

## 5. ANALYSIS.md integrity check

**ANALYSIS.md remains factually correct. No contradiction found, no cited evidence path missing.**

| ANALYSIS.md claim | Evidence path | Status |
|---|---|---|
| 39 modules / ~8k lines under `src/atria_echotrace/` | `src/atria_echotrace/` | ✅ intact (133 files incl. assets) |
| 282 tests, 14 test files | `tests/` | ✅ intact (40 files) |
| Six API routers, endpoints as listed | `api/*.py` | ✅ unchanged |
| Both prompt templates verbatim | `ml/prompts.py` | ✅ unchanged |
| Five-stage model in `app.js:758` | `web/js/app.js` | ✅ unchanged |
| 200-frame benchmark: median 4.98 mm, 200/200 parsed | `outputs/benchmark/full200/{pointwise.csv,overlays/manifest.csv}` | ✅ present |
| 200 overlays, 0 failures | `outputs/benchmark/full200/overlays/` | ✅ 202 files (200 PNG + index.html + manifest.csv) |
| Provenance of the evidence chain | `outputs/benchmark/provenance/` | ✅ all 11 scripts present |
| `eval_1784997417` n=3 Dice 0.672 | `outputs/evaluations/eval_1784997417.json` | ✅ present |
| 22,048-frame corpus loads/validates | `datasets/.../unified_processed/unified_processed/` | ✅ intact — **the one claim D1 could have broken, and it did not** |
| `atria train` ≠ published adapter config | `BENCHMARK_PLAN.md §1`, `ml/train.py` | ✅ unchanged (still under-documented in README, as reported) |
| LA selectable but adapters LV-only | `app.js:1221`, `api/inference.py:44` | ✅ unchanged |

The only ANALYSIS-adjacent wrinkle is D2: if `echonet_processed/metadata.csv` changed in
content today, no ANALYSIS claim rests on that specific file — the corpus claim rests on the
unified copy, which is untouched. I state this as bounded rather than certain because no
hash baseline exists to compare against.

---

## 6. Approval package

**KEEP** — everything marked CORE/SUPPORT/EVIDENCE/TOOLING/SUPERSEDED above: the app, tests,
launchers, docker, sample-dataset, weights, datasets (incl. the intact unified corpus), all
`outputs/benchmark` evidence and provenance, `outputs/evaluations`, `outputs/revisions`,
`.superseded/`, `.claude/`, and all canonical docs.

**ARCHIVE** — `outputs/benchmark/pilot20/` (3 files, 0.08 MB) → suggest
`outputs/benchmark/.superseded_runs/pilot20/`, or simply a one-line `README` in
`outputs/benchmark/` stating that `pilot20_fixed` supersedes it. Archiving is optional; the
documenting alternative costs nothing and removes the same confusion risk.

**DELETE (safe, zero unique content)**
- `outputs/benchmark/__pycache__/` — 2 bytecode files
- `…\Temp\claude\C--Users-…-ATRIA-Platform\{session}\tasks\` — 27 subagent transcripts, 6.76 MB

**UNKNOWN — needs your call**
- `datasets/classification_scripts/` (3 zero-byte files) and `datasets/classified_datasets/`
  (empty) — your new work from today; are the empty files intentional placeholders?
- `datasets/processed_datasets/camus_processed/` — keep the 257.8 MB zip, or was the
  extracted corpus deliberately removed and the zip is next?
- `echonet_processed/metadata.csv` — was today's modification intentional?
- `outputs/uploads/` (11 test PNGs) — safe to clear, or keep as UI-test fixtures?
- `code-chat/Session1.docx` — still outstanding from the prior audit.

---

## 7. Stop

**No actions taken.** Nothing was modified, moved, deleted, or run. Awaiting your approval
before any cleanup.
