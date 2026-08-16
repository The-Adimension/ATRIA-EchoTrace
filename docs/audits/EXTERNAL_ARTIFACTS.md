# EXTERNAL ARTIFACTS — scan outside the project directory

Read-only. Nothing outside `ATRIA-Platform\` was moved, deleted, or modified to produce
this document. Scanned 2026-08-04.

**Note on method:** two background research agents were dispatched to do this scan and to
cross-check the source-tree inventory; both were killed mid-run by a session usage limit
("You've hit your session limit · resets 6pm Africa/Cairo") before producing output. The
scan below was done directly instead (PowerShell `Get-ChildItem` against the specific
locations Phase 2b names), which is what the results below reflect — no agent output was
used or is quoted anywhere in this document.

---

## 1. The most significant finding: a session scratchpad with unique benchmark code

`C:\Users\sheha\AppData\Local\Temp\claude\C--Users-sheha-Desktop-The-Adimension-ATRIA-Platform\`
holds one scratchpad folder per Claude Code session ever opened against this project —
**54 session folders total**. 53 of them are empty. One,
`a560eeb4-2788-44cc-a159-dd3198615fd8\scratchpad\`, contains 13 files, 11 of which are
**real scripts with no copy anywhere inside the project tree**:

| File | Size | Modified | Function (from this session's own record of the work) | Unique vs project? |
|---|---|---|---|---|
| `stage_testset.py` | 3.2 KB | 08-02 18:28 | Staged the official 200-frame CAMUS test subgroup into `outputs/benchmark/camus_test/`, verified against `subgroup_testing.txt` | **Yes** — this is *how* `camus_test/` was built; not reproducible from anything in-project |
| `calibrate.py` | 6.3 KB | 08-02 18:30 | Computed the discretisation-ceiling calibration (`ceiling.json`) | **Yes** — the output JSON is in-project, the script that produced it is not |
| `ef_gate.py` | 4.4 KB | 08-02 18:32 | Computed the EF integrity gate (`ef_gate.json`) that you accepted as "Step 0 — CLEAR" | **Yes** |
| `gate.py` | 10.0 KB | 08-02 21:58 | The Step-0 integrity gate logic itself (largest script here) | **Yes** |
| `pilot.py` | 3.9 KB | 08-02 18:42 | First (interrupted/broken) 20-frame pilot run | Yes, but superseded by the next row |
| `pilot_fixed.py` | 4.0 KB | 08-02 23:05 | Corrected pilot run that actually produced `pilot20_fixed/` | **Yes** — feeds `rescore_pointwise.py`'s `RUNS` dict directly (`A2 pilot20`) |
| `run_bc.py` | 6.9 KB | 08-03 00:00 | Ran conditions B (bf16) and C (geometry prompt) | **Yes** — produced `pilot20_fixed/raw_predictions_{B,C}.jsonl`, which `rescore_pointwise.py` still reads today |
| `score_abc.py` | 7.3 KB | 08-03 00:26 | The A-vs-B-vs-C comparison you explicitly requested | **Yes** |
| `score_pilot.py` | 8.6 KB | 08-02 18:43 | Early pilot scorer, largely superseded by `score_abc.py` | Yes, lower value |
| `prove_remap.py` | 2.9 KB | 08-02 23:01 | Proved the vision-tower LoRA key-remap fix before it was written into `ml/engine.py` | **Yes** — the evidence trail behind a now-shipped fix |
| `batch_probe.py` | 3.6 KB | 08-03 13:38 | Probed whether batched inference changes throughput/answers — informed `run200.py`'s design | Yes, informational |
| `run200.py` | 6.0 KB | 08-03 13:49 | Earlier copy of the 200-frame runner | **No** — an in-project copy exists at `outputs/benchmark/run200.py` and is the newer, final version (17:28 vs 13:49) |
| `__pycache__/*.pyc` (3 files) | ~35 KB | — | Compiled bytecode | No — zero source value |

**Recommendation: ARCHIVE the 10 unique `.py` files** (everything above except the
duplicate `run200.py` and `__pycache__`) into the project, e.g. under a new
`outputs/benchmark/provenance/` folder, so the full chain — how the test set was staged,
how the ceiling/EF gate were calibrated, how A/B/C were run and compared, how the adapter
fix was proven — survives outside a temp directory that OS/tooling housekeeping could
clear at any time. **No move performed in this pass** — this is a recommendation only.

---

## 2. Claude Code's own project-state directory (infrastructure, not project artifacts)

`C:\Users\sheha\.claude\projects\C--Users-sheha-Desktop-The-Adimension-ATRIA-Platform\`

| Item | Type | Size | Tied to ATRIA? | Unique content? | Recommendation |
|---|---|---|---|---|---|
| `memory\*.md` (7 files) + `MEMORY.md` | memory files | ~11 KB total | yes | yes — the persistent-memory records this and prior sessions wrote | **KEEP as-is** — this is the memory system's own storage location by design, not a project artifact to relocate |
| `a560eeb4-2788-44cc-a159-dd3198615fd8.jsonl` | conversation transcript | 14.1 MB | yes — the session that did the original benchmark/pilot work | yes, technically (full action-by-action record) | **KEEP as-is** — Claude Code's own session log; not meant for manual archival into a code repo |
| `314d83c3-d158-4cea-a41e-58e6b0c37134.jsonl` | conversation transcript | 8.1 MB | yes — the current session | yes | **KEEP as-is**, same reasoning |
| `a560eeb4-…/`, `314d83c3-…/` (subfolders) | internal session state | not deep-scanned | yes | uncertain — not opened, per "read/list only, don't exhaustively mine" | **UNKNOWN / leave alone** — internal harness state, not source material |

---

## 3. `%TEMP%` root

Only one ATRIA-named item found: `atria_inspect\` — **empty**, directory timestamp
**2026-05-13**, six weeks before any dated material inside this project (the earliest
project file dates are late June 2026). Almost certainly unrelated leftover from something
else entirely, or a since-cleared probe from long before this engagement.

| Path | Type | Size | Modified | Tied to ATRIA? | Recommendation |
|---|---|---|---|---|---|
| `%TEMP%\atria_inspect\` | empty dir | 0 | 2026-05-13 | **no** (predates this project) | DELETE (empty, zero risk) or ignore — your call, no data either way |

`%LOCALAPPDATA%\Temp` is the same location as `%TEMP%` on this machine — not scanned twice.

---

## 4. `AppData\Local` and `AppData\Roaming`, top level

No entries matching `atria`/`echotrace` other than the already-covered `.claude` tree.
Nothing further found.

---

## 5. Desktop and Downloads — found, but explicitly out of scope

Your Phase 2b instructions scope Desktop/Downloads to items "clearly named for this
project **and** referenced by existing logs/scripts inside the project." A name-filtered
scan found a substantial amount of pre-existing ATRIA/EchoTrace material on both:

- **Desktop**: 5 files — two Product Discovery Sheet drafts (`.docx`, including
  "(Repaired)" copies) and their Word lock files.
- **Downloads**: ~50 files — marketing videos and audio (`ATRIA-EchoTrace-*.mp4`,
  `EchoTrace-HITL-*.mp4/.gif`, narration `.m4a`), multiple dated/numbered Colab notebook
  exports (`atria_g3_t4x2*.ipynb`, up to 5 numbered revisions), a PDF, and an HTML export.

**None of these are referenced by any script, log, or doc inside the project** — the
project already carries its own copies of the source material that matters
(`notebook_as_py.txt`, `Notebook-readme.txt`, `ATRIA-EchoTrace-FT-HF-MedGemma.txt` at the
project root). These Desktop/Downloads files read as your own pre-existing research and
product-marketing material sitting in its normal location, not as artifacts this — or any
— Claude Code session produced. Per your own scoping rule and the standing instruction not
to perform broad home-directory cleanup unrelated to this project:

**Recommendation: no action, not evaluated further, out of scope for this audit.** Listed
here only because the phase instructions required checking; full individual file listing
omitted as it would exceed what was asked.

---

## 6. Summary table for CLEANUP_PLAN.md

| Path | Recommendation |
|---|---|
| `…\Temp\claude\…\a560eeb4-…\scratchpad\{stage_testset,calibrate,ef_gate,gate,pilot,pilot_fixed,run_bc,score_abc,score_pilot,prove_remap,batch_probe}.py` | **ARCHIVE-INTO-PROJECT** |
| `…\Temp\claude\…\a560eeb4-…\scratchpad\run200.py`, `__pycache__\` | DELETE (duplicate / bytecode, zero value) |
| `.claude\projects\…\memory\`, `*.jsonl` | KEEP as-is (infrastructure) |
| `%TEMP%\atria_inspect\` | DELETE (empty) or ignore |
| Desktop/Downloads ATRIA material | out of scope, no action |
