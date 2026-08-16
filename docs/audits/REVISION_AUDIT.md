# REVISION AUDIT — ATRIA EchoTrace vs the original assignment

Scope: documentation and classification only. No new features, no benchmark expansion, no
re-training, no refactors proposed or performed in this phase.

---

## 1. Original deliverable checklist vs current state

Source: the original assignment (notebook → production, multi-platform, click-and-run) and
`DELIVERY.md §1`, cross-checked against the filesystem in this pass rather than trusted
blind.

| Deliverable | Docs claim | Verified this pass |
|---|---|---|
| Clean `src/` package, installable, `atria` CLI, 7 subcommands | delivered | **confirmed** — `pyproject.toml` `where=["src"]`, `cli.py` 512 lines, 7 subcommands present |
| Backend implementing every notebook phase | delivered | **confirmed at the module level** — every one of the 39 files under `src/atria_echotrace/` carries a docstring citing exact `notebook_as_py.txt` line ranges (CODE_INVENTORY.md §1) |
| Buildless SPA workstation | delivered | **confirmed present** — 13 files under `web/`, vendored Preact/htm/fonts, no `node_modules`, no build step |
| Config, logging, error handling, upload/path hardening | delivered | **confirmed** — `config.py::display_path()`, `data/frames.py` traversal guards, `api/revisions.py` path-traversal guard all present and reviewed |
| Cross-platform launchers + Docker | launchers verified, Docker authored only | **unchanged** — Docker still unexecuted (not installed on this machine), consistent with DELIVERY §4 |
| Docs: README, notebook-phase mapping, data lineage | delivered | **confirmed present**, all cross-reference each other consistently |
| Test suite over real data/weights | delivered, 282 tests | **not re-run this pass** (out of scope — this phase is inventory, not verification); file-level presence of all 14 test files confirmed |
| Runs anywhere: local-first weights, no absolute paths | delivered, verified on a fresh copy | unchanged from DELIVERY.md's own claim; not independently re-verified this pass |
| Platform composition (5 stages, enter at any point) | delivered | **confirmed** — Stages panel wiring present in `web/js/app.js`, `api/meta.py`, `api/deps.py::ingest_available` |

**Everything in this table matches what the project's own authoritative docs already
claimed.** This audit did not find a deliverable that regressed.

---

## 2. Complete and verified

Everything in §1, plus — **not yet reflected in DELIVERY.md/HANDOVER.md, but real and
complete as of this session**:

- The **vision-tower LoRA key-remap fix** (`ml/engine.py::repair_legacy_adapter_keys`),
  regression-tested by `tests/test_adapter_completeness.py`.
- **GAP_ANALYSIS Round 3, N1–N5, all closed** — corpus export precedence, evaluation
  lifecycle test coverage, dtype-policy docstring, canvas accessibility.
- **A 200-frame benchmark against the adapter's actual task.** DELIVERY.md §2 currently
  states *"No full-split benchmark exists … Nothing on this page is a benchmark"* — that
  was true when written (2026-08-02) and is **no longer the full picture**: this session
  ran all 200 frames of the official CAMUS test split (50 patients, LV, config A2) and
  re-scored them against the adapter's real contract — structured `[y,x]` coordinate
  generation, not segmentation — using point-to-polyline distance in mm as the primary
  metric. Results live in `outputs/benchmark/full200/` (`pointwise.csv`,
  `overlays/manifest.csv`, `overlays/index.html`, 200 rendered QA overlays, 0 failures).
  This is real, complete, more rigorous than anything currently quoted in DELIVERY.md, and
  **currently invisible to a reader who only reads the canonical docs**.

---

## 3. Partial, messy, or misleading in the tree

Ranked by how much a future reader (including a future session) could be misled:

1. **DELIVERY.md/HANDOVER.md's accuracy section is now stale, not wrong.** They correctly
   disclose n=1/n=3 as smoke values and correctly say no full-split benchmark exists — but
   a 200-frame benchmark now exists and isn't mentioned. A reader following HANDOVER's own
   "read these before touching anything" instruction would not discover it. This is the
   single highest-value gap this audit found.
2. **Two scoring philosophies sit side by side in `outputs/benchmark/` with no in-tree
   signal of which is authoritative.** `score200.py` (Dice/IoU/HD95/EF, vs both
   reference standards) and `rescore_pointwise.py` (point-to-curve, the corrected framing)
   both read the same `raw_predictions.jsonl` and both look equally canonical to a cold
   reader. `score200.py` is not wrong — it was reframed as measuring "downstream
   constructions" rather than the adapter's actual task — but nothing in the file itself
   says so.
3. **`PLAN.md §3`'s file-structure diagram under-represents the shipped package by one
   entry.** It lists `ml/` as "runtime, prompts, engine, datasets, train, evaluate,
   publish" — `ml/reference/camus_ef.py` (added this session, vendored CAMUS EF code) is
   absent from the diagram even though it is genuinely present under `src/` today.
4. **`code-chat/Session1.docx`** sits in the project root, 575 KB, undocumented by any
   canonical doc, not gitignored, not classified as either live or superseded. Harmless on
   its own, but it's the one file in the tree this audit could not classify with evidence.
5. **The top-level `atria_echotrace/` prototype is 5.04 GB**, almost all of it
   (5.02 GB) its own abandoned `venv/`. The docs correctly call the *code* dead; none of
   them mention that 99.6% of that path's size is a venv, which is the more actionable
   fact for anyone deciding whether to reclaim disk space.

Nothing above is a code defect. All five are documentation/organization gaps.

---

## 4. Intentionally out of scope (unchanged — confirmed, not re-litigated)

Directly from `DELIVERY.md §4` and `README.md "Not built"`, re-confirmed present and
unchanged by this audit: authentication/multi-user/database, PACS/DICOM-network
integration, video/temporal modelling (RV, valves, strain), training sweeps/distributed
training, active learning, Kubernetes/CI/telemetry, client-side build tooling, versioned/
appendable corpus, browser-driven training/evaluation/publishing, arbitrary LoRA path in
the UI, upload resizing, correcting the EchoNet ED/ES transposition, unifying
per-stage quantisation, and Docker execution (not installed on this machine). LA contour
tracing remains out of scope per your own confirmation earlier in this engagement — the
adapter was tuned on LV only.

These are not gaps. They are recorded decisions with stated reasons, and this audit found
no evidence any of them should be revisited.

---

## 5. Minimal revision items (documentation-only — no code, no new evaluation)

Ordered by value, each independently approvable:

1. **Add a short section to `DELIVERY.md` (or `HANDOVER.md §5`) pointing to the 200-frame
   point-to-curve benchmark** — location (`outputs/benchmark/full200/`), headline numbers,
   and the corrected task framing (structured coordinate generation, not segmentation).
   *~15 lines, no code touched.*
2. **One-line docstring note in `score200.py`** stating its Dice/IoU/EF framing was
   superseded by `rescore_pointwise.py` per the corrected task understanding, so a cold
   reader doesn't mistake it for the current primary evaluation. *1 line.*
3. **Add `ml/reference/` to `PLAN.md §3`'s file-structure diagram.** *1 line.*
4. **Resolve `code-chat/Session1.docx`'s status** — this needs you, not more inspection: is
   it worth keeping (and if so, what is it, so it can be documented) or is it stray. See
   CLEANUP_PLAN.md.
5. **Archive the 10 unique scratchpad scripts** identified in `EXTERNAL_ARTIFACTS.md §1`
   into the project (e.g. `outputs/benchmark/provenance/`) so the calibration/gate/pilot
   provenance behind the current results survives outside a temp directory.

None of these touch `src/atria_echotrace/`, the vendored preprocessors, the inference
path, or any test. All five are either pure documentation or file relocation.
