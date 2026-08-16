# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-08-16

First public release of the full platform. The repository previously held only the research
notebook; this release adds the packaged application, the benchmark evidence, and the showcase.

### Added

- **The application** — `src/atria_echotrace/`: FastAPI backend, buildless browser workstation,
  ML engine, ingest, export. Runs offline; weights resolve locally first.
- **282 tests** — including `test_ingest.py` (asserts the vendored preprocessors stay byte-identical
  to their source) and `test_adapter_completeness.py` (fails if adapter tensors silently unbind).
- **The dataset contract** — `datasets/` ships structure, scripts and documentation; the raw and
  derived data never enter version control. See `datasets/README.md`.
- **`evidence/`** — the CSVs, JSONL and adapter configs behind every published number.
- **`tools/`** — the benchmark and training-log harness that produced that evidence, plus the
  archival provenance scripts.
- **`showcase/`** — the built showcase page, including the interactive correction editor, the
  200-frame specimen ladder, the six-run training record and three narrated videos.
- **`docs/`** — architecture, data workflow, training, evaluation, limitations, reproducing,
  adapting to your own data, and the audit trail.

### Results published

- Official CAMUS test split, 50 patients / 200 frames, never trained on:
  **200/200 parsed**, median point-to-curve **4.98 mm** (IQR 3.37–7.52),
  Dice **0.744 ± 0.135**, IoU 0.609 ± 0.156.
- Six fine-tuning runs across two open datasets, with the learning-rate ablation that produced the
  published recipe. Shipped CAMUS adapter: **128 minutes** on one GPU.

### Limitations surfaced

- **25/200 (12.5%)** generated polygons self-intersect.
- Two-chamber views weaker than four-chamber (5.45 vs 4.76 mm).
- EchoNet ED/ES labels transposed in 99.0% of cases — flagged, deliberately not corrected.
- Dice figures beyond n = 200 are small-n; 0.8924 is an n = 1 smoke value and is not headlined.

### Contributed to the community

- **A PEFT adapter can load partially, in silence** — 324 of 802 checkpoint tensors addressed a
  module path renamed in transformers 5.x, leaving ~40% of the fine-tuning inert with no error.
  Detection and repair pattern documented in `docs/architecture.md`.
- **bfloat16 is mandatory for Gemma-family models** — float16 yields `<pad>`-only output.

### Notes

- The README previously described the project as "Prototype / Advanced MVP" with no performance
  metrics. That is superseded by the evidence above.
- Research use only. Not a cleared or approved medical device.
