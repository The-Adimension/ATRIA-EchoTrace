# Reproducing

## Install

```bash
git clone https://github.com/The-Adimension/ATRIA-EchoTrace.git
cd ATRIA-EchoTrace
pip install -e .            # review tier: no weights, no GPU
pip install -e ".[ai]"      # + local inference and training
pip install -e ".[ingest]"  # + SimpleITK / OpenCV for raw dataset preprocessing
atria doctor                # verifies Python, torch, CUDA, and weight resolution
```

## Weights

Both adapters are **gated** on Hugging Face — access requests are reviewed, which keeps research
artifacts traceable and ensures users acknowledge the research-only intended use.

| | |
|---|---|
| Base model | [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it) |
| CAMUS adapter | [`The-Adimension/EchoTrace-MedGemma-CAMUS`](https://huggingface.co/The-Adimension/EchoTrace-MedGemma-CAMUS) · DOI 10.57967/hf/9541 |
| EchoNet adapter | [`The-Adimension/EchoTrace-MedGemma-EchoNet`](https://huggingface.co/The-Adimension/EchoTrace-MedGemma-EchoNet) · DOI 10.57967/hf/9540 |

Place them under `models/` and `adapters/` (both gitignored). The application looks there first
and falls back to the Hugging Face cache.

**A CPU-only torch wheel is the most common failure.** `atria doctor` prints the torch build; if
it reads `+cpu` with `CUDA None`, reinstall from the CUDA index. "GPU not detected" is almost
never a hardware problem.

## Tests

```bash
pytest                      # 282 tests
pytest -m "not real_data"   # skip tests needing the full corpus or weights
```

`tests/test_ingest.py` asserts the vendored preprocessors are byte-identical to
`datasets/data_processing_scripts/`, and — when the real CAMUS data is present — that a fresh
3-patient ingest reproduces all 12 shipped entries exactly, splits and EF included.

## Reproducing the corpus

See [data-workflow.md](data-workflow.md). Acquire CAMUS and EchoNet yourself, then:

```bash
atria ingest camus   --source <camus>   --output datasets/processed_datasets/camus_processed
atria ingest echonet --source <echonet> --output datasets/processed_datasets/echonet_processed
atria ingest unified --camus-processed   datasets/processed_datasets/camus_processed \
                     --echonet-processed datasets/processed_datasets/echonet_processed \
                     --output            datasets/processed_datasets/unified_processed
```

## Reproducing the benchmark

```bash
python tools/benchmark/run200.py
python tools/benchmark/score200.py
python tools/benchmark/rescore_pointwise.py
python tools/benchmark/make_overlays.py
```

Outputs land in `outputs/` (gitignored). The published copies live in
[`evidence/benchmark/`](../evidence/benchmark/) — compare against those.

The harness flushes per-frame JSONL as it goes, so an interrupt costs one frame rather than the
run. Median inference is 86 s/frame, so a full 200-frame benchmark is a multi-hour job.

## Reproducing the training curves

```bash
python tools/training/extract_tb_logs.py        # TensorBoard events → per-step CSV
python tools/training/generate_run_reports.py   # loss, token accuracy, LR, throughput plots
```

Extracted metrics for all six runs are already in
[`evidence/training/`](../evidence/training/).

## What is not reproducible

`atria train` **ports** the notebook recipe; it is not a byte-exact replay of the Colab runs that
produced the published adapters. It reproduces the method, not the weights. The exact artifacts
come from Hugging Face.

Wall-clock figures are HuggingFace's recorded `train_runtime` and include evaluation and
checkpointing. The GPU model is **not** a recorded field — folder names say "blackwell", but that
is a naming convention, not evidence.
