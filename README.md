# ATRIA EchoTrace

[![Live showcase](https://img.shields.io/badge/showcase-atria.theadimension.com-0b6bcb?style=flat-square)](https://atria.theadimension.com/)
[![Mirror](https://img.shields.io/badge/mirror-github.io-5b6675?style=flat-square)](https://the-adimension.github.io/ATRIA-EchoTrace/)
[![CI](https://github.com/The-Adimension/ATRIA-EchoTrace/actions/workflows/ci.yml/badge.svg)](https://github.com/The-Adimension/ATRIA-EchoTrace/actions/workflows/ci.yml)
[![Deploy](https://github.com/The-Adimension/ATRIA-EchoTrace/actions/workflows/pages.yml/badge.svg)](https://github.com/The-Adimension/ATRIA-EchoTrace/actions/workflows/pages.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-1a7f45?style=flat-square)](LICENSE)
[![Base model](https://img.shields.io/badge/base-MedGemma%201.5%204B-c02b2b?style=flat-square)](https://huggingface.co/google/medgemma-1.5-4b-it)
[![Colab](https://img.shields.io/badge/notebook-open%20in%20Colab-f9ab00?style=flat-square)](https://colab.research.google.com/github/The-Adimension/ATRIA-EchoTrace/blob/main/ATRIA-EchoTrace_GoogleColab-CompleteNotebook_20260715.ipynb)
[![Research use only](https://img.shields.io/badge/research%20use%20only-not%20a%20medical%20device-9a6400?style=flat-square)](docs/limitations.md)

**Fine-tuning MedGemma 1.5 4B to trace the heart as editable coordinates, not an opaque mask.**

A cardiologist cannot argue with a segmentation mask. ATRIA EchoTrace adapts
[MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it) with QLoRA so it emits the
left-ventricular endocardial border as an **ordered JSON list of 30 boundary points**, normalised
to a `[0, 1000]` grid. That single change makes the output something a clinician can read, drag,
correct — and hand back as the next generation's training label.

> **ATRIA** — Artifact Transformation & Resource Interoperability in Artificial Intelligence
> **EchoTrace** — tracing the echocardiographic frame, point by point

**Research use only. Not a cleared or approved medical device.** Every contour is a proposal
requiring review by a qualified echocardiographer.

---

## Results

Evaluated on the **official CAMUS test split — 50 patients, 200 frames, never seen in training.**

| | |
|---|---|
| Outputs parsed to a valid polygon | **200 / 200** |
| Median point-to-curve distance | **4.98 mm** (IQR 3.37–7.52) |
| Dice / IoU vs the expert mask | **0.744 ± 0.135** / 0.609 ± 0.156 |
| Within one point of the 30-point target | **178 / 200** |
| Accuracy by acquisition quality | 4.61 mm Good · 6.40 Medium · 6.79 Poor |
| Training cost, shipped CAMUS adapter | **128 minutes**, one GPU |

Known limitations are published rather than buried: **25 of 200 polygons (12.5%) self-intersect**,
two-chamber views are weaker than four-chamber (5.45 vs 4.76 mm), and the worst case is a
well-formed ventricle traced in the wrong place. See [docs/limitations.md](docs/limitations.md).

### See it

**[→ atria.theadimension.com](https://atria.theadimension.com/)** — the full showcase: the
interactive correction editor, all 200 held-out predictions rankable best-to-worst, the six-run
training record, and three narrated videos.

Also mirrored at **[the-adimension.github.io/ATRIA-EchoTrace](https://the-adimension.github.io/ATRIA-EchoTrace/)**,
and served from [`showcase/`](showcase/) in this repository — the same static bundle, deployable
to any host.

---

## Quick start

Runs with no weights and no downloads — the bundled 50-frame sample is a byte-exact subset of the
held-out test split.

```bash
git clone https://github.com/The-Adimension/ATRIA-EchoTrace.git
cd ATRIA-EchoTrace
./run.sh                # Windows: run.cmd
```

That opens the workstation on the sample dataset. To add local inference and training:

```bash
pip install -e ".[ai]"
atria doctor            # checks Python, torch, CUDA, and where weights resolve from
```

Weights are gated and resolve locally first — see [docs/reproducing.md](docs/reproducing.md).

---

## The pipeline

```
acquire → ingest → classify → train → evaluate → trace & revise → export → retrain
```

Every stage is independently enterable from the CLI, and each produces artifacts the next
consumes unchanged:

```bash
atria --dataset-dir <dir> ingest        # raw downloads → the three-artifact contract
atria --dataset-dir <dir> classify      # derive classification sets
atria --dataset-dir <dir> train --output-dir ./my-adapter
atria --dataset-dir <dir> evaluate --adapter ./my-adapter
atria --dataset-dir <dir> serve         # the workstation
atria --dataset-dir <dir> export-corpus # corrections become training data
```

> `--dataset-dir` is a **global** option and must precede the subcommand.

Full walkthrough: [docs/data-workflow.md](docs/data-workflow.md) ·
[docs/adapting-to-your-data.md](docs/adapting-to-your-data.md)

---

## Three ways in

| | |
|---|---|
| **Read it** | [`ATRIA-EchoTrace_GoogleColab-CompleteNotebook_20260715.ipynb`](ATRIA-EchoTrace_GoogleColab-CompleteNotebook_20260715.ipynb) — the research notebook the platform was derived from |
| **Run it** | [Open in Colab](https://colab.research.google.com/github/The-Adimension/ATRIA-EchoTrace/blob/main/ATRIA-EchoTrace_GoogleColab-CompleteNotebook_20260715.ipynb) — executes without gated weights |
| **Operate it** | The `atria` CLI and the packaged workstation, above |

---

## Repository layout

```
src/atria_echotrace/   the application — API, workstation, ML engine, ingest, export
tests/                 282 tests
datasets/              the acquisition contract: scripts ship, data never does
sample-dataset/        50 held-out frames, bundled so the app runs immediately
tools/                 the harness that produced the evidence
evidence/              the CSVs and configs behind every number published here
showcase/              the built showcase page (also served via GitHub Pages)
docs/                  architecture, training, evaluation, limitations, audits
docker/                review tier, CPU inference, GPU inference
```

**`datasets/` is a contract, not a payload.** CAMUS and EchoNet-Dynamic are third-party datasets
under their own licences — you acquire them, place them where
[`datasets/README.md`](datasets/README.md) documents, and the vendored preprocessors reproduce the
training corpus exactly.

---

## What is offered back

Two findings that cost us time and may save yours:

**A PEFT adapter can load partially, in silence.** These adapters were trained when Gemma 3 nested
the vision encoder as `vision_tower.vision_model.encoder`; transformers 5.x flattened it to
`vision_tower.encoder`. **324 of the checkpoint's 802 tensors addressed a path that no longer
existed.** PEFT warns and continues, leaving every vision-tower `lora_B` at zero — roughly 40% of
the fine-tuning inert while the model still produced plausible contours. Now detected, repaired at
load, and guarded by `tests/test_adapter_completeness.py`.

**bfloat16 is mandatory for Gemma-family models.** float16 makes MedGemma emit nothing but `<pad>`
tokens. Measured, not theorised.

---

## Citing

See [CITATION.cff](CITATION.cff). The adapters carry DOIs
[10.57967/hf/9541](https://doi.org/10.57967/hf/9541) (CAMUS) and
[10.57967/hf/9540](https://doi.org/10.57967/hf/9540) (EchoNet), both Apache-2.0 and gated.

Built on the **DEITY Principles Framework** (Anwer, *EHJ Imaging Methods & Practice* 2026) —
Data, Ethics, Informatics, Technology, You.

**Datasets:** CAMUS (Leclerc et al., *IEEE TMI* 2019) · EchoNet-Dynamic (Ouyang et al., *Nature*
2020), used under their respective terms.

---

**The Adimension** — Shehab Anwer, MD, Founder.
We welcome collaboration with echocardiography laboratories willing to contribute corrected
tracings, and with HAI-DEF developers working on structured-output adaptation of MedGemma.

*To AïA, your legacy continues to inspire every step of this journey.*
