# Data workflow — acquire, ingest, classify, train

The chain from a third-party download to a fine-tuned adapter, and the guarantees at each step.

```
   YOU ACQUIRE                    ATRIA PRODUCES
┌────────────────┐
│ CAMUS_public   │──┐
│ (CREATIS/Lyon) │  │  atria ingest camus     ┌──────────────────┐
└────────────────┘  ├────────────────────────▶│ camus_processed  │──┐
┌────────────────┐  │                         │  2,000 frames    │  │  merge_datasets()
│ EchoNet-Dynamic│──┘  atria ingest echonet   └──────────────────┘  ├──▶ unified_processed
│ (Stanford AIMI)│─────────────────────────── ┌──────────────────┐  │    22,048 frames
└────────────────┘                            │ echonet_processed│──┘    11,024 cases
                                              │ 20,048 frames    │
                                              └──────────────────┘
                                                       │
                    atria classify ◀───────────────────┤
                    classified_datasets/               │
                                                       ▼
                                            atria train  →  LoRA adapter
                                                       │
                                            atria evaluate  →  evidence/
                                                       │
                                            atria serve  →  clinician corrects
                                                       │
                                            atria export-corpus  →  back to the top
```

## Step 0 — acquire

Neither dataset is ours to redistribute. Register, accept the licence, download, and place the
files exactly as [`datasets/README.md`](../datasets/README.md) documents. Nothing else in this
pipeline works until that layout exists.

## Step 1 — ingest

`atria ingest` runs the **vendored reference preprocessors** at
[`src/atria_echotrace/data/ingest/reference/`](../src/atria_echotrace/data/ingest/reference/README.md),
copied byte-for-byte from `datasets/data_processing_scripts/` and marked *do not edit*.

Why vendored rather than reimplemented: the application originally reimplemented this preprocessing
from the published dataset formats. That reimplementation scored Dice 0.994 against the real
output — close, and wrong in ways that mattered. It read MetaImage rather than the NIfTI CAMUS
actually ships, invented splits instead of reading `database_split/subgroup_*.txt`, and looked for
an `LVef` config key that does not exist. Using the real scripts removes the entire class of
divergence.

`tests/test_ingest.py` asserts source and vendored copy remain byte-identical, and that a fresh
3-patient CAMUS ingest reproduces all 12 shipped entries exactly — splits and EF included.

### The EchoNet frame-index derivation

EchoNet publishes no ED/ES frame indices. Each video has exactly two traced frames in
`VolumeTracings.csv`; the preprocessor takes the **lower frame number as `esf`** and the **higher
as `edf`**, writing them into a copy of `FileList.csv` and preserving the original as
`original_FileList.csv`. This reproduces the shipped corpus for **20,048 / 20,048 cases (100%)**.

## Step 2 — the contract

Ingest output, and anything you write yourself, must satisfy:

```
<dataset>/frames/<stem>.png      one RGB frame per stem, any resolution
<dataset>/tracings.json          authoritative
<dataset>/metadata.csv           optional; split wins over tracings
<dataset>/manifest.json          optional; provenance, checksums, case pairing
```

Polygons are `[[y, x]]` normalised to `[0, 1000]` — the model's own output format, so no conversion
sits between training data and inference. **No code assumes a vertex count, winding direction or
explicit closure**, because CAMUS and EchoNet differ on all three.

The resampling is **index subsampling, not arc length**: `np.linspace(0, len(contour)-1, 30,
dtype=int)` over `find_contours` output. Because `linspace` includes both endpoints of a closed
ring, the closing vertex is duplicated — which is exactly why CAMUS polygons satisfy
`p[0] == p[29]` and EchoNet's do not.

## Step 3 — classify (optional)

Derives classification-task datasets from that same corpus. Two products per task: a mapping CSV,
or an ImageFolder tree.

**CAMUS quality is per view, not per patient** — the two views disagree in 208 of 500 patients
(41.6%). Any pipeline attaching quality to the patient record mislabels two frames in five before
training begins.

## Step 4 — train, evaluate, correct, retrain

See [training.md](training.md), [evaluation.md](evaluation.md) and
[adapting-to-your-data.md](adapting-to-your-data.md).

## Findings surfaced, not corrected

**EchoNet ED/ES labels are transposed in 9,922 of 10,024 cases (99.0%).** The platform flags this
at every layer and **deliberately does not correct it**: relabelling at inference would push every
request outside the distribution the adapter was fitted to. CAMUS shows 0 / 1,000 transposed.

**Physical areas in cm² are withheld when pixel spacing is unknown** rather than computed from a
placeholder. EchoNet ships no spacing; CAMUS ships a real 0.308 mm/px.
