# datasets/ — the acquisition contract

**This directory ships structure and scripts. It never ships data.**

CAMUS and EchoNet-Dynamic are third-party datasets released under their own terms. They are not
The Adimension's to redistribute. You acquire them yourself, place them where this file documents,
and the vendored preprocessors turn them into the corpus the published adapters were trained on.

```
datasets/
├── data_processing_scripts/     ← ships    the real preprocessors
├── classification_scripts/      ← ships    classification-set derivation
├── original_datasets_and_repos/ ← YOU PLACE the raw downloads here
├── processed_datasets/          ← WRITTEN  by `atria ingest`
└── classified_datasets/         ← WRITTEN  by `atria classify`
```

The last three are empty in a fresh clone (a `.gitkeep` holds the shape) and are excluded by
`.gitignore`. That is deliberate: the repository tells you where things go; your machine holds
them.

---

## 1. Acquire

### CAMUS

Cardiac Acquisitions for Multi-structure Ultrasound Segmentation — 500 patients, 2CH and 4CH,
end-diastole and end-systole, with expert LV and LA tracings and real pixel spacing.

- Source: <https://www.creatis.insa-lyon.fr/Challenge/camus/>
- Register and accept the licence with CREATIS / Université de Lyon.
- Citation: Leclerc et al., *IEEE Transactions on Medical Imaging*, 2019.

Place it so this path exists:

```
datasets/original_datasets_and_repos/camus_public/
├── database_nifti/        patient0001 … patient0500  (.nii.gz)
└── database_split/        subgroup_training.txt · subgroup_validation.txt · subgroup_testing.txt
```

The **official 400 / 50 / 50 split is read from `database_split/`** — it is never invented. Every
accuracy figure published in this repository is measured on `subgroup_testing.txt`.

### EchoNet-Dynamic

10,031 apical-4-chamber videos with two traced frames each.

- Source: <https://echonet.github.io/dynamic/>
- Register with Stanford AIMI and accept the research use agreement.
- Citation: Ouyang et al., *Nature*, 2020.

Place it so this path exists:

```
datasets/original_datasets_and_repos/echonet_dynamic/
├── Videos/                 *.avi
├── FileList.csv
└── VolumeTracings.csv
```

**On `FileList.csv`:** use the file exactly as Stanford ships it. EchoNet publishes no ED/ES frame
indices, so the preprocessor derives them from `VolumeTracings.csv` (lower frame number → `esf`,
higher → `edf`) and writes them into a copy, preserving your original as
`original_FileList.csv`. This reproduces the shipped corpus for **20,048 / 20,048** cases. Do not
substitute a `FileList.csv` that already carries `esf`/`edf` columns from elsewhere — the
derivation is what makes the result reproducible.

---

## 2. Ingest

```bash
pip install -e ".[ingest]"        # SimpleITK for CAMUS NIfTI, OpenCV for EchoNet AVI

atria ingest camus   --source datasets/original_datasets_and_repos/camus_public \
                     --output datasets/processed_datasets/camus_processed
atria ingest echonet --source datasets/original_datasets_and_repos/echonet_dynamic \
                     --output datasets/processed_datasets/echonet_processed
atria ingest unified --camus-processed   datasets/processed_datasets/camus_processed \
                     --echonet-processed datasets/processed_datasets/echonet_processed \
                     --output            datasets/processed_datasets/unified_processed
```

Yielding **22,048 frames / 11,024 cases** — CAMUS 2,000 (2CH+4CH, LV+LA, 0.308 mm/px) and
EchoNet 20,048 (4CH, LV only, spacing unknown).

These commands run the **actual scripts that produced the adapters' training corpus**, vendored
byte-for-byte at [`src/atria_echotrace/data/ingest/reference/`](../src/atria_echotrace/data/ingest/reference/README.md)
and marked *do not edit*. Their output is identical to that corpus rather than an approximation of
it — asserted by `tests/test_ingest.py` whenever the real data is present.

---

## 3. Classify (optional)

```bash
atria --dataset-dir datasets/processed_datasets/unified_processed classify camus-quality
atria --dataset-dir datasets/processed_datasets/unified_processed classify camus-ef
atria --dataset-dir datasets/processed_datasets/unified_processed classify echonet-ef
```

Or run the standalone scripts in `classification_scripts/` directly. Each task has two forms — a
`metadata` variant writing a mapping CSV, and a `dirs` variant building an ImageFolder tree.

**CAMUS image quality is graded per *view*, not per patient.** The two views disagree in
**208 of 500 patients (41.6%)**, so attaching quality to the patient record mislabels two frames
in five. The scripts here grade per view.

---

## 4. The dataset contract

Any preprocessor producing these artifacts drives the whole application — this is what makes
"bring your own data" real rather than aspirational:

```
<dataset>/frames/<stem>.png      one RGB frame per stem, any resolution
<dataset>/tracings.json          {stem: {...}}   authoritative
<dataset>/metadata.csv           optional: key,split,…   (split wins over tracings)
<dataset>/manifest.json          optional: provenance, checksums, case pairing
```

Each `tracings.json` entry:

```jsonc
{
  "patient_id": "patient0258", "view": "4CH", "instant": "ED",
  "image_h": 552, "image_w": 669,
  "spacing_h": 0.308, "spacing_w": 0.308,   // mm/px; 1.0 means "unknown"
  "lv_polygon": [[631, 479], …],            // [[y, x]] normalised to [0, 1000]
  "la_polygon": null,                       // null when unavailable
  "split": "test", "source": "camus",
  "ef": 64.06                               // optional
}
```

**No code assumes a vertex count, a winding direction, or an explicitly closed contour** — the two
datasets differ on all three, and the application handles both conventions alike.

---

## What ships, and what does not

| | Tracked | Why |
|---|---|---|
| `data_processing_scripts/*.py` | **yes** | Ours. The provenance of the corpus |
| `classification_scripts/**/*.py` | **yes** | Ours |
| `original_datasets_and_repos/` | no | Third-party, licence-gated, tens of GB |
| `processed_datasets/` | no | Derived from third-party data |
| `classified_datasets/` | no | Derived again (~1.2 GB) |

`sample-dataset/` at the repository root is the one exception — 50 frames, a byte-exact subset of
the **held-out test split**, small enough to bundle so the application runs with zero downloads.
