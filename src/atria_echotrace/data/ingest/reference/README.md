# Reference preprocessors — vendored verbatim, do not edit

These two files are the **actual scripts that produced the corpus the published LoRA
adapters were fine-tuned on**. They are copied here byte-for-byte from
`datasets/data_processing_scripts/` and must not be modified: any edit breaks the
guarantee that `atria ingest` reproduces the training data exactly.

| File | SHA-256 (first 16) | Size |
|---|---|---|
| `preprocess_camus.py` | `9653f13287d4a3e6…` | 14 729 bytes |
| `preprocess_echonet.py` | `837ebebb8187e01e…` | 31 082 bytes |

Verified: re-running `preprocess_camus.extract_contour_polygon` on
`patient0001_4CH_ED` reproduces that entry in the shipped `unified_processed/tracings.json`
identically. `tests/test_ingest.py` asserts this whenever the real data is present.

## Why vendored rather than reimplemented

The application originally reimplemented this preprocessing from the published dataset
formats. That reimplementation was close but not identical (Dice 0.994 against the real
output) and was wrong in ways that mattered: it read MetaImage rather than the NIfTI
that CAMUS actually ships, invented splits instead of reading
`database_split/subgroup_*.txt`, and looked for an `LVef` config key that does not
exist. Using the real scripts removes the entire class of divergence.

See [RESEARCH.md §8](../../../../../RESEARCH.md) for the full lineage and the
measurements behind that decision.

## Entry points used by `atria ingest`

- `preprocess_camus.preprocess_camus(camus_root, output_dir, num_points, lv_label, la_label)`
- `preprocess_echonet.preprocess_echonet(echonet_root, output_dir, num_points, max_videos, target_size)`
- `preprocess_echonet.merge_datasets(camus_processed, echonet_processed, unified_dir, log)`

## Notes for callers

- `preprocess_echonet` imports OpenCV at module scope, so it is only importable with the
  `ingest` extra installed. `preprocess_camus` needs only SimpleITK.
- Both call `logging.basicConfig(...)`, which reconfigures root logging. That is
  acceptable in a one-shot CLI, and is why the server never imports these modules.
- Both write a `preprocessing_log.txt` into their output directory.
