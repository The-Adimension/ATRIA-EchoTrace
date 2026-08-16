# processed_datasets/ — written by `atria ingest`

Empty by design. Nothing in this directory is tracked.

```
camus_processed/     2,000 frames   LV + LA, real 0.308 mm/px spacing
echonet_processed/  20,048 frames   LV only, spacing unknown
unified_processed/  22,048 frames / 11,024 cases   ← what training consumes
```

Each holds the three-artifact contract: `frames/`, `tracings.json`, `metadata.csv`
(+ optional `manifest.json`). See [`../README.md`](../README.md) §4.

Produced by the vendored reference preprocessors, which reproduce the published training corpus
exactly rather than approximating it.
