# classified_datasets/ — written by `atria classify`

Empty by design. Nothing in this directory is tracked (~1.2 GB when populated).

```
camus_quality/       Good · Medium · Poor          graded PER VIEW, not per patient
camus_ef_5pct/       ejection fraction, 5% bins
echonet_ef_5pct/     ejection fraction, 5% bins
```

Two products per task: a **mapping CSV** (`metadata` variant) and an **ImageFolder tree**
(`dirs` variant). Scripts: [`../classification_scripts/`](../classification_scripts/).
