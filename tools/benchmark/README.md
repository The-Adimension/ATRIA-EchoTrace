# outputs/benchmark

`pilot20_fixed/` supersedes `pilot20/`; current scorers read `pilot20_fixed` only.

`pilot20/` has been moved to `.superseded_runs/pilot20/` — it is the pre-fix pilot, kept
for provenance. `rescore_pointwise.py`'s `RUNS` dict reads `pilot20_fixed` exclusively.
