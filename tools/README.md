# tools/ — the harness that produced the evidence

These are the scripts that generated everything in [`../evidence/`](../evidence/). They are kept
so the numbers can be re-derived rather than taken on trust.

They write to `outputs/`, which is gitignored. Compare results against `evidence/`.

```
benchmark/    the 200-frame evaluation
training/     TensorBoard extraction and run reports
provenance/   archival — the one-off investigations, kept for audit
```

## benchmark/

| Script | Does |
|---|---|
| `run200.py` | Inference over the held-out split → per-frame JSONL, flushed as it goes |
| `score200.py` | Dice / IoU / MAD / HD95 / NSD against the expert mask |
| `rescore_pointwise.py` | **Point-to-curve distance — the primary metric** |
| `make_overlays.py` | Renders all 200 overlays from persisted predictions, CPU-only |

Run them in that order. `run200.py` needs the AI tier and the gated weights; the rest work from
the persisted JSONL, so re-scoring and re-rendering cost **no GPU replay**.

`rescore_pointwise.py` implements point-to-curve as the shortest distance to the expert
*polyline* — edge projection with the parameter clipped to `[0,1]` — validated against five
known-answer geometric cases before use. It is not distance to the nearest vertex.

## training/

| Script | Does |
|---|---|
| `extract_tb_logs.py` | TensorBoard event files → per-step CSV and JSON |
| `generate_run_reports.py` | Loss curves, token accuracy, LR schedule, grad-norm/entropy, throughput, dashboard |

## provenance/ — archival, not maintained

Eleven one-off scripts from the investigations that established the results: configuration probes,
calibration, EF gating, the pilot runs, and `prove_remap.py`, which demonstrated the adapter
key-remap defect.

**These are kept for audit, not for reuse.** They encode paths and assumptions from the sessions
that produced them and are not expected to run unmodified. If you want to reproduce a result, use
`benchmark/` and `training/` above.
