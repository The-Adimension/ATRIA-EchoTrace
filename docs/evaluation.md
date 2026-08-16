# Evaluation

Primary evaluation: the **official CAMUS test split — 50 patients, 200 frames, never trained on**.
Quality mix 94 Good · 76 Medium · 30 Poor. Config: 4-bit NF4 + double-quant, bf16, training-matched
prompt. Raw data: [`evidence/benchmark/`](../evidence/benchmark/).

## Match the metric to the output

This adapter was **first** scored with Dice, IoU and Simpson-biplane ejection fraction — the
standard segmentation battery. That was the wrong instrument. The model emits *coordinates*, so
region overlap measures a downstream rasterisation the model never performed.

It was re-scored with **point-to-curve distance**: the shortest Euclidean distance from each
predicted point to the expert **polyline** (piecewise-linear, edge projection with the parameter
clipped to `[0,1]`) — not distance to the nearest vertex, and not rasterised mask-boundary
distance. The implementation was validated against five known-answer geometric cases before use.

## Primary results

| | |
|---|---|
| Output parsed to a valid polygon | **200 / 200** |
| Coordinates in bounds | **200 / 200** |
| **Median point-to-curve** | **4.98 mm** (mean 5.97) |
| Interquartile range | 3.37 – 7.52 mm |
| Best / worst frame | 1.20 / 18.59 mm |
| Within one point of the 30-point target | **178 / 200** (89%) |
| Inference latency | median 86 s/frame |

## Stratified (per-frame medians)

| Stratum | Median | n | Stratum | Median | n |
|---|---|---|---|---|---|
| Good | **4.61 mm** | 94 | 4CH | 4.76 mm | 100 |
| Medium | **6.40 mm** | 76 | 2CH | 5.45 mm | 100 |
| Poor | **6.79 mm** | 30 | ED / ES | 4.74 / 5.49 mm | 100 |

Accuracy degrades **monotonically with acquisition quality** — the behaviour a clinician expects
from a system genuinely reading the image, and stronger evidence of a sane model than the headline
number alone.

## Standard segmentation metrics (n = 200)

Against the original expert mask from `_gt.nii.gz`:

| Metric | Mean ± SD | Median | Range |
|---|---|---|---|
| **Dice** | **0.744 ± 0.135** | 0.777 | 0.263 – 0.940 |
| **IoU** | **0.609 ± 0.156** | 0.635 | 0.151 – 0.886 |
| MAD | 6.22 ± 3.08 mm | 5.42 | 1.50 – 16.80 |
| HD95 | 13.11 ± 5.88 mm | 11.71 | 4.00 – 34.03 |
| NSD @ 2 mm | 0.227 ± 0.157 | 0.202 | 0.000 – 0.670 |

Dice by view: 4CH 0.765 ± 0.113 · 2CH 0.723 ± 0.152. By quality: Good 0.782 · Medium 0.701 ·
Poor 0.735 — note this is *not* monotonic, unlike the point-to-curve figures, and the Poor
subgroup is small (n = 30). Reported as measured rather than smoothed.

## The representation is not the bottleneck

Representing an expert mask as 30 index-uniform points costs **Dice 0.9905** and **0.214 mm** MAD.
Near-lossless. A Dice of 0.744 is therefore a property of the model, not the output format.

## Why we do not lead with Dice 0.8924

That figure — and 0.7026 for EchoNet — are **n = 1 smoke values** printed by a regression test, and
the CAMUS one is a four-chamber frame, the best-performing view. They are observations, not
accuracy. The most defensible multi-frame Dice available is **0.672 ± 0.153 at n = 3**.

## Every frame was rendered and reviewed

All 200 predictions were drawn at native resolution and inspected — expert reference in green,
prediction in red, predicted vertices marked. That review is where the real limitations surfaced;
see [limitations.md](limitations.md). The full gallery ships in
[`showcase/gallery/`](../showcase/gallery/).

## Reproduce it

```bash
python tools/benchmark/run200.py               # inference to per-frame JSONL
python tools/benchmark/score200.py             # Dice / IoU / MAD / HD95 / NSD
python tools/benchmark/rescore_pointwise.py    # the primary metric
python tools/benchmark/make_overlays.py        # 200 overlays, CPU-only, no GPU replay
```
