# Limitations

Published rather than buried. Each was found by looking at outputs, and each matters to anyone
deciding whether to build on this.

## One polygon in eight crosses itself

**25 of 200 (12.5%)** generated polygons self-intersect — a structural-validity failure a
downstream area or volume calculation would silently mishandle. This was not surfaced anywhere
until the metrics lock.

## Two-chamber views are weaker

**5.45 mm** against 4.76 mm for four-chamber. On the only matched multi-frame Dice run the gap is
starker: **0.887 on 4CH** against **0.583 / 0.546 on 2CH**.

## The worst case is confidently wrong

At 18.59 mm the model draws a **well-formed ventricle in the wrong place** — correct point count,
no self-intersection, displaced by roughly a chamber width. It did not produce noise; it
confidently traced the wrong structure. No scalar metric expresses that. It is the clearest
argument for mandatory human review, and it is only visible as a picture.

## Dice figures are small-n

The most defensible is **0.672 ± 0.153 at n = 3**. Do not quote 0.8924 — that is n = 1 on a
cherry-picked view.

## Inference is not interactive

Median **86 s/frame**. A proposal generator for review, not a real-time tool.

## EchoNet labels are transposed

End-diastole and end-systole are swapped in **9,922 of 10,024 cases (99.0%)**. The platform
**flags this and deliberately does not correct it** — relabelling at inference would push every
request outside the distribution the adapter was fitted to.

## Physical units are withheld, not invented

Areas in cm² are reported only when pixel spacing is known. CAMUS ships a real 0.308 mm/px;
EchoNet ships none, and a placeholder of 1.0 means *unknown* rather than *one millimetre*.

## Never claimed

- Any EF or volume accuracy — the adapter emits coordinates, not volumes
- Any left-atrium result — the published adapters are **LV-only**
- A full-split (2,752-frame) benchmark — never run
- "Docker verified" — the compose files are authored, never executed
- That `atria train` reproduces the published adapters — it ports the notebook recipe, which differs
- "Clinically validated" or "production-ready" — neither is true

---

**Research use only. Not a cleared or approved medical device.** Every contour is a proposal
requiring review by a qualified echocardiographer. No clinical decision should rest on its output.
