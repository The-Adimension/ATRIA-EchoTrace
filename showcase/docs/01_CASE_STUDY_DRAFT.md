# Case Study Draft — submission-ready

Structured to mirror the [Visilant showcase](https://developers.google.com/health-ai-developer-foundations/showcase/visilant),
which is the closest competitive reference: Problem → Solution → How open models from
HAI-DEF helped → Real-world impact → Next steps. Section lengths match theirs (~200 / 180 /
150 / 120 / 80 words) so it drops into Google's page template without editing.

Two variants of the title are given; **Option A is recommended.**

---

## Title

**Option A (recommended)**
> **How The Adimension is teaching MedGemma to trace the heart, built with HAI-DEF**

**Option B (method-forward)**
> **How ATRIA EchoTrace turns cardiac contouring into structured coordinate generation with MedGemma**

### One-sentence positioning

> ATRIA EchoTrace fine-tunes MedGemma 1.5 4B to emit the left-ventricular endocardial
> border as editable JSON coordinates rather than an opaque mask — making every AI proposal
> something a cardiologist can read, correct, and feed back as training data.

---

## The problem

Tracing the endocardial border is the single most repeated manual act in
echocardiography. Every ejection fraction, every volume, every strain measurement begins
with a clinician drawing a line around a chamber that is moving, noisy, and often only
partly visible. It is slow, it varies between observers, and it varies within the same
observer on different days.

Automated segmentation has been attempted for years, and it usually returns a **mask** —
a filled region of pixels. A mask is difficult to correct: a clinician who disagrees with
one part of the boundary cannot easily nudge it, and the disagreement is not captured in
any form the model can learn from. The correction is lost.

There is a second, quieter problem. Cardiac image quality is not uniform. In the CAMUS
dataset, image quality is graded per acquisition window, and **it differs between the
2-chamber and 4-chamber views in 208 of 500 patients** — 41.6%. Any system that treats
quality as a property of the patient rather than of the individual acquisition is
mislabelling two frames in five before it starts.

## The solution: ATRIA EchoTrace

ATRIA EchoTrace reframes the task. Instead of predicting a region, the model predicts
**an ordered list of 30 boundary points as JSON** — `[[y, x], …]`, normalised to a
`[0, 1000]` grid so the contract is resolution-independent.

That single change makes the output *interpretable structure*. A clinician sees the
proposed contour with its vertices exposed, drags any point that is wrong, and saves. The
saved revision records both polygons — the model's proposal and the human's correction —
and `atria export-corpus` turns those corrections into the same three-artifact training
contract the original corpus uses. **The clinician's correction becomes the next
generation's ground truth.** The loop closes.

The platform runs as a single local process: a FastAPI backend serving both a JSON API
and a buildless browser workstation, with dual end-diastole / end-systole canvases,
vertex-level editing, live chamber metrics, and export. Weights resolve locally first, so
it runs fully offline.

## How open models from HAI-DEF helped

MedGemma 1.5 4B made the reframing possible. A conventional segmentation network has no
mechanism to emit JSON; a general vision-language model does, but lacks the medical
grounding. MedGemma provided both — and being open-weight meant it could be adapted where
the data lives.

Adaptation used **QLoRA (4-bit NF4, bfloat16 compute) with LoRA rank 32**, applied not
only to the language tower but across the **vision tower's attention and MLP layers** —
the visual encoder has to learn what an endocardial border looks like, not just how to
format an answer. Training ran on a single pre-Ampere consumer GPU.

Two findings from that work are offered back to the HAI-DEF community:

**bfloat16 is mandatory for Gemma-family models.** Substituting float16 — the obvious
choice on a card without native bf16 support — makes MedGemma emit nothing but `<pad>`
tokens. Measured, not theorised.

**A PEFT adapter can load partially, in silence.** These adapters were trained when Gemma 3
nested the vision encoder as `vision_tower.vision_model.encoder`; transformers 5.x
flattened it to `vision_tower.encoder`. **324 of the checkpoint's 802 tensors therefore
addressed a module path that no longer existed.** PEFT emits a non-fatal warning and
continues, leaving every vision-tower `lora_B` at its zero initialisation — roughly **40%
of the adapter contributing nothing**, while the model still loaded and produced
plausible contours. Any HAI-DEF developer who fine-tunes a vision tower and later upgrades
transformers can hit this. It is now detected, repaired at load time, and guarded by a
regression test that fails if it returns.

## Real-world results

Evaluated on the **official CAMUS test split — 50 patients, 200 frames, never seen in
training** (quality mix: 94 Good, 76 Medium, 30 Poor):

| Measure | Result |
|---|---|
| Outputs parsed to a valid polygon | **200 / 200** |
| Median point-to-curve distance to the expert tracing | **4.98 mm** (mean 5.97) |
| Point count | median **31**; 178/200 (89%) land on 30–31 points |
| Best / worst frame | 1.20 mm / 18.59 mm |
| Accuracy by acquisition quality | 4.61 mm (Good) · 6.40 (Medium) · 6.79 (Poor) |

The metric matters as much as the number. This model was **first** scored with Dice, IoU
and Simpson-biplane ejection fraction — the standard segmentation battery. That was the
wrong instrument: the adapter emits coordinates, not masks or volumes, and region overlap
measures a downstream construction rather than the model's actual output. It was re-scored
with **point-to-curve distance** — the shortest distance from each predicted point to the
expert curve — which is what "did the points land on the tracing?" actually means.

Every one of the 200 frames was also **rendered for visual inspection** (reference in
green, prediction in red, predicted vertices marked) and reviewed. That review is where
the honest limitations surface: the worst case is not noise but a **well-formed ventricle
traced in the wrong place** — correct point count, no self-intersection, confidently
wrong. A scalar Dice score hides that failure mode completely; a picture does not.

Three further limitations are published rather than buried. Accuracy tracks acquisition
quality exactly as a clinician would expect — **4.61 mm on Good studies, 6.40 on Medium,
6.79 on Poor** — and 2-chamber views are weaker than 4-chamber (5.45 vs 4.76 mm). **25 of
the 200 generated polygons (12.5%) self-intersect**, a structural-validity failure that a
downstream area calculation would silently mishandle. On the only matched multi-frame
Dice evaluation the adapter scored **0.887 on 4-chamber and 0.583 / 0.546 on
2-chamber** — a view gap invisible in any single headline number. And in EchoNet-Dynamic,
end-diastole and end-systole labels are transposed in **99% of cases**; the platform
**flags this and deliberately does not correct it**, because relabelling at inference
would push every request outside the distribution the adapter was fitted to.

## Built on the DEITY Principles

ATRIA EchoTrace is the reference implementation of the **DEITY Principles Framework**
(Anwer, *EHJ-IMP* 2026) — Data, Ethics, Informatics, Technology, You. The framework is not
a label applied afterwards; each principle forced a decision that can be checked:

- **Data** — the original preprocessing scripts are vendored byte-identically and re-run to
  reproduce the training corpus exactly. Provenance is executable.
- **Ethics** — the EchoNet label transposition is surfaced and flagged at every layer, never
  silently repaired. Physical areas in cm² are **withheld** when pixel spacing is unknown
  rather than computed from a placeholder.
- **Informatics** — the output is human-readable structure a clinician can diff and edit.
- **Technology** — 4-bit quantisation, one consumer GPU, offline-capable, local-first.
- **You** — the correction is the product. Human judgement enters the training set by
  design, not by exception.

## Next steps

ATRIA EchoTrace is **research software and explicitly not a cleared or approved medical
device**; every contour is a proposal requiring review by a qualified echocardiographer.
The immediate priorities are extending validation beyond CAMUS to the full EchoNet split,
closing the 2-chamber performance gap, and retraining on clinician-corrected contours to
test whether the human-in-the-loop cycle measurably improves the model.

**The Adimension welcomes collaboration** with echocardiography laboratories willing to
contribute corrected tracings, and with HAI-DEF developers working on structured-output
adaptation of MedGemma. The platform, the benchmark harness, and the provenance scripts
are open.

---

## Notes for the submitter (delete before sending)

- **Do not add the 0.8924 Dice figure.** It is n = 1 on a single 4-chamber frame. If a
  Dice number is requested, give **0.672 ± 0.153 (n = 3, matched)** and label the 0.8924 a
  single-frame smoke value.
- The QLoRA description above states rank 32, NF4, bf16 — **confirm against the gated
  `adapter_config.json` before submitting** (extraction plan in `04_SUBMISSION_CHECKLIST.md`).
  Epochs / batch / learning rate are deliberately omitted here because the published
  adapters were trained with different values than the notebook default; do not quote them
  until read from the checkpoint.
- Keep "research use only" in the body, not only in a footer. Visilant earned clinical
  language with an RCT; this entry earns credibility with candour.
