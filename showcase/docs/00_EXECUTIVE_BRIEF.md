# Executive Brief — Seizing the HAI-DEF Showcase Opportunity

**To:** Shehab Anwer, MD — Founder, The Adimension
**Re:** Google HAI-DEF Showcase acceptance — strategy and content package
**Date:** 2026-08-13

---

## 1. The one-line strategy

**Lead with the thing none of the current showcase entries have: a published, reproducible
benchmark of a novel task, plus the honest account of a bug that would have inflated it.**

Every existing HAI-DEF case study leads with reach (50,000 patients screened, rural
deployment, offline operation). You cannot win on deployment scale today, and you should
not try. You can win on something Google's own showcase page is visibly missing:
**methodological rigour on an open model, documented end to end.**

## 2. What changed since your forum post — and why it is the whole story

On the Google AI Developers Forum (28 June 2026) you wrote, correctly and candidly:

> *"we have not yet published formal quantitative benchmarks (Dice, IoU, Hausdorff
> distance, etc.)"*

**That gap is now closed.** The platform in this repository has since produced a complete
200-frame evaluation against the official CAMUS test split, and — more importantly —
produced it *twice*, because the first framing was wrong and was corrected.

That correction is your strongest narrative asset, and most teams would hide it:

1. The adapter was first scored with **Dice/IoU and Simpson-biplane ejection fraction** —
   the standard segmentation playbook.
2. That was the wrong instrument. The adapter does not emit a mask or an EF. It emits a
   **JSON list of `[y, x]` coordinates**. Scoring coordinate generation with a
   region-overlap metric measures a downstream construction, not the model.
3. It was re-scored with **point-to-curve distance in millimetres** — the metric that
   actually answers "do the predicted points land on the expert tracing?"

**Result: median 4.98 mm point-to-curve, 200/200 outputs parsed, on 50 held-out patients.**

Alongside that, a second finding is arguably more valuable to the HAI-DEF *community* than
to you: **40% of the published adapter was silently inert.** A PEFT checkpoint trained when
Gemma 3 nested the vision encoder as `vision_tower.vision_model.encoder` addressed a module
path that transformers 5.x had flattened to `vision_tower.encoder`. **324 of 802 tensors**
loaded into nothing. PEFT emitted a non-fatal warning and continued; every vision-tower
`lora_B` stayed at its zero initialisation. The model loaded, ran, and produced
plausible-looking contours — while nearly half the fine-tuning contributed nothing.

**This is a trap any HAI-DEF developer fine-tuning a vision tower across a transformers
major version can hit, and it fails silently.** Publishing it is a genuine contribution to
Google's ecosystem, and it is exactly the kind of thing that gets a showcase entry
remembered rather than skimmed.

## 3. What to emphasise

| Emphasise | Why |
|---|---|
| **Novel task**: contour generation as *structured coordinate output*, not segmentation | No other showcase entry does this. It is a genuinely different use of a VLM |
| **The 200-frame benchmark** with the corrected metric | Closes your own stated gap; nothing else in the showcase publishes a held-out benchmark this explicitly |
| **The silent adapter-loading defect** | A reusable warning for the whole HAI-DEF community; demonstrates engineering seriousness |
| **The HITL loop that closes**: corrections become training data (`export-corpus`) | Directly instantiates **You** in DEITY; most "human-in-the-loop" claims stop at review |
| **DEITY as the operating discipline**, not a badge | You have a peer-reviewed framework; show it *governing* engineering decisions |
| **Honest negative findings** (EchoNet ED/ES transposition; 2CH weaker than 4CH) | Credibility. Google's page currently shows no limitations anywhere |

## 4. What to de-emphasise — firmly

| De-emphasise | Why |
|---|---|
| **Dice 0.8924** | It is **n = 1**, a single 4CH frame — the best-performing view. The only matched multi-frame run is **n = 3, Dice 0.672 ± 0.153**. Quoting 0.8924 as accuracy is the single biggest credibility risk in the whole submission |
| Any hint of clinical readiness | Research use only. Visilant earned its clinical language with an RCT; you have not run one and should not imply otherwise |
| "Production-grade platform" as a headline | True of the software, but it invites "deployed where? on whom?" — which you cannot yet answer. Lead with method, mention the platform as evidence of rigour |
| EF / volume estimates | Correctly reframed as out-of-scope for the adapter's actual task. Do not reintroduce them |
| LA (left atrium) results | The published adapters are **LV-only**. LA is selectable but out of distribution |

## 5. The DEITY framing — use it structurally, not decoratively

Your paper defines DEITY as Data · Ethics · Informatics · Technology · You. The
temptation is to add a paragraph saying "we follow DEITY." **Do the opposite**: show each
principle forcing a concrete, checkable engineering decision. That is what makes it a
framework rather than a slogan.

| Principle | The engineering decision it forced |
|---|---|
| **Data** | The original preprocessing scripts are vendored byte-identically and re-run to reproduce the training corpus exactly. Provenance is executable, not asserted |
| **Ethics** | ED/ES transposition affecting ~91% of training frames was **surfaced and flagged, never silently corrected** — correcting it at inference would push every request out of the adapter's training distribution |
| **Informatics** | The output *is* interpretable structure: JSON coordinates a human can read, edit and diff — not an opaque mask |
| **Technology** | 4-bit NF4 on one consumer GPU; local-first weights; runs fully offline |
| **You** | The clinician's correction becomes the label. `export-corpus` turns revisions into the same contract training consumes — the loop closes |

## 6. Positioning against the current field

| Entry | Their strength | Your differentiator |
|---|---|---|
| **Visilant** | 50,000 patients, RCT starting, 95%+ screening accuracy | You cannot match reach. You *can* match rigour and exceed transparency |
| **Crane AI Labs** | Offline, rural Uganda | Different axis entirely |
| **Impact Challenge winners** (EpiCast, Sunny, FieldScreen) | Resource-constrained settings, offline, local language | These won on *access*. You win on *method* |

**The gap in Google's showcase is a methodology-and-rigour entry.** Every current item is
an access/deployment story. None publishes a held-out benchmark, a corrected metric, or a
negative finding. Occupy that space — it is unoccupied and it is where a cardiologist with
a peer-reviewed framework and a reproducible platform is strongest.

## 7. Three risks, and how to handle each

1. **The n=1 Dice number is already public** in earlier materials. Do not delete it —
   relabel it everywhere as a *single-frame smoke value* and put the n=3 and 200-frame
   numbers next to it. Being the person who corrected their own headline number is a
   stronger position than being the person who quietly dropped it.
2. **The public GitHub README understates the work.** It still says *"Prototype / Advanced
   MVP"* and *"no specific performance metrics."* If a Google reviewer reads it before the
   submission, the gap between it and your claims looks careless. **Update it first** — see
   `04_SUBMISSION_CHECKLIST.md`, item 1.
3. **The adapters are gated.** That is defensible and correct for research artifacts, but
   say *why* in one sentence, and make sure the reviewer has a non-gated path to
   understanding the work (Colab, the Space, the overlay gallery).

## 8. What I recommend you do, in order

1. **Update the public GitHub README** so it matches reality (benchmark, stage names, the
   adapter-repair finding). One hour, highest leverage.
2. **Submit the case study** in `01_CASE_STUDY_DRAFT.md` via
   <https://services.google.com/fb/forms/hai-def-showcase/>.
3. **Offer Google the adapter-loading defect as a separate technical note.** Post it to the
   HAI-DEF forum thread you already have open. It is a community contribution and it makes
   the showcase entry easy for Google to say yes to.
4. **Do not wait for the RCT or clinical deployment.** Your entry is a *methods* entry.
   Submit it as such, now, while MedGemma 1.5 fine-tuning is current.

---

### One honest caveat on this brief

Every metric in this package was verified by running the code in this repository — the
verification log is in `05_VERIFICATION_LOG.md`. The three items I could **not** verify are
flagged there: the exact published-adapter hyperparameters (the checkpoints are gated and
must be read locally), the current state of your X/LinkedIn/YouTube assets (not publicly
indexed in my searches), and the live status of the Hugging Face Space (returned HTTP 401).
Fill those three in before submitting.
