# Competitive Analysis — the HAI-DEF Showcase field

Based on direct reading of the [showcase index](https://developers.google.com/health-ai-developer-foundations/showcase),
the [Visilant case study](https://developers.google.com/health-ai-developer-foundations/showcase/visilant),
and the [MedGemma Impact Challenge winners](https://blog.google/innovation-and-ai/technology/health/med-gemma-impact-challenge/).

---

## 1. How the showcase is organised

Four content tiers, in descending prestige:

| Tier | Count | Examples |
|---|---|---|
| **Real-world clinical applications** | 4 | Visilant, Crane AI Labs, Nimblemind, AskCPG by Qmed Asia |
| Technical solutions and tools | 2 | — |
| Community prototypes | 9 | Impact Challenge winners |
| First-party demo applications | 7 | Google's own demos |

**Target tier 1.** You have a working platform, a peer-reviewed framework, real datasets
and a held-out benchmark. Tier 3 would undersell it. But be aware that tier 1's current
occupants all have *deployment*, which you do not — so you must give Google a different
reason to place you there. That reason is methodological rigour (see §4).

## 2. The Visilant template, section by section

The structure to copy, with measured lengths:

| Section | ~Words | What it does |
|---|---|---|
| Title: *"How [Org] is [action], built with [tech]"* | — | Names the organisation and the outcome |
| **Problem** | 200 | Global statistic → root cause → why prior AI fell short |
| **The Solution: [Product]** | 180 | Names the product; describes the workflow; mentions expert-in-the-loop and safety guardrails |
| **How open models from HAI-DEF helped** | 150 | Contrasts old approach (multiple CNNs) with fine-tuned MedGemma; cites dataset size; gives one accuracy metric |
| **Real-world impact and adoption** | 120 | Executive quote, named partners, patient volume |
| **Next steps** | 80 | Forward momentum — RCT, regulatory path, scale |

**Tonal rules observed:** clinically credible but accessible; mission-driven vocabulary
("underserved", "treatable"); metrics presented big-to-small (1 billion affected → 200,000
images → 50,000 screened → 95%+ accuracy); human oversight woven through rather than
appended; **no explicit call-to-action** — it closes on momentum.

`01_CASE_STUDY_DRAFT.md` follows this exactly.

## 3. What Google actually rewards

From the Impact Challenge winners — EpiCast (disease surveillance in West Africa), Sunny
(private skin self-screening), FieldScreen AI (on-device TB screening), Tracer (catching
diagnostic discrepancies):

| Signal | Evidence |
|---|---|
| **Access for the underserved** | Nearly every winner targets a resource-constrained setting |
| **Offline / on-device** | An explicit specialty award category |
| **Novel-task fine-tuning** | Also an explicit specialty award category ← **your lane** |
| **Structured output from unstructured input** | EpiCast (free text → WHO IDSR signals); Tracer (notes → hypotheses) ← **exactly your pattern** |
| **Human oversight** | Universal, in every entry |

**Two of Google's own award categories — novel-task fine-tuning and structured output —
describe ATRIA EchoTrace precisely.** Use their vocabulary.

## 4. The gap in the field, and why it is yours

Reading all four tier-1 entries together, a consistent absence emerges:

- **No entry publishes a held-out benchmark** with a named split and per-frame data.
- **No entry names a metric limitation** or explains why one metric was chosen over another.
- **No entry reports a negative finding** or a bug in the open-model tooling.
- **No entry has a peer-reviewed conceptual framework** behind it.
- Google's own index page shows **no metrics at all** in the item descriptions.

The showcase is, today, a collection of *access and deployment* stories. It has no
*methodology* story.

> **Strategic conclusion:** do not compete on reach. Compete on rigour, and make the entry
> useful to other HAI-DEF developers. An entry that hands the community a reusable warning
> (the silent adapter-loading failure) and a worked example of choosing the right metric is
> something Google can point other developers at — which is what a *developer foundations*
> programme actually needs.

## 5. Head-to-head positioning

| | Visilant | ATRIA EchoTrace |
|---|---|---|
| Clinical domain | Ophthalmology, front-of-eye | Cardiology, echocardiography |
| Scale | 50,000 patients | 200 held-out frames, 50 patients |
| Validation | RCT initiating | Held-out benchmark, published metric rationale |
| Output | Triage report | **Editable coordinate structure** |
| Human role | Reviews the output | **Corrections become training data** |
| Framework | Clinical guardrails | **Peer-reviewed DEITY framework** |
| Transparency | Metrics stated | **Limitations and a negative finding published** |

**Do not claim parity on scale.** Claim a different axis, and be explicit that you are
early: "200 frames, 50 patients, research use only" said plainly is more credible than any
hedge — and it is exactly the honesty Google's responsible-AI framing rewards.

## 6. Submission mechanics

- **Form:** <https://services.google.com/fb/forms/hai-def-showcase/> (linked twice from the
  showcase index, for different submission types).
- **Existing relationship:** your forum thread already drew a reply from **Fereshteh Mahvar
  at Google**. Reference that thread in the submission — it converts a cold form into a
  continuation of an open conversation.
- **Implicit expectations** inferred from the published entries: a named organisation, a
  one-line description, a thumbnail image, a "Read More" body in the six-section structure,
  and at least one concrete number.

---

## Sources

- [HAI-DEF Showcase index](https://developers.google.com/health-ai-developer-foundations/showcase)
- [Visilant case study](https://developers.google.com/health-ai-developer-foundations/showcase/visilant)
- [MedGemma Impact Challenge winners](https://blog.google/innovation-and-ai/technology/health/med-gemma-impact-challenge/)
- [HAI-DEF overview](https://developers.google.com/health-ai-developer-foundations) · [MedGemma](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [ATRIA EchoTrace forum thread](https://discuss.ai.google.dev/t/atria-echotrace-fine-tuning-medgemma-1-5-for-polygon-based-heart-structure-contouring/172907)
