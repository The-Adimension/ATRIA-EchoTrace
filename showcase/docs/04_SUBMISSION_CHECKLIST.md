# Submission Checklist, Media Package & Next Actions

Form: <https://services.google.com/fb/forms/hai-def-showcase/>

---

## PART A — Do these before submitting

### 1. 🔴 Update the public GitHub README (highest leverage, ~1 hour)

It currently says *"No specific performance metrics… are stated"* and *"Prototype /
Advanced MVP"*. Both are now false. Minimum edits:

- [ ] Add the 200-frame benchmark table (200/200 parsed, median 4.98 mm, 50 patients)
- [ ] Add the adapter-repair finding (324/802 tensors) — it is a community contribution
- [ ] Update status from "Prototype / Advanced MVP" to reflect the platform's real state
- [ ] Use the current functional stage names: Preprocess · Classify · Train · Evaluate ·
      Trace & Revise · Export Corpus · Publish Adapter
- [ ] State the license explicitly (the README currently does not)
- [ ] Link the DEITY paper DOI

### 2. 🟡 Extract the exact published-adapter hyperparameters

The case study deliberately omits epochs / batch / learning rate, because the published
adapters were **not** trained with the notebook defaults. Do not guess. Extract them:

```bash
# The adapters are already local in this repo
python - <<'PY'
import json, pathlib
for name in ("atria-echotrace-camus", "atria-echotrace-echonet"):
    p = pathlib.Path("adapters") / name / "adapter_config.json"
    if p.is_file():
        c = json.loads(p.read_text())
        print(name, {k: c.get(k) for k in
              ("r", "lora_alpha", "lora_dropout", "target_modules", "modules_to_save")})
PY
```

For epochs / batch / LR, read `training_args.bin` (a torch pickle) or the tensorboard
metrics in `metadata.json` from the same folder. **`BENCHMARK_PLAN.md §1` already records
values derived this way — reconcile against the checkpoint before publishing any number.**

- [ ] Confirm LoRA rank / alpha / dropout / target modules
- [ ] Confirm epochs, batch, learning rate, max_length
- [ ] Decide whether to publish them (recommended: yes — it aids reproduction)

### 3. 🟡 Confirm the assets I could not reach

- [ ] **Hugging Face Space** — returned HTTP 401 to my fetch. Is it public and running? A
      live demo is the single most valuable non-gated asset for a reviewer
- [ ] **Colab notebook** — confirm it still executes end-to-end
- [ ] **Supply URLs** for the X/Twitter threads, LinkedIn articles and YouTube videos
      referenced in your brief — none were publicly indexed in my searches
- [ ] Confirm the DEITY citation year: OUP shows advance publication 2025, PMC/issue 2026.
      Your citation (2026, 4(1), qyaf038) matches the issue — just be consistent

### 4. 🟢 Prepare the gated-adapter explanation

One sentence, in the submission:

> The adapters are gated to keep research artifacts traceable and ensure users acknowledge
> the research-only intended use; access requests are reviewed, and both carry DOIs
> (10.57967/hf/9541, 10.57967/hf/9540) and an Apache-2.0 license.

- [ ] Confirm gating is *review-and-grant*, not closed — reviewers must be able to get in

---

## PART B — Media package

### Priority 1 — the assets that carry the argument

| Asset | Where | Why it wins |
|---|---|---|
| **Overlay gallery** | `outputs/benchmark/full200/overlays/index.html` | 200 real predictions, filterable, best/worst. **Export the best-5 and worst-5 as a strip** — including a bad one is the credibility move |
| **Benchmark table** | From `pointwise.csv` | The headline numbers |
| **4-panel comparison figure** | `render/figures.py` output | Original / model / clinician / overlay — the notebook's own visual, ideal as the page thumbnail |
| **HITL editing clip** | Your existing videos | Shows a clinician dragging a vertex — this is **You** in DEITY, made visible in 10 seconds |

### Priority 2 — supporting

- Stages panel screenshot (shows the full lifecycle and live readiness)
- Architecture diagram: raw data → preprocess → classify → train → evaluate → trace & revise → export corpus → publish
- DEITY graphical abstract from the EHJ-IMP paper
- Colab link (non-gated entry point)

### Suggested thumbnail
The **4-panel comparison figure**, cropped to model-vs-clinician. It communicates the
entire value proposition — AI proposes, human corrects — without a caption.

---

## PART C — What to put in each form field

| Field | Content |
|---|---|
| Organisation | The Adimension |
| Contact | Shehab Anwer, MD — Founder |
| Application name | ATRIA EchoTrace |
| One-line description | Fine-tuned MedGemma 1.5 that traces the left-ventricular endocardial border as editable JSON coordinates, with clinician corrections feeding back as training data |
| HAI-DEF model used | MedGemma 1.5 4B (`google/medgemma-1.5-4b-it`), QLoRA-adapted |
| Category | Real-world clinical application — *or* Technical solution. **Mention the novel-task fine-tuning and structured-output angles explicitly**, both are Google's own award categories |
| Body | `01_CASE_STUDY_DRAFT.md` |
| Links | GitHub · HF collection · both adapter DOIs · Colab · Space · forum thread · DEITY DOI |
| Metrics | 200/200 parsed · median 4.98 mm · 50 held-out patients · 22,048-frame corpus |
| Responsible AI | Research use only; not a cleared device; mandatory clinician review; limitations published (2CH gap, EchoNet transposition, worst-case failure mode) |

---

## PART D — Ordered next actions

| # | Action | Effort | Why now |
|---|---|---|---|
| 1 | Update the public GitHub README | 1 h | A reviewer will read it; the mismatch is the biggest own-goal risk |
| 2 | Extract and confirm adapter hyperparameters | 30 min | Needed before any training claim is published |
| 3 | Verify Space + Colab are live | 20 min | Non-gated reviewer paths |
| 4 | Export best-5 / worst-5 overlay strip + 4-panel figure | 30 min | The visual argument |
| 5 | Submit the form with `01_CASE_STUDY_DRAFT.md` | 30 min | — |
| 6 | Post the adapter-loading defect to the HAI-DEF forum as a technical note | 1 h | Community contribution; makes saying yes easy; reactivates the Mahvar thread |
| 7 | *(Optional, later)* Run the full EchoNet split benchmark | GPU hours | Extends validation beyond CAMUS |

**Do not gate the submission on item 7.** Submit on the CAMUS benchmark you have.

---

## PART E — Risk register

| Risk | Mitigation |
|---|---|
| Quoting **Dice 0.8924** as accuracy | It is n=1 on the best-performing view. Always pair with n=3 (0.672 ± 0.153) and label it a single-frame smoke value |
| Implying clinical readiness | Keep "research use only, not a cleared device" in the body text |
| Gated adapters read as closed | State the reason and confirm requests are granted |
| README/claims mismatch | Item 1 |
| Over-claiming the platform | Describe it as a research platform demonstrating the method, not a deployed product |
| Publishing unverified hyperparameters | Item 2 — extract from the checkpoint, do not quote the notebook defaults |
| LA (left atrium) claims | Adapters are **LV-only**. Do not present LA results |
