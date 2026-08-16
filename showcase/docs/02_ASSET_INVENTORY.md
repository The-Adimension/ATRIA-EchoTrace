# Complete Inventory of Public and Local Assets

Everything found in research, plus the local artifacts that are the strongest evidence.
Status column: 🟢 verified this session · 🟡 exists, needs your check · 🔴 gap to fix.

---

## 1. Public assets — found and confirmed

| Asset | URL | What it gives the showcase | Status |
|---|---|---|---|
| **Google AI Developers Forum thread** | [discuss.ai.google.dev/t/…/172907](https://discuss.ai.google.dev/t/atria-echotrace-fine-tuning-medgemma-1-5-for-polygon-based-heart-structure-contouring/172907) | Your own public statement of the task, datasets and the *absence* of benchmarks (28 Jun 2026); reply from **Fereshteh Mahvar (Google)** suggesting a service endpoint. Establishes an existing relationship with the HAI-DEF team | 🟢 |
| **CAMUS adapter** | [The-Adimension/EchoTrace-MedGemma-CAMUS](https://huggingface.co/The-Adimension/EchoTrace-MedGemma-CAMUS) | PEFT/LoRA on `google/medgemma-1.5-4b-it`; **DOI 10.57967/hf/9541**; Apache-2.0; 82 downloads; **gated** | 🟢 |
| **EchoNet adapter** | [The-Adimension/EchoTrace-MedGemma-EchoNet](https://huggingface.co/The-Adimension/EchoTrace-MedGemma-EchoNet) | **DOI 10.57967/hf/9540**; Apache-2.0; 78 downloads; **gated** | 🟢 |
| **HF collection** | [The-Adimension/atria-echotrace](https://huggingface.co/collections/The-Adimension/atria-echotrace-6a5884b6c17ec3aab49d0875) | Single landing point for both adapters | 🟢 |
| **HF Space** | [ShehabAnwer/The-Adimension-ATRIA-EchoTrace](https://huggingface.co/spaces/ShehabAnwer/The-Adimension-ATRIA-EchoTrace) | Linked as demo space from both model cards | 🟡 returned **HTTP 401** to my fetch — confirm it is public and running |
| **GitHub repository** | [The-Adimension/ATRIA-EchoTrace](https://github.com/The-Adimension/ATRIA-EchoTrace) | Public code, DEITY framing, HITL description, disclaimer | 🔴 **understates the work** — see §3 |
| **Colab notebook** | [colab.research.google.com/drive/1qofahQ8…](https://colab.research.google.com/drive/1qofahQ8LztTrB_Us9j1Iyz2aYeS2_2rH?usp=sharing) | Reviewer can run it without gated weights — **the most important non-gated entry point** | 🟡 confirm it still runs |
| **DEITY paper** | [10.1093/ehjimp/qyaf038](https://doi.org/10.1093/ehjimp/qyaf038) · [PMC12922539](https://pmc.ncbi.nlm.nih.gov/articles/PMC12922539/) | Peer-reviewed foundation. **No other showcase entry has one** | 🟢 |
| **The Adimension site** | [theadimension.com](https://www.theadimension.com/) | Brand home | 🟡 |
| **LinkedIn** | [ch.linkedin.com/in/shehabanwermd](https://ch.linkedin.com/in/shehabanwermd) | Author credibility: cardiologist, Section Editor for Echocardiography, *Int J Cardiovasc Imaging* | 🟢 |
| **Google Scholar** | [scholar.google.com/citations?user=W-rpYiUAAAAJ](https://scholar.google.com/citations?user=W-rpYiUAAAAJ&hl=en) | Publication record in cardiac mechanics, echo, ML | 🟢 |

**Not located in public search:** the X/Twitter threads, LinkedIn articles, and the YouTube
overview / HITL videos referenced in your brief. They may exist but are not indexed against
the queries I ran. **You must supply these URLs directly** — see `04_SUBMISSION_CHECKLIST.md`.

---

## 2. Local assets — verified this session, not yet public

**These are your differentiators. None of them is visible to anyone outside this machine.**

| Asset | Path | Value to the showcase |
|---|---|---|
| **200-frame benchmark** | `outputs/benchmark/full200/pointwise.csv` | The headline result: 200/200 parsed, median 4.98 mm |
| **Visual QA gallery** | `outputs/benchmark/full200/overlays/index.html` | **200 rendered overlays**, filterable, best/worst bands. The single most persuasive visual asset you have |
| **Per-frame data** | `outputs/benchmark/full200/overlays/manifest.csv` | Per-frame class, point count, distance — reviewable evidence |
| **Provenance scripts** | `outputs/benchmark/provenance/` (11 scripts) | How the test split was staged, calibrated and gated — reproducibility |
| **Staged test split** | `outputs/benchmark/camus_test/` (200 frames + `PROVENANCE.md`) | Proof the split is the dataset authors' official one |
| **Adapter-repair fix + test** | `src/atria_echotrace/ml/engine.py`, `tests/test_adapter_completeness.py` | The 324/802-tensor finding, fixed and regression-guarded |
| **Classification track** | `datasets/classified_datasets/` | 3 tasks: quality (3 classes), CAMUS EF (16 bins), EchoNet EF (19 bins) |
| **Stage registry** | `src/atria_echotrace/api/stages.py` | Registry-driven lifecycle, state-aware UI |
| **Test suite** | `tests/` | 282 tests (254 pass / 9 skip on the review tier) |
| **Training corpus** | `datasets/processed_datasets/unified_processed/` | 22,048 frames / 11,024 cases, reproduced byte-identically by the vendored scripts |

---

## 3. The gap that must be closed before submitting

**The public GitHub README does not describe the platform that now exists.** Fetched
2026-08-13, it states:

- *"No specific performance metrics, accuracy rates, or comparative benchmarks are stated"*
- Status: *"Prototype / Advanced MVP"*

Both were accurate when written. Neither is accurate now. A Google reviewer who reads the
repository before or after the case study will find a mismatch between a submission
claiming a 200-frame benchmark and a README claiming none exists.

**Fix this first.** It is the highest-leverage hour in this entire package.

---

## 4. Claims from the forum post — reconciled against reality

| Forum claim (Jun–Jul 2026) | Status now |
|---|---|
| "2000 PNGs from DICOM extract… 1500-1600 training, 200 eval, 100-200 testing" | 🟢 CAMUS portion of the corpus is 2,000 frames. Official split is 400/50/50 patients = 1,600/200/200 frames |
| "20,000 images (from 10,000 AVI, 112×112)" | 🟢 EchoNet portion is 20,048 frames from 10,024 videos |
| "QLoRA to the vision tower's attention and MLP layers across nearly all 27 encoder layers" | 🟢 Consistent with the adapter's own config; **this is also exactly what the key-nesting bug silenced** |
| "we have not yet published formal quantitative benchmarks" | 🔴 **Now obsolete — this is your news** |
| "NOT a cleared or approved medical device" | 🟢 Keep this framing verbatim |

---

## Sources

- [Google AI Developers Forum — ATRIA EchoTrace thread](https://discuss.ai.google.dev/t/atria-echotrace-fine-tuning-medgemma-1-5-for-polygon-based-heart-structure-contouring/172907)
- [EchoTrace-MedGemma-CAMUS](https://huggingface.co/The-Adimension/EchoTrace-MedGemma-CAMUS) · [EchoTrace-MedGemma-EchoNet](https://huggingface.co/The-Adimension/EchoTrace-MedGemma-EchoNet) · [ATRIA-EchoTrace collection](https://huggingface.co/collections/The-Adimension/atria-echotrace-6a5884b6c17ec3aab49d0875)
- [GitHub: The-Adimension/ATRIA-EchoTrace](https://github.com/The-Adimension/ATRIA-EchoTrace)
- [The Adimension / DEITY paper, EHJ-IMP](https://academic.oup.com/ehjimp/advance-article/doi/10.1093/ehjimp/qyaf038/8102118) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12922539/)
- [HAI-DEF Showcase](https://developers.google.com/health-ai-developer-foundations/showcase) · [Visilant case study](https://developers.google.com/health-ai-developer-foundations/showcase/visilant)
- [MedGemma Impact Challenge winners](https://blog.google/innovation-and-ai/technology/health/med-gemma-impact-challenge/)
- [MedGemma model page](https://developers.google.com/health-ai-developer-foundations/medgemma) · [Google Research MedGemma blog](https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/)
- [theadimension.com](https://www.theadimension.com/) · [LinkedIn](https://ch.linkedin.com/in/shehabanwermd) · [Google Scholar](https://scholar.google.com/citations?user=W-rpYiUAAAAJ&hl=en)
