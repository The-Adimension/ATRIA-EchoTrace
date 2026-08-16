# Reply to Richa Tiwari (HAI-DEF Program Lead)

Paste-ready. Adjust the bracketed items before sending.

---

**Subject:** Re: ATRIA-EchoTrace — HAI-DEF Showcase, Technical Solutions and Tools

Dear Richa,

Thank you — we're delighted, and **yes, we're very happy to publish a technical blog post**
for the "Technical Solutions and Tools" section.

The post is drafted and covers exactly the three areas you asked for:

**1. QLoRA fine-tuning on MedGemma 1.5 4B for 30-point polygon coordinates.** We've
documented the full configuration recovered directly from the shipped checkpoints rather
than from our notebook defaults — LoRA r 32 / α 32 / dropout 0.05 across ten target modules,
including the SigLIP vision tower's `fc1`, `fc2` and `out_proj`; effective batch 64
(4 × 16 grad-accum); bf16 with gradient checkpointing; linear schedule from 1e-4 with 3 %
warmup; `max_length` 1024; 5 epochs on a single GPU. Both published adapters turned out to
share an identical recipe — of 134 `SFTConfig` fields, only the output paths differ.

**2. Standard segmentation accuracy numbers — yes, we have them.** Since our June forum
post (where I noted we had not yet published formal benchmarks) we've completed a full
evaluation on the **official CAMUS test split: 50 patients, 200 frames, never seen in
training**:

- **Dice 0.744 ± 0.135**, **IoU 0.609 ± 0.156** (n = 200, vs the original expert masks)
- MAD 6.22 mm · HD95 13.11 mm · NSD@2mm 0.227
- **200/200 outputs parsed** to a valid polygon and in bounds
- Median **point-to-curve distance 4.98 mm** (IQR 3.37–7.52)

One methodological note we've been explicit about in the post: because the adapter emits
*coordinates* rather than a mask, we treat point-to-curve distance as the primary metric and
report Dice/IoU alongside it for comparability with the segmentation literature. We also
measured the discretisation ceiling — representing an expert mask as 30 points costs only
Dice 0.9905 — which confirms the representation isn't the limiting factor.

We also publish the limitations rather than omit them: a 2CH/4CH performance gap, a 12.5 %
self-intersection rate, and a worst-case failure mode that is a *plausible ventricle traced
in the wrong place* — which is the clearest argument we have for mandatory human review.

**3. Developer utility and enablement.** The post has a "what you can reuse tomorrow"
section, and one finding we think is genuinely valuable to the wider HAI-DEF community:

> While validating our own adapters we discovered that **324 of 802 checkpoint tensors were
> loading into nothing, silently.** The adapters were trained when Gemma 3 nested the vision
> encoder as `vision_tower.vision_model.encoder`; transformers 5.x flattened it to
> `vision_tower.encoder`. PEFT emits a non-fatal warning and continues, leaving every
> vision-tower `lora_B` at zero initialisation — so roughly **40 % of the fine-tuning was
> inert while the model still loaded and produced plausible contours.**

Any developer who fine-tunes a vision tower and later upgrades `transformers` can hit this,
and it is invisible unless you assert adapter completeness. We've included the detection and
repair pattern, and we'd be glad for HAI-DEF to surface it more prominently if useful — it
seems like the kind of thing worth a warning in the fine-tuning docs.

The post is **visual, not just tabular**. It includes a schematic of the end-to-end pipeline
(with the human-in-the-loop return path drawn explicitly), a histogram of the point-to-curve
distance across all 200 held-out frames, and **27 rendered predictions** — three at full size
plus a 24-frame ladder sampled evenly by rank from best to worst. That ladder is deliberate:
it shows tight agreement at the top, gradual drift through the middle, and the handful of
gross mislocalisations at the end, so a reader sees the real distribution of behaviour rather
than a curated best case. The complete 200-image gallery ships with the repository.

**Hosting:** we'll publish at [theadimension.com/... or Medium — confirm] and send you the
final URL. [Add your preferred timeline, e.g. "we can have it live within a week."]

On testimonials — understood, we'll orient those toward developer utility and enablement
rather than clinical outcomes, consistent with our research-use-only framing.

Thank you again for the opportunity and for the clear guidance on framing.

Best regards,
**Shehab Anwer, MD**
Founder, The Adimension
[email] · [links]

---

## Before you send — quick checks

- [ ] Decide the hosting URL (own site vs Medium) and state a timeline
- [ ] Confirm the Hugging Face Space is public and running (it returned HTTP 401 to an
      anonymous fetch — a reviewer needs at least one non-gated entry point)
- [ ] Confirm the Colab notebook still executes end to end
- [ ] Update the public GitHub README first — it still reads *"Prototype / Advanced MVP"*
      with *"no specific performance metrics"*, which now contradicts this reply
- [ ] Decide whether to offer the PEFT finding as a **separate** short note on the forum
      thread — it's a strong standalone contribution and reactivates the existing thread
- [ ] Export a few overlay PNGs (best / typical / worst) to embed in the published post;
      the raw gallery is a local HTML file and will need hosting or conversion
