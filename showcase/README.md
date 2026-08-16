# full_showcase — deployable bundle

Everything Google HAI-DEF needs, in one folder. **Copy this directory to your web server as-is.**
No build step, no dependencies, no server-side code — plain static files.

```bash
# example
rsync -av full_showcase/ user@server:/var/www/atria-echotrace/
# or
scp -r full_showcase/* user@server:/var/www/html/atria-echotrace/
```

Then link Google to `https://<your-domain>/atria-echotrace/`.

---

## What's in here

```
full_showcase/
├── index.html          THE PAGE — CSS + 230 images + 8 figures + the brand mark +
│                       the interactive HITL editor and the 200-frame specimen
│                       ladder, all inline
├── media/
│   ├── logo.svg              the Adimension mark (also inlined into the page)
│   ├── explain-web.mp4       3m01s overview, 1080p, narrated
│   ├── hitl-notebook-web.mp4 57s notebook walkthrough, narrated
│   ├── webapp-web.mp4        16s workstation capture, silent, loops
│   └── poster-*.webp         one poster per video
├── gallery/
│   ├── index.html      the 200-frame evaluation gallery, filterable
│   └── *.png           200 rendered overlays at native resolution
├── ft_runs/
│   ├── run_reports/    6 fine-tuning runs: loss curves, token accuracy, LR schedule,
│   │                   grad-norm/entropy, throughput, dashboard, per-step CSV,
│   │                   hyperparameters, and the raw TensorBoard event file
│   ├── extracted_data/ all runs' metrics as CSV/JSON + run overview
│   └── *.py            the extraction and report-generation scripts
├── data/
│   ├── pointwise.csv              per-frame point-to-curve results (primary metric)
│   ├── per_frame.csv              per-frame Dice / IoU / MAD / HD95 / NSD
│   ├── scores.json                full scoring output incl. per-patient EF
│   ├── overlays_manifest.csv      per-frame class, point count, distance, overlay path
│   ├── adapter_config_*.json      the released LoRA configurations
│   └── training_metadata_*.json   recorded training summaries
└── docs/               the full written package (strategy, metrics lock, hyperparameters)
```

**309 files, ~99 MB** — the 200 gallery PNGs (28.4 MB), the three videos (56.7 MB) and the run
reports (10.5 MB). `index.html` is 1.6 MB on its own because every image it shows is inlined,
including 200 thumbnails for the specimen ladder.

**A visitor downloads about 1 MB of that on first load.** The two narrated videos are
`preload="none"` behind poster images, so 56 MB of media costs ~950 KB until someone presses play
— measured, not assumed.

---

## Internal links the page relies on

`index.html` links to `gallery/index.html`, `ft_runs/` and `data/` with **relative paths**, so
the bundle works at any URL depth. Nothing points at a local filesystem path.

If your host does not auto-index directories, the `ft_runs/` and `data/` links will 404 —
either enable directory listing for those two paths, or replace those two `<a href>` targets in
`index.html` with links to specific files.

## Video and branding

All three clips were re-encoded from the originals, which are kept outside this bundle in
`../media-source/`. **Do not replace these with the originals** — the source files total 406 MB and
none of them is web-playable:

| | Source | Shipped | Notes |
|---|---|---|---|
| 3-minute overview | 362 MB @ 16 Mbit/s | **53 MB** | 1080p, CRF 23, narrated |
| Notebook walkthrough | 33 MB | **2.4 MB** | 720p, CRF 26, narrated |
| Workstation capture | 9.9 MB | **0.7 MB** | audio stripped — the source is digital silence (−91 dB) |

**Every source file had `moov` at the very end**, so a browser had to download the whole file
before the first frame. All three shipped files are `-movflags +faststart` (`moov` at byte 32),
verified. If you ever re-encode, that flag is not optional.

`logo.svg` uses `fill="currentColor"`, so one file serves the dark hero and the light footer; it is
inlined into the page and reused as the favicon. The old `logo-white.svg` had byte-identical path
data and is retired to `../media-source/`.

## The interactive editor (§2 "Try it")

Section 2 embeds a working human-in-the-loop editor: three real held-out frames with the model's
actual predicted polygon, whose vertices the reader can drag. Coordinates, the JSON array and the
point-to-curve distance all recompute live in the browser.

- **No model runs.** The frames, the expert reference and the prediction are baked in as data;
  the only computation is the distance metric, which is ~40 lines of vanilla JS.
- **It is the same arithmetic as the benchmark.** On load each frame reports exactly the median
  the Python harness recorded (1.20 / 4.98 / 18.59 mm), which cross-validates the two
  implementations.
- Pointer events are used throughout, so dragging works on touch devices as well as mouse.

## The training record (§5)

Section 5 is the pipeline evidence: a table of all **six fine-tuning runs across two open
datasets**, plus two figures — every run on shared axes, and the CAMUS learning-rate ablation that
produced the published recipe. The table, both figures and every number in the prose are generated
at build time from `ft_runs/extracted_data/*.csv`; nothing is transcribed by hand.

Two claims it exists to support:

- **The recipe was found, not guessed** — 1e-5 → eval 0.752, 2e-4/8 epochs → 0.390,
  1e-4/5 epochs → **0.259**, all on the identical 1,600-frame split.
- **It transferred without retuning** — Runs 5 and 6 differ in two of 134 `SFTConfig` fields
  (the output paths), across a 10× difference in corpus size.

Elapsed times are HuggingFace's recorded `train_runtime`. Run 4 terminated without writing a final
summary, so its runtime is left blank rather than estimated, and the 36-hour total covers only the
five runs that logged one.

## The specimen ladder (§6, "Walk the whole benchmark")

Section 6 embeds all **200 held-out predictions** as a single rankable strip, running **vertically
beside the frame it controls** — best at the top, worst at the bottom — so the image, the strip and
the readout are all in one glance. Drag it, click it and use ↑ ↓ (or PageUp/PageDown for ten at a
time), or press *Sweep* to auto-walk best → worst. The frame, its metadata and its warnings update
live.

- The image sits in a **fixed-aspect stage** (`aspect-ratio` + `object-fit: contain`). The 200
  overlays span 20 distinct pixel sizes and aspect ratios from 0.92 to 1.21, so without this the
  whole section changes height on every hover.
- Bar length is the error, colour is acquisition quality, the shaded band is the IQR and the dashed
  line is the median.
- Self-intersecting polygons are drawn at full opacity so the 25 structural failures are findable.
- **Images load progressively.** A 168 px WebP thumbnail is inlined for every frame so the strip
  responds instantly and works with no network; once you settle on a frame, the full-resolution
  render is fetched from `gallery/` and swapped in. Sweeping deliberately does *not* fetch — it
  stays on thumbnails, so auto-walking all 200 frames costs zero requests instead of ~28 MB.
- If `gallery/` is missing (someone opens `index.html` on its own), the thumbnails simply remain.
  Nothing breaks and no image is left blank — verified by serving the file with no siblings.
- The static 24-frame ladder below it is retained deliberately: it is what a reader sees in print,
  in a PDF export, or with JavaScript disabled.

## Fonts and privacy

The page loads **no external resources** — no CDN, no web fonts, no analytics, no third-party
scripts, and deliberately **no YouTube or Vimeo embed**. Nothing a reader does here is visible to
anyone but your own server. Images are inlined as WebP data-URIs; the diagrams and data figures are
inline SVG; the scripts are the editor and the ladder, both self-contained.

The only requests it makes are same-origin siblings inside this bundle: the video files and their
posters, and the ladder's full-resolution upgrades from `gallery/`. Both are enhancements, never
dependencies — the page is fully readable if either is missing.

## Browser support

WebP data-URIs and inline SVG: Chrome, Edge, Firefox, Safari 14+. No polyfill needed.

---

## Provenance of every number on the page

| Claim | Source in this bundle |
|---|---|
| 200/200 parsed · median 4.98 mm · IQR 3.37–7.52 | `data/pointwise.csv`, `data/overlays_manifest.csv` |
| Dice 0.744 ± 0.135 · IoU 0.609 ± 0.156 | `data/per_frame.csv` |
| LoRA r32 / α32 / 10 target modules | `data/adapter_config_*.json` |
| 5 epochs · batch 4×16 · lr 1e-4 · bf16 · max_length 1024 | `ft_runs/run_reports/*/hyperparameters_and_sample_size.json` |
| Loss curves, token accuracy, wall-clock | `ft_runs/run_reports/Run_5_*`, `Run_6_*` (`metrics_data.csv`) |
| Six-run table, Figures 4–5, elapsed times | `ft_runs/extracted_data/all_runs_metrics.csv`, `finetuning_runs_overview.csv` |
| 324/802 adapter tensors inert | described in `docs/` — reproduced by the platform's `test_adapter_completeness.py` |

The two released adapters correspond to:

- **CAMUS** → `ft_runs/run_reports/Run_6_CAMUS_32LoRA_Blackwell_1600samples/`
- **EchoNet** → `ft_runs/run_reports/Run_5_20K_Blackwell_16576samples/`

---

**Research use only — not a cleared or approved medical device.**
© The Adimension · Shehab Anwer, MD
