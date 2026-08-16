"""Visual review gallery from persisted predictions. CPU/rendering only, no inference.

Per frame: the native-resolution echo image with the reference polyline in GREEN and the
predicted polygon in RED, predicted vertices marked, and a caption strip carrying the
identity and the point-to-curve metrics. Native resolution is preserved — the caption is
appended below the image, never scaled into it.

Outputs: outputs/benchmark/full200/overlays/{*.png, manifest.csv, index.html}
"""

import csv
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "src")
from atria_echotrace.data.frames import load_frame  # noqa: E402
from atria_echotrace.render.overlays import draw_polygons_on_image, to_pixel_points  # noqa: E402

BENCH = pathlib.Path("outputs/benchmark/camus_test")
FULL = pathlib.Path("outputs/benchmark/full200")
OUT = FULL / "overlays"
OUT.mkdir(parents=True, exist_ok=True)

GREEN = (0, 255, 0, 255)   # reference — fixed convention
RED = (255, 0, 0, 255)     # prediction — fixed convention
CAPTION_H = 68

tracings = json.loads((BENCH / "tracings.json").read_text())
meta = {r["key"]: r for r in csv.DictReader((BENCH / "metadata.csv").open())}
recs = [json.loads(l) for l in (FULL / "raw_predictions.jsonl").read_text().splitlines() if l.strip()]


def denorm(poly, h, w):
    a = np.asarray(poly, dtype=float)
    return np.column_stack([a[:, 0] / 1000.0 * h, a[:, 1] / 1000.0 * w])


def point_to_polyline(points, polyline):
    p, v = np.asarray(points, float), np.asarray(polyline, float)
    a, b = v, np.roll(v, -1, axis=0)
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom == 0] = 1e-12
    ap = p[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("psi,si->ps", ap, ab) / denom[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(p[:, None, :] - closest, axis=2).min(axis=1)


def get_font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = get_font(15)
FONT_SMALL = get_font(12)

rows, failures = [], []
for r in recs:
    stem = r["stem"]
    try:
        if not r.get("parsed") or not r.get("polygon"):
            raise ValueError("no parseable polygon in the persisted record")
        e = tracings[stem]
        h, w, sh = e["image_h"], e["image_w"], e["spacing_h"]

        image = load_frame(BENCH / "frames" / f"{stem}.png")

        # metrics (recomputed here so the caption cannot drift from the picture)
        pred_px = denorm(r["polygon"], h, w)
        ref_px = denorm(e["lv_polygon"], h, w)
        d_mm = point_to_polyline(pred_px, ref_px) * sh
        med_mm = float(np.median(d_mm))
        pct2 = float((d_mm <= 2.0).mean() * 100)

        # reference under prediction, so the prediction is never hidden
        canvas = draw_polygons_on_image(
            image, [(e["lv_polygon"], GREEN), (r["polygon"], RED)], line_width=2
        )

        # predicted vertices, so point placement is visible and not just the stroke
        draw = ImageDraw.Draw(canvas)
        for x, y in to_pixel_points(r["polygon"], w, h):   # (x, y), per its docstring
            draw.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=(255, 210, 0), outline=(140, 0, 0))

        # caption strip appended BELOW: the image itself is never resized
        out_img = Image.new("RGB", (canvas.width, canvas.height + CAPTION_H), (14, 18, 28))
        out_img.paste(canvas, (0, 0))
        cap = ImageDraw.Draw(out_img)
        q = meta.get(stem, {}).get("image_quality", "?")
        cap.text((8, canvas.height + 6),
                 f"{stem}   {r['view']} {r['instant']}   quality={q}",
                 fill=(235, 240, 250), font=FONT)
        cap.text((8, canvas.height + 26),
                 f"points={r['vertices']}   median point-to-curve={med_mm:.2f} mm   "
                 f"within 2mm={pct2:.0f}%   Dice(secondary)={r.get('dice_vs_polygon', float('nan')):.3f}",
                 fill=(150, 205, 255), font=FONT_SMALL)
        cap.text((8, canvas.height + 44),
                 "green = reference    red = prediction    yellow dots = predicted vertices",
                 fill=(140, 150, 170), font=FONT_SMALL)

        name = f"{med_mm:07.3f}mm__{stem}.png"   # filename sorts worst-last
        out_img.save(OUT / name)
        rows.append({"stem": stem, "view": r["view"], "instant": r["instant"],
                     "quality": q, "n_points": r["vertices"],
                     "median_pt_mm": round(med_mm, 3), "pct_within_2mm": round(pct2, 1),
                     "overlay_path": f"overlays/{name}"})
    except Exception as exc:  # noqa: BLE001 - failures are listed, never silently skipped
        failures.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})

rows.sort(key=lambda x: x["median_pt_mm"])

with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
    wtr = csv.DictWriter(fh, fieldnames=["stem", "view", "instant", "quality", "n_points",
                                         "median_pt_mm", "pct_within_2mm", "overlay_path"])
    wtr.writeheader()
    wtr.writerows(rows)

# ------------------------------------------------------------------ gallery
med_all = np.median([r["median_pt_mm"] for r in rows]) if rows else 0
cards = []
for i, r in enumerate(rows):
    band = "best" if i < 5 else ("worst" if i >= len(rows) - 5 else "mid")
    fname = r["overlay_path"].split("/")[-1]
    cards.append(
        f'<figure class="card {band}" data-mm="{r["median_pt_mm"]}" data-view="{r["view"]}" '
        f'data-instant="{r["instant"]}" data-quality="{r["quality"]}">'
        f'<a href="{fname}" target="_blank"><img src="{fname}" loading="lazy"></a>'
        f'<figcaption><b>{r["stem"]}</b><br>{r["view"]} {r["instant"]} · {r["quality"]}<br>'
        f'<span class="mm">{r["median_pt_mm"]:.2f} mm</span> · {r["n_points"]} pts · '
        f'{r["pct_within_2mm"]:.0f}% &le;2mm</figcaption></figure>'
    )

html = f"""<meta charset="utf-8"><title>ATRIA overlay review — {len(rows)} frames</title>
<style>
 body{{background:#0e121c;color:#e8ecf5;font:14px/1.5 system-ui,sans-serif;margin:0;padding:18px}}
 h1{{font-size:19px;margin:0 0 4px}} .sub{{color:#93a0b8;margin-bottom:14px}}
 .bar{{position:sticky;top:0;background:#0e121cf0;padding:10px 0;border-bottom:1px solid #26304a;
      margin-bottom:14px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;z-index:9}}
 button{{background:#1b2437;color:#cfe0ff;border:1px solid #2e3b57;border-radius:6px;
        padding:5px 11px;cursor:pointer;font:13px system-ui}}
 button:hover{{border-color:#4d7ecb}} button.on{{background:#2a3c60;border-color:#5b8ad8}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
 .card{{margin:0;background:#141b2b;border:1px solid #232e46;border-radius:8px;overflow:hidden}}
 .card.best{{border-color:#2f7d4f}} .card.worst{{border-color:#a2404a}}
 img{{width:100%;display:block;background:#000}}
 figcaption{{padding:7px 9px;font-size:11.5px;color:#9fb0cc;line-height:1.45}}
 .mm{{color:#ffd166;font-weight:600}} .legend{{margin-left:auto;color:#93a0b8;font-size:12px}}
</style>
<h1>ATRIA EchoTrace — overlay review</h1>
<div class="sub">{len(rows)} frames · CAMUS official test split · config A2 (4-bit, training prompt) ·
 sorted by median point-to-curve distance (best first) · overall median {med_all:.2f} mm</div>
<div class="bar">
  <button class="on" onclick="f(this,'all')">all</button>
  <button onclick="f(this,'best')">best 5</button>
  <button onclick="f(this,'worst')">worst 5</button>
  <button onclick="f(this,'2CH')">2CH</button>
  <button onclick="f(this,'4CH')">4CH</button>
  <button onclick="f(this,'ED')">ED</button>
  <button onclick="f(this,'ES')">ES</button>
  <button onclick="f(this,'Good')">Good</button>
  <button onclick="f(this,'Medium')">Medium</button>
  <button onclick="f(this,'Poor')">Poor</button>
  <span class="legend">green = reference &nbsp;·&nbsp; red = prediction &nbsp;·&nbsp; yellow dots = predicted vertices</span>
</div>
<div class="grid">{''.join(cards)}</div>
<script>
function f(btn, key) {{
  document.querySelectorAll('.bar button').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelectorAll('.card').forEach(c => {{
    const show = key === 'all' || c.classList.contains(key) ||
      c.dataset.view === key || c.dataset.instant === key || c.dataset.quality === key;
    c.style.display = show ? '' : 'none';
  }});
}}
</script>"""
(OUT / "index.html").write_text(html, encoding="utf-8")

print(f"overlays written : {len(rows)}")
print(f"gallery          : {OUT/'index.html'}")
print(f"manifest         : {OUT/'manifest.csv'}")
if failures:
    print(f"\nFAILED TO RENDER ({len(failures)}):")
    for f_ in failures:
        print(f"  {f_['stem']}: {f_['error']}")
else:
    print("failures         : none")

if rows:
    print("\nBEST 5 (closest to the reference tracing)")
    for r in rows[:5]:
        print(f"  {r['median_pt_mm']:6.2f} mm  {r['stem']:28} {r['view']} {r['instant']} "
              f"{r['quality']:6} {r['n_points']:2}pts  ->  {r['overlay_path']}")
    mid = len(rows) // 2
    print("\nTYPICAL (median of the distribution)")
    for r in rows[mid - 1:mid + 2]:
        print(f"  {r['median_pt_mm']:6.2f} mm  {r['stem']:28} {r['view']} {r['instant']} "
              f"{r['quality']:6} {r['n_points']:2}pts  ->  {r['overlay_path']}")
    print("\nWORST 5 (furthest from the reference tracing)")
    for r in rows[-5:]:
        print(f"  {r['median_pt_mm']:6.2f} mm  {r['stem']:28} {r['view']} {r['instant']} "
              f"{r['quality']:6} {r['n_points']:2}pts  ->  {r['overlay_path']}")
