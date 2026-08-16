"""Score the full 200-frame CAMUS benchmark (config A2, official test subgroup).

Metric groups A-F from BENCHMARK_PLAN, against BOTH reference standards:
  (a) the 30-point training polygon - "did it learn the task it was set?"
  (b) the original _gt.nii.gz expert mask - "is the contour clinically accurate?"

SUPERSEDED FRAMING: this scores the adapter as if it were a segmentation/EF model.
The adapter's actual contract is structured [y,x] coordinate generation, not
segmentation or EF prediction; see rescore_pointwise.py for the corrected,
point-to-curve re-score against that real task. The numbers here aren't wrong,
just not the primary evaluation.
"""

import collections
import csv
import json
import pathlib
import statistics as st
import sys

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import find_contours

sys.path.insert(0, "src")
from atria_echotrace.domain.geometry import polygon_to_mask  # noqa: E402
from atria_echotrace.ml.reference import camus_ef  # noqa: E402

# numpy 2 removed 2-D np.cross; identical operation, vendored file untouched.
_np_cross = np.cross
def _cross2d(a, b, *A, **K):
    a_, b_ = np.asarray(a), np.asarray(b)
    if a_.shape[-1] == 2 and b_.shape[-1] == 2 and not A and not K:
        return a_[..., 0] * b_[..., 1] - a_[..., 1] * b_[..., 0]
    return _np_cross(a, b, *A, **K)
camus_ef.np.cross = _cross2d

BENCH = pathlib.Path("outputs/benchmark/camus_test")
FULL = pathlib.Path("outputs/benchmark/full200")
NIFTI = pathlib.Path("datasets/original_datasets_and_repos/camus_public/database_nifti")
LV = 1

tracings = json.loads((BENCH / "tracings.json").read_text())
meta = {r["key"]: r for r in csv.DictReader((BENCH / "metadata.csv").open())}
recs = [json.loads(l) for l in (FULL / "raw_predictions.jsonl").read_text().splitlines() if l.strip()]
by_stem = {r["stem"]: r for r in recs}


def boundary_mm(mask, sh, sw):
    cs = find_contours(mask.astype(float), 0.5)
    if not cs:
        return np.empty((0, 2))
    p = np.vstack(cs)
    return np.column_stack([p[:, 0] * sh, p[:, 1] * sw])


def distances(a, b):
    if len(a) == 0 or len(b) == 0:
        return None
    dab, dba = cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]
    both = np.concatenate([dab, dba])
    return {"mad": float(both.mean()), "hd": float(max(dab.max(), dba.max())),
            "hd95": float(np.percentile(both, 95)), "nsd2": float((both <= 2.0).mean())}


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    d = a.sum() + b.sum()
    return 1.0 if d == 0 else float(2 * np.logical_and(a, b).sum() / d)


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def proper_crossing(p1, q1, p2, q2):
    def o(a, b, c):
        v = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        return 0 if v == 0 else (1 if v > 0 else -1)
    o1, o2, o3, o4 = o(p1, q1, p2), o(p1, q1, q2), o(p2, q2, p1), o(p2, q2, q1)
    return all(x != 0 for x in (o1, o2, o3, o4)) and o1 != o2 and o3 != o4


def self_intersects(poly):
    p = [tuple(map(float, pt)) for pt in poly]
    n = len(p)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if proper_crossing(p[i], p[(i+1) % n], p[j], p[(j+1) % n]):
                return True
    return False


rows = []
for stem, r in by_stem.items():
    if not r.get("parsed"):
        continue
    e = tracings[stem]
    h, w, sh, sw = e["image_h"], e["image_w"], e["spacing_h"], e["spacing_w"]
    pred = polygon_to_mask(r["polygon"], h, w)
    ref_poly = polygon_to_mask(e["lv_polygon"], h, w)
    gt, _ = camus_ef.sitk_load(NIFTI / r["patient"] / f"{stem}_gt.nii.gz")
    ref_mask = (gt == LV).astype(np.uint8)

    row = {"stem": stem, "patient": r["patient"], "view": r["view"], "instant": r["instant"],
           "quality": meta[stem]["image_quality"], "sex": meta[stem]["sex"],
           "vertices": r["vertices"], "seconds": r["seconds"],
           "self_intersects": self_intersects(r["polygon"]),
           "dice_poly": dice(pred, ref_poly), "dice": dice(pred, ref_mask),
           "iou": iou(pred, ref_mask)}
    for tag, ref in (("poly", ref_poly), ("", ref_mask)):
        d = distances(boundary_mm(pred, sh, sw), boundary_mm(ref, sh, sw))
        if d:
            row.update({(f"{k}_{tag}" if tag else k): v for k, v in d.items()})
    rows.append(row)


def agg(sub, key):
    v = [r[key] for r in sub if key in r and r[key] == r[key]]
    return (st.fmean(v), st.pstdev(v)) if v else (float("nan"), float("nan"))


def line(label, sub, keys=("dice", "mad", "hd95", "nsd2")):
    cells = []
    for k in keys:
        m, s = agg(sub, k)
        cells.append(f"{m:.3f}±{s:.3f}" if k in ("dice", "nsd2") else f"{m:.2f}±{s:.2f}")
    return f"  {label:26} n={len(sub):3}  " + "  ".join(f"{c:>14}" for c in cells)


print("=" * 96)
print("FULL BENCHMARK - CAMUS official test subgroup, 50 patients / 200 frames, LV, config A2")
print("=" * 96)

parsed = [r for r in recs if r.get("parsed")]
v30 = [r for r in rows if r["vertices"] == 30]
inb = [r for r in parsed if all(0 <= y <= 1000 and 0 <= x <= 1000 for y, x in r["polygon"])]
selfx = [r for r in rows if r["self_intersects"]]
print("\nGROUP A - output validity")
print(f"  valid polygon        : {len(parsed)}/{len(recs)}  ({100*len(parsed)/len(recs):.1f}%)")
print(f"  exactly 30 vertices  : {len(v30)}/{len(rows)}  ({100*len(v30)/len(rows):.1f}%)")
print(f"  coords in bounds     : {len(inb)}/{len(parsed)}  ({100*len(inb)/len(parsed):.1f}%)")
print(f"  self-intersecting    : {len(selfx)}/{len(rows)}  ({100*len(selfx)/len(rows):.1f}%)")
vc = collections.Counter(r["vertices"] for r in rows)
print(f"  vertex distribution  : {dict(sorted(vc.items()))}")
sec = [r["seconds"] for r in rows]
print(f"  latency              : median {st.median(sec):.0f}s")

print("\nGROUPS B & C - agreement (vs EXPERT MASK)")
print(f"  {'stratum':26} {'':6}  {'Dice':>14}  {'MAD mm':>14}  {'HD95 mm':>14}  {'NSD@2mm':>14}")
print(line("ALL", rows))
print(f"  {'-'*90}")
for v in ("2CH", "4CH"):
    print(line(f"view = {v}", [r for r in rows if r["view"] == v]))
for i in ("ED", "ES"):
    print(line(f"instant = {i}", [r for r in rows if r["instant"] == i]))
for q in ("Good", "Medium", "Poor"):
    print(line(f"quality = {q}", [r for r in rows if r["quality"] == q]))
for s_ in ("M", "F"):
    print(line(f"sex = {s_}", [r for r in rows if r["sex"] == s_]))

m_poly, s_poly = agg(rows, "dice_poly")
m_mask, s_mask = agg(rows, "dice")
print(f"\n  vs 30-pt polygon: Dice {m_poly:.4f}±{s_poly:.4f}   "
      f"vs expert mask: Dice {m_mask:.4f}±{s_mask:.4f}")

print("\nGROUP D - clinical: biplane Simpson EF (vendored CAMUS implementation)")
patients = sorted({r["patient"] for r in rows})
ef_rows = []
for pid in patients:
    need = [f"{pid}_{v}_{i}" for v in ("2CH", "4CH") for i in ("ED", "ES")]
    if not all(s in by_stem and by_stem[s].get("parsed") for s in need):
        continue
    m, sp = {}, {}
    for v in ("2CH", "4CH"):
        for i in ("ED", "ES"):
            e = tracings[f"{pid}_{v}_{i}"]
            m[(v, i)] = polygon_to_mask(by_stem[f"{pid}_{v}_{i}"]["polygon"], e["image_h"], e["image_w"])
        e = tracings[f"{pid}_{v}_ED"]
        sp[v] = (e["spacing_h"], e["spacing_w"])
    try:
        edv, esv = camus_ef.compute_left_ventricle_volumes(
            m[("2CH", "ED")], m[("2CH", "ES")], sp["2CH"],
            m[("4CH", "ED")], m[("4CH", "ES")], sp["4CH"])
        ef = round(100 * (edv - esv) / edv)
    except Exception:  # noqa: BLE001
        continue
    ref = next(float(l.split(":")[1]) for l in
               (NIFTI / pid / "Info_4CH.cfg").read_text().splitlines() if l.startswith("EF:"))
    ef_rows.append({"patient": pid, "ef": ef, "ref": ref, "delta": ef - ref,
                    "edv": round(edv, 1), "esv": round(esv, 1)})

if ef_rows:
    d = [r["delta"] for r in ef_rows]
    ad = [abs(x) for x in d]
    print(f"  patients with complete biplane : {len(ef_rows)}/{len(patients)}")
    print(f"  mean |EF error|                : {st.fmean(ad):.1f} EF points   (CAMUS ref 5.6)")
    print(f"  bias                           : {st.fmean(d):+.1f}   SD {st.pstdev(d):.1f}")
    print(f"  95% limits of agreement        : {st.fmean(d)-1.96*st.pstdev(d):+.1f} to "
          f"{st.fmean(d)+1.96*st.pstdev(d):+.1f} EF points")
    within5 = sum(1 for x in ad if x <= 5)
    print(f"  within +/-5 EF points          : {within5}/{len(ef_rows)} ({100*within5/len(ef_rows):.0f}%)")
    xs = [r["ref"] for r in ef_rows]; ys = [r["ef"] for r in ef_rows]
    if len(xs) > 2 and st.pstdev(xs) > 0 and st.pstdev(ys) > 0:
        mx, my = st.fmean(xs), st.fmean(ys)
        num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
        den = (sum((a-mx)**2 for a in xs) * sum((b-my)**2 for b in ys)) ** 0.5
        print(f"  Pearson r (AI vs reference EF) : {num/den:.3f}   (CAMUS ref 0.80)")

    def cat(ef):
        return "severe" if ef < 30 else "moderate" if ef < 40 else "mild" if ef < 50 else "normal"
    agree = sum(1 for r in ef_rows if cat(r["ef"]) == cat(r["ref"]))
    print(f"\nGROUP E - EF category agreement : {agree}/{len(ef_rows)} "
          f"({100*agree/len(ef_rows):.0f}%)")
    conf = collections.Counter((cat(r["ref"]), cat(r["ef"])) for r in ef_rows)
    order = ["normal", "mild", "moderate", "severe"]
    print(f"  {'ref \\\\ AI':12}" + "".join(f"{c:>10}" for c in order))
    for a in order:
        print(f"  {a:12}" + "".join(f"{conf.get((a,b),0):>10}" for b in order))

json.dump({"frames": rows, "ef": ef_rows}, (FULL / "scores.json").open("w"), indent=2)
with (FULL / "per_frame.csv").open("w", newline="", encoding="utf-8") as fh:
    wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wtr.writeheader()
    wtr.writerows(rows)
print(f"\nwritten -> {FULL/'scores.json'} and per_frame.csv")
