"""Score the pilot: metric groups A-D against both reference standards.

Group A  output validity      - could a generative model even answer?
Group B  region agreement     - Dice, IoU
Group C  boundary accuracy    - MAD/ASSD, HD, HD95 in millimetres
Group D  clinical measurement - biplane Simpson EDV/ESV/EF via the vendored CAMUS code

Two references throughout:
  (a) the 30-point training polygon - "did it learn the task it was set?"
  (b) the original expert mask      - "is the contour clinically accurate?"
"""

import json
import pathlib
import statistics
import sys

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import find_contours

sys.path.insert(0, "src")
from atria_echotrace.domain.geometry import polygon_to_mask  # noqa: E402
from atria_echotrace.ml.reference import camus_ef  # noqa: E402

# numpy 2 removed 2-D np.cross; restore the identical operation without touching the
# vendored file (proven equal to the 3-D z-component before use).
_np_cross = np.cross


def _cross2d(a, b, *args, **kwargs):
    a_, b_ = np.asarray(a), np.asarray(b)
    if a_.shape[-1] == 2 and b_.shape[-1] == 2 and not args and not kwargs:
        return a_[..., 0] * b_[..., 1] - a_[..., 1] * b_[..., 0]
    return _np_cross(a, b, *args, **kwargs)


_u, _v = np.array([1.0, 2.0]), np.array([[3.0, 4.0], [5.0, 6.0]])
assert np.allclose(
    _cross2d(_u, _v),
    _np_cross(np.append(_u, 0.0), np.column_stack([_v, np.zeros(len(_v))]))[:, 2],
)
camus_ef.np.cross = _cross2d

BENCH = pathlib.Path("outputs/benchmark/camus_test")
OUT = pathlib.Path("outputs/benchmark/pilot20")
NIFTI = pathlib.Path("datasets/original_datasets_and_repos/camus_public/database_nifti")
LV = 1


def boundary_mm(mask, sh, sw):
    cs = find_contours(mask.astype(float), 0.5)
    if not cs:
        return np.empty((0, 2))
    p = np.vstack(cs)
    return np.column_stack([p[:, 0] * sh, p[:, 1] * sw])


def distances(a, b):
    if len(a) == 0 or len(b) == 0:
        return None
    dab = cKDTree(b).query(a)[0]
    dba = cKDTree(a).query(b)[0]
    both = np.concatenate([dab, dba])
    return {"mad_mm": float(both.mean()), "hd_mm": float(max(dab.max(), dba.max())),
            "hd95_mm": float(np.percentile(both, 95)),
            "nsd2mm": float((both <= 2.0).mean())}


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    d = a.sum() + b.sum()
    return 1.0 if d == 0 else float(2 * np.logical_and(a, b).sum() / d)


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def self_intersects(poly):
    """Any non-adjacent edge pair crossing => anatomically impossible contour."""
    p = np.asarray(poly, float)
    n = len(p)

    def seg(a, b, c, d):
        def o(u, v, w):
            return np.sign((v[0]-u[0])*(w[1]-u[1]) - (v[1]-u[1])*(w[0]-u[0]))
        return o(a,b,c) != o(a,b,d) and o(c,d,a) != o(c,d,b)

    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if seg(p[i], p[(i+1) % n], p[j], p[(j+1) % n]):
                return True
    return False


def cfg_ef(pid):
    for line in (NIFTI / pid / "Info_4CH.cfg").read_text().splitlines():
        if line.startswith("EF:"):
            return float(line.split(":", 1)[1])
    return None


records = [json.loads(l) for l in (OUT / "raw_predictions.jsonl").read_text().splitlines() if l.strip()]
tracings = json.loads((BENCH / "tracings.json").read_text())
meta = {r["stem"]: r for r in records}

# --------------------------------------------------------------- Group A
print("=" * 76)
print("GROUP A - OUTPUT VALIDITY (could the model answer at all?)")
print("=" * 76)
n = len(records)
parsed = [r for r in records if r.get("parsed")]
v30 = [r for r in parsed if r["vertices"] == 30]
inb = [r for r in parsed if all(0 <= y <= 1000 and 0 <= x <= 1000 for y, x in r["polygon"])]
selfx = [r for r in parsed if self_intersects(r["polygon"])]
secs = [r["seconds"] for r in parsed]
print(f"  frames attempted     : {n}")
print(f"  valid polygon        : {len(parsed)}/{n}  ({100*len(parsed)/n:.0f}%)")
print(f"  exactly 30 vertices  : {len(v30)}/{len(parsed)}  ({100*len(v30)/len(parsed):.0f}%)")
print(f"  all coords in bounds : {len(inb)}/{len(parsed)}  ({100*len(inb)/len(parsed):.0f}%)")
print(f"  self-intersecting    : {len(selfx)}/{len(parsed)}" + (f"  <-- {[r['stem'] for r in selfx]}" if selfx else ""))
print(f"  latency              : median {statistics.median(secs):.0f}s  range {min(secs):.0f}-{max(secs):.0f}s")
if parsed and parsed[0].get("input_tokens"):
    print(f"  tokens               : in {parsed[0]['input_tokens']}, out ~{statistics.median(r['generated_tokens'] for r in parsed):.0f}")

# --------------------------------------------------------------- Groups B & C
print()
print("=" * 76)
print("GROUPS B & C - AGREEMENT vs BOTH REFERENCES")
print("=" * 76)
rows = []
for r in parsed:
    e = tracings[r["stem"]]
    h, w, sh, sw = e["image_h"], e["image_w"], e["spacing_h"], e["spacing_w"]
    pred = polygon_to_mask(r["polygon"], h, w)
    ref_poly = polygon_to_mask(e["lv_polygon"], h, w)
    gt, _ = camus_ef.sitk_load(NIFTI / r["patient"] / f"{r['stem']}_gt.nii.gz")
    ref_mask = (gt == LV).astype(np.uint8)

    row = {"stem": r["stem"], "view": r["view"], "instant": r["instant"],
           "dice_poly": dice(pred, ref_poly), "iou_poly": iou(pred, ref_poly),
           "dice_mask": dice(pred, ref_mask), "iou_mask": iou(pred, ref_mask)}
    for name, ref in (("poly", ref_poly), ("mask", ref_mask)):
        d = distances(boundary_mm(pred, sh, sw), boundary_mm(ref, sh, sw))
        if d:
            row.update({f"{k}_{name}": v for k, v in d.items()})
    rows.append(row)


def stat(key, subset=None):
    vals = [r[key] for r in (subset or rows) if key in r and r[key] == r[key]]
    return f"{statistics.fmean(vals):.4f} ± {statistics.pstdev(vals):.4f}" if vals else "n/a"


print(f"  {'metric':22} {'vs 30-pt polygon':>22} {'vs expert mask':>22}")
for label, a, b in (("Dice", "dice_poly", "dice_mask"), ("IoU", "iou_poly", "iou_mask"),
                    ("MAD (mm)", "mad_mm_poly", "mad_mm_mask"),
                    ("HD95 (mm)", "hd95_mm_poly", "hd95_mm_mask"),
                    ("HD (mm)", "hd_mm_poly", "hd_mm_mask"),
                    ("NSD@2mm", "nsd2mm_poly", "nsd2mm_mask")):
    print(f"  {label:22} {stat(a):>22} {stat(b):>22}")

print()
print("  stratified (Dice vs expert mask):")
for field, values in (("view", ("2CH", "4CH")), ("instant", ("ED", "ES"))):
    for v in values:
        sub = [r for r in rows if r[field] == v]
        if sub:
            print(f"    {field}={v:4}  n={len(sub):2}  Dice {stat('dice_mask', sub)}   MAD {stat('mad_mm_mask', sub)} mm")

# --------------------------------------------------------------- Group D
print()
print("=" * 76)
print("GROUP D - CLINICAL: biplane Simpson EF (vendored CAMUS implementation)")
print("=" * 76)
patients = sorted({r["patient"] for r in parsed})
ef_rows = []
for pid in patients:
    needed = [f"{pid}_{v}_{i}" for v in ("2CH", "4CH") for i in ("ED", "ES")]
    if not all(s in meta and meta[s].get("parsed") for s in needed):
        print(f"  {pid}: incomplete (a frame failed) - EF not computable")
        continue
    masks, sp = {}, {}
    for v in ("2CH", "4CH"):
        for i in ("ED", "ES"):
            e = tracings[f"{pid}_{v}_{i}"]
            masks[(v, i)] = polygon_to_mask(meta[f"{pid}_{v}_{i}"]["polygon"], e["image_h"], e["image_w"])
        sp[v] = (tracings[f"{pid}_{v}_ED"]["spacing_h"], tracings[f"{pid}_{v}_ED"]["spacing_w"])
    try:
        edv, esv = camus_ef.compute_left_ventricle_volumes(
            masks[("2CH", "ED")], masks[("2CH", "ES")], sp["2CH"],
            masks[("4CH", "ED")], masks[("4CH", "ES")], sp["4CH"])
        ef = round(100 * (edv - esv) / edv)
    except Exception as exc:  # noqa: BLE001
        print(f"  {pid}: EF failed ({type(exc).__name__})")
        continue
    ref = cfg_ef(pid)
    ef_rows.append({"patient": pid, "ef_pred": ef, "ef_ref": ref, "delta": ef - ref,
                    "edv": round(edv, 1), "esv": round(esv, 1)})

print(f"  {'patient':14}{'AI EF':>7}{'ref EF':>8}{'delta':>7}{'EDV mL':>9}{'ESV mL':>8}")
for r in ef_rows:
    print(f"  {r['patient']:14}{r['ef_pred']:>7}{r['ef_ref']:>8.0f}{r['delta']:>+7.0f}{r['edv']:>9.1f}{r['esv']:>8.1f}")
if ef_rows:
    d = [abs(r["delta"]) for r in ef_rows]
    print(f"\n  mean |EF error| : {statistics.fmean(d):.1f} EF points   (CAMUS reference: 5.6)")
    print(f"  bias            : {statistics.fmean([r['delta'] for r in ef_rows]):+.1f} EF points")

json.dump({"frames": rows, "ef": ef_rows}, (OUT / "scores.json").open("w"), indent=2)
print(f"\nwritten -> {OUT/'scores.json'}")
