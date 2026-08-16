"""Re-score the persisted runs against the adapter's ACTUAL task. CPU only, no inference.

The adapter's contract: given an echo frame, emit a parseable JSON list of [y, x] pairs
(normalised to [0, 1000]) tracing the LV endocardial border. It is structured coordinate
generation — not segmentation, not EF.

PRIMARY metrics
  1. output validity     - JSON fence, parseable, well-formed pairs, in-bounds, self-intersection
  2. point count         - distribution, exactly-30 rate, outliers
  3. point-to-curve      - each predicted point's Euclidean distance to the reference
                           POLYLINE (piecewise-linear), in px and mm; plus the reverse
                           direction for coverage

References, kept separate and never collapsed:
  (a) the 30-point training polygon  - what the model was trained to match  [PRIMARY]
  (b) the dense contour from _gt.nii.gz - the clinical boundary             [parallel]
"""

import csv
import json
import pathlib
import re
import statistics as st
import sys

import numpy as np
from skimage.measure import find_contours

sys.path.insert(0, "src")
from atria_echotrace.ml.reference import camus_ef  # noqa: E402  (sitk_load only)

BENCH = pathlib.Path("outputs/benchmark/camus_test")
NIFTI = pathlib.Path("datasets/original_datasets_and_repos/camus_public/database_nifti")
LV = 1

RUNS = {
    "A2 full200 (primary)": "outputs/benchmark/full200/raw_predictions.jsonl",
    "A2 pilot20": "outputs/benchmark/pilot20_fixed/raw_predictions.jsonl",
    "B2 pilot20 bf16": "outputs/benchmark/pilot20_fixed/raw_predictions_B.jsonl",
    "C  pilot20 geom-prompt": "outputs/benchmark/pilot20_fixed/raw_predictions_C.jsonl",
}

tracings = json.loads((BENCH / "tracings.json").read_text())
meta = {r["key"]: r for r in csv.DictReader((BENCH / "metadata.csv").open())}


# ----------------------------------------------------------------- geometry
def denorm(poly, h, w):
    """[y,x] in [0,1000] -> pixel coordinates. Distances must be measured here, not in
    normalised space: frames are not square, so normalised units distort the aspect."""
    a = np.asarray(poly, dtype=float)
    return np.column_stack([a[:, 0] / 1000.0 * h, a[:, 1] / 1000.0 * w])


def point_to_polyline(points, polyline, closed=True):
    """Min Euclidean distance from each point to a piecewise-linear curve.

    Not distance-to-nearest-vertex: the curve between vertices counts, which is what
    "does the point land on the tracing" means.
    """
    p = np.asarray(points, dtype=float)
    v = np.asarray(polyline, dtype=float)
    a = v
    b = np.roll(v, -1, axis=0) if closed else v[1:]
    if not closed:
        a = v[:-1]
    ab = b - a                                   # (S,2)
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom == 0] = 1e-12
    ap = p[:, None, :] - a[None, :, :]           # (P,S,2)
    t = np.einsum("psi,si->ps", ap, ab) / denom[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(p[:, None, :] - closest, axis=2)
    return d.min(axis=1)


def proper_crossing(p1, q1, p2, q2):
    def o(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
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
            if proper_crossing(p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]):
                return True
    return False


FENCE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def output_compliance(raw):
    """Structured-output compliance judged from the RAW text, not the app's parser
    (which has a lenient fallback that would mask malformed output)."""
    out = {"fence": False, "json_ok": False, "has_key": False, "pairs_ok": False, "n": 0}
    m = FENCE.search(raw or "")
    if not m:
        return out
    out["fence"] = True
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return out
    out["json_ok"] = True
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return out
    poly = data[0].get("polygon_2d")
    if poly is None:
        return out
    out["has_key"] = True
    if isinstance(poly, list) and poly:
        out["n"] = len(poly)
        out["pairs_ok"] = all(
            isinstance(pt, (list, tuple)) and len(pt) == 2
            and all(isinstance(c, (int, float)) for c in pt)
            for pt in poly
        )
    return out


gt_cache: dict[str, np.ndarray] = {}


def dense_reference(stem):
    """Marching-squares contour of the expert mask, in pixel coords (hundreds of points)."""
    if stem not in gt_cache:
        arr, _ = camus_ef.sitk_load(NIFTI / stem.split("_")[0] / f"{stem}_gt.nii.gz")
        cs = find_contours((arr == LV).astype(float), 0.5)
        gt_cache[stem] = max(cs, key=len) if cs else np.empty((0, 2))
    return gt_cache[stem]


def summarise(values, unit=""):
    if not values:
        return "n/a"
    q1, q3 = np.percentile(values, [25, 75])
    return (f"med {np.median(values):.2f}{unit} "
            f"[IQR {q1:.2f}-{q3:.2f}]  mean {np.mean(values):.2f}  p95 {np.percentile(values,95):.2f}")


for run_name, path in RUNS.items():
    p = pathlib.Path(path)
    if not p.exists():
        continue
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    comp = [output_compliance(r.get("raw_response", "")) for r in recs]
    rows, all_fwd_px, all_fwd_mm, all_rev_px, all_dense_mm = [], [], [], [], []

    for r, c in zip(recs, comp):
        if not r.get("parsed") or not r.get("polygon"):
            continue
        stem = r["stem"]
        e = tracings[stem]
        h, w, sh, sw = e["image_h"], e["image_w"], e["spacing_h"], e["spacing_w"]

        pred_px = denorm(r["polygon"], h, w)
        ref_px = denorm(e["lv_polygon"], h, w)

        fwd = point_to_polyline(pred_px, ref_px, closed=True)          # predicted -> reference
        rev = point_to_polyline(ref_px, pred_px, closed=True)          # reference -> predicted
        dense = dense_reference(stem)
        fwd_dense = point_to_polyline(pred_px, dense, closed=True) if len(dense) else np.array([])

        in_bounds = bool(np.all((pred_px[:, 0] >= 0) & (pred_px[:, 0] <= h)
                                & (pred_px[:, 1] >= 0) & (pred_px[:, 1] <= w)))
        rows.append({
            "stem": stem, "view": r["view"], "instant": r["instant"],
            "quality": meta[stem]["image_quality"] if stem in meta else "",
            "n_points": len(r["polygon"]), "in_bounds": in_bounds,
            "self_intersects": self_intersects(r["polygon"]),
            "fwd_med_px": float(np.median(fwd)), "fwd_med_mm": float(np.median(fwd) * sh),
            "rev_med_px": float(np.median(rev)),
            "fwd_dense_med_mm": float(np.median(fwd_dense) * sh) if len(fwd_dense) else float("nan"),
        })
        all_fwd_px.extend(fwd.tolist())
        all_fwd_mm.extend((fwd * sh).tolist())
        all_rev_px.extend(rev.tolist())
        if len(fwd_dense):
            all_dense_mm.extend((fwd_dense * sh).tolist())

    n = len(recs)
    print("=" * 92)
    print(f"{run_name}   ({n} frames)")
    print("=" * 92)

    print("\n1. OUTPUT VALIDITY  (structured coordinate generation)")
    print(f"   ```json fence present        : {sum(c['fence'] for c in comp)}/{n}")
    print(f"   JSON parses                  : {sum(c['json_ok'] for c in comp)}/{n}")
    print(f"   has polygon_2d key           : {sum(c['has_key'] for c in comp)}/{n}")
    print(f"   all entries well-formed [y,x]: {sum(c['pairs_ok'] for c in comp)}/{n}")
    print(f"   in-bounds after denorm       : {sum(r['in_bounds'] for r in rows)}/{len(rows)}")
    print(f"   self-intersecting polyline   : {sum(r['self_intersects'] for r in rows)}/{len(rows)}")

    counts = [r["n_points"] for r in rows]
    dist = {c: counts.count(c) for c in sorted(set(counts))}
    print("\n2. POINT COUNT")
    print(f"   distribution                 : {dist}")
    print(f"   exactly 30                   : {counts.count(30)}/{len(counts)} "
          f"({100*counts.count(30)/len(counts):.0f}%)")
    print(f"   median {int(np.median(counts))}   range {min(counts)}-{max(counts)}   "
          f"within 30±1: {sum(1 for c in counts if 29 <= c <= 31)}/{len(counts)} "
          f"({100*sum(1 for c in counts if 29 <= c <= 31)/len(counts):.0f}%)")

    print("\n3. POINT-TO-CURVE FIDELITY   [PRIMARY: vs the 30-point training polyline]")
    print(f"   predicted -> reference, px   : {summarise(all_fwd_px)}")
    print(f"   predicted -> reference, mm   : {summarise(all_fwd_mm)}")
    print(f"   reference -> predicted, px   : {summarise(all_rev_px)}   (coverage)")
    tot = len(all_fwd_mm)
    for tol in (1.0, 2.0, 5.0):
        k = sum(1 for d in all_fwd_mm if d <= tol)
        print(f"   within {tol:.0f} mm                  : {k}/{tot}  ({100*k/tot:.1f}%)")
    print(f"   per-frame median (mm)        : {summarise([r['fwd_med_mm'] for r in rows])}")
    if all_dense_mm:
        print(f"\n   [parallel: vs dense expert contour from _gt.nii.gz]")
        print(f"   predicted -> dense ref, mm   : {summarise(all_dense_mm)}")
        k = sum(1 for d in all_dense_mm if d <= 2.0)
        print(f"   within 2 mm                  : {k}/{len(all_dense_mm)} "
              f"({100*k/len(all_dense_mm):.1f}%)")

    if "full200" in path:
        print("\n   stratified (per-frame median, predicted -> reference, mm)")
        for field, vals in (("view", ("2CH", "4CH")), ("instant", ("ED", "ES")),
                            ("quality", ("Good", "Medium", "Poor"))):
            for v in vals:
                sub = [r["fwd_med_mm"] for r in rows if r[field] == v]
                if sub:
                    print(f"     {field}={v:7} n={len(sub):3}  median {np.median(sub):.2f} mm")
        out = pathlib.Path("outputs/benchmark/full200/pointwise.csv")
        with out.open("w", newline="", encoding="utf-8") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\n   per-frame CSV -> {out}")
    print()
