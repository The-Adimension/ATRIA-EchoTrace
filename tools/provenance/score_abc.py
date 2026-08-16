"""Three-way comparison on the repaired adapter: A2 vs B2 vs C.

A2  4-bit NF4 + double-quant, training-matched prompt (true view + instant)
B2  unquantised bf16,          training-matched prompt
C   unquantised bf16,          geometry-focused prompt (no view, no instant)

A2 vs B2 -> quantisation effect.   B2 vs C -> loss of view/instant conditioning.
All three on the identical 20 frames with a fully loaded adapter.
"""

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

_c = np.cross
def _c2(a, b, *A, **K):
    a_, b_ = np.asarray(a), np.asarray(b)
    if a_.shape[-1] == 2 and b_.shape[-1] == 2 and not A and not K:
        return a_[..., 0] * b_[..., 1] - a_[..., 1] * b_[..., 0]
    return _c(a, b, *A, **K)
camus_ef.np.cross = _c2

BENCH = pathlib.Path("outputs/benchmark/camus_test")
FIXED = pathlib.Path("outputs/benchmark/pilot20_fixed")
NIFTI = pathlib.Path("datasets/original_datasets_and_repos/camus_public/database_nifti")
PILOT = ["patient0052", "patient0189", "patient0225", "patient0238", "patient0266"]

tracings = json.loads((BENCH / "tracings.json").read_text())


def load(path):
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    return {json.loads(l)["stem"]: json.loads(l)
            for l in p.read_text().splitlines() if l.strip()}


CONDS = {
    "A2 4bit/train-prompt": load(FIXED / "raw_predictions.jsonl"),
    "B2 bf16/train-prompt": load(FIXED / "raw_predictions_B.jsonl"),
    "C  bf16/geom-prompt": load(FIXED / "raw_predictions_C.jsonl"),
}
CONDS = {k: v for k, v in CONDS.items() if v}

gt_cache: dict[str, np.ndarray] = {}
def gt_mask(stem):
    if stem not in gt_cache:
        arr, _ = camus_ef.sitk_load(NIFTI / stem.split("_")[0] / f"{stem}_gt.nii.gz")
        gt_cache[stem] = (arr == 1).astype(np.uint8)
    return gt_cache[stem]


def boundary_mm(m, sh, sw):
    cs = find_contours(m.astype(float), 0.5)
    if not cs:
        return np.empty((0, 2))
    p = np.vstack(cs)
    return np.column_stack([p[:, 0] * sh, p[:, 1] * sw])


def metrics(pred_poly, stem):
    e = tracings[stem]
    h, w, sh, sw = e["image_h"], e["image_w"], e["spacing_h"], e["spacing_w"]
    pred = polygon_to_mask(pred_poly, h, w)
    ref = gt_mask(stem)
    a, b = pred.astype(bool), ref.astype(bool)
    dice = 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum()) if a.sum() + b.sum() else 1.0
    pa, pb = boundary_mm(pred, sh, sw), boundary_mm(ref, sh, sw)
    if len(pa) == 0 or len(pb) == 0:
        return {"dice": dice}
    dab, dba = cKDTree(pb).query(pa)[0], cKDTree(pa).query(pb)[0]
    both = np.concatenate([dab, dba])
    return {"dice": float(dice), "mad": float(both.mean()),
            "hd95": float(np.percentile(both, 95)),
            "hd": float(max(dab.max(), dba.max())),
            "nsd2": float((both <= 2.0).mean())}


def ef_for(records):
    out = []
    for pid in PILOT:
        need = [f"{pid}_{v}_{i}" for v in ("2CH", "4CH") for i in ("ED", "ES")]
        if not all(s in records and records[s].get("parsed") for s in need):
            continue
        m, sp = {}, {}
        for v in ("2CH", "4CH"):
            for i in ("ED", "ES"):
                e = tracings[f"{pid}_{v}_{i}"]
                m[(v, i)] = polygon_to_mask(records[f"{pid}_{v}_{i}"]["polygon"],
                                            e["image_h"], e["image_w"])
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
        out.append({"patient": pid, "ef": ef, "ref": ref, "delta": ef - ref,
                    "edv": edv, "esv": esv})
    return out


scored = {}
for name, recs in CONDS.items():
    rows = []
    for stem, r in recs.items():
        if not r.get("parsed"):
            continue
        row = {"stem": stem, "view": r["view"], "instant": r["instant"],
               "vertices": r["vertices"], "seconds": r["seconds"]}
        row.update(metrics(r["polygon"], stem))
        rows.append(row)
    scored[name] = {"rows": rows, "recs": recs, "ef": ef_for(recs)}


def agg(rows, key):
    v = [r[key] for r in rows if key in r]
    return (st.fmean(v), st.pstdev(v)) if v else (float("nan"), float("nan"))


print("=" * 88)
print("THREE-WAY COMPARISON - repaired adapter, identical 20 frames (vs expert mask)")
print("=" * 88)
hdr = f"  {'metric':22}" + "".join(f"{n:>22}" for n in scored)
print(hdr)
for label, key, fmt in (("parse rate", None, None), ("exactly-30 vertices", None, None),
                        ("Dice", "dice", "{:.4f} ± {:.4f}"),
                        ("MAD (mm)", "mad", "{:.3f} ± {:.3f}"),
                        ("HD95 (mm)", "hd95", "{:.2f} ± {:.2f}"),
                        ("NSD@2mm", "nsd2", "{:.3f} ± {:.3f}"),
                        ("latency (s)", "seconds", "{:.0f} ± {:.0f}")):
    cells = []
    for name, d in scored.items():
        if label == "parse rate":
            n = len(d["recs"]); ok = len(d["rows"])
            cells.append(f"{ok}/{n}")
        elif label == "exactly-30 vertices":
            cells.append(f"{sum(1 for r in d['rows'] if r['vertices'] == 30)}/{len(d['rows'])}")
        else:
            m, s = agg(d["rows"], key)
            cells.append(fmt.format(m, s))
    print(f"  {label:22}" + "".join(f"{c:>22}" for c in cells))

print()
print("  vertex-count distribution")
for name, d in scored.items():
    counts = sorted({r["vertices"] for r in d["rows"]})
    dist = ", ".join(f"{c}:{sum(1 for r in d['rows'] if r['vertices'] == c)}" for c in counts)
    print(f"    {name:22} {dist}")

print()
print("  stratification (Dice)")
for field, vals in (("view", ("2CH", "4CH")), ("instant", ("ED", "ES"))):
    for v in vals:
        cells = []
        for name, d in scored.items():
            sub = [r for r in d["rows"] if r[field] == v]
            m, s = agg(sub, "dice")
            cells.append(f"{m:.4f}")
        print(f"    {field}={v:5}" + "".join(f"{c:>22}" for c in cells))

print()
print("  clinical: ejection fraction")
for name, d in scored.items():
    if not d["ef"]:
        print(f"    {name:22} no complete patients")
        continue
    absd = [abs(r["delta"]) for r in d["ef"]]
    bias = st.fmean(r["delta"] for r in d["ef"])
    print(f"    {name:22} n={len(d['ef'])}  mean|error| {st.fmean(absd):5.1f} pts   "
          f"bias {bias:+6.1f}   (CAMUS ref 5.6)")

print()
print("  small-ES-cavity failure mode (patient0189: true ESV 15 mL, EF 69)")
for name, d in scored.items():
    r = next((x for x in d["ef"] if x["patient"] == "patient0189"), None)
    if r:
        print(f"    {name:22} ESV {r['esv']:6.1f} mL   EF {r['ef']:3} (ref 69)   delta {r['delta']:+.0f}")

json.dump({k: {"rows": v["rows"], "ef": v["ef"]} for k, v in scored.items()},
          (FIXED / "three_way.json").open("w"), indent=2)
print(f"\nwritten -> {FIXED/'three_way.json'}")
