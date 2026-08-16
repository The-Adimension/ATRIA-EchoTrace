"""Steps 2 and 3 — free calibration. No GPU, no model.

Step 2  Discretisation ceiling: how much accuracy is lost purely by representing an
        expert mask as 30 index-uniform polygon points? This is the best score any model
        could achieve, and must be known before attributing error to the model.

Step 3  EF GATE: does the vendored CAMUS Simpson implementation reproduce the reference
        EF values shipped with the dataset? If not, EF error cannot be attributed to the
        model rather than to our arithmetic, and the benchmark must stop.
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
from atria_echotrace.ml.reference.camus_ef import (  # noqa: E402
    compute_left_ventricle_volumes,
    sitk_load,
)

BENCH = pathlib.Path("outputs/benchmark/camus_test")
NIFTI = pathlib.Path("datasets/original_datasets_and_repos/camus_public/database_nifti")
LV_LABEL = 1  # CAMUS: 1=LV endo, 2=myocardium, 3=LA (preprocess_camus.py LABELS)


def boundary_points_mm(mask: np.ndarray, spacing_h: float, spacing_w: float) -> np.ndarray:
    """Sub-pixel boundary of a binary mask, in millimetres."""
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.empty((0, 2))
    pts = np.vstack(contours)  # (row, col) = (h, w)
    return np.column_stack([pts[:, 0] * spacing_h, pts[:, 1] * spacing_w])


def surface_distances(a: np.ndarray, b: np.ndarray) -> dict[str, float] | None:
    """Symmetric boundary distances in mm: MAD/ASSD, Hausdorff, HD95."""
    if len(a) == 0 or len(b) == 0:
        return None
    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    both = np.concatenate([d_ab, d_ba])
    return {
        "mad_mm": float(both.mean()),
        "hd_mm": float(max(d_ab.max(), d_ba.max())),
        "hd95_mm": float(np.percentile(both, 95)),
    }


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def parse_cfg(path: pathlib.Path) -> dict:
    info = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


tracings = json.loads((BENCH / "tracings.json").read_text())
stems = sorted(tracings)
patients = sorted({s.split("_")[0] for s in stems})
print(f"evaluation set: {len(patients)} patients, {len(stems)} frames\n")

# ---------------------------------------------------------------- Step 2: ceiling
print("=" * 74)
print("STEP 2 - DISCRETISATION CEILING  (30-point polygon vs original expert mask)")
print("=" * 74)

rows = []
gt_masks: dict[str, np.ndarray] = {}
for stem in stems:
    entry = tracings[stem]
    pid = stem.split("_")[0]
    gt, _ = sitk_load(NIFTI / pid / f"{stem}_gt.nii.gz")
    lv = (gt == LV_LABEL).astype(np.uint8)
    gt_masks[stem] = lv

    h, w = entry["image_h"], entry["image_w"]
    sh, sw = entry["spacing_h"], entry["spacing_w"]
    poly = polygon_to_mask(entry["lv_polygon"], h, w)

    d = surface_distances(
        boundary_points_mm(poly, sh, sw), boundary_points_mm(lv, sh, sw)
    )
    rows.append(
        {
            "stem": stem,
            "view": entry["view"],
            "instant": entry["instant"],
            "dice": dice(poly, lv),
            **(d or {"mad_mm": float("nan"), "hd_mm": float("nan"), "hd95_mm": float("nan")}),
        }
    )


def summarise(values):
    values = [v for v in values if v == v]
    return f"{statistics.fmean(values):.4f} +/- {statistics.pstdev(values):.4f}"


for label, key in (("Dice", "dice"), ("MAD  (mm)", "mad_mm"), ("HD   (mm)", "hd_mm"), ("HD95 (mm)", "hd95_mm")):
    print(f"  {label:12} {summarise([r[key] for r in rows])}")
print()
for instant in ("ED", "ES"):
    sub = [r for r in rows if r["instant"] == instant]
    print(f"  {instant}: Dice {summarise([r['dice'] for r in sub])}   MAD {summarise([r['mad_mm'] for r in sub])} mm")

(BENCH / "ceiling.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

# ---------------------------------------------------------------- Step 3: EF gate
print()
print("=" * 74)
print("STEP 3 - EF GATE  (vendored CAMUS Simpson vs the dataset's own EF values)")
print("=" * 74)

results = []
for pid in patients:
    try:
        masks = {
            (v, i): gt_masks[f"{pid}_{v}_{i}"] for v in ("2CH", "4CH") for i in ("ED", "ES")
        }
    except KeyError:
        continue
    sp = {
        v: (tracings[f"{pid}_{v}_ED"]["spacing_h"], tracings[f"{pid}_{v}_ED"]["spacing_w"])
        for v in ("2CH", "4CH")
    }
    edv, esv = compute_left_ventricle_volumes(
        masks[("2CH", "ED")], masks[("2CH", "ES")], sp["2CH"],
        masks[("4CH", "ED")], masks[("4CH", "ES")], sp["4CH"],
    )
    ef = round(100 * (edv - esv) / edv)
    ref = parse_cfg(NIFTI / pid / "Info_4CH.cfg")
    ref_ef = float(ref["EF"])
    results.append({"patient": pid, "ef_computed": ef, "ef_reference": ref_ef,
                    "edv": round(edv, 1), "esv": round(esv, 1), "delta": ef - ref_ef})

deltas = [r["delta"] for r in results]
exact = sum(1 for d in deltas if abs(d) < 0.5)
within1 = sum(1 for d in deltas if abs(d) <= 1)
within2 = sum(1 for d in deltas if abs(d) <= 2)
print(f"  patients evaluated : {len(results)}")
print(f"  exact match        : {exact}/{len(results)}  ({100*exact/len(results):.0f}%)")
print(f"  within +/-1 EF pt  : {within1}/{len(results)}  ({100*within1/len(results):.0f}%)")
print(f"  within +/-2 EF pts : {within2}/{len(results)}  ({100*within2/len(results):.0f}%)")
print(f"  mean delta         : {statistics.fmean(deltas):+.3f} EF points")
print(f"  max |delta|        : {max(abs(d) for d in deltas):.1f} EF points")
worst = sorted(results, key=lambda r: -abs(r["delta"]))[:5]
print("  largest deviations :", ", ".join(f"{r['patient']}({r['delta']:+.0f})" for r in worst))

(BENCH / "ef_gate.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

passed = within1 == len(results)
print()
print("  GATE:", "PASS - Simpson implementation reproduces the reference EF"
      if passed else "REVIEW - deviations exceed +/-1 EF point; inspect before proceeding")
