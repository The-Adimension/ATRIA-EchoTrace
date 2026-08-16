"""Step 3 — the EF gate.

The vendored CAMUS code is numpy-1 era: it calls `np.cross` on 2-D vectors, which numpy
2.0 removed. Rather than edit code that is vendored verbatim, the equivalent operation is
injected into the module's namespace at import time. The vendored file stays byte-identical.

The substitution is provably the same number: for 2-D u, v the cross product is the
z-component of the 3-D cross, u0*v1 - u1*v0. Verified against the 3-D route before use.
"""

import json
import pathlib
import statistics
import sys

import numpy as np

sys.path.insert(0, "src")
from atria_echotrace.ml.reference import camus_ef  # noqa: E402

_numpy_cross = np.cross


def cross_2d_compatible(a, b, *args, **kwargs):
    """`np.cross` with numpy<2 semantics for 2-D inputs; delegates otherwise."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if a_arr.shape[-1] == 2 and b_arr.shape[-1] == 2 and not args and not kwargs:
        return a_arr[..., 0] * b_arr[..., 1] - a_arr[..., 1] * b_arr[..., 0]
    return _numpy_cross(a, b, *args, **kwargs)


# Prove the shim before relying on it.
_u = np.array([1.0, 2.0])
_v = np.array([[3.0, 4.0], [5.0, 6.0]])
_expected = _numpy_cross(np.append(_u, 0.0), np.column_stack([_v, np.zeros(len(_v))]))[:, 2]
assert np.allclose(cross_2d_compatible(_u, _v), _expected), "2-D cross shim is not equivalent"

camus_ef.np.cross = cross_2d_compatible  # noqa: it is the module's own numpy handle

BENCH = pathlib.Path("outputs/benchmark/camus_test")
NIFTI = pathlib.Path("datasets/original_datasets_and_repos/camus_public/database_nifti")
LV_LABEL = 1


def parse_cfg(path):
    info = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


tracings = json.loads((BENCH / "tracings.json").read_text())
patients = sorted({s.split("_")[0] for s in tracings})

results = []
for pid in patients:
    masks, spacing = {}, {}
    ok = True
    for view in ("2CH", "4CH"):
        for instant in ("ED", "ES"):
            stem = f"{pid}_{view}_{instant}"
            if stem not in tracings:
                ok = False
                break
            gt, _ = camus_ef.sitk_load(NIFTI / pid / f"{stem}_gt.nii.gz")
            masks[(view, instant)] = (gt == LV_LABEL).astype(np.uint8)
        spacing[view] = (tracings[f"{pid}_{view}_ED"]["spacing_h"],
                         tracings[f"{pid}_{view}_ED"]["spacing_w"])
    if not ok:
        continue

    edv, esv = camus_ef.compute_left_ventricle_volumes(
        masks[("2CH", "ED")], masks[("2CH", "ES")], spacing["2CH"],
        masks[("4CH", "ED")], masks[("4CH", "ES")], spacing["4CH"],
    )
    ef = round(100 * (edv - esv) / edv)
    ref = parse_cfg(NIFTI / pid / "Info_4CH.cfg")
    ref_ef = float(ref["EF"])
    results.append({
        "patient": pid, "ef_computed": ef, "ef_reference": ref_ef,
        "edv_ml": round(edv, 1), "esv_ml": round(esv, 1),
        "delta": ef - ref_ef, "quality": ref.get("ImageQuality"),
    })

deltas = [r["delta"] for r in results]
n = len(results)
exact = sum(1 for d in deltas if abs(d) < 0.5)
w1 = sum(1 for d in deltas if abs(d) <= 1)
w2 = sum(1 for d in deltas if abs(d) <= 2)

print("=" * 74)
print("STEP 3 - EF GATE: vendored CAMUS Simpson vs the dataset's own EF values")
print("=" * 74)
print(f"  patients evaluated  : {n}")
print(f"  exact match         : {exact}/{n}  ({100*exact/n:.0f}%)")
print(f"  within +/-1 EF pt   : {w1}/{n}  ({100*w1/n:.0f}%)")
print(f"  within +/-2 EF pts  : {w2}/{n}  ({100*w2/n:.0f}%)")
print(f"  mean delta          : {statistics.fmean(deltas):+.3f} EF points")
print(f"  median |delta|      : {statistics.median(abs(d) for d in deltas):.1f}")
print(f"  max |delta|         : {max(abs(d) for d in deltas):.1f}")
print(f"  EDV range           : {min(r['edv_ml'] for r in results):.0f}-{max(r['edv_ml'] for r in results):.0f} mL")
print(f"  reference EF range  : {min(r['ef_reference'] for r in results):.0f}-{max(r['ef_reference'] for r in results):.0f} %")
worst = sorted(results, key=lambda r: -abs(r["delta"]))[:6]
print("  largest deviations  : " + ", ".join(f"{r['patient'].replace('patient','p')}({r['delta']:+.0f})" for r in worst))

(BENCH / "ef_gate.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print()
verdict = "PASS" if w1 == n else ("PASS (with noted rounding spread)" if w2 == n else "REVIEW")
print(f"  GATE VERDICT: {verdict}")
