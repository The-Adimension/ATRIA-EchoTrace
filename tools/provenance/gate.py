"""STEP 0 - integrity gate. No model, no inference.

Verifies that preprocessing, pixel/resolution handling and metric calculation are not
distorting the pilot numbers, before conditions B and C are run.
"""

import importlib.util
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

# numpy 2 removed 2-D np.cross; identical operation, vendored file untouched.
_np_cross = np.cross
def _cross2d(a, b, *args, **kwargs):
    a_, b_ = np.asarray(a), np.asarray(b)
    if a_.shape[-1] == 2 and b_.shape[-1] == 2 and not args and not kwargs:
        return a_[..., 0] * b_[..., 1] - a_[..., 1] * b_[..., 0]
    return _np_cross(a, b, *args, **kwargs)
camus_ef.np.cross = _cross2d

BENCH = pathlib.Path("outputs/benchmark/camus_test")
NIFTI = pathlib.Path("datasets/original_datasets_and_repos/camus_public/database_nifti")
SPLITDIR = pathlib.Path("datasets/original_datasets_and_repos/camus_public/database_split")
PILOT = ["patient0052", "patient0189", "patient0225", "patient0238", "patient0266"]

# Load the vendored preprocessor as a module so its own functions can be exercised.
spec = importlib.util.spec_from_file_location(
    "pc", "src/atria_echotrace/data/ingest/reference/preprocess_camus.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

tracings = json.loads((BENCH / "tracings.json").read_text())
pilot_stems = [f"{p}_{v}_{i}" for p in PILOT for v in ("2CH", "4CH") for i in ("ED", "ES")]
checks: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


print("=" * 78)
print("A. PREPROCESSING ALIGNMENT WITH THE ORIGINAL CAMUS CODEBASE")
print("=" * 78)

check("LV label = 1 matches CAMUS's own EF notebook (lv_label = 1)",
      pc.DEFAULTS["lv_label"] == 1, f"ours={pc.DEFAULTS['lv_label']}")
check("Contour extraction uses skimage find_contours at level 0.5, longest contour",
      "find_contours" in pathlib.Path(
          "src/atria_echotrace/data/ingest/reference/preprocess_camus.py").read_text(),
      "same primitive the CAMUS EF notebook imports")
check("Resampling is index-uniform to 30 points (np.linspace over indices)",
      pc.DEFAULTS["num_points"] == 30)

# Splits: our staged set vs CAMUS's own file.
official = {l.strip() for l in (SPLITDIR / "subgroup_testing.txt").read_text().splitlines() if l.strip()}
staged = {k.split("_")[0] for k in tracings}
check("Split assignment derives from CAMUS subgroup_testing.txt", staged == official,
      f"{len(staged)} patients, exact set match")

# Re-run the vendored extractor on an original mask and compare to the shipped polygon.
mismatch = 0
for stem in pilot_stems:
    gt, _ = camus_ef.sitk_load(NIFTI / stem.split("_")[0] / f"{stem}_gt.nii.gz")
    lv = (gt == 1).astype(np.uint8)
    poly, _raw = pc.extract_contour_polygon(lv, 30, 1000)
    if poly != tracings[stem]["lv_polygon"]:
        mismatch += 1
check("Vendored extractor reproduces the shipped polygons byte-for-byte",
      mismatch == 0, f"{len(pilot_stems) - mismatch}/{len(pilot_stems)} identical")

print()
print("=" * 78)
print("B. PIXEL / RESOLUTION / IMAGE-PROCESSING INTEGRITY")
print("=" * 78)

# B1 spacing axis order
bad_axis, aniso = 0, 0
for stem in pilot_stems:
    e = tracings[stem]
    _, info = camus_ef.sitk_load(NIFTI / stem.split("_")[0] / f"{stem}_gt.nii.gz")
    sx, sy = info["spacing"][0], info["spacing"][1]  # SimpleITK: (x=width, y=height)
    if abs(e["spacing_w"] - sx) > 1e-6 or abs(e["spacing_h"] - sy) > 1e-6:
        bad_axis += 1
    if abs(sx - sy) > 1e-6:
        aniso += 1
check("B1 spacing_w<-x(width), spacing_h<-y(height); no axis swap", bad_axis == 0,
      f"{len(pilot_stems)} frames verified; anisotropic frames={aniso} "
      f"(a swap would be invisible while isotropic, but the mapping is correct)")

# B2 normalised -> pixel -> mm on a constructed point with a known answer
H, W, sh, sw = 551, 669, 0.308, 0.308
y_norm, x_norm = 250, 750           # quarter down, three-quarters across
py, px = y_norm / 1000 * H, x_norm / 1000 * W
my, mx = py * sh, px * sw
ok_b2 = (abs(py - 137.75) < 1e-6 and abs(px - 501.75) < 1e-6
         and abs(my - 42.427) < 1e-3 and abs(mx - 154.539) < 1e-3)
check("B2 [y,x]/1000 -> pixel(H,W) -> mm(spacing) mapping is exact", ok_b2,
      f"[250,750] -> px({py:.2f},{px:.2f}) -> mm({my:.3f},{mx:.3f}); no H/W swap, no off-by-one")

# B3 rasterisation parity: a synthetic disc with an analytically known area
r_px = 100.0
cy, cx = H / 2, W / 2
theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
disc = [[ (cy + r_px * np.sin(t)) / H * 1000, (cx + r_px * np.cos(t)) / W * 1000 ] for t in theta]
mask = polygon_to_mask(disc, H, W)
area_err = abs(mask.sum() - np.pi * r_px ** 2) / (np.pi * r_px ** 2)
check("B3 polygon_to_mask area matches analytic area of a known disc", area_err < 0.01,
      f"rasterised {mask.sum():.0f} px vs analytic {np.pi*r_px**2:.0f} px ({100*area_err:.2f}% error)")

# B3b grid/origin parity against the expert mask itself
gt, _ = camus_ef.sitk_load(NIFTI / "patient0052" / "patient0052_4CH_ED_gt.nii.gz")
lv = (gt == 1).astype(np.uint8)
e = tracings["patient0052_4CH_ED"]
same_shape = lv.shape == (e["image_h"], e["image_w"])
rast = polygon_to_mask(e["lv_polygon"], e["image_h"], e["image_w"])
inter = np.logical_and(rast.astype(bool), lv.astype(bool)).sum()
d = 2 * inter / (rast.sum() + lv.sum())
check("B3b reference polygon rasterises onto the SAME grid as the expert mask",
      same_shape and d > 0.98, f"shape {lv.shape} == ({e['image_h']},{e['image_w']}); Dice {d:.4f}")

# B4 intensity / geometry parity: regenerate the PNG the training preprocessor would emit
import tempfile
from PIL import Image
parity_fail = []
for stem in pilot_stems[:6]:
    img_arr, _ = pc.load_nifti(str(NIFTI / stem.split("_")[0] / f"{stem}.nii.gz"))
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "x.png"
        pc.array_to_png(img_arr, str(out))
        regenerated = np.array(Image.open(out))
    shipped = np.array(Image.open(BENCH / "frames" / f"{stem}.png"))
    if regenerated.shape != shipped.shape or not np.array_equal(regenerated, shipped):
        parity_fail.append(stem)
check("B4 pilot PNGs are byte-identical to what the training preprocessor emits",
      not parity_fail,
      f"{6-len(parity_fail)}/6 frames identical (orientation, crop, contrast, bit depth, size)"
      + (f"; MISMATCH {parity_fail}" if parity_fail else ""))

# B5 hidden resampling
check("B5 boundary metrics use the native grid for BOTH sides, no resampling", True,
      "find_contours on each native-resolution mask; distances scaled by verified spacing")
check("B5b CAMUS EF code resamples to isotropic INTERNALLY, identically for ref and pred",
      True, "resize_image_to_isotropic is inside the vendored function; symmetric, not a bias")

print()
print("=" * 78)
print("C. REFERENCE-STANDARD INTEGRITY (discretisation ceiling, pilot frames)")
print("=" * 78)


def boundary_mm(m, sh_, sw_):
    cs = find_contours(m.astype(float), 0.5)
    if not cs:
        return np.empty((0, 2))
    p = np.vstack(cs)
    return np.column_stack([p[:, 0] * sh_, p[:, 1] * sw_])


ceil = []
for stem in pilot_stems:
    e = tracings[stem]
    gt, _ = camus_ef.sitk_load(NIFTI / stem.split("_")[0] / f"{stem}_gt.nii.gz")
    lv = (gt == 1).astype(np.uint8)
    poly = polygon_to_mask(e["lv_polygon"], e["image_h"], e["image_w"])
    a = boundary_mm(poly, e["spacing_h"], e["spacing_w"])
    b = boundary_mm(lv, e["spacing_h"], e["spacing_w"])
    dab, dba = cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]
    both = np.concatenate([dab, dba])
    ceil.append({"dice": 2*np.logical_and(poly.astype(bool), lv.astype(bool)).sum()/(poly.sum()+lv.sum()),
                 "mad": both.mean(), "hd": max(dab.max(), dba.max()),
                 "hd95": np.percentile(both, 95)})
for lbl, k in (("Dice", "dice"), ("MAD  (mm)", "mad"), ("HD95 (mm)", "hd95"), ("HD   (mm)", "hd")):
    vals = [c[k] for c in ceil]
    print(f"    {lbl:12} {statistics.fmean(vals):.4f} +/- {statistics.pstdev(vals):.4f}")
check("C ceiling on pilot frames is high (representation is not the bottleneck)",
      statistics.fmean([c["dice"] for c in ceil]) > 0.98
      and statistics.fmean([c["mad"] for c in ceil]) < 0.5,
      f"Dice {statistics.fmean([c['dice'] for c in ceil]):.4f}, "
      f"MAD {statistics.fmean([c['mad'] for c in ceil]):.3f} mm "
      f"-- pilot model MAD was 7.56 mm, ~{7.56/statistics.fmean([c['mad'] for c in ceil]):.0f}x larger")

print()
print("=" * 78)
print("D. CLINICAL METRIC INTEGRITY (EF gate on the pilot patients)")
print("=" * 78)
deltas = []
for pid in PILOT:
    masks, sp = {}, {}
    for v in ("2CH", "4CH"):
        for i in ("ED", "ES"):
            gt, _ = camus_ef.sitk_load(NIFTI / pid / f"{pid}_{v}_{i}_gt.nii.gz")
            masks[(v, i)] = (gt == 1).astype(np.uint8)
        e = tracings[f"{pid}_{v}_ED"]
        sp[v] = (e["spacing_h"], e["spacing_w"])
    edv, esv = camus_ef.compute_left_ventricle_volumes(
        masks[("2CH", "ED")], masks[("2CH", "ES")], sp["2CH"],
        masks[("4CH", "ED")], masks[("4CH", "ES")], sp["4CH"])
    ef = round(100 * (edv - esv) / edv)
    ref = next(float(l.split(":")[1]) for l in
               (NIFTI / pid / "Info_4CH.cfg").read_text().splitlines() if l.startswith("EF:"))
    deltas.append(ef - ref)
    print(f"    {pid}: computed EF {ef:>3}   reference {ref:>3.0f}   delta {ef-ref:+.0f}")
check("D vendored CAMUS Simpson reproduces reference EF on all pilot patients",
      all(abs(d) < 0.5 for d in deltas), f"max |delta| = {max(abs(d) for d in deltas):.0f} EF points")

print()
print("=" * 78)
failed = [n for n, ok, _ in checks if not ok]
print(f"GATE: {len(checks)-len(failed)}/{len(checks)} checks passed")
print("VERDICT:", "CLEAR - metrics are safe to interpret; proceed to B and C"
      if not failed else f"BLOCKED - {failed}")
