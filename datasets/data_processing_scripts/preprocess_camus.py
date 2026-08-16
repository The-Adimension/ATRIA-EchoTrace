"""
CAMUS Dataset Preprocessor for MedGemma LV/LA Tracing Fine-tuning
=================================================================

Extracts ED/ES frames from NIfTI as PNGs and LV+LA contour polygons as JSON
from the CAMUS public echocardiography dataset.

Input:  CAMUS_public/database_nifti/patient0001..0500/
Output: camus_processed/
        ├── frames/                    # PNG files (grayscale→RGB)
        │   ├── patient0001_2CH_ED.png
        │   ├── patient0001_2CH_ES.png
        │   ├── patient0001_4CH_ED.png
        │   └── ...
        ├── tracings.json              # All 30-point polygons + image metadata
        ├── metadata.csv               # Patient-level clinical data + splits
        └── preprocessing_log.txt      # Processing summary and errors

Usage:
    python preprocess_camus.py
    python preprocess_camus.py --camus_root ./CAMUS_public --output_dir ./camus_processed --num_points 30
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from skimage.measure import find_contours

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "camus_root": os.path.join(os.path.dirname(__file__), "CAMUS_public"),
    "output_dir": os.path.join(os.path.dirname(__file__), "camus_processed"),
    "num_points": 30,
    "lv_label": 1,
    "la_label": 3,
}

VIEWS = ["2CH", "4CH"]
INSTANTS = ["ED", "ES"]

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_nifti(filepath: str):
    """Load a NIfTI file and return (numpy_array, spacing_tuple)."""
    image = sitk.ReadImage(str(filepath))
    arr = np.squeeze(sitk.GetArrayFromImage(image))
    spacing = image.GetSpacing()[:2]  # (width_spacing, height_spacing)
    return arr, spacing


def array_to_png(arr: np.ndarray, out_path: str):
    """Convert a float/uint array to uint8 RGB PNG and save."""
    arr_f = arr.astype(np.float32)
    low, high = arr_f.min(), arr_f.max()
    if high - low > 0:
        arr_u8 = ((arr_f - low) / (high - low) * 255).astype(np.uint8)
    else:
        arr_u8 = np.zeros_like(arr_f, dtype=np.uint8)
    img = Image.fromarray(arr_u8, mode="L").convert("RGB")
    img.save(out_path)
    return img.size  # (width, height)


def extract_contour_polygon(
    mask: np.ndarray, num_points: int, norm_scale: int = 1000
):
    """
    Extract a closed contour from a binary mask and return as a normalized
    polygon with `num_points` evenly-spaced points.

    Returns:
        polygon: list of [y, x] ints in [0, norm_scale], or None if no contour
        raw_point_count: int, number of points in the original contour
    """
    contours = find_contours(mask, 0.5)
    if not contours:
        return None, 0

    # Take the longest contour (main boundary)
    contour = max(contours, key=len)
    raw_count = len(contour)

    # Subsample evenly
    indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
    sub = contour[indices]

    # Normalize to [0, norm_scale]
    h, w = mask.shape
    polygon = [
        [int(np.clip(pt[0] / h * norm_scale, 0, norm_scale)),
         int(np.clip(pt[1] / w * norm_scale, 0, norm_scale))]
        for pt in sub
    ]
    return polygon, raw_count


def parse_cfg(cfg_path: str) -> dict:
    """Parse a CAMUS Info_*.cfg file into a dict."""
    info = {}
    try:
        with open(cfg_path, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, val = line.split(":", 1)
                    val = val.strip()
                    # Try numeric conversion
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    info[key.strip()] = val
    except FileNotFoundError:
        pass
    return info


def load_splits(split_dir: str) -> dict:
    """Load CAMUS train/val/test split files → {patient_id: split_name}."""
    split_map = {}
    for split_name, filename in [
        ("train", "subgroup_training.txt"),
        ("val", "subgroup_validation.txt"),
        ("test", "subgroup_testing.txt"),
    ]:
        filepath = os.path.join(split_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                for line in f:
                    pid = line.strip()
                    if pid:
                        split_map[pid] = split_name
    return split_map


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def preprocess_camus(
    camus_root: str,
    output_dir: str,
    num_points: int = 30,
    lv_label: int = 1,
    la_label: int = 3,
):
    """Process all CAMUS patients and produce frames/, tracings.json, metadata.csv."""

    db_nifti = os.path.join(camus_root, "database_nifti")
    db_split = os.path.join(camus_root, "database_split")
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(output_dir, "preprocessing_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("preprocess_camus")

    log.info("=" * 60)
    log.info("CAMUS Preprocessing")
    log.info("=" * 60)
    log.info(f"Source:      {db_nifti}")
    log.info(f"Output:      {output_dir}")
    log.info(f"Num points:  {num_points}")
    log.info(f"LV label:    {lv_label}")
    log.info(f"LA label:    {la_label}")

    # Load splits
    split_map = load_splits(db_split)
    log.info(f"Split map:   {len(split_map)} patients loaded")
    for s in ["train", "val", "test"]:
        count = sum(1 for v in split_map.values() if v == s)
        log.info(f"  {s}: {count}")

    # Discover patient folders
    patient_dirs = sorted([
        d for d in os.listdir(db_nifti)
        if d.startswith("patient") and os.path.isdir(os.path.join(db_nifti, d))
    ])
    log.info(f"Patients found: {len(patient_dirs)}")

    # Accumulators
    all_tracings = {}
    metadata_rows = []
    stats = defaultdict(int)
    t0 = time.time()

    for pi, patient_id in enumerate(patient_dirs):
        patient_dir = os.path.join(db_nifti, patient_id)
        split = split_map.get(patient_id, "unknown")

        if (pi + 1) % 50 == 0 or pi == 0:
            log.info(f"Processing {pi+1}/{len(patient_dirs)}: {patient_id}")

        for view in VIEWS:
            # Parse cfg for metadata
            cfg_path = os.path.join(patient_dir, f"Info_{view}.cfg")
            cfg = parse_cfg(cfg_path)

            for instant in INSTANTS:
                key = f"{patient_id}_{view}_{instant}"
                img_nifti = os.path.join(patient_dir, f"{key}.nii.gz")
                gt_nifti = os.path.join(patient_dir, f"{key}_gt.nii.gz")

                # Check files exist
                if not os.path.exists(img_nifti):
                    log.warning(f"SKIP {key}: image not found")
                    stats["skipped_no_image"] += 1
                    continue
                if not os.path.exists(gt_nifti):
                    log.warning(f"SKIP {key}: GT mask not found")
                    stats["skipped_no_gt"] += 1
                    continue

                try:
                    # Load image and save as PNG
                    img_arr, spacing = load_nifti(img_nifti)
                    h, w = img_arr.shape
                    png_path = os.path.join(frames_dir, f"{key}.png")
                    array_to_png(img_arr, png_path)

                    # Load GT mask
                    gt_arr, _ = load_nifti(gt_nifti)

                    # Extract LV contour
                    lv_mask = (gt_arr == lv_label).astype(np.uint8)
                    lv_poly, lv_raw = extract_contour_polygon(lv_mask, num_points)

                    # Extract LA contour
                    la_mask = (gt_arr == la_label).astype(np.uint8)
                    la_poly, la_raw = extract_contour_polygon(la_mask, num_points)

                    if lv_poly is None:
                        log.warning(f"SKIP {key}: no LV contour found in GT")
                        stats["skipped_no_lv"] += 1
                        continue

                    # Build tracing entry
                    entry = {
                        "patient_id": patient_id,
                        "view": view,
                        "instant": instant,
                        "image_file": f"{key}.png",
                        "image_h": int(h),
                        "image_w": int(w),
                        "spacing_h": float(spacing[1]),
                        "spacing_w": float(spacing[0]),
                        "lv_polygon": lv_poly,
                        "la_polygon": la_poly,
                        "lv_points_raw": lv_raw,
                        "la_points_raw": la_raw,
                        "split": split,
                    }
                    all_tracings[key] = entry
                    stats["success"] += 1
                    if la_poly is None:
                        stats["missing_la"] += 1

                    # Build metadata row
                    metadata_rows.append({
                        "key": key,
                        "patient_id": patient_id,
                        "view": view,
                        "instant": instant,
                        "split": split,
                        "image_h": int(h),
                        "image_w": int(w),
                        "ef": cfg.get("EF", ""),
                        "age": cfg.get("Age", ""),
                        "sex": cfg.get("Sex", ""),
                        "image_quality": cfg.get("ImageQuality", ""),
                        "has_lv": lv_poly is not None,
                        "has_la": la_poly is not None,
                        "lv_points": num_points if lv_poly else 0,
                        "la_points": num_points if la_poly else 0,
                    })

                except Exception as e:
                    log.error(f"ERROR {key}: {e}")
                    stats["errors"] += 1

    elapsed = time.time() - t0

    # Save tracings.json
    tracings_path = os.path.join(output_dir, "tracings.json")
    with open(tracings_path, "w") as f:
        json.dump(all_tracings, f, indent=None, separators=(",", ":"))
    log.info(f"Saved tracings: {tracings_path} ({len(all_tracings)} entries)")

    # Save metadata.csv
    csv_path = os.path.join(output_dir, "metadata.csv")
    if metadata_rows:
        fieldnames = list(metadata_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)
    log.info(f"Saved metadata: {csv_path} ({len(metadata_rows)} rows)")

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("PREPROCESSING COMPLETE")
    log.info("=" * 60)
    log.info(f"Time elapsed:        {elapsed:.1f}s")
    log.info(f"Patients processed:  {len(patient_dirs)}")
    log.info(f"Images extracted:    {stats['success']}")
    log.info(f"  Missing LA:        {stats['missing_la']}")
    log.info(f"  Skipped (no img):  {stats['skipped_no_image']}")
    log.info(f"  Skipped (no GT):   {stats['skipped_no_gt']}")
    log.info(f"  Skipped (no LV):   {stats['skipped_no_lv']}")
    log.info(f"  Errors:            {stats['errors']}")
    log.info("")

    # Split breakdown
    split_counts = defaultdict(lambda: defaultdict(int))
    for entry in all_tracings.values():
        split_counts[entry["split"]][entry["view"] + "_" + entry["instant"]] += 1
    for split_name in ["train", "val", "test", "unknown"]:
        if split_name in split_counts:
            total = sum(split_counts[split_name].values())
            breakdown = ", ".join(
                f"{k}={v}" for k, v in sorted(split_counts[split_name].items())
            )
            log.info(f"  {split_name:8s}: {total:4d} images  ({breakdown})")

    log.info("")
    log.info(f"Output directory: {output_dir}")
    log.info(f"  frames/         {stats['success']} PNGs")
    log.info(f"  tracings.json   {len(all_tracings)} entries")
    log.info(f"  metadata.csv    {len(metadata_rows)} rows")

    return all_tracings, metadata_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess CAMUS dataset for MedGemma LV/LA tracing"
    )
    parser.add_argument(
        "--camus_root", type=str, default=DEFAULTS["camus_root"],
        help="Path to CAMUS_public/ directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULTS["output_dir"],
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--num_points", type=int, default=DEFAULTS["num_points"],
        help="Number of polygon points per contour (default: 30)"
    )
    parser.add_argument(
        "--lv_label", type=int, default=DEFAULTS["lv_label"],
        help="GT mask label for LV cavity (default: 1)"
    )
    parser.add_argument(
        "--la_label", type=int, default=DEFAULTS["la_label"],
        help="GT mask label for left atrium (default: 3)"
    )
    args = parser.parse_args()

    preprocess_camus(
        camus_root=args.camus_root,
        output_dir=args.output_dir,
        num_points=args.num_points,
        lv_label=args.lv_label,
        la_label=args.la_label,
    )
