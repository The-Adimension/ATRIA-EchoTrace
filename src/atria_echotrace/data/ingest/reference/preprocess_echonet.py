"""
EchoNet-Dynamic Preprocessor for Unified LV Tracing Fine-tuning
================================================================

Mirrors CAMUS-style output: PNG frames + tracings.json + metadata.csv

Extracts ED/ES frames from EchoNet AVI videos and converts VolumeTracings
chord format into 30-point normalized LV polygons (no LA — not annotated).

Input:  EchoNet-Dynamic/ with FileList.csv, VolumeTracings.csv, Videos/
Output: echonet_processed/
        ├── frames/                         # PNG files (RGB)
        │   ├── echonet_0X..._4CH_ED.png
        │   ├── echonet_0X..._4CH_ES.png
        │   └── ...
        ├── tracings.json                   # 30-point LV polygons + metadata
        ├── metadata.csv                    # Same schema as camus_processed
        └── preprocessing_log.txt

Optionally merges with existing camus_processed/ into unified_processed/
(writes combined tracings.json + metadata.csv; frames stay in their dirs).

Usage:
    python preprocess_echonet.py
    python preprocess_echonet.py --echonet_root ./EchoNet-Dynamic --output_dir ./echonet_processed
    python preprocess_echonet.py --merge_camus ./camus_processed --unified_dir ./unified_processed
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless machines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "echonet_root": os.path.join(os.path.dirname(__file__), "CAMUS_public/../EchoNet-Dynamic"),
    "output_dir": os.path.join(os.path.dirname(__file__), "echonet_processed"),
    "num_points": 30,
}

# Metadata CSV columns — matches camus_processed schema + source
METADATA_FIELDS = [
    "key", "patient_id", "view", "instant", "split",
    "image_h", "image_w", "ef", "age", "sex", "image_quality",
    "has_lv", "has_la", "lv_points", "la_points", "source",
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def build_lv_polygon(
    trace_rows: np.ndarray,
    img_h: int,
    img_w: int,
    num_points: int,
    norm_scale: int = 1000,
):
    """
    Convert EchoNet VolumeTracings chord rows into a normalised polygon.

    Uses the same construction as echo.py (EchoNet-Dynamic-Repo):
        - Skip row 0 (long-axis apex→mitral marker)
        - Left boundary:  x1[1:], y1[1:]
        - Right boundary: x2[1:], y2[1:] (reversed)
        - Concatenate to form a closed polygon

    Returns:
        polygon : list of [y_norm, x_norm] ints in [0, norm_scale], matching
                  CAMUS tracing format, or None if too few points.
        raw_count : int, number of polygon points before resampling.
    """
    if len(trace_rows) < 2:
        return None, 0

    x1, y1, x2, y2 = (
        trace_rows[:, 0], trace_rows[:, 1],
        trace_rows[:, 2], trace_rows[:, 3],
    )
    x = np.concatenate((x1[1:], np.flip(x2[1:])))
    y = np.concatenate((y1[1:], np.flip(y2[1:])))

    raw_count = len(x)
    if raw_count < 3:
        return None, raw_count

    # Resample evenly to num_points (same approach as preprocess_camus.py)
    indices = np.linspace(0, raw_count - 1, num_points, dtype=int)
    x_sub = x[indices]
    y_sub = y[indices]

    # Normalise to [0, norm_scale] as [y, x] pairs
    polygon = [
        [
            int(np.clip(y_pt / img_h * norm_scale, 0, norm_scale)),
            int(np.clip(x_pt / img_w * norm_scale, 0, norm_scale)),
        ]
        for y_pt, x_pt in zip(y_sub, x_sub)
    ]
    return polygon, raw_count


def extract_frame_png(video_path: str, frame_idx: int, out_path: str, target_size: int = None):
    """
    Extract a single frame from an AVI file and save as RGB PNG.

    If target_size is given, the frame is upscaled to (target_size × target_size)
    using Lanczos resampling before saving.  Stored image_h / image_w in 
    tracings.json will reflect the upscaled dimensions, which keeps the
    resolution-independent [0, 1000] polygon coordinates fully aligned.

    Returns (width, height) on success, None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb, mode="RGB")
    if target_size is not None:
        pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
    pil_img.save(out_path)
    w, h = pil_img.size
    return w, h


# ---------------------------------------------------------------------------
# FileList augmentation
# ---------------------------------------------------------------------------


def derive_frame_indices(tracings_csv):
    """
    Derive the (esf, edf) frame indices for every video from VolumeTracings.csv.

    The official Stanford release does not publish these columns; each video simply has
    exactly two traced frames in VolumeTracings.csv. The rule used here is **temporal**:

        esf = lower frame number      edf = higher frame number

    Verified: this reproduces all 20 048 EchoNet polygons of the shipped corpus exactly
    (100 %), so a corpus rebuilt from the official release stays byte-compatible with the
    published EchoNet adapter. EchoNet's own loader (EchoNet-Dynamic-Repo/echonet/
    datasets/echo.py, LargeIndex/SmallIndex) instead picks by *file-appearance* order;
    the two agree for 99 % of videos but disagree for 102, where file order is not
    temporal order, so file order reproduces only 98.98 % and is not used.

    This is **not** the physiological ordering: measured over the 10 024 traced videos,
    the frame labelled ED here carries the *smaller* cross-sectional area in ~99 % of
    cases (the documented EchoNet ED/ES transposition). It is kept because it is the
    convention the shipped corpus and adapter were built under. Assigning by measured
    area instead would be physiologically correct but would relabel ~99 % of the corpus
    and push every EchoNet request outside the adapter's training distribution, so it
    remains a future opt-in that would require retraining.

    Returns:
        dict[str, tuple[int, int]] mapping FileName -> (esf, edf).
    """
    frames = defaultdict(set)
    with open(tracings_csv, "r") as f:
        header = f.readline().strip().split(",")
        assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"], (
            f"Unexpected VolumeTracings header: {header}"
        )
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 6:
                continue
            frames[parts[0]].add(int(float(parts[5])))

    # EchoNet drops videos with fewer than two traced frames (echo.py keep-filter).
    return {
        name: (min(found), max(found))
        for name, found in frames.items()
        if len(found) >= 2
    }


def ensure_filelist_has_frame_indices(filelist_csv, tracings_csv, log):
    """
    Guarantee FileList.csv carries esf/edf, deriving them when the file is the original.

    A user who downloaded only the official Stanford release has no esf/edf columns. In
    that case the original is backed up verbatim to ``original_FileList.csv`` and a new
    FileList.csv is written with every original column plus the two derived ones, so the
    rest of this script - and anything else reading FileList.csv - is unchanged.

    A FileList that already carries both columns is left untouched.
    """
    df = pd.read_csv(filelist_csv)
    if "esf" in df.columns and "edf" in df.columns:
        log.info("FileList already carries esf/edf; leaving it untouched.")
        return

    derived = derive_frame_indices(tracings_csv)
    backup = os.path.join(os.path.dirname(filelist_csv) or ".", "original_FileList.csv")
    if not os.path.exists(backup):
        shutil.copy2(filelist_csv, backup)
        log.info(f"Backed up the original FileList to {backup}")
    else:
        log.info(f"Backup already present, not overwriting: {backup}")

    # Int64 (nullable) keeps these written as "46", not "46.0", for the videos that have
    # fewer than two traced frames and therefore no index.
    df["esf"] = df["FileName"].map(lambda n: derived.get(n, (None, None))[0]).astype("Int64")
    df["edf"] = df["FileName"].map(lambda n: derived.get(n, (None, None))[1]).astype("Int64")
    missing = int(df["esf"].isna().sum())
    df.to_csv(filelist_csv, index=False)
    log.info(
        f"Derived esf/edf from VolumeTracings for {len(df) - missing}/{len(df)} videos "
        f"(esf=lower frame number, edf=higher; reproduces the shipped corpus exactly). "
        f"NOTE: this is the corpus/adapter convention, not the physiological one - the "
        f"frame labelled ED is the smaller in ~99% of videos. Assigning by measured area "
        f"is a future opt-in and would require retraining. Wrote {filelist_csv}"
    )
    if missing:
        log.warning(f"{missing} video(s) have <2 traced frames and will be skipped.")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def preprocess_echonet(
    echonet_root: str,
    output_dir: str,
    num_points: int = 30,
    max_videos: int = None,  # type: ignore[assignment]
    target_size: int = None,  # type: ignore[assignment]
):
    """
    Process all EchoNet-Dynamic videos → echonet_processed/ in CAMUS format.

    target_size: if set, frames are Lanczos-upscaled to (target_size × target_size).
    """
    videos_dir = os.path.join(echonet_root, "Videos")
    filelist_csv = os.path.join(echonet_root, "FileList.csv")
    tracings_csv = os.path.join(echonet_root, "VolumeTracings.csv")
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(output_dir, "preprocessing_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("preprocess_echonet")

    log.info("=" * 60)
    log.info("EchoNet-Dynamic Preprocessing")
    log.info("=" * 60)
    log.info(f"Source:      {echonet_root}")
    log.info(f"Output:      {output_dir}")
    log.info(f"Num points:  {num_points}")
    log.info(f"Target size: {target_size if target_size else 'native (112×112)'}")

    # ---------- Ensure FileList carries esf/edf ----------
    # The official Stanford release does not publish these columns; they must be derived
    # from VolumeTracings.csv. Doing it here means a user who downloaded only the
    # official release can run this script end-to-end.
    ensure_filelist_has_frame_indices(filelist_csv, tracings_csv, log)

    # ---------- Load FileList ----------
    df = pd.read_csv(filelist_csv)
    df["Split"] = df["Split"].str.lower()
    log.info(f"FileList:    {len(df)} videos")
    if max_videos is not None:
        df = df.head(max_videos)
        log.info(f"  (capped to {max_videos} for this run)")
    for s in ["train", "val", "test"]:
        log.info(f"  {s}: {(df['Split'] == s).sum()}")

    # ---------- Load VolumeTracings ----------
    # trace[filename][frame_int] = list of (x1, y1, x2, y2)
    trace = defaultdict(lambda: defaultdict(list))
    with open(tracings_csv, "r") as f:
        header = f.readline().strip().split(",")
        assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"], (
            f"Unexpected VolumeTracings header: {header}"
        )
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 6:
                continue
            filename, x1, y1, x2, y2, frame = parts
            trace[filename][int(float(frame))].append(
                (float(x1), float(y1), float(x2), float(y2))
            )
    log.info(f"Traced videos: {len(trace)}")

    # ---------- Process each video ----------
    all_tracings = {}
    metadata_rows = []
    stats = defaultdict(int)
    t0 = time.time()

    for pi, row in enumerate(df.itertuples(index=False)):
        filename = str(row.FileName)
        split = row.Split

        # Skip rows where esf/edf are missing (NaN in FileList.csv)
        try:
            esf = int(row.esf)   # end-systole frame index
            edf = int(row.edf)   # end-diastole frame index
        except (ValueError, TypeError):
            log.warning(f"SKIP {filename}: missing esf/edf frame index")
            stats["skipped_no_trace"] += 1
            continue

        ef = float(row.EF)
        img_h = int(row.FrameHeight)
        img_w = int(row.FrameWidth)

        if (pi + 1) % 1000 == 0 or pi == 0:
            log.info(f"Processing {pi + 1}/{len(df)}: {filename}")

        video_path = os.path.join(videos_dir, f"{filename}.avi")
        if not os.path.exists(video_path):
            log.warning(f"SKIP {filename}: video not found")
            stats["skipped_no_video"] += 1
            continue

        patient_id = f"echonet_{filename}"

        for instant, frame_idx in [("ES", esf), ("ED", edf)]:
            key = f"{patient_id}_4CH_{instant}"

            # Verify tracing exists for this frame
            if frame_idx not in trace[filename]:
                log.warning(f"SKIP {key}: no tracing for frame {frame_idx}")
                stats["skipped_no_trace"] += 1
                continue

            t_rows = np.array(trace[filename][frame_idx])  # (n, 4)

            # Build LV polygon from chords
            lv_poly, lv_raw = build_lv_polygon(t_rows, img_h, img_w, num_points)
            if lv_poly is None:
                log.warning(f"SKIP {key}: degenerate polygon ({lv_raw} pts)")
                stats["skipped_bad_polygon"] += 1
                continue

            # Extract frame
            png_path = os.path.join(frames_dir, f"{key}.png")
            size = extract_frame_png(video_path, frame_idx, png_path, target_size)
            if size is None:
                log.warning(f"SKIP {key}: frame extraction failed")
                stats["skipped_frame_error"] += 1
                continue

            w_actual, h_actual = size

            all_tracings[key] = {
                "patient_id": patient_id,
                "view": "4CH",
                "instant": instant,
                "image_file": f"{key}.png",
                "image_h": h_actual,
                "image_w": w_actual,
                "spacing_h": 1.0,
                "spacing_w": 1.0,
                "lv_polygon": lv_poly,
                "la_polygon": None,
                "lv_points_raw": lv_raw,
                "la_points_raw": 0,
                "split": split,
                "source": "echonet",
                "ef": ef,
            }
            stats["success"] += 1

            metadata_rows.append({
                "key": key,
                "patient_id": patient_id,
                "view": "4CH",
                "instant": instant,
                "split": split,
                "image_h": h_actual,
                "image_w": w_actual,
                "ef": ef,
                "age": "",
                "sex": "",
                "image_quality": "",
                "has_lv": True,
                "has_la": False,
                "lv_points": num_points,
                "la_points": 0,
                "source": "echonet",
            })

    elapsed = time.time() - t0

    # Save outputs
    tracings_path = os.path.join(output_dir, "tracings.json")
    with open(tracings_path, "w") as f:
        json.dump(all_tracings, f, indent=None, separators=(",", ":"))
    log.info(f"Saved tracings: {tracings_path} ({len(all_tracings)} entries)")

    csv_path = os.path.join(output_dir, "metadata.csv")
    if metadata_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(metadata_rows)
    log.info(f"Saved metadata: {csv_path} ({len(metadata_rows)} rows)")

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("PREPROCESSING COMPLETE")
    log.info("=" * 60)
    log.info(f"Time elapsed:          {elapsed:.1f}s")
    log.info(f"Videos in FileList:    {len(df)}")
    log.info(f"Images extracted:      {stats['success']}")
    log.info(f"  Skipped (no video):  {stats['skipped_no_video']}")
    log.info(f"  Skipped (no trace):  {stats['skipped_no_trace']}")
    log.info(f"  Skipped (bad poly):  {stats['skipped_bad_polygon']}")
    log.info(f"  Skipped (frame err): {stats['skipped_frame_error']}")
    log.info("")
    split_counts = defaultdict(int)
    for entry in all_tracings.values():
        split_counts[entry["split"]] += 1
    for s in ["train", "val", "test"]:
        log.info(f"  {s:8s}: {split_counts[s]:5d} images")
    log.info(f"  total:    {stats['success']:5d} images")
    log.info("")
    log.info(f"Output: {output_dir}")

    return all_tracings, metadata_rows


# ---------------------------------------------------------------------------
# Merge utility
# ---------------------------------------------------------------------------


def merge_datasets(
    camus_processed: str,
    echonet_processed: str,
    unified_dir: str,
    log,
):
    """
    Combine camus_processed/ and echonet_processed/ into unified_processed/.

    Copies ALL frames into unified_processed/frames/ so the folder is self-contained
    and can be zipped and uploaded to Colab unchanged (just set CAMUS_ROOT to the
    unified_processed/ path in the notebook).

    Schema is identical to camus_processed — every entry has the same fields;
    EchoNet entries have la_polygon=null and a 'source'='echonet' field.
    """
    unified_frames_dir = os.path.join(unified_dir, "frames")
    os.makedirs(unified_frames_dir, exist_ok=True)

    # ---------- Load tracings ----------
    with open(os.path.join(camus_processed, "tracings.json")) as f:
        camus_tracings = json.load(f)
    for entry in camus_tracings.values():
        entry.setdefault("source", "camus")
    log.info(f"CAMUS tracings:   {len(camus_tracings)} entries")

    with open(os.path.join(echonet_processed, "tracings.json")) as f:
        echonet_tracings = json.load(f)
    log.info(f"EchoNet tracings: {len(echonet_tracings)} entries")

    # ---------- Copy frames ----------
    log.info("Copying CAMUS frames...")
    camus_frames_src = os.path.join(camus_processed, "frames")
    n_camus = 0
    for fname in os.listdir(camus_frames_src):
        if fname.endswith(".png"):
            shutil.copy2(
                os.path.join(camus_frames_src, fname),
                os.path.join(unified_frames_dir, fname),
            )
            n_camus += 1
    log.info(f"  Copied {n_camus} CAMUS frames")

    log.info("Copying EchoNet frames...")
    echonet_frames_src = os.path.join(echonet_processed, "frames")
    n_echonet = 0
    for fname in os.listdir(echonet_frames_src):
        if fname.endswith(".png"):
            shutil.copy2(
                os.path.join(echonet_frames_src, fname),
                os.path.join(unified_frames_dir, fname),
            )
            n_echonet += 1
    log.info(f"  Copied {n_echonet} EchoNet frames")
    log.info(f"  Total frames in unified: {n_camus + n_echonet}")

    # ---------- Write combined tracings.json ----------
    unified = {**camus_tracings, **echonet_tracings}
    unified_path = os.path.join(unified_dir, "tracings.json")
    with open(unified_path, "w") as f:
        json.dump(unified, f, indent=None, separators=(",", ":"))
    log.info(f"Unified tracings: {unified_path} ({len(unified)} entries)")

    # ---------- Write combined metadata.csv ----------
    rows = []
    with open(os.path.join(camus_processed, "metadata.csv"), newline="") as f:
        for row in csv.DictReader(f):
            row.setdefault("source", "camus")
            rows.append(row)
    with open(os.path.join(echonet_processed, "metadata.csv"), newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if rows:
        seen = set(METADATA_FIELDS)
        all_fields = list(METADATA_FIELDS)
        for row in rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    all_fields.append(k)
        for row in rows:
            for fn in all_fields:
                row.setdefault(fn, "")

        unified_csv = os.path.join(unified_dir, "metadata.csv")
        with open(unified_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Unified metadata: {unified_csv} ({len(rows)} rows)")

    log.info(f"Unified dataset ready: {unified_dir}")
    return unified, unified_dir


# ---------------------------------------------------------------------------
# Visual verification
# ---------------------------------------------------------------------------

NORM_SCALE = 1000


def _draw_polygon_on_image(
    img_arr: np.ndarray,
    polygon: list,
    color_fill,
    color_outline,
) -> np.ndarray:
    """Draw a filled + outlined polygon onto an RGB numpy array."""
    pil_img = Image.fromarray(img_arr).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    h, w = img_arr.shape[:2]
    pts = [
        (p[1] / NORM_SCALE * w, p[0] / NORM_SCALE * h)
        for p in polygon
    ]
    if len(pts) >= 3:
        draw.polygon(pts, fill=color_fill, outline=color_outline)
        for x, y in pts:
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color_outline)
    result = Image.alpha_composite(pil_img, overlay).convert("RGB")
    return np.array(result)


def verify_output(
    tracings: dict,
    frames_dir: str,
    verify_dir: str,
    log,
    n_samples: int = 8,
):
    """
    Save a grid of sample frames with LV polygon overlays to verify_dir/.

    Picks up to n_samples entries, balanced between CAMUS and EchoNet sources.
    Each sample shows: original | LV overlay | (LA overlay if available).
    Saves individual per-sample PNGs and one combined grid PNG.
    """
    os.makedirs(verify_dir, exist_ok=True)

    # Separate by source for balanced sampling
    camus_keys = [k for k, v in tracings.items() if v.get("source", "camus") == "camus"]
    echonet_keys = [k for k, v in tracings.items() if v.get("source", "echonet") == "echonet"]

    half = n_samples // 2
    sample_keys = []
    # Pick evenly spaced indices for reproducibility
    if camus_keys:
        idx = np.linspace(0, len(camus_keys) - 1, min(half, len(camus_keys)), dtype=int)
        sample_keys += [camus_keys[i] for i in idx]
    if echonet_keys:
        idx = np.linspace(0, len(echonet_keys) - 1, min(half, len(echonet_keys)), dtype=int)
        sample_keys += [echonet_keys[i] for i in idx]

    log.info(f"Saving {len(sample_keys)} verification samples to {verify_dir}/")

    grid_items = []  # (composite_image_arr, title)

    for key in sample_keys:
        entry = tracings[key]
        png_path = os.path.join(frames_dir, entry["image_file"])
        if not os.path.exists(png_path):
            log.warning(f"  verify: missing frame {png_path}")
            continue

        img_arr = np.array(Image.open(png_path).convert("RGB"))
        source = entry.get("source", "camus")
        lv_poly = entry.get("lv_polygon")
        la_poly = entry.get("la_polygon")

        # Composite: original + LV + LA (if present)
        panels = [img_arr]
        panel_labels = ["original"]

        if lv_poly:
            lv_panel = _draw_polygon_on_image(
                img_arr,
                lv_poly,
                color_fill=(0, 200, 80, 60),
                color_outline=(0, 230, 60, 255),
            )
            panels.append(lv_panel)
            panel_labels.append("LV")

        if la_poly:
            la_panel = _draw_polygon_on_image(
                img_arr,
                la_poly,
                color_fill=(255, 160, 0, 60),
                color_outline=(255, 180, 0, 255),
            )
            panels.append(la_panel)
            panel_labels.append("LA")

        # Pad all panels to same size
        max_h = max(p.shape[0] for p in panels)
        max_w = max(p.shape[1] for p in panels)
        padded = []
        for p in panels:
            ph, pw = p.shape[:2]
            pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            pad[:ph, :pw] = p
            padded.append(pad)

        composite = np.concatenate(padded, axis=1)
        grid_items.append((composite, f"{source}  {key}\n{entry['view']}_{entry['instant']}"))

        # Save individual PNG
        individual_path = os.path.join(verify_dir, f"{key}_verify.png")
        fig, axes = plt.subplots(1, len(padded), figsize=(4 * len(padded), 4))
        if len(padded) == 1:
            axes = [axes]
        for ax, panel, label in zip(axes, padded, panel_labels):
            ax.imshow(panel)
            ax.set_title(label, fontsize=9)
            ax.axis("off")
        fig.suptitle(
            f"{source.upper()} | {key} | {entry['view']}_{entry['instant']}",
            fontsize=9, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(individual_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # Combined grid PNG
    if grid_items:
        n_rows = len(grid_items)
        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = [axes]
        for ax, (composite, title) in zip(axes, grid_items):
            ax.imshow(composite)
            ax.set_title(title, fontsize=8, loc="left")
            ax.axis("off")
        plt.suptitle("Unified Dataset — Verification Samples", fontsize=12, fontweight="bold")
        plt.tight_layout()
        grid_path = os.path.join(verify_dir, "_verify_grid.png")
        plt.savefig(grid_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved grid: {grid_path}")

    log.info(f"  Verification complete — {len(grid_items)} samples saved")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess EchoNet-Dynamic for unified LV/LA tracing fine-tuning"
    )
    parser.add_argument(
        "--echonet_root", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "EchoNet-Dynamic"),
        help="Path to EchoNet-Dynamic/ directory",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "echonet_processed"),
        help="Output directory for processed EchoNet data",
    )
    parser.add_argument(
        "--num_points", type=int, default=30,
        help="Polygon points per contour (default: 30, matches CAMUS preprocessing)",
    )
    parser.add_argument(
        "--merge_camus", type=str, default=None,
        metavar="CAMUS_PROCESSED_DIR",
        help="Path to existing camus_processed/ to merge into unified_processed/",
    )
    parser.add_argument(
        "--unified_dir", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "unified_processed"),
        help="Output directory for unified dataset (used with --merge_camus)",
    )
    parser.add_argument(
        "--max_videos", type=int, default=None,
        metavar="N",
        help="Process only first N videos (for quick validation; omit for full run)",
    )
    parser.add_argument(
        "--verify_samples", type=int, default=8,
        metavar="N",
        help="Number of samples to render in verification images (default: 8)",
    )
    parser.add_argument(
        "--echonet_target_size", type=int, default=None,
        metavar="N",
        help="Upscale EchoNet frames to N×N using Lanczos (default: native 112×112). Recommended: 224",
    )
    args = parser.parse_args()

    all_tracings, metadata_rows = preprocess_echonet(
        echonet_root=args.echonet_root,
        output_dir=args.output_dir,
        num_points=args.num_points,
        max_videos=args.max_videos,
        target_size=args.echonet_target_size,
    )

    log = logging.getLogger("preprocess_echonet")

    if args.merge_camus:
        log.info("")
        log.info("=" * 60)
        log.info("Merging CAMUS + EchoNet → unified_processed")
        log.info("=" * 60)
        unified_tracings, unified_dir = merge_datasets(
            camus_processed=args.merge_camus,
            echonet_processed=args.output_dir,
            unified_dir=args.unified_dir,
            log=log,
        )

        log.info("")
        log.info("=" * 60)
        log.info("Generating verification samples")
        log.info("=" * 60)
        verify_output(
            tracings=unified_tracings,
            frames_dir=os.path.join(unified_dir, "frames"),
            verify_dir=os.path.join(unified_dir, "verify_samples"),
            log=log,
            n_samples=args.verify_samples,
        )
    else:
        # Verify echonet-only output
        log.info("")
        log.info("=" * 60)
        log.info("Generating verification samples (EchoNet only)")
        log.info("=" * 60)
        verify_output(
            tracings=all_tracings,
            frames_dir=os.path.join(args.output_dir, "frames"),
            verify_dir=os.path.join(args.output_dir, "verify_samples"),
            log=log,
            n_samples=args.verify_samples,
        )
