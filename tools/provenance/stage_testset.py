"""Step 1 — stage the official CAMUS test subgroup as a standalone dataset.

Selection is by `split == "test"` in the processed data, which `preprocess_camus.py`
derived from CAMUS's own `database_split/subgroup_testing.txt`. Never by patient number:
the official splits are scattered (patient0027 .. patient0276).
"""

import csv
import io
import json
import pathlib
import zipfile
from collections import Counter

ZIP = pathlib.Path(
    "datasets/processed_datasets/camus_processed/camus_processed.zip"
)
OUT = pathlib.Path("outputs/benchmark/camus_test")
SPLIT_FILE = pathlib.Path(
    "datasets/original_datasets_and_repos/camus_public/database_split/subgroup_testing.txt"
)

official = {line.strip() for line in SPLIT_FILE.read_text().splitlines() if line.strip()}
print(f"official subgroup_testing.txt : {len(official)} patients")

z = zipfile.ZipFile(ZIP)
tracings = json.loads(z.read("tracings.json"))
meta = list(csv.DictReader(io.StringIO(z.read("metadata.csv").decode("utf-8"))))

selected = {k: v for k, v in tracings.items() if v.get("split") == "test"}
patients = sorted({k.split("_")[0] for k in selected})

# The selection must agree with CAMUS's own file, or the provenance claim is wrong.
assert set(patients) == official, (
    f"split=test disagrees with subgroup_testing.txt: "
    f"only-in-data={sorted(set(patients) - official)[:5]} "
    f"only-in-file={sorted(official - set(patients))[:5]}"
)
print(f"split=='test' selection      : {len(patients)} patients, {len(selected)} frames")
print("cross-check against CAMUS's own file: EXACT MATCH")

(OUT / "frames").mkdir(parents=True, exist_ok=True)
for stem in selected:
    (OUT / "frames" / f"{stem}.png").write_bytes(z.read(f"frames/{stem}.png"))

(OUT / "tracings.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

rows = [r for r in meta if r["key"] in selected]
with (OUT / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Provenance, so nobody has to re-derive why this set was chosen.
(OUT / "PROVENANCE.md").write_text(
    "# CAMUS benchmark evaluation set\n\n"
    f"- {len(patients)} patients, {len(selected)} frames\n"
    "- Selected by `split == \"test\"` in `camus_processed.zip`, which "
    "`preprocess_camus.py` derived from CAMUS's official "
    "`database_split/subgroup_testing.txt`.\n"
    "- Verified to match that file exactly.\n"
    "- **Never trained on**: the published CAMUS adapter consumed the full 1 600-frame "
    "`train` split (5 epochs x 25 steps x effective batch 64), which excludes every "
    "patient here.\n"
    "- Structure under evaluation: **LV only** (the adapter was not tuned for LA).\n",
    encoding="utf-8",
)

print()
print(f"staged to {OUT}")
print(f"  PNGs written : {len(list((OUT / 'frames').glob('*.png')))}")
for field in ("view", "instant", "image_quality"):
    print(f"  {field:14}: {dict(Counter(r[field] for r in rows))}")
complete = Counter(r["patient_id"] for r in rows)
print(f"  patients with all 4 frames: {sum(1 for v in complete.values() if v == 4)}/{len(patients)}")
