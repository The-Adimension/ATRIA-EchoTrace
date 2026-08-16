"""CAMUS ejection fraction in 5-point bins -> mapping file. Copies no images.

Collection A. Writes datasets/classified_datasets/camus_ef_5pct/mapping.csv, linking
every PNG to its class.

Bins are fixed 5-point bands: 0_5, 5_10, 10_15 … 95_100. EF is a patient-level value in
CAMUS (identical across 2CH and 4CH in all 500 patients), so both views of a patient
share a bin.

Standalone::

    python datasets/classification_scripts/camus/classify_camus_ef_5pct_metadata.py --dry-run

All paths resolve from the repository root; nothing needs editing after a clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from atria_echotrace.data.classify import run_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_cli("camus_ef_5pct", "metadata", prog=Path(__file__).name))
