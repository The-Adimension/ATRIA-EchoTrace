"""CAMUS image quality (Good/Medium/Poor) -> mapping file. Copies no images.

Collection A. Writes datasets/classified_datasets/camus_quality/mapping.csv, linking
every PNG to its class.

Quality is a property of the acquisition window, not the patient: it differs between 2CH
and 4CH in 208 of 500 patients, so labels are keyed on (patient, view).

Standalone::

    python datasets/classification_scripts/camus/classify_camus_quality_metadata.py --dry-run

All paths resolve from the repository root; nothing needs editing after a clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from atria_echotrace.data.classify import run_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_cli("camus_quality", "metadata", prog=Path(__file__).name))
