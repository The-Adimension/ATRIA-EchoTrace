"""CAMUS image quality (Good/Medium/Poor) -> one directory per class.

Collection B. Writes datasets/classified_datasets/camus_quality/<class>/*.png in the
classic ImageFolder layout.

Quality is a property of the acquisition window, not the patient: it differs between 2CH
and 4CH in 208 of 500 patients, so labels are keyed on (patient, view).

Standalone::

    python datasets/classification_scripts/camus/classify_camus_quality_dirs.py --dry-run
    python datasets/classification_scripts/camus/classify_camus_quality_dirs.py --link

All paths resolve from the repository root; nothing needs editing after a clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from atria_echotrace.data.classify import run_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_cli("camus_quality", "dirs", prog=Path(__file__).name))
