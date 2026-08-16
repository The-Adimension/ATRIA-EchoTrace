"""EchoNet ejection fraction in 5-point bins -> one directory per class.

Collection B. Writes datasets/classified_datasets/echonet_ef_5pct/<bin>/*.png in the
classic ImageFolder layout.

Bins are fixed 5-point bands: 0_5, 5_10, 10_15 … 95_100. EF comes from the authentic
FileList.csv and is the only clinical class dimension EchoNet publishes — it carries no
sex, age or image-quality data.

Standalone::

    python datasets/classification_scripts/echonet/classify_echonet_ef_5pct_dirs.py --dry-run
    python datasets/classification_scripts/echonet/classify_echonet_ef_5pct_dirs.py --link

All paths resolve from the repository root; nothing needs editing after a clone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from atria_echotrace.data.classify import run_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_cli("echonet_ef_5pct", "dirs", prog=Path(__file__).name))
