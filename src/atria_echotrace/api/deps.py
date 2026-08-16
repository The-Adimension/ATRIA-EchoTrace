"""Shared application state and FastAPI dependencies.

The dataset repository and inference engine are process singletons: the repository
caches parsed JSON, and the engine holds several gigabytes of weights, so neither may
be rebuilt per request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import HTTPException, status

from ..config import Settings, settings
from ..data.dataset import DatasetError, DatasetRepository


#: Set by :func:`set_settings` when the CLI or a test supplies configuration that
#: differs from the module-level default.
_override: Settings | None = None


def get_settings() -> Settings:
    """Active configuration."""
    return _override if _override is not None else settings


def set_settings(value: Settings | None) -> None:
    """Install (or clear) a configuration override and drop dependent caches."""
    global _override
    _override = value
    get_repository.cache_clear()


@lru_cache(maxsize=1)
def get_repository() -> DatasetRepository:
    """Process-wide dataset repository."""
    return DatasetRepository(get_settings().dataset_dir)


def reset_caches() -> None:
    """Clear cached singletons. Used by tests that point at a different dataset."""
    set_settings(None)
    get_repository.cache_clear()


def repository() -> DatasetRepository:
    """FastAPI dependency that surfaces dataset misconfiguration as HTTP 503."""
    repo = get_repository()
    try:
        repo.frames  # noqa: B018 - triggers the cached load and validates the contract
    except DatasetError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return repo


def ml_available() -> bool:
    """True when the optional ``[ai]`` extra is importable."""
    from importlib.util import find_spec

    return all(find_spec(name) is not None for name in ("torch", "transformers", "peft"))


def ingest_available() -> bool:
    """True when the optional ``[ingest]`` extra is importable.

    Stage-0 readiness for the launcher: SimpleITK reads CAMUS NIfTI, OpenCV reads
    EchoNet AVI. Both are needed before `atria ingest` can process raw data.
    """
    from importlib.util import find_spec

    return all(find_spec(name) is not None for name in ("SimpleITK", "cv2"))


def classification_sources() -> dict[str, bool]:
    """Which original label sources are on this machine.

    Classification-set derivation needs no extra package — only the original metadata
    that carries the class labels. Reported per dataset so the launcher can say which
    of the two is actually derivable here.
    """
    from ..config import PROJECT_ROOT

    originals = PROJECT_ROOT / "datasets" / "original_datasets_and_repos"
    return {
        "camus": (originals / "camus_public" / "database_nifti").is_dir(),
        "echonet": (originals / "echonet_dynamic" / "FileList.csv").is_file(),
    }


def require_ml() -> None:
    """Guard for endpoints that need the model tier.

    Raises:
        HTTPException: 503 with install instructions when the extra is absent.
    """
    if not ml_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The AI tier is not installed. Install it with "
                '`pip install -e ".[ai]"` (torch, transformers, peft, bitsandbytes), '
                "then restart the server."
            ),
        )


def get_engine() -> Any:
    """Return the inference engine singleton, importing the ML tier lazily.

    Importing at module scope would make the whole API depend on torch.
    """
    require_ml()
    from ..ml.engine import get_inference_engine

    return get_inference_engine()
