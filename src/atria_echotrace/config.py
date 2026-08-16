"""Application configuration.

Replaces the notebook's ``@param`` form fields and Colab ``userdata.get('HF_TOKEN')``
(notebook_as_py.txt L112-250, L173-186) with environment-driven settings.

Secrets are read from the environment or a local ``.env`` only — never from a file
committed to the repository, and never echoed by an API endpoint or a log line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .domain.structures import NORM_SCALE

#: Base vision-language model (notebook L227). Gated on Hugging Face: users must
#: accept the Health AI Developer Foundations terms before the weights resolve.
BASE_MODEL_ID: Final[str] = "google/medgemma-1.5-4b-it"

#: The two published LoRA adapters, plus the un-adapted base model.
#: Mirrors ``AVAILABLE_ADAPTERS`` in the author's deployed Space.
ADAPTERS: Final[dict[str, dict[str, str | None]]] = {
    "base": {
        "id": "base",
        "label": "MedGemma 1.5 alone (no adapter)",
        "repo": None,
        "dataset": None,
        "doi": None,
    },
    "camus": {
        "id": "camus",
        "label": "MedGemma + ATRIA-EchoTrace-CAMUS (apical 2CH/4CH)",
        "repo": "The-Adimension/EchoTrace-MedGemma-CAMUS",
        "dataset": "camus",
        "doi": "10.57967/hf/9541",
    },
    "echonet": {
        "id": "echonet",
        "label": "MedGemma + ATRIA-EchoTrace-EchoNet (apical 4CH)",
        "repo": "The-Adimension/EchoTrace-MedGemma-EchoNet",
        "dataset": "echonet",
        "doi": "10.57967/hf/9540",
    },
}


#: Directory searched for adapter checkpoints shipped alongside the repository.
#: A local copy is preferred over the gated Hugging Face repo, so the AI tier works
#: offline and without a token once the weights are present.
LOCAL_ADAPTERS_DIRNAME: Final[str] = "adapters"

#: Directory searched for a local copy of the base model, e.g. a plain
#: ``git clone`` / ``hf download`` of google/medgemma-1.5-4b-it. Both the base model
#: and the adapters are gated, so supporting a hand-placed directory is what lets the
#: AI tier run with no token and no network at all.
LOCAL_MODELS_DIRNAME: Final[str] = "models"


def _find_project_root() -> Path:
    """Locate the repository root by marker files, falling back to the CWD.

    Editable installs resolve through ``src/``; a non-editable install has no
    repository, so the working directory is used instead.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "sample-dataset").is_dir():
            return parent
    return Path.cwd()


PROJECT_ROOT: Final[Path] = _find_project_root()


def display_path(path: Path | str | None, root: Path | None = None) -> str | None:
    """Render a filesystem path safely for a client response or the browser.

    Absolute paths must never leave the process: they disclose the operating-system
    user name and the machine's directory layout to anything that can reach the API,
    and they end up in screenshots and bug reports. Anything inside the project is
    shown relative to it (``models/medgemma-1.5-4b-it``); anything outside is reduced
    to its last two components with a leading ellipsis, which is enough to recognise a
    location without revealing where it lives.

    Server-side logs and the local ``atria doctor`` still print full paths — there the
    reader is the operator running the command.
    """
    if path is None:
        return None
    resolved = Path(path)
    base = (root or PROJECT_ROOT).resolve()
    try:
        absolute = resolved.resolve()
    except OSError:  # pragma: no cover - unresolvable path
        absolute = resolved
    if absolute == base:
        return "."
    if absolute.is_relative_to(base):
        return absolute.relative_to(base).as_posix()
    tail = absolute.parts[-2:]
    return ".../" + "/".join(tail) if tail else "..."


def local_adapter_dir(adapter_id: str, root: Path | None = None) -> Path | None:
    """Return a local checkpoint directory for ``adapter_id``, if one is present.

    Looks for ``<root>/adapters/atria-echotrace-<id>/adapter_config.json``. Returning
    a local path lets the engine attach an adapter with no network access and no
    Hugging Face token, which matters because both published adapters are gated.
    """
    base = (root or PROJECT_ROOT) / LOCAL_ADAPTERS_DIRNAME / f"atria-echotrace-{adapter_id}"
    return base if (base / "adapter_config.json").is_file() else None


def local_model_dir(model_id: str, root: Path | None = None) -> Path | None:
    """Return a hand-placed copy of ``model_id`` under ``models/``, if present.

    Accepts the three layouts people actually produce when they download a gated
    model by hand, for ``google/medgemma-1.5-4b-it``:

        models/medgemma-1.5-4b-it/          (repo name only — the common case)
        models/google--medgemma-1.5-4b-it/  (Hugging Face cache-style)
        models/google/medgemma-1.5-4b-it/   (full owner/name path)

    ``config.json`` is the marker of a usable Transformers directory; a folder
    without one is ignored rather than half-loaded.
    """
    base = (root or PROJECT_ROOT) / LOCAL_MODELS_DIRNAME
    name = model_id.split("/")[-1]
    for candidate in (base / name, base / model_id.replace("/", "--"), base / model_id):
        if (candidate / "config.json").is_file():
            return candidate
    return None


def hf_cache_dir(model_id: str) -> Path | None:
    """Return the Hugging Face cache snapshot for ``model_id``, if it is complete.

    Checked on the filesystem rather than through ``huggingface_hub`` so the review
    tier — which has no Hugging Face libraries — can still report accurately where
    the weights would come from.
    """
    import os

    home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    hub = home / "hub" if (home / "hub").is_dir() else home
    repo = hub / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not repo.is_dir():
        return None
    for snapshot in sorted(repo.iterdir(), reverse=True):
        if (snapshot / "config.json").is_file():
            return snapshot
    return None


class Settings(BaseSettings):
    """Runtime configuration. Every field is overridable by an ``ATRIA_*`` env var."""

    model_config = SettingsConfigDict(
        env_prefix="ATRIA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- server ---
    host: str = Field(default="127.0.0.1", description="Bind address; loopback by default.")
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = Field(default="INFO")

    # --- data locations ---
    dataset_dir: Path = Field(default=PROJECT_ROOT / "sample-dataset")
    output_dir: Path = Field(default=PROJECT_ROOT / "outputs")

    # --- model ---
    base_model_id: str = Field(default=BASE_MODEL_ID)
    default_adapter: str = Field(default="camus")
    max_new_tokens: int = Field(default=1024, ge=16, le=4096)
    #: Force CPU even when CUDA is present (useful for reproducing CPU results).
    force_cpu: bool = Field(default=False)
    #: Load weights from the local Hugging Face cache only, never hitting the network.
    offline: bool = Field(default=False)

    #: Read from HF_TOKEN / HUGGINGFACE_HUB_TOKEN as well as ATRIA_HF_TOKEN, so the
    #: standard Hugging Face environment variables work unchanged.
    hf_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ATRIA_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"),
    )

    # --- upload limits (clinician-supplied frames) ---
    max_upload_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    max_upload_pixels: int = Field(default=40_000_000, ge=1024)

    norm_scale: int = Field(default=NORM_SCALE)

    # --- derived paths ---
    @property
    def frames_dir(self) -> Path:
        return self.dataset_dir / "frames"

    @property
    def tracings_path(self) -> Path:
        return self.dataset_dir / "tracings.json"

    @property
    def manifest_path(self) -> Path:
        return self.dataset_dir / "manifest.json"

    @property
    def metadata_path(self) -> Path:
        """Optional ``metadata.csv`` (notebook L216); absent from ``sample-dataset``."""
        return self.dataset_dir / "metadata.csv"

    @property
    def revisions_dir(self) -> Path:
        return self.output_dir / "revisions"

    @property
    def uploads_dir(self) -> Path:
        return self.output_dir / "uploads"

    @property
    def evaluations_dir(self) -> Path:
        return self.output_dir / "evaluations"

    def ensure_output_dirs(self) -> None:
        """Create writable output directories. Never touches the dataset directory."""
        for path in (self.revisions_dir, self.uploads_dir, self.evaluations_dir):
            path.mkdir(parents=True, exist_ok=True)

    # --- weight resolution ---------------------------------------------------
    def resolve_base_model(self) -> tuple[str, str]:
        """Decide where the base model will be loaded from.

        Precedence, most self-sufficient first:

        1. ``models/<name>/`` in the project — no token, no network.
        2. The Hugging Face cache — no network, but needed a token once.
        3. The Hub itself — needs a token and accepted licence terms.

        Returns:
            ``(reference, source)`` where ``reference`` is what to hand to
            ``from_pretrained`` (a directory path or the repo id) and ``source`` is
            one of ``"local"``, ``"cache"`` or ``"hub"``.
        """
        local = local_model_dir(self.base_model_id)
        if local is not None:
            return str(local), "local"
        if hf_cache_dir(self.base_model_id) is not None:
            return self.base_model_id, "cache"
        return self.base_model_id, "hub"

    def weights_report(self) -> dict[str, object]:
        """Where every weight would come from, for the UI and `atria doctor`.

        Answers the operator's real question — "will this work, and if not what do I
        do?" — without loading anything.
        """
        reference, source = self.resolve_base_model()
        # Paths are project-relative: this report is served over HTTP and rendered in
        # the browser, so it must not disclose the machine's directory layout.
        base: dict[str, object] = {
            "id": self.base_model_id,
            "source": source,
            "path": display_path(reference) if source == "local" else None,
            "ready": source in ("local", "cache"),
        }
        if source == "local":
            base["detail"] = f"Local copy at {display_path(reference)}"
        elif source == "cache":
            base["detail"] = "Present in the Hugging Face cache"
        else:
            base["detail"] = (
                f"Not found locally. Either download {self.base_model_id} into "
                f"{LOCAL_MODELS_DIRNAME}/, or accept its licence and sign in "
                "(`hf auth login`) so it can be fetched."
            )

        adapters = []
        for entry in ADAPTERS.values():
            if not entry["repo"]:
                continue
            local = local_adapter_dir(str(entry["id"]))
            adapters.append(
                {
                    "id": entry["id"],
                    "source": "local" if local else "hub",
                    "path": display_path(local),
                    "ready": local is not None,
                    "detail": (
                        f"Local checkpoint at {display_path(local)}"
                        if local
                        else f"Not found locally. Download {entry['repo']} into "
                        f"{LOCAL_ADAPTERS_DIRNAME}/atria-echotrace-{entry['id']}/, "
                        "or sign in to fetch it."
                    ),
                }
            )

        return {
            "base": base,
            "adapters": adapters,
            # Folder names only. The client needs to know *what to create*, not where
            # this installation happens to live on disk.
            "models_dir": f"{LOCAL_MODELS_DIRNAME}/",
            "adapters_dir": f"{LOCAL_ADAPTERS_DIRNAME}/",
            "has_token": self.token_value() is not None,
        }

    def token_value(self) -> str | None:
        """Return the raw token, or ``None``. Callers must not log the result."""
        return self.hf_token.get_secret_value() if self.hf_token else None


settings = Settings()
