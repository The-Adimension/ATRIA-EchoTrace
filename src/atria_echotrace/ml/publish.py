"""Publish a trained LoRA adapter to the Hugging Face Hub.

Faithful port of the notebook's "Interactive HuggingFace Adapter Transfer" cell
(notebook_as_py.txt L1580-1754): scan a checkpoint folder, pre-select the files needed
for adapter inference, create the repository if absent, optionally mark it gated by
injecting front-matter into ``README.md``, then upload the selected files.

The ipywidgets UI becomes CLI arguments. Because this pushes content to a public
service, :func:`publish_adapter` never runs unattended from a bare command: the CLI
requires an explicit confirmation flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..logging_setup import get_logger

logger = get_logger("ml.publish")

#: Filename prefixes the notebook pre-selected as needed for adapter inference (L1654).
RECOMMENDED_PREFIXES = ("adapter", "tokenizer", "chat_", "processor")
#: Weight formats excluded by the notebook's filter, except the explicit exception below.
EXCLUDED_SUFFIXES = (".pt", ".pth", ".bin")
#: The one .bin the notebook keeps (L1655).
ALWAYS_INCLUDE = "adapter_model.bin"

GATED_README = (
    "---\ngated: true\n---\n# Gated Adapter Repository\n\n"
    "Access to this model is restricted. Please request access."
)


@dataclass
class PublishPlan:
    """What would be uploaded, without performing any upload."""

    folder: Path
    repo_id: str
    gated: bool
    files: list[str]
    total_bytes: int

    def describe(self) -> str:
        lines = [
            f"Repository : {self.repo_id}",
            f"Source     : {self.folder}",
            f"Gated      : {'yes' if self.gated else 'no'}",
            f"Files      : {len(self.files)} ({self.total_bytes / 1e6:.1f} MB)",
        ]
        lines += [f"  - {name}" for name in self.files]
        return "\n".join(lines)


def is_recommended(filename: str) -> bool:
    """Whether the notebook's heuristic pre-selects this file.

    Reproduces the original expression's precedence exactly (L1654-1655):
    ``(prefix match and not .pt and not .pth and not .bin) or name == "adapter_model.bin"``.
    """
    return (
        any(filename.startswith(prefix) for prefix in RECOMMENDED_PREFIXES)
        and not filename.endswith(EXCLUDED_SUFFIXES)
    ) or filename == ALWAYS_INCLUDE


def scan_folder(folder: Path, include_all: bool = False) -> list[str]:
    """List candidate files in a checkpoint folder (port of ``scan_folder``, L1630-1661).

    Raises:
        FileNotFoundError: if the folder does not exist.
        RuntimeError: if it contains no files, or none that look like adapter artefacts.
    """
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Checkpoint folder not found: {folder}")

    names = sorted(item.name for item in folder.iterdir() if item.is_file())
    if not names:
        raise RuntimeError(f"Checkpoint folder is empty: {folder}")

    selected = names if include_all else [name for name in names if is_recommended(name)]
    if not selected:
        raise RuntimeError(
            f"No adapter artefacts found in {folder}. Expected files starting with "
            f"{', '.join(RECOMMENDED_PREFIXES)}. Found: {', '.join(names[:12])}. "
            "Pass include_all to upload everything."
        )
    return selected


def plan_publish(
    folder: Path,
    repo_id: str,
    gated: bool = True,
    include_all: bool = False,
) -> PublishPlan:
    """Build an upload plan without contacting the Hub.

    Raises:
        ValueError: if ``repo_id`` is not ``owner/name``.
    """
    if not repo_id or repo_id.count("/") != 1 or any(not part for part in repo_id.split("/")):
        raise ValueError(
            f"Invalid repo id {repo_id!r}. Expected 'owner/name', e.g. "
            "'The-Adimension/EchoTrace-MedGemma-CAMUS'."
        )
    folder = Path(folder).expanduser()
    files = scan_folder(folder, include_all=include_all)
    total = sum((folder / name).stat().st_size for name in files)
    return PublishPlan(
        folder=folder, repo_id=repo_id, gated=gated, files=files, total_bytes=total
    )


def publish_adapter(plan: PublishPlan, token: str | None = None) -> str:
    """Create the repository if needed and upload the planned files.

    Port of ``upload_files`` (L1666-1738), including the gated-README handling: an
    existing README has ``gated: true`` injected into its front matter, otherwise a
    minimal gated README is generated.

    Returns:
        The repository URL.

    Raises:
        RuntimeError: if no token is available, or the Hub rejects the operation.
    """
    from huggingface_hub import HfApi, get_token

    resolved = token or get_token()
    if not resolved:
        raise RuntimeError(
            "No Hugging Face token available. Run `hf auth login`, or set HF_TOKEN. "
            "A token with write access to the target namespace is required."
        )

    api = HfApi(token=resolved)
    files = list(plan.files)

    logger.info("Verifying repository %s (creating if absent)", plan.repo_id)
    try:
        api.create_repo(repo_id=plan.repo_id, repo_type="model", exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - surfaced with context
        raise RuntimeError(f"Could not create or access {plan.repo_id}: {exc}") from exc

    if plan.gated:
        logger.info("Applying gated configuration to repository metadata")
        readme_path = plan.folder / "README.md"
        if "README.md" in files and readme_path.is_file():
            content = readme_path.read_text(encoding="utf-8")
            if "gated:" not in content:
                if content.startswith("---"):
                    content = content.replace("---\n", "---\ngated: true\n", 1)
                else:
                    content = "---\ngated: true\n---\n" + content
        else:
            content = GATED_README
        try:
            api.upload_file(
                path_or_fileobj=content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=plan.repo_id,
                repo_type="model",
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Could not upload gated README: {exc}") from exc
        # Avoid uploading README twice.
        if "README.md" in files:
            files.remove("README.md")

    logger.info("Uploading %d file(s) to %s", len(files), plan.repo_id)
    for index, name in enumerate(files, start=1):
        logger.info("[%d/%d] %s", index, len(files), name)
        try:
            api.upload_file(
                path_or_fileobj=str(plan.folder / name),
                path_in_repo=name,
                repo_id=plan.repo_id,
                repo_type="model",
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Upload of {name} failed: {exc}") from exc

    url = f"https://huggingface.co/{plan.repo_id}"
    logger.info("Upload complete: %s", url)
    return url
