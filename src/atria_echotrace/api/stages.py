"""The lifecycle-stage registry — the single source of truth for what this platform does.

Every stage the platform exposes is registered here exactly once, with a **stable machine
id** and a **functional title**. Neither encodes position: the old ``0 / 0b / A / B / B→A
/ C`` labels made order part of a stage's identity, so inserting a capability meant
renaming its neighbours. Display order is the list order below and nothing else, so adding
a stage is one new entry and no edits anywhere else.

The UI renders entirely from ``GET /api/meta/stages``; no stage title, description or
command is written in JavaScript.

Each entry carries:

``id``          stable identifier, never displayed, safe to key on
``title``       human-readable functional name
``summary``     one paragraph for the panel
``command``     the CLI invocation, when the stage belongs on a command line
``here``        true for the stage the workstation itself is
``ready``       ``(caps) -> bool``: can this machine run it at all
``blocked``     why not, when ``ready`` is false
``state``       ``() -> list[fact]``: what actually exists on disk right now

A *fact* is ``{"label", "present", "detail"}``. Facts are the state-awareness the Stages
panel needs to stop being a static menu: they answer "has this stage been run, and what
did it leave behind" without the operator going to a terminal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..config import PROJECT_ROOT, display_path

DATASETS = PROJECT_ROOT / "datasets"
PROCESSED_ROOT = DATASETS / "processed_datasets"
CLASSIFIED_ROOT = DATASETS / "classified_datasets"
BENCHMARK_ROOT = PROJECT_ROOT / "outputs" / "benchmark"


def _fact(label: str, present: bool, detail: str = "") -> dict[str, Any]:
    return {"label": label, "present": bool(present), "detail": detail}


def _corpora() -> list[Path]:
    """Processed corpora on disk, in any of the layouts `atria ingest` can produce.

    Deliberately a bounded scan of known layouts rather than ``rglob``: this runs on
    every panel open, and walking a 22 000-frame tree to answer "does a corpus exist"
    made the endpoint take nearly two seconds.
    """
    if not PROCESSED_ROOT.is_dir():
        return []
    candidates = [
        *(PROCESSED_ROOT / name for name in ("camus_processed", "echonet_processed")),
        PROCESSED_ROOT / "unified_processed" / "unified_processed",
        PROCESSED_ROOT / "unified_processed",
        *(d for d in PROCESSED_ROOT.iterdir() if d.is_dir()),
    ]
    seen: list[Path] = []
    for candidate in candidates:
        if (
            candidate not in seen
            and (candidate / "metadata.csv").is_file()
            and (candidate / "frames").is_dir()
        ):
            seen.append(candidate)
    return seen


def _frame_count(corpus: Path) -> int:
    """Row count of metadata.csv — one file read, versus stat-ing every PNG."""
    try:
        with (corpus / "metadata.csv").open(encoding="utf-8", newline="") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return 0


def _preprocess_state() -> list[dict[str, Any]]:
    corpora = _corpora()
    detail = ", ".join(f"{c.name} ({_frame_count(c)} frames)" for c in corpora)
    return [
        _fact(
            "processed corpus",
            bool(corpora),
            detail or f"none under {display_path(PROCESSED_ROOT)}",
        )
    ]


def _classify_state() -> list[dict[str, Any]]:
    tasks = [d for d in sorted(CLASSIFIED_ROOT.glob("*")) if d.is_dir()] \
        if CLASSIFIED_ROOT.is_dir() else []
    facts = [
        _fact(
            "classified sets",
            bool(tasks),
            ", ".join(t.name for t in tasks) if tasks
            else f"none under {display_path(CLASSIFIED_ROOT)}",
        )
    ]
    for task in tasks:
        classes = [d for d in task.iterdir() if d.is_dir()]
        mapping = (task / "mapping.csv").is_file()
        detail = []
        if mapping:
            detail.append("mapping.csv")
        if classes:
            detail.append(f"{len(classes)} class dirs")
        facts.append(_fact(task.name, True, " · ".join(detail) or "empty"))
    return facts


def _adapters_present() -> list[str]:
    root = PROJECT_ROOT / "adapters"
    return [d.name for d in sorted(root.glob("*")) if d.is_dir()] if root.is_dir() else []


def _train_state() -> list[dict[str, Any]]:
    adapters = _adapters_present()
    return [
        _fact("local adapters", bool(adapters),
              ", ".join(adapters) if adapters else "none in adapters/"),
        _fact("processed corpus to train on", bool(_corpora()),
              f"{len(_corpora())} found" if _corpora() else "run Preprocess first"),
    ]


def _evaluate_state(settings: Any) -> list[dict[str, Any]]:
    runs_dir = Path(settings.evaluations_dir)
    runs = sorted(runs_dir.glob("*.json")) if runs_dir.is_dir() else []
    benchmark = BENCHMARK_ROOT / "full200"
    overlays = benchmark / "overlays" / "index.html"
    return [
        _fact("evaluation runs", bool(runs), f"{len(runs)} in {display_path(runs_dir)}"
              if runs else "none yet — start one from this panel"),
        _fact("200-frame CAMUS benchmark", (benchmark / "pointwise.csv").is_file(),
              f"{display_path(benchmark)} — median 4.98 mm point-to-curve, 200/200 parsed"
              if (benchmark / "pointwise.csv").is_file() else "not present"),
        _fact("visual QA gallery", overlays.is_file(),
              f"{display_path(overlays)} — 200 overlays" if overlays.is_file()
              else "not present"),
    ]


def _revisions(settings: Any) -> int:
    root = Path(settings.revisions_dir)
    return len([d for d in root.glob("rev_*") if d.is_dir()]) if root.is_dir() else 0


def _trace_state(settings: Any) -> list[dict[str, Any]]:
    n = _revisions(settings)
    return [_fact("saved revisions", bool(n),
                  f"{n} in {display_path(settings.revisions_dir)}" if n
                  else "none yet — trace a frame and save")]


def _export_state(settings: Any) -> list[dict[str, Any]]:
    n = _revisions(settings)
    return [_fact("revisions available to export", bool(n),
                  f"{n} ready" if n else "save a revision first")]


def _publish_state() -> list[dict[str, Any]]:
    adapters = _adapters_present()
    return [_fact("adapters available to publish", bool(adapters),
                  ", ".join(adapters) if adapters else "none in adapters/")]


#: Display order is this list's order. Identity is ``id``, never position.
REGISTRY: list[dict[str, Any]] = [
    {
        "id": "preprocess",
        "title": "Preprocess",
        "summary": (
            "Raw CAMUS NIfTI or EchoNet AVI into the standard contract: PNG frames, "
            "tracings.json, metadata.csv. Runs the original preprocessing scripts, "
            "vendored unmodified, so the output matches the corpus the adapters were "
            "trained on."
        ),
        "command": "atria ingest camus --source <raw> --output <dir>",
        "ready": lambda caps: bool(caps["tiers"]["ingest"]),
        "blocked": "needs the [ingest] extra",
        "state": lambda settings: _preprocess_state(),
    },
    {
        "id": "classify",
        "title": "Classify",
        "summary": (
            "Derive classification-task datasets from the same processed corpus: CAMUS "
            "image quality (per acquisition window) and CAMUS/EchoNet ejection fraction "
            "in 5-point bins. Two independent products per task — a mapping.csv linking "
            "every PNG to its class, or one directory per class in ImageFolder layout. "
            "Labels are copied from the original dataset metadata, never invented."
        ),
        "command": "atria classify all metadata\natria classify all dirs --link",
        "ready": lambda caps: any(
            (caps.get("classification_sources") or {}).values()
        ),
        "blocked": "needs the original dataset metadata",
        "state": lambda settings: _classify_state(),
    },
    {
        "id": "train",
        "title": "Train",
        "summary": (
            "Fine-tune MedGemma with QLoRA on a prepared corpus. A multi-hour GPU job, "
            "so it lives on the command line. Note it ports the notebook's recipe and "
            "will not reproduce the published adapters — see README."
        ),
        # --dataset-dir is a GLOBAL option and must precede the subcommand; the
        # post-subcommand form fails with "unrecognized arguments".
        "command": "atria --dataset-dir <dir> train --output-dir <ckpt>",
        "ready": lambda caps: bool(caps["tiers"]["ai"]),
        "blocked": "needs the [ai] extra",
        "state": lambda settings: _train_state(),
    },
    {
        "id": "evaluate",
        "title": "Evaluate",
        "summary": (
            "Score a hold-out split for parse rate, Dice and IoU, with best/worst "
            "comparison figures. Runs in the background and can be started from this "
            "panel or the command line."
        ),
        "command": "atria evaluate --adapter camus --split test --max-samples 50",
        "ready": lambda caps: bool(caps["tiers"]["ai"]),
        "blocked": "needs the [ai] extra",
        "state": _evaluate_state,
    },
    {
        "id": "trace_revise",
        "title": "Trace & Revise",
        "summary": (
            "This screen. Upload your own frame or pick a dataset case, let the adapter "
            "propose a contour, correct it vertex by vertex, and save the revision with "
            "its figures and metrics."
        ),
        "command": None,
        "here": True,
        "ready": lambda caps: True,
        "blocked": None,
        "state": _trace_state,
    },
    {
        "id": "export_corpus",
        "title": "Export Corpus",
        "summary": (
            "Turn saved revisions into a trainable dataset — your corrections become the "
            "ground truth, in the same three-artifact contract Train consumes."
        ),
        "command": "atria export-corpus --out <dir>",
        "ready": lambda caps: True,
        "blocked": None,
        "state": _export_state,
    },
    {
        "id": "publish",
        "title": "Publish Adapter",
        "summary": (
            "Upload an approved adapter to the Hugging Face Hub, gated by default. "
            "Deliberately not a button: publishing is irreversible and this API is "
            "unauthenticated."
        ),
        "command": "atria publish-adapter --folder <ckpt> --repo owner/name --yes",
        "ready": lambda caps: True,
        "blocked": None,
        "state": lambda settings: _publish_state(),
    },
]


def resolve(caps: dict[str, Any], settings: Any) -> list[dict[str, Any]]:
    """Render the registry for this machine, right now.

    Readiness and disk facts are computed per request so the panel reflects reality
    rather than what was true when the page loaded.
    """
    out = []
    for entry in REGISTRY:
        ready = bool(entry["ready"](caps))
        state_fn: Callable[[Any], list[dict[str, Any]]] = entry["state"]
        out.append(
            {
                "id": entry["id"],
                "title": entry["title"],
                "summary": entry["summary"],
                "command": entry.get("command"),
                "here": bool(entry.get("here")),
                "ready": ready,
                "blocked": None if ready else entry.get("blocked"),
                "state": state_fn(settings),
            }
        )
    return out
