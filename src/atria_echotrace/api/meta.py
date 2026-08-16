"""Capability, disclaimer and dataset-health endpoints.

The disclaimers and DEITY framework are content from the notebook's markdown cells
(notebook_as_py.txt L31-70) and are served to the UI so the mandatory-human-oversight
requirement is visible in the application, not only in documentation.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from .. import __version__
from ..config import ADAPTERS, Settings, display_path, local_adapter_dir
from ..data.dataset import DatasetRepository
from .deps import (
    classification_sources,
    get_settings,
    ingest_available,
    ml_available,
    repository,
)

router = APIRouter(prefix="/api/meta", tags=["meta"])

#: Verbatim from the notebook's "Important Disclaimers" cell (L31-58).
DISCLAIMERS: list[dict[str, str]] = [
    {
        "id": "intended-use",
        "title": "I. INTENDED USE",
        "text": (
            "ATRIA EchoTrace is currently classified as a Prototype / Advanced Minimum "
            "Viable Product (MVP). The platform, including its MedGemma-based models, "
            "LoRA adapters, pipelines, and user interfaces, is explicitly intended for "
            "research, academic collaboration, and internal validation. It is NOT a "
            "cleared or approved medical device by regulatory bodies."
        ),
    },
    {
        "id": "non-clinical-use",
        "title": "II. NON-CLINICAL USE",
        "text": (
            "Not for Direct Clinical Diagnosis. ATRIA EchoTrace is an AI-powered "
            "annotation and drafting tool designed to support and augment medical "
            "research and dataset creation. The software must not be used as a "
            "standalone diagnostic tool, for direct patient care, or as the sole basis "
            "for clinical decision-making. Any downstream metrics derived from its "
            "outputs (such as ejection fraction, cardiac volume, or strain "
            "calculations) are strictly for research and validation purposes."
        ),
    },
    {
        "id": "human-oversight",
        "title": "III. MANDATORY HUMAN OVERSIGHT",
        "text": (
            "In strict adherence to the DEITY Principles Framework (specifically the "
            '"You" pillar), this system relies on a Human-in-the-Loop (HITL) '
            "architecture. All AI-generated structural contours (LV/LA) and "
            "annotations are preliminary proposals. They must be independently "
            "reviewed, verified, and revised by qualified echocardiographers, "
            "cardiologists, or trained medical personnel."
        ),
    },
    {
        "id": "data-privacy",
        "title": "IV. DATA PRIVACY",
        "text": (
            "Users of ATRIA EchoTrace are solely responsible for ensuring that all "
            "echocardiographic inputs (DICOM, AVI, PNG) are properly anonymized and "
            "de-identified prior to processing."
        ),
    },
    {
        "id": "regulatory-compliance",
        "title": "V. REGULATORY COMPLIANCE",
        "text": (
            "Users must ensure their use of the platform complies with all applicable "
            "local, national, and international healthcare data privacy regulations "
            "(e.g., HIPAA, GDPR)."
        ),
    },
    {
        "id": "as-is",
        "title": 'VI. "AS IS" PROVISION',
        "text": (
            "The ATRIA EchoTrace codebase, models, artifacts, and documentation are "
            'provided "AS IS" without any warranties of accuracy, reliability, or '
            "clinical fitness, either express or implied."
        ),
    },
    {
        "id": "liability",
        "title": "VII. LIMITATION OF LIABILITY",
        "text": (
            "The developers, contributors, and affiliated research initiatives assume "
            "no liability for any direct, indirect, or consequential damages, or "
            "clinical outcomes arising from the use, misuse, or inability to use this "
            "platform."
        ),
    },
]

#: Citations from the notebook's References cell (L60-70).
CITATIONS: list[dict[str, str]] = [
    {
        "label": "The Adimension & DEITY Principles",
        "text": (
            "Anwer, S. (2026). The Adimension: Bridging human ingenuity and machine "
            "intelligence through the DEITY principles framework. European Heart "
            "Journal - Imaging Methods and Practice, 4(1), qyaf038."
        ),
        "url": "https://doi.org/10.1093/ehjimp/qyaf038",
    },
    {
        "label": "CAMUS Dataset",
        "text": (
            "Leclerc, S., et al. (2019). Deep learning for segmentation using an open "
            "large-scale dataset in 2D echocardiography. IEEE Transactions on Medical "
            "Imaging, 38(9), 2198-2210."
        ),
        "url": "https://doi.org/10.1109/tmi.2019.2900516",
    },
    {
        "label": "EchoNet-Dynamic",
        "text": (
            "Ouyang, D., et al. (2020). Video-based AI for beat-to-beat assessment of "
            "cardiac function. Nature, 580(7802), 252-256."
        ),
        "url": "https://doi.org/10.1038/s41586-020-2145-8",
    },
    {
        "label": "Google MedGemma 1.5",
        "text": "Google. (2026). MedGemma 1.5: Technical reports and model card.",
        "url": "https://huggingface.co/google/medgemma-1.5-4b-it",
    },
    {
        "label": "LoRA: Low-Rank Adaptation",
        "text": (
            "Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language "
            "models. arXiv."
        ),
        "url": "https://doi.org/10.48550/arXiv.2106.09685",
    },
]

#: The five DEITY pillars as tabulated in the project README.
DEITY_PILLARS: list[dict[str, Any]] = [
    {
        "pillar": "DATA",
        "points": [
            "Multi-format ready",
            "Preprocessed PNG frames",
            "JSON 30-pt normalized polygons (LV/LA)",
            "Metadata CSV",
            "Distributed imaging view, modality, and cardiac cycle stage",
        ],
    },
    {
        "pillar": "ETHICS",
        "points": [
            "Open CAMUS/EchoNet datasets",
            "Resource accessibility",
            "Local-first implementation",
            "Ground-truth evolution loop",
            "Continuous human oversight",
        ],
    },
    {
        "pillar": "INFORMATICS",
        "points": [
            "Structured outputs: JSON coordinates",
            "Built-in visualization",
            "Configurable parameterization",
            "Ablation-ready sample controls",
            "Full pipeline traceability",
        ],
    },
    {
        "pillar": "TECHNOLOGY",
        "points": [
            "Foundational base model: MedGemma-1.5-4B-it",
            "Parameter-efficient fine tuning for specialized adapters",
            "Resource-efficient implementation via bitsandbytes 4-bit",
            "Modular Python pipeline",
            "Data-driven integration readiness",
        ],
    },
    {
        "pillar": "YOU",
        "points": [
            "Adapter-human interactive revision UI",
            "Human edits directly enable iterative model development",
            "Standardized reusable outputs",
            "Open-source codebase for the community",
        ],
    },
]


@router.get("/capabilities")
def capabilities(
    settings: Settings = Depends(get_settings),
    repo: DatasetRepository = Depends(repository),
) -> dict[str, Any]:
    """Report what this installation can actually do.

    The UI drives affordances from this: the AI panel is disabled with a reason
    rather than failing when clicked.
    """
    has_ml = ml_available()
    device: dict[str, Any] = {"available": False, "reason": "AI tier not installed"}
    model_state: dict[str, Any] = {"state": "unavailable"}

    if has_ml:
        from ..ml.runtime import describe_device
        from ..ml.engine import get_inference_engine

        device = describe_device(force_cpu=settings.force_cpu)
        model_state = get_inference_engine().status()

    return {
        "version": __version__,
        # review = always; ai = the [ai] extra; ingest = the [ingest] extra, i.e.
        # whether raw CAMUS/EchoNet data can be preprocessed on this machine.
        "tiers": {"review": True, "ai": has_ml, "ingest": ingest_available()},
        # Classification-set derivation needs no extra package, only the original
        # metadata carrying the class labels. Per dataset, so the launcher can say
        # which of the two is derivable on this machine.
        "classification_sources": classification_sources(),
        "device": device,
        "model": model_state,
        "base_model_id": settings.base_model_id,
        # Where every weight would come from, so the UI can tell the operator whether
        # the AI tier will work and, if not, exactly what to do about it.
        "weights": settings.weights_report(),
        # A locally-present checkpoint needs neither a token nor network access, so
        # the UI can say so instead of implying a gated download is required.
        "adapters": [
            {**entry, "available_locally": local_adapter_dir(str(entry["id"])) is not None}
            for entry in ADAPTERS.values()
        ],
        "default_adapter": settings.default_adapter,
        "structures": ["LV", "LA"],
        "norm_scale": settings.norm_scale,
        "dataset": {
            # Project-relative: an absolute path here would disclose the OS user name
            # and directory layout to every client.
            "dir": display_path(settings.dataset_dir),
            "n_frames": len(repo.frames),
            "n_cases": len(repo.cases),
        },
        "has_token": settings.token_value() is not None,
    }


@router.get("/stages")
def stages(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """The lifecycle-stage registry, resolved for this machine.

    The UI renders entirely from this: no stage title, description or command is written
    in JavaScript, so registering a stage in ``api/stages.py`` is the only edit needed to
    add one. Each entry carries live ``state`` facts read from disk, which is what lets
    the Stages panel show whether a stage has actually been run.
    """
    from . import stages as registry

    caps = {
        "tiers": {"review": True, "ai": ml_available(), "ingest": ingest_available()},
        "classification_sources": classification_sources(),
    }
    return {"stages": registry.resolve(caps, settings)}


@router.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe. Deliberately free of dependencies so it answers while the
    model loads or the dataset is misconfigured — and free of filesystem detail, since
    liveness probes are the most widely-reachable endpoint."""
    return {"status": "ok", "version": __version__}


@router.get("/dataset-report")
def dataset_report(repo: DatasetRepository = Depends(repository)) -> dict[str, Any]:
    """Dataset validation report (notebook sanity-check cells L371-389, L273-280)."""
    report = repo.validate().as_dict()
    report["dataset_dir"] = display_path(report["dataset_dir"])
    return report


@router.get("/disclaimers")
def disclaimers() -> dict[str, Any]:
    """Notebook disclaimers, DEITY pillars and citations."""
    return {"disclaimers": DISCLAIMERS, "deity": DEITY_PILLARS, "citations": CITATIONS}
