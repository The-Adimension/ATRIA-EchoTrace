"""Torch runtime configuration and device capability policy.

The notebook hard-coded an Ampere/Blackwell-class setup: ``bfloat16`` compute dtype
with 4-bit NF4 quantisation (notebook_as_py.txt L139-150, L641-647, L883-888). Native
``bfloat16`` needs CUDA compute capability >= 8.0, and the development machine here is a
Turing Quadro RTX 5000 (sm_75), so a dtype decision is unavoidable for the
multi-platform requirement.

**Measured on that GPU (RESEARCH.md §2.2):** substituting ``float16`` — the obvious
choice for a pre-Ampere card — makes MedGemma 1.5 emit nothing but ``<pad>`` tokens.
Gemma-family models overflow in fp16, so the substitution silently destroys the output
rather than merely slowing it. ``bfloat16`` produces coherent generations on the same
GPU via emulation, at roughly 5 tokens/second.

The policy therefore keeps the notebook's ``bfloat16`` on every CUDA device and treats
native support as a *performance* property, not a correctness one.

Policy
------
======================================  ================================  =============
Device                                  Quantisation                      Compute dtype
======================================  ================================  =============
CUDA, native bf16 (sm_80+)              4-bit NF4 + double quant          bfloat16
CUDA, emulated bf16 (e.g. Turing)       4-bit NF4 + double quant          bfloat16 (slow)
CPU / MPS                               none (bitsandbytes is CUDA-only)  float32
======================================  ================================  =============
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from ..logging_setup import get_logger

logger = get_logger("ml.runtime")

#: Verbatim from the notebook's CUDA memory configuration cell (L144-147).
CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"

#: CUDA build to recommend when a GPU is present but torch cannot use it. cu126 is the
#: broadest-compatibility choice: it still supports Turing (sm_75) and works with older
#: drivers. Newer builds exist; this is a default, not a ceiling.
RECOMMENDED_TORCH_BACKEND = "cu126"

#: What to run to replace a CPU-only wheel with a CUDA one.
CUDA_REINSTALL_HINT = (
    f'uv pip install --torch-backend={RECOMMENDED_TORCH_BACKEND} --force-reinstall torch  '
    f"(or: pip install --force-reinstall torch "
    f"--index-url https://download.pytorch.org/whl/{RECOMMENDED_TORCH_BACKEND})"
)


def probe_nvidia_smi() -> dict[str, Any]:
    """Ask the NVIDIA driver what hardware it sees, independently of torch.

    This is the only way to tell "no GPU in this machine" apart from "GPU present but
    the installed torch cannot address it" — the two states produce an identical
    ``torch.cuda.is_available() == False`` (RESEARCH: the CPU-wheel trap).

    Returns:
        ``{"present": bool, "gpus": [str], "driver": str | None, "detail": str}``.
        Never raises: a missing or broken ``nvidia-smi`` simply means "not present".
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            "present": False,
            "gpus": [],
            "driver": None,
            "detail": "nvidia-smi is not on PATH.",
        }
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env-specific
        return {"present": False, "gpus": [], "driver": None, "detail": f"nvidia-smi failed: {exc}"}

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return {
            "present": False,
            "gpus": [],
            "driver": None,
            "detail": detail[0] if detail else "nvidia-smi returned a non-zero exit code.",
        }

    gpus: list[str] = []
    driver: str | None = None
    for line in completed.stdout.strip().splitlines():
        if not line.strip():
            continue
        name, _, version = line.partition(",")
        gpus.append(name.strip())
        driver = driver or version.strip() or None
    return {
        "present": bool(gpus),
        "gpus": gpus,
        "driver": driver,
        "detail": f"{len(gpus)} NVIDIA GPU(s) reported by the driver." if gpus else "No GPU rows.",
    }


def diagnose_cpu_fallback(force_cpu: bool = False) -> dict[str, str]:
    """Explain *why* the engine is about to run on the CPU.

    ``torch.cuda.is_available()`` collapses four very different situations into one
    ``False``. Reporting them all as "no CUDA device detected" sends a user with a
    perfectly good GPU to check their hardware, when the real fix is a one-line
    reinstall. ``torch.version.cuda`` is the field that separates them: ``None`` means
    the installed wheel was built without CUDA at all.

    Returns:
        ``{"cause": <slug>, "reason": <sentence>, "remedy": <sentence or "">}``.
    """
    import torch

    if force_cpu:
        return {
            "cause": "forced",
            "reason": "Running on CPU because the configuration forces it (ATRIA_FORCE_CPU).",
            "remedy": "Unset ATRIA_FORCE_CPU to use a GPU if one is available.",
        }

    build = getattr(torch.version, "cuda", None)
    gpu = probe_nvidia_smi()
    build_label = f"torch {torch.__version__}"

    if not gpu["present"]:
        return {
            "cause": "no_gpu",
            "reason": (
                f"Running on CPU: no NVIDIA GPU is visible to the driver ({gpu['detail']}). "
                "Expect minutes per frame."
            ),
            "remedy": "",
        }

    hardware = f"{gpu['gpus'][0]}" + (f", driver {gpu['driver']}" if gpu["driver"] else "")

    if build is None:
        return {
            "cause": "cpu_only_build",
            "reason": (
                f"An NVIDIA GPU is present ({hardware}) but PyTorch cannot use it: "
                f"{build_label} is a CPU-only build. On Windows and macOS, PyPI serves "
                "CPU-only wheels, so a plain `pip install torch` never enables CUDA."
            ),
            "remedy": f"Reinstall a CUDA build: {CUDA_REINSTALL_HINT}",
        }

    return {
        "cause": "driver_mismatch",
        "reason": (
            f"An NVIDIA GPU is present ({hardware}) and {build_label} was built for "
            f"CUDA {build}, but it still reports no usable device. The driver is usually "
            "older than the build requires, or the GPU is hidden from this process "
            "(CUDA_VISIBLE_DEVICES, a container without --gpus, or a remote session)."
        ),
        "remedy": (
            "Update the NVIDIA driver, check CUDA_VISIBLE_DEVICES, or install a torch "
            f"built for an older CUDA (e.g. --torch-backend=cu118)."
        ),
    }


def configure_torch_allocator() -> None:
    """Set ``PYTORCH_CUDA_ALLOC_CONF`` (notebook L139-150).

    Must run before CUDA is initialised to take effect. An existing value set by the
    operator is respected rather than overwritten.
    """
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if existing:
        logger.debug("PYTORCH_CUDA_ALLOC_CONF already set to %s; leaving as-is", existing)
        return
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
    logger.debug("PYTORCH_CUDA_ALLOC_CONF=%s", CUDA_ALLOC_CONF)


def select_device_policy(force_cpu: bool = False) -> dict[str, Any]:
    """Choose device, dtype and quantisation for the current machine.

    Capability is read from GPU 0 while the engine loads with ``device_map="auto"``,
    which may shard across several GPUs; on a *heterogeneous* multi-GPU host the dtype
    decision is therefore taken from one card and applied to all — harmless in practice,
    since bfloat16 is the correct choice on every CUDA generation this targets.

    Returns:
        A mapping with ``device``, ``compute_dtype`` (a torch dtype),
        ``quantization`` (``"nf4"`` or ``None``), and diagnostic fields.

    Raises:
        ImportError: if torch is not installed (the ``[ai]`` extra is absent).
    """
    import torch

    if force_cpu or not torch.cuda.is_available():
        diagnosis = diagnose_cpu_fallback(force_cpu=force_cpu)
        if diagnosis["cause"] == "cpu_only_build":
            # Worth a warning, not a debug line: the operator has GPU hardware sitting
            # idle and a one-command fix, and inference is ~100x slower meanwhile.
            logger.warning("%s %s", diagnosis["reason"], diagnosis["remedy"])
        return {
            "device": "cpu",
            # float32 on CPU: bfloat16 matmuls are poorly supported on many CPUs and
            # 4-bit quantisation is CUDA-only, so there is nothing to gain here.
            "compute_dtype": torch.float32,
            "compute_dtype_name": "float32",
            "quantization": None,
            "bf16_supported": False,
            "gpu_name": None,
            "total_memory_gb": None,
            "capability": None,
            "cpu_cause": diagnosis["cause"],
            "remedy": diagnosis["remedy"],
            "torch_version": torch.__version__,
            "torch_cuda_build": getattr(torch.version, "cuda", None),
            "reason": diagnosis["reason"],
        }

    capability = torch.cuda.get_device_capability(0)
    # torch.cuda.is_bf16_supported() defaults to including_emulation=True, so it
    # answers True on Turing (sm_75), where bfloat16 is emulated rather than native.
    # The native answer is needed only to warn about speed — it must NOT switch the
    # dtype, see below.
    try:
        native_bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        # Older torch has no such keyword; compute capability is the authority.
        native_bf16 = capability >= (8, 0)

    # bfloat16 on every CUDA device, native or emulated. Measured on a Quadro RTX 5000
    # (sm_75): float16 makes MedGemma 1.5 generate nothing but <pad> tokens, because
    # Gemma-family activations overflow the fp16 range. bfloat16 has the same exponent
    # range as fp32 and generates correctly on the same card. Correctness outranks the
    # emulation slowdown, so the notebook's dtype is kept everywhere (RESEARCH.md §2.2).
    compute_dtype = torch.bfloat16
    return {
        "device": "cuda",
        "compute_dtype": compute_dtype,
        "compute_dtype_name": "bfloat16",
        "quantization": "nf4",
        "bf16_supported": native_bf16,
        "gpu_name": torch.cuda.get_device_name(0),
        "total_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        ),
        "capability": f"sm_{capability[0]}{capability[1]}",
        "cpu_cause": None,
        "remedy": "",
        "torch_version": torch.__version__,
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "reason": (
            "CUDA with native bfloat16; using the notebook's configuration exactly."
            if native_bf16
            else f"CUDA compute capability sm_{capability[0]}{capability[1]} predates "
            "native bfloat16 (needs sm_80+), so bfloat16 runs emulated: correct but "
            "slow (~5 tokens/s measured). float16 is not substituted because "
            "Gemma-family models overflow in fp16 and emit only padding."
        ),
    }


def describe_device(force_cpu: bool = False) -> dict[str, Any]:
    """JSON-serialisable form of :func:`select_device_policy` for the API."""
    try:
        policy = select_device_policy(force_cpu=force_cpu)
    except ImportError as exc:
        return {"available": False, "reason": f"torch is not installed: {exc}"}
    return {
        "available": True,
        "device": policy["device"],
        "compute_dtype": policy["compute_dtype_name"],
        "quantization": policy["quantization"],
        "bf16_supported": policy["bf16_supported"],
        "gpu_name": policy["gpu_name"],
        "total_memory_gb": policy["total_memory_gb"],
        "capability": policy["capability"],
        # Why the CPU, when it is the CPU: 'cpu_only_build' means a GPU is present and
        # only the wheel is wrong, which is a one-command fix rather than a hardware
        # problem. None on CUDA.
        "cpu_cause": policy.get("cpu_cause"),
        "remedy": policy.get("remedy", ""),
        "torch_version": policy.get("torch_version"),
        "torch_cuda_build": policy.get("torch_cuda_build"),
        "reason": policy["reason"],
    }


def build_quantization_config(policy: dict[str, Any]) -> Any | None:
    """Build the ``BitsAndBytesConfig`` for a policy, or ``None`` on CPU.

    4-bit NF4 with double quantisation, matching the notebook's **HITL interface** cell
    (L1273-1275) — the stage this application is the production form of.

    The notebook's quantisation settings vary deliberately between cells: the training
    cell disables quantisation outright (``load_in_4bit=False``, L641-647), the
    batch-evaluation cell uses 4-bit without double quantisation (L883-888), and the
    HITL cell uses 4-bit *with* it. That spread is ablation, not inconsistency — those
    runs explored different adapters and datasets. Serving therefore pins the HITL
    values, because they are the configuration the published adapters were measured
    under here (CAMUS Dice 0.8924 / EchoNet 0.7026); training keeps its own cell's
    settings in :mod:`.train`. Neither is imposed on the other.

    Transformers v5 removed the bare ``load_in_4bit=True`` shortcut, so a config object
    is mandatory.
    """
    if policy.get("quantization") != "nf4":
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=policy["compute_dtype"],
    )
