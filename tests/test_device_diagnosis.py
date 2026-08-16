"""Why the engine fell back to the CPU — the four states must be distinguishable.

`torch.cuda.is_available()` collapses "no GPU", "CPU-only wheel", "driver mismatch" and
"forced" into a single False. Reporting them all as "no CUDA device detected" sent users
with a working GPU to check their hardware when the real fix was a wheel reinstall. These
tests pin the classifier; they need no GPU and no torch build of any particular kind.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from atria_echotrace.ml import runtime

torch = pytest.importorskip("torch", reason="the [ai] extra is not installed")


def _fake_smi(monkeypatch, *, on_path: bool, returncode: int = 0, stdout: str = "") -> None:
    monkeypatch.setattr(
        runtime.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if on_path else None
    )
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout, stderr=""),
    )


def _fake_torch_cuda_build(monkeypatch, build: str | None) -> None:
    monkeypatch.setattr(runtime, "__name__", runtime.__name__)  # no-op, keeps intent clear
    monkeypatch.setattr(torch.version, "cuda", build, raising=False)


# --------------------------------------------------------------- nvidia-smi probe
def test_probe_reports_absent_when_nvidia_smi_is_missing(monkeypatch) -> None:
    _fake_smi(monkeypatch, on_path=False)
    result = runtime.probe_nvidia_smi()
    assert result["present"] is False
    assert result["gpus"] == []
    assert "PATH" in result["detail"]


def test_probe_parses_names_and_driver(monkeypatch) -> None:
    _fake_smi(
        monkeypatch,
        on_path=True,
        stdout="Quadro RTX 5000, 610.47\nNVIDIA A100-SXM4-40GB, 610.47\n",
    )
    result = runtime.probe_nvidia_smi()
    assert result["present"] is True
    assert result["gpus"] == ["Quadro RTX 5000", "NVIDIA A100-SXM4-40GB"]
    assert result["driver"] == "610.47"


def test_probe_treats_a_failing_nvidia_smi_as_absent(monkeypatch) -> None:
    """A driver that will not answer is indistinguishable from no GPU, and must not raise."""
    _fake_smi(monkeypatch, on_path=True, returncode=9, stdout="couldn't communicate\n")
    result = runtime.probe_nvidia_smi()
    assert result["present"] is False


def test_probe_survives_a_crashing_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def explode(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

    monkeypatch.setattr(runtime.subprocess, "run", explode)
    assert runtime.probe_nvidia_smi()["present"] is False


# ------------------------------------------------------------ the four causes
def test_forced_cpu_is_named_as_a_choice(monkeypatch) -> None:
    diagnosis = runtime.diagnose_cpu_fallback(force_cpu=True)
    assert diagnosis["cause"] == "forced"
    assert "ATRIA_FORCE_CPU" in diagnosis["remedy"]


def test_no_gpu_present_is_reported_honestly(monkeypatch) -> None:
    _fake_smi(monkeypatch, on_path=False)
    diagnosis = runtime.diagnose_cpu_fallback()
    assert diagnosis["cause"] == "no_gpu"
    assert "no NVIDIA GPU" in diagnosis["reason"]
    # Nothing to fix: this machine genuinely has no GPU.
    assert diagnosis["remedy"] == ""


def test_cpu_only_wheel_on_a_gpu_machine_is_identified(monkeypatch) -> None:
    """The regression this whole module exists for."""
    _fake_smi(monkeypatch, on_path=True, stdout="Quadro RTX 5000, 610.47\n")
    _fake_torch_cuda_build(monkeypatch, None)

    diagnosis = runtime.diagnose_cpu_fallback()
    assert diagnosis["cause"] == "cpu_only_build"
    # It must name the hardware, so nobody goes hunting for a missing GPU.
    assert "Quadro RTX 5000" in diagnosis["reason"]
    assert "CPU-only build" in diagnosis["reason"]
    # And it must hand over a runnable fix.
    assert "torch-backend" in diagnosis["remedy"] or "index-url" in diagnosis["remedy"]
    assert runtime.RECOMMENDED_TORCH_BACKEND in diagnosis["remedy"]


def test_cuda_build_that_still_cannot_see_the_gpu_blames_the_driver(monkeypatch) -> None:
    _fake_smi(monkeypatch, on_path=True, stdout="Quadro RTX 5000, 470.10\n")
    _fake_torch_cuda_build(monkeypatch, "12.6")

    diagnosis = runtime.diagnose_cpu_fallback()
    assert diagnosis["cause"] == "driver_mismatch"
    assert "CUDA 12.6" in diagnosis["reason"]
    assert "CUDA_VISIBLE_DEVICES" in diagnosis["reason"]
    # Never blames the wheel when the wheel is fine.
    assert "CPU-only" not in diagnosis["reason"]


# --------------------------------------------------------- surfaced to clients
def test_describe_device_carries_the_cause_and_remedy(monkeypatch) -> None:
    _fake_smi(monkeypatch, on_path=True, stdout="Quadro RTX 5000, 610.47\n")
    _fake_torch_cuda_build(monkeypatch, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    described = runtime.describe_device()
    assert described["device"] == "cpu"
    assert described["cpu_cause"] == "cpu_only_build"
    assert described["remedy"]
    assert described["torch_cuda_build"] is None
    assert described["torch_version"]


def test_describe_device_reports_no_cause_on_cuda(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    described = runtime.describe_device()
    assert described["device"] == "cuda"
    assert described["cpu_cause"] is None
    assert described["remedy"] == ""
    assert described["torch_cuda_build"] is not None
