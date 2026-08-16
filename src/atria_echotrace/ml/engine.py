"""MedGemma 1.5 + LoRA inference engine.

Production form of the notebook's HITL model plumbing: the cached global loader
``load_model`` (notebook_as_py.txt L1257-1282) and the per-frame inference callback
``process_image_backend`` (L1284-1316). The generation call is byte-for-byte the
notebook's: greedy decoding, ``max_new_tokens=1024``, ``pad_token_id=eos``, and
decoding only the newly generated span.

Differences from the notebook, all required by a long-lived server:

* **Explicit lifecycle.** Loading is an action (``load``/``load_async``) with observable
  state, not an import-time side effect. An 8 GB load cannot happen inside a request.
* **Device policy.** dtype and quantisation come from :mod:`.runtime` instead of being
  hard-coded to bfloat16, so pre-Ampere GPUs and CPU-only machines work (RESEARCH.md §2.2).
* **Serialised generation.** A lock ensures one generation at a time; concurrent
  requests queue rather than exhausting device memory.
* **One adapter at a time.** Switching adapters reloads the model. PEFT can hold
  several adapters simultaneously, but these adapters carry ``modules_to_save`` for
  ``lm_head`` and ``embed_tokens`` (2.84 GB each), so keeping two resident risks
  exhausting memory on the CPU and mid-range GPU targets. Base weights come from the
  local Hugging Face cache, making a switch disk-bound rather than network-bound.
"""

from __future__ import annotations

import gc
import threading
import time
from enum import Enum
from typing import Any, Callable

from PIL import Image

from ..config import ADAPTERS, Settings, local_adapter_dir, settings as default_settings
from ..domain.geometry import parse_polygon, sanitize_polygon
from ..logging_setup import get_logger
from .prompts import PromptVariant, build_messages, build_prompt
from .runtime import build_quantization_config, configure_torch_allocator, select_device_policy

logger = get_logger("ml.engine")

ProgressCallback = Callable[[float, str], None]


def repair_legacy_adapter_keys(model: Any, adapter_ref: str) -> dict[str, Any]:
    """Re-attach LoRA weights whose module paths predate the installed transformers.

    The published adapters were trained when Gemma 3 nested the vision encoder as
    ``vision_tower.vision_model.encoder``. Transformers 5.x flattened that to
    ``vision_tower.encoder``, so every vision-tower tensor in the checkpoint addresses a
    module path that no longer exists. PEFT emits a "missing adapter keys" warning and
    carries on, leaving ``lora_B`` at its zero initialisation — i.e. **the entire vision
    half of the adapter silently contributes nothing**, which on a spatial task is the
    half that matters most.

    This detects that exact situation (legacy nesting present in the checkpoint, absent
    from the live model), rewrites the keys and re-applies them. A checkpoint whose keys
    already match is left untouched.

    Returns:
        ``{"remapped": int, "vision_lora_b": int, "vision_lora_b_active": int,
        "fully_loaded": bool}`` — reported through :meth:`InferenceEngine.status` so a
        partial load can never again be invisible.
    """
    from peft import load_peft_weights, set_peft_model_state_dict

    def vision_lora_b(module: Any) -> list[Any]:
        return [p for n, p in module.named_parameters()
                if "lora_B" in n and "vision_tower" in n]

    tensors = vision_lora_b(model)
    stats = {
        "remapped": 0,
        "vision_lora_b": len(tensors),
        "vision_lora_b_active": sum(1 for p in tensors if float(p.abs().sum()) > 0),
        "fully_loaded": True,
    }
    if not tensors or stats["vision_lora_b_active"] == len(tensors):
        return stats  # nothing to repair

    try:
        weights = load_peft_weights(adapter_ref, device="cpu")
    except Exception as exc:  # noqa: BLE001 - repair is best-effort, never fatal
        logger.warning("Could not re-read %s to repair adapter keys: %s", adapter_ref, exc)
        stats["fully_loaded"] = False
        return stats

    module_paths = {name for name, _ in model.named_modules()}
    legacy = ".vision_tower.vision_model."
    if not any(legacy in key for key in weights) or any(
        ".vision_model." in path for path in module_paths
    ):
        stats["fully_loaded"] = False
        return stats  # a different mismatch; do not guess

    remapped = {key.replace(legacy, ".vision_tower."): value for key, value in weights.items()}
    stats["remapped"] = sum(1 for key in weights if legacy in key)
    set_peft_model_state_dict(model, remapped, adapter_name="default")

    tensors = vision_lora_b(model)
    stats["vision_lora_b_active"] = sum(1 for p in tensors if float(p.abs().sum()) > 0)
    stats["fully_loaded"] = stats["vision_lora_b_active"] == len(tensors)
    if stats["fully_loaded"]:
        logger.info(
            "Repaired %d legacy vision-tower adapter keys; %d/%d LoRA tensors now active.",
            stats["remapped"], stats["vision_lora_b_active"], len(tensors),
        )
    else:
        logger.warning(
            "Adapter is only partially loaded: %d/%d vision-tower LoRA tensors active. "
            "Predictions will be degraded.",
            stats["vision_lora_b_active"], len(tensors),
        )
    return stats


class ModelState(str, Enum):
    """Observable engine lifecycle states."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


#: Substrings identifying a Hub reachability/authorisation failure, as opposed to a
#: genuine problem with the weights themselves.
_HUB_ACCESS_MARKERS = (
    "gated",
    "401",
    "403",
    "unauthorized",
    "authentication",
    "connection",
    "timed out",
    "timeout",
    "offline",
    "could not reach",
    "name resolution",
    "temporary failure",
)


def _is_hub_access_error(exc: Exception) -> bool:
    """Whether a load failure looks like "cannot reach or authorise the Hub"."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _HUB_ACCESS_MARKERS)


class ModelNotReady(RuntimeError):
    """Raised when inference is requested before the model finished loading."""


class AdapterError(ValueError):
    """Raised for an unknown adapter id."""


def resolve_adapter(adapter_id: str | None) -> dict[str, Any]:
    """Look up an adapter in the registry.

    Accepts a registry key (``base``/``camus``/``echonet``), a full repo id, or a
    local filesystem path to a checkpoint directory — the notebook's HITL cell took a
    local ``lora_path`` (L1397), so local checkpoints remain first-class.

    Raises:
        AdapterError: if the value matches neither a registry key nor a plausible
            repo id or existing path.
    """
    if adapter_id in (None, "", "base"):
        return dict(ADAPTERS["base"])
    if adapter_id in ADAPTERS:
        entry = dict(ADAPTERS[adapter_id])
        # Prefer a checkpoint committed under adapters/: it needs neither network
        # access nor a token, and both published repos are gated.
        local = local_adapter_dir(adapter_id)
        if local is not None:
            entry["repo"] = str(local)
            entry["source"] = "local"
        return entry

    for entry in ADAPTERS.values():
        if entry["repo"] and entry["repo"] == adapter_id:
            return dict(entry)

    from pathlib import Path

    if Path(adapter_id).expanduser().is_dir():
        return {
            "id": adapter_id,
            "label": f"Local checkpoint: {adapter_id}",
            "repo": str(Path(adapter_id).expanduser()),
            "dataset": None,
            "doi": None,
        }
    if "/" in adapter_id:
        return {
            "id": adapter_id,
            "label": f"Custom adapter: {adapter_id}",
            "repo": adapter_id,
            "dataset": None,
            "doi": None,
        }
    raise AdapterError(
        f"Unknown adapter {adapter_id!r}. Expected one of {sorted(ADAPTERS)}, "
        "a Hugging Face repo id, or a local checkpoint directory."
    )


class InferenceEngine:
    """Holds the loaded model and serves contour predictions."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        self._model: Any = None
        self._processor: Any = None
        self._adapter: dict[str, Any] | None = None
        #: How completely the adapter's weights attached — see
        #: :func:`repair_legacy_adapter_keys`. A partial load must stay visible.
        self._adapter_load: dict[str, Any] = {}
        self._policy: dict[str, Any] | None = None
        #: "local" | "cache" | "hub" — where the base weights were actually read from.
        self._weights_source: str | None = None

        self._state = ModelState.UNLOADED
        self._progress = 0.0
        self._message = "Model not loaded."
        self._error: str | None = None
        self._loaded_at: float | None = None
        self._load_seconds: float | None = None

        # Guards state transitions; held only briefly.
        self._state_lock = threading.Lock()
        # Serialises loading and generation on the device.
        self._device_lock = threading.RLock()
        self._load_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ status
    def _set_state(
        self,
        state: ModelState | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._state_lock:
            if state is not None:
                self._state = state
            if progress is not None:
                self._progress = max(0.0, min(1.0, progress))
            if message is not None:
                self._message = message
            self._error = error

    def status(self) -> dict[str, Any]:
        """Current lifecycle snapshot, safe to poll frequently."""
        with self._state_lock:
            return {
                "state": self._state.value,
                "progress": round(self._progress, 3),
                "message": self._message,
                "error": self._error,
                "adapter": self._adapter,
                "adapter_load": self._adapter_load,
                "base_model_id": self.settings.base_model_id,
                "device": None if self._policy is None else self._policy["device"],
                "compute_dtype": None
                if self._policy is None
                else self._policy["compute_dtype_name"],
                "quantization": None if self._policy is None else self._policy["quantization"],
                "weights_source": self._weights_source,
                "loaded_at": self._loaded_at,
                "load_seconds": self._load_seconds,
            }

    @property
    def is_ready(self) -> bool:
        with self._state_lock:
            return self._state is ModelState.READY

    # ----------------------------------------------------------------- loading
    def load_async(self, adapter_id: str | None = None) -> dict[str, Any]:
        """Start loading in a background thread and return immediately.

        Re-requesting the adapter that is already loaded is a no-op.

        Raises:
            AdapterError: unknown adapter id.
        """
        adapter = resolve_adapter(adapter_id)

        with self._state_lock:
            if self._state is ModelState.LOADING:
                return self.status()
            if (
                self._state is ModelState.READY
                and self._adapter is not None
                and self._adapter["id"] == adapter["id"]
            ):
                return self.status()
            self._state = ModelState.LOADING
            self._progress = 0.0
            self._message = f"Queued load of {adapter['label']}"
            self._error = None

        thread = threading.Thread(
            target=self._load_worker,
            args=(adapter,),
            name="atria-model-load",
            daemon=True,
        )
        self._load_thread = thread
        thread.start()
        return self.status()

    def _load_worker(self, adapter: dict[str, Any]) -> None:
        try:
            self._load(adapter)
        except Exception as exc:  # noqa: BLE001 - surfaced through status()
            logger.exception("Model load failed")
            self._set_state(
                state=ModelState.ERROR,
                progress=0.0,
                message="Model load failed.",
                error=self._explain_load_error(exc, adapter),
            )

    def _explain_load_error(self, exc: Exception, adapter: dict[str, Any]) -> str:
        """Turn a load failure into an instruction the operator can act on."""
        text = str(exc)
        lowered = text.lower()
        if "gated" in lowered or "401" in lowered or "403" in lowered:
            # Only remote repositories can be gated; a local checkpoint directory
            # never is, so naming it here would send the operator down a false trail.
            targets = []
            if self.settings.resolve_base_model()[1] != "local":
                targets.append(self.settings.base_model_id)
            if adapter.get("repo") and adapter.get("source") != "local":
                targets.append(str(adapter["repo"]))
            if targets:
                return (
                    "Access denied by Hugging Face for: "
                    + ", ".join(targets)
                    + ". Accept the terms for each while signed in, then provide a token "
                    "with read access (run `hf auth login`, or set HF_TOKEN). "
                    "Alternatively, download them into the project's models/ and "
                    "adapters/ folders, which needs no token — see the Weights panel. "
                    "Original error: " + text
                )
            # Everything resolves locally, so a gated/401 message cannot be about
            # access to these weights. Saying "access denied" would misdirect.
            return (
                "The load failed while contacting Hugging Face even though every weight "
                "resolves to a local folder. Set ATRIA_OFFLINE=1 to forbid Hub access "
                "entirely. Original error: " + text
            )
        if "out of memory" in lowered or "cuda error" in lowered:
            return (
                "The device ran out of memory while loading. Try ATRIA_FORCE_CPU=1, "
                "or free GPU memory and retry. Original error: " + text
            )
        if "no module named" in lowered:
            return (
                'The AI tier is incompletely installed. Run: pip install -e ".[ai]". '
                "Original error: " + text
            )
        return text

    def load(
        self,
        adapter_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Load synchronously. Returns the resulting status."""
        adapter = resolve_adapter(adapter_id)
        self._load(adapter, progress)
        return self.status()

    def _load(self, adapter: dict[str, Any], progress: ProgressCallback | None = None) -> None:
        """Load base weights and optionally attach a LoRA adapter."""

        def report(fraction: float, message: str) -> None:
            self._set_state(state=ModelState.LOADING, progress=fraction, message=message)
            logger.info("[load %3d%%] %s", int(fraction * 100), message)
            if progress is not None:
                progress(fraction, message)

        started = time.time()
        configure_torch_allocator()

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        with self._device_lock:
            # Release any previous model before allocating the next one.
            self._release_model()

            policy = select_device_policy(force_cpu=self.settings.force_cpu)
            self._policy = policy
            report(0.05, f"Device: {policy['device']} ({policy['reason']})")

            token = self.settings.token_value()
            # A hand-placed copy under models/ wins over the cache and the Hub, so a
            # fully offline, token-free install works (RESEARCH.md §9).
            model_id, weights_source = self.settings.resolve_base_model()
            self._weights_source = weights_source
            if weights_source == "local":
                report(0.10, f"Using local base model at {model_id}")

            quantization = build_quantization_config(policy)

            def load_base(local_only: bool):
                """Load processor + base weights, optionally cache-only."""
                processor = AutoProcessor.from_pretrained(
                    model_id, token=token, local_files_only=local_only
                )
                # Left padding for generation (notebook L877, L1272).
                processor.tokenizer.padding_side = "left"

                load_kwargs: dict[str, Any] = {
                    # MedGemma's card specifies eager attention for Gemma 3.
                    "attn_implementation": "eager",
                    # Transformers v5 renamed torch_dtype to dtype and defaults to
                    # "auto"; passing it explicitly avoids silent precision drift.
                    "dtype": policy["compute_dtype"],
                    "token": token,
                    "local_files_only": local_only,
                }
                if quantization is not None:
                    load_kwargs["quantization_config"] = quantization
                    load_kwargs["device_map"] = "auto"
                return processor, AutoModelForImageTextToText.from_pretrained(
                    model_id, **load_kwargs
                )

            report(0.15, f"Loading processor for {model_id}")
            report(0.30, f"Loading base weights for {model_id} (this can take minutes)")
            try:
                self._processor, model = load_base(self.settings.offline)
            except Exception as exc:
                # The base model is gated, so an unauthenticated machine gets a 401 even
                # when the weights are already cached. Retrying cache-only makes a warm
                # cache sufficient, which is the normal state on a clinical workstation.
                if self.settings.offline or not _is_hub_access_error(exc):
                    raise
                logger.warning(
                    "Hub access failed (%s). Retrying from the local Hugging Face "
                    "cache only.",
                    str(exc).splitlines()[0][:160],
                )
                report(0.30, "Hub unavailable; loading from the local cache")
                self._processor, model = load_base(True)

            if policy["device"] == "cpu":
                model = model.to("cpu")

            repo = adapter.get("repo")
            if repo:
                report(0.80, f"Attaching LoRA adapter {repo}")
                from peft import PeftModel

                # A local checkpoint directory is always read cache-only; there is
                # nothing to fetch and a Hub round-trip would only add a failure mode.
                adapter_local_only = self.settings.offline or adapter.get("source") == "local"
                try:
                    model = PeftModel.from_pretrained(
                        model, repo, token=token, local_files_only=adapter_local_only
                    )
                except Exception as exc:
                    if adapter_local_only or not _is_hub_access_error(exc):
                        raise
                    logger.warning(
                        "Hub access failed for adapter %s. Retrying from the local "
                        "cache only.",
                        repo,
                    )
                    model = PeftModel.from_pretrained(
                        model, repo, token=token, local_files_only=True
                    )

                # The checkpoint may address module paths from an older transformers
                # layout; without this the vision-tower LoRA loads as zeros in silence.
                self._adapter_load = repair_legacy_adapter_keys(model, repo)
            else:
                self._adapter_load = {"fully_loaded": True, "remapped": 0,
                                      "vision_lora_b": 0, "vision_lora_b_active": 0}

            model.eval()
            self._model = model
            self._adapter = adapter

            elapsed = time.time() - started
            self._loaded_at = time.time()
            self._load_seconds = round(elapsed, 1)
            self._set_state(
                state=ModelState.READY,
                progress=1.0,
                message=(
                    f"Ready: {adapter['label']} on {policy['device']} "
                    f"({policy['compute_dtype_name']}"
                    f"{', 4-bit NF4' if policy['quantization'] else ''}) "
                    f"in {elapsed:.1f}s"
                ),
            )
            logger.info(
                "Model ready: %s on %s in %.1fs", adapter["label"], policy["device"], elapsed
            )

    def _release_model(self) -> None:
        """Drop references and free device memory."""
        if self._model is None and self._processor is None:
            return
        logger.info("Releasing currently loaded model")
        self._model = None
        self._processor = None
        self._adapter = None
        self._adapter_load = {}
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch absent or CUDA unavailable
            pass

    def unload(self) -> dict[str, Any]:
        """Free the model (notebook's "Free Memory for Evaluation" cell, L853-862)."""
        with self._device_lock:
            self._release_model()
            self._policy = None
            self._loaded_at = None
            self._load_seconds = None
            self._set_state(
                state=ModelState.UNLOADED, progress=0.0, message="Model not loaded."
            )
        return self.status()

    # --------------------------------------------------------------- inference
    def predict(
        self,
        image: Image.Image,
        target_structure: str = "LV",
        view: str | None = None,
        instant: str | None = None,
        prompt_variant: PromptVariant | None = None,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Predict a normalised contour for one frame.

        Args:
            image: The frame; converted to RGB, as the notebook does.
            target_structure: ``"LV"`` or ``"LA"``.
            view: ``"2CH"``/``"4CH"``, used by the training prompt variant.
            instant: ``"ED"``/``"ES"``, used by the training prompt variant.
            prompt_variant: Force a template; defaults to ``training`` when view and
                instant are known (RESEARCH.md §0.4).
            max_new_tokens: Generation cap; defaults to the configured 1024.

        Returns:
            Mapping with ``polygon`` (normalised ``[[y, x], ...]``), ``raw_response``,
            ``prompt_variant``, ``inference_seconds``, ``adapter``, ``device`` and
            ``vertices``.

        Raises:
            ModelNotReady: if the model is not loaded.
            ValueError: for an unknown structure or an unsatisfiable prompt variant.
            RuntimeError: if generation produced no parseable polygon.
        """
        if not self.is_ready:
            raise ModelNotReady(
                f"Model is not ready (state={self.status()['state']}). "
                "POST /api/model/load first."
            )

        prompt_text, variant_used = build_prompt(
            target_structure=target_structure,
            view=view,
            instant=instant,
            variant=prompt_variant,
        )
        frame = image.convert("RGB") if image.mode != "RGB" else image
        started = time.time()

        import torch

        with self._device_lock:
            if self._model is None or self._processor is None:  # pragma: no cover
                raise ModelNotReady("Model was unloaded while the request was queued.")

            processor = self._processor
            model = self._model

            messages = build_messages(frame, prompt_text)
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = processor(text=text, images=[frame], return_tensors="pt").to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens or self.settings.max_new_tokens,
                    do_sample=False,  # greedy, as the notebook (L913, L1305)
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

        elapsed = time.time() - started
        parsed = parse_polygon(response, target_structure)
        polygon = sanitize_polygon(parsed, self.settings.norm_scale)

        if polygon is None:
            raise RuntimeError(
                f"The model returned no parseable {target_structure.upper()} polygon "
                f"with at least 3 valid vertices. First 300 characters of the raw "
                f"response: {response[:300]!r}"
            )

        logger.info(
            "Predicted %s contour: %d vertices in %.1fs (adapter=%s, prompt=%s)",
            target_structure.upper(),
            len(polygon),
            elapsed,
            (self._adapter or {}).get("id"),
            variant_used,
        )
        return {
            "polygon": polygon,
            "vertices": len(polygon),
            "raw_response": response,
            "prompt_variant": variant_used,
            "prompt": prompt_text,
            "target_structure": target_structure.upper(),
            "inference_seconds": round(elapsed, 2),
            # The notebook printed the input token count to diagnose silent truncation
            # against its 2048-token training cap (L1042-1048). Nothing truncates at
            # inference here, but the count stays the cheapest way to see a prompt
            # drifting towards that cap.
            "input_tokens": int(input_len),
            "generated_tokens": int(output_ids.shape[1] - input_len),
            "adapter": self._adapter,
            "device": (self._policy or {}).get("device"),
        }


_engine: InferenceEngine | None = None
_engine_lock = threading.Lock()


def get_inference_engine(settings: Settings | None = None) -> InferenceEngine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = InferenceEngine(settings)
        return _engine


def reset_inference_engine() -> None:
    """Drop the singleton. Used by tests."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            _engine.unload()
        _engine = None
