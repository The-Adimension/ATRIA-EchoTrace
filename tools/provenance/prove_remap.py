"""Prove the vision-tower key remap works, before changing any application code."""

import sys

import torch

sys.path.insert(0, "src")
from atria_echotrace.config import Settings  # noqa: E402
from atria_echotrace.logging_setup import configure  # noqa: E402
from atria_echotrace.ml.engine import InferenceEngine  # noqa: E402

configure("ERROR")


def vision_lora_b_stats(model):
    tensors = [(n, p) for n, p in model.named_parameters()
               if "lora_B" in n and "vision_tower" in n]
    nonzero = sum(1 for _, p in tensors if float(p.abs().sum()) > 0)
    return len(tensors), nonzero


engine = InferenceEngine(Settings())
engine.load("camus")
model = engine._model
total, nz_before = vision_lora_b_stats(model)
print(f"  BEFORE repair: {nz_before}/{total} vision lora_B non-zero")

# --- the repair ------------------------------------------------------------------
from peft import load_peft_weights, set_peft_model_state_dict  # noqa: E402

adapter_dir = engine._adapter["repo"]
raw = load_peft_weights(adapter_dir, device="cpu")
print(f"  checkpoint tensors: {len(raw)}")

module_names = {n for n, _ in model.named_modules()}
# Only remap when the checkpoint carries a nesting level the live model does not have.
needs_remap = any(".vision_model." in k for k in raw) and not any(
    ".vision_model." in n for n in module_names
)
print(f"  legacy 'vision_model.' nesting present in checkpoint, absent in model: {needs_remap}")

if needs_remap:
    remapped = {k.replace(".vision_tower.vision_model.", ".vision_tower."): v
                for k, v in raw.items()}
    changed = sum(1 for k in raw if ".vision_tower.vision_model." in k)
    print(f"  remapped {changed} keys")
    info = set_peft_model_state_dict(model, remapped, adapter_name="default")
    missing = [k for k in getattr(info, "unexpected_keys", []) if "vision_tower" in k]
    print(f"  set_peft_model_state_dict unexpected vision keys: {len(missing)}")

total, nz_after = vision_lora_b_stats(model)
print(f"  AFTER  repair: {nz_after}/{total} vision lora_B non-zero")
print()
print("  RESULT:", "REPAIR WORKS - vision tower is now active"
      if nz_after == total and total > 0 else "REPAIR FAILED")

# Sanity: does it still produce a contour, and does it differ from the crippled run?
if nz_after == total:
    import json
    import pathlib
    from atria_echotrace.data.frames import load_frame
    from atria_echotrace.domain.geometry import polygon_dice

    bench = pathlib.Path("outputs/benchmark/camus_test")
    tr = json.loads((bench / "tracings.json").read_text())
    stem = "patient0052_2CH_ED"
    e = tr[stem]
    r = engine.predict(load_frame(bench / "frames" / f"{stem}.png"), "LV",
                       view=e["view"], instant=e["instant"])
    d = polygon_dice(r["polygon"], e["lv_polygon"], e["image_h"], e["image_w"])
    print(f"\n  smoke: {stem} -> {r['vertices']} pts, Dice {d:.4f} "
          f"(condition A crippled gave 0.7531)")
engine.unload()
