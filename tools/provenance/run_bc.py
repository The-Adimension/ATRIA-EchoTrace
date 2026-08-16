"""Conditions B and C on the identical 20 pilot frames. Measurement only.

B  unquantised bf16 + training-matched prompt (true view + instant)
C  unquantised bf16 + the geometry-focused prompt supplied verbatim

Only the controlled variable is changed, by patching in this harness:
  quantisation -> ml.runtime.select_device_policy returns quantization=None
  prompt (C)   -> ml.engine.build_prompt returns the supplied text
Application defaults are untouched; nothing here is written back to the package.
Same frames, same order, same adapter, same decoding (greedy, max_new_tokens default).
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, "src")
from atria_echotrace.config import Settings  # noqa: E402
from atria_echotrace.data.frames import load_frame  # noqa: E402
from atria_echotrace.domain.geometry import polygon_dice  # noqa: E402
from atria_echotrace.logging_setup import configure  # noqa: E402
from atria_echotrace.ml import engine as engine_mod  # noqa: E402
from atria_echotrace.ml import runtime as runtime_mod  # noqa: E402

configure("ERROR")

CONDITION = sys.argv[1]  # "B" or "C"
assert CONDITION in {"B", "C"}

BENCH = pathlib.Path("outputs/benchmark/camus_test")
OUT = pathlib.Path("outputs/benchmark/pilot20_fixed")
RAW = OUT / f"raw_predictions_{CONDITION}.jsonl"
PATIENTS = ["patient0052", "patient0189", "patient0225", "patient0238", "patient0266"]

# --- controlled variable 1: no quantisation -------------------------------------
_original_policy = runtime_mod.select_device_policy


def unquantised_policy(force_cpu: bool = False):
    policy = _original_policy(force_cpu=force_cpu)
    policy["quantization"] = None          # bf16 weights, no NF4
    policy["reason"] = "EXPERIMENT: quantisation disabled (unquantised bf16)"
    return policy


runtime_mod.select_device_policy = unquantised_policy
engine_mod.select_device_policy = unquantised_policy

# --- controlled variable 2 (C only): the supplied prompt, verbatim ---------------
CONDITION_C_PROMPT = (
    "Instructions:\n"
    "The following user query will require outputting polygon coordinates for the left "
    "ventricle endocardium. The format of polygon coordinates is [[y0, x0], [y1, x1], ...] "
    "where each point is [y, x] representing the boundary. Always normalize the x and y "
    "coordinates to the range [0, 1000], meaning that a point at 15% of the image width "
    "would be associated with an x coordinate of 150. You MUST output a single parseable "
    'json list of objects enclosed into ```json...``` brackets, for instance ```json'
    '[{"polygon_2d": [[100, 200], [150, 250]], "label": "the left ventricle endocardium"}]``` '
    "is a valid output. Now answer to the user query.\n\n"
    "Query:\n"
    "This is an echocardiogram view. Trace the left ventricle endocardium. Output the final "
    'answer in the format "Final Answer: X" where X is a JSON list of objects with '
    '"polygon_2d" and "label" keys. Answer:'
)

if CONDITION == "C":
    def fixed_prompt(target_structure="LV", view=None, instant=None, variant=None):
        return CONDITION_C_PROMPT, "geometry-focused"

    engine_mod.build_prompt = fixed_prompt

tracings = json.loads((BENCH / "tracings.json").read_text())
stems = [f"{p}_{v}_{i}" for p in PATIENTS for v in ("2CH", "4CH") for i in ("ED", "ES")]

done = set()
if RAW.exists():
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["stem"])

print(f"CONDITION {CONDITION}: {len(stems)} frames, unquantised bf16, "
      f"{'training-matched prompt' if CONDITION == 'B' else 'geometry-focused prompt'}")
if done:
    print(f"resuming: {len(done)} already done")

settings = Settings()
engine = engine_mod.InferenceEngine(settings)
t0 = time.time()
engine.load("camus")
status = engine.status()
assert status["quantization"] is None, "quantisation was NOT disabled - abort"

# The engine sets device_map only when quantising, so an unquantised load lands on the
# CPU. Place it on the GPU here (harness-only; the package is untouched).
import torch  # noqa: E402

device = next(engine._model.parameters()).device
if device.type != "cuda":
    print(f"loaded on {device}; moving to cuda ...", flush=True)
    engine._model.to("cuda")
    device = next(engine._model.parameters()).device
print(f"model ready in {status['load_seconds']}s on {device} "
      f"({status['compute_dtype']}, quant={status['quantization']}), "
      f"GPU alloc {torch.cuda.memory_allocated()/2**30:.1f} GiB\n", flush=True)
assert device.type == "cuda", "model is not on the GPU - abort"

# The adapter must be COMPLETE. A silent partial load is exactly the defect that
# invalidated the first pilot; never measure without checking.
load_info = status.get("adapter_load") or {}
print(
    f"adapter: remapped={load_info.get('remapped')} "
    f"vision LoRA {load_info.get('vision_lora_b_active')}/{load_info.get('vision_lora_b')} "
    f"fully_loaded={load_info.get('fully_loaded')}\n",
    flush=True,
)
assert load_info.get("fully_loaded"), "adapter NOT fully loaded - abort"

with RAW.open("a", encoding="utf-8") as sink:
    for n, stem in enumerate(stems, 1):
        if stem in done:
            continue
        e = tracings[stem]
        image = load_frame(BENCH / "frames" / f"{stem}.png")
        rec = {"stem": stem, "patient": stem.split("_")[0], "condition": CONDITION,
               "view": e["view"], "instant": e["instant"],
               "image_h": e["image_h"], "image_w": e["image_w"],
               "spacing_h": e["spacing_h"], "spacing_w": e["spacing_w"]}
        try:
            # C deliberately withholds view/instant from the prompt; the patched
            # build_prompt ignores them, so passing them through changes nothing.
            r = engine.predict(image, "LV", view=e["view"], instant=e["instant"])
            rec.update(parsed=True, polygon=r["polygon"], vertices=r["vertices"],
                       seconds=r["inference_seconds"], prompt_variant=r["prompt_variant"],
                       input_tokens=r.get("input_tokens"),
                       generated_tokens=r.get("generated_tokens"),
                       raw_response=r["raw_response"], prompt=r["prompt"])
            d = polygon_dice(r["polygon"], e["lv_polygon"], e["image_h"], e["image_w"])
            rec["dice_vs_polygon"] = round(d, 4)
            print(f"  [{n:2}/{len(stems)}] {stem:26} {r['vertices']:3} pts  "
                  f"Dice {d:.4f}  {r['inference_seconds']:.0f}s")
        except Exception as exc:  # noqa: BLE001
            rec.update(parsed=False, error=f"{type(exc).__name__}: {exc}"[:400])
            print(f"  [{n:2}/{len(stems)}] {stem:26} FAILED: {type(exc).__name__}")
        sink.write(json.dumps(rec) + "\n")
        sink.flush()

engine.unload()
print(f"\ncondition {CONDITION} complete in {(time.time()-t0)/60:.1f} min -> {RAW}")
