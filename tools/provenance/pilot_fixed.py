"""20-frame pilot: 5 patients x 4 frames, exercising the full metric chain end to end.

Deliberately 5 whole patients rather than 20 loose frames, so biplane EF (which needs
2CH+4CH at ED+ES) is exercised too. Results are appended to JSONL as they arrive, so a
crash or interrupt never loses completed inferences.
"""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from atria_echotrace.config import Settings  # noqa: E402
from atria_echotrace.data.frames import load_frame  # noqa: E402
from atria_echotrace.domain.geometry import polygon_dice, polygon_iou, polygon_to_mask  # noqa: E402
from atria_echotrace.logging_setup import configure  # noqa: E402
from atria_echotrace.ml.engine import InferenceEngine  # noqa: E402

configure("ERROR")

BENCH = pathlib.Path("outputs/benchmark/camus_test")
OUT = pathlib.Path("outputs/benchmark/pilot20_fixed")
OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "raw_predictions.jsonl"

# 2 Good, 2 Medium, 1 Poor; EF 5 -> 69, i.e. severe dysfunction through normal.
PATIENTS = ["patient0052", "patient0189", "patient0225", "patient0238", "patient0266"]

tracings = json.loads((BENCH / "tracings.json").read_text())
stems = [f"{p}_{v}_{i}" for p in PATIENTS for v in ("2CH", "4CH") for i in ("ED", "ES")]
assert all(s in tracings for s in stems), "pilot stems missing from the staged set"
print(f"pilot: {len(PATIENTS)} patients, {len(stems)} frames\n")

done = set()
if RAW.exists():
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["stem"])
    print(f"resuming: {len(done)} already inferred\n")

settings = Settings()
engine = InferenceEngine(settings)
t0 = time.time()
engine.load("camus")
status = engine.status()
print(f"model ready in {status['load_seconds']}s "
      f"({status['device']} {status['compute_dtype']} {status['quantization']}, "
      f"weights={status['weights_source']})\n")

with RAW.open("a", encoding="utf-8") as sink:
    for n, stem in enumerate(stems, 1):
        if stem in done:
            continue
        entry = tracings[stem]
        image = load_frame(BENCH / "frames" / f"{stem}.png")
        record = {
            "stem": stem,
            "patient": stem.split("_")[0],
            "view": entry["view"],
            "instant": entry["instant"],
            "image_h": entry["image_h"],
            "image_w": entry["image_w"],
            "spacing_h": entry["spacing_h"],
            "spacing_w": entry["spacing_w"],
        }
        try:
            result = engine.predict(
                image, "LV", view=entry["view"], instant=entry["instant"]
            )
            record.update(
                parsed=True,
                polygon=result["polygon"],
                vertices=result["vertices"],
                seconds=result["inference_seconds"],
                prompt_variant=result["prompt_variant"],
                input_tokens=result.get("input_tokens"),
                generated_tokens=result.get("generated_tokens"),
                raw_response=result["raw_response"],
            )
            dice = polygon_dice(result["polygon"], entry["lv_polygon"],
                                entry["image_h"], entry["image_w"])
            record["dice_vs_polygon"] = round(dice, 4)
            print(f"  [{n:2}/{len(stems)}] {stem:26} {result['vertices']:2} pts  "
                  f"Dice {dice:.4f}  {result['inference_seconds']:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - a failure is data, not a crash
            record.update(parsed=False, error=f"{type(exc).__name__}: {exc}"[:400])
            print(f"  [{n:2}/{len(stems)}] {stem:26} FAILED: {type(exc).__name__}")
        sink.write(json.dumps(record) + "\n")
        sink.flush()

engine.unload()
print(f"\ninference complete in {(time.time()-t0)/60:.1f} min -> {RAW}")
