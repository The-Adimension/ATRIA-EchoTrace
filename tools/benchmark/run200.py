"""Full benchmark: CAMUS official test subgroup, 50 patients / 200 frames, LV, config A2.

A2 = 4-bit NF4 + double-quant, training-matched prompt (true view + instant) — the
configuration selected as primary from the three-way pilot.

Resumable: every frame is appended to JSONL and flushed immediately, so an interrupt
costs at most one frame. Re-running skips what is already done.

Batch size is a command-line argument. It changes throughput only if the probe showed
batched output to be byte-identical to single-frame output; otherwise pass 1.
"""

import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, "src")
from atria_echotrace.config import Settings  # noqa: E402
from atria_echotrace.data.frames import load_frame  # noqa: E402
from atria_echotrace.domain.geometry import (  # noqa: E402
    parse_polygon,
    polygon_dice,
    sanitize_polygon,
)
from atria_echotrace.logging_setup import configure  # noqa: E402
from atria_echotrace.ml.engine import InferenceEngine  # noqa: E402
from atria_echotrace.ml.prompts import build_messages, build_prompt  # noqa: E402

configure("ERROR")

# Keep the machine awake for the lifetime of this process only.
# Sleeping mid-run does not kill the job outright — it leaves the CUDA context degraded,
# and the per-frame time climbs (87s -> 484 -> 503 -> 623) until the process dies. This
# is a per-process request that Windows drops automatically on exit; it changes no
# system-wide power setting.
if sys.platform == "win32":
    import ctypes

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    if ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        print("  sleep suppressed for the duration of this run", flush=True)
    else:
        print("  WARNING: could not suppress sleep; the run may degrade if the machine sleeps",
              flush=True)

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 1
BENCH = pathlib.Path("outputs/benchmark/camus_test")
OUT = pathlib.Path("outputs/benchmark/full200")
OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "raw_predictions.jsonl"

tracings = json.loads((BENCH / "tracings.json").read_text())
stems = sorted(tracings)  # deterministic order
assert len(stems) == 200, f"expected 200 frames, found {len(stems)}"

done = set()
if RAW.exists():
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["stem"])
todo = [s for s in stems if s not in done]

print(f"FULL BENCHMARK  config A2 (4-bit NF4, training prompt)  batch={BATCH}")
print(f"  frames: {len(stems)} total, {len(done)} already done, {len(todo)} to run")

engine = InferenceEngine(Settings())
t0 = time.time()
engine.load("camus")
status = engine.status()
load = status.get("adapter_load") or {}
print(
    f"  model ready in {status['load_seconds']}s "
    f"({status['device']} {status['compute_dtype']} quant={status['quantization']})"
)
print(
    f"  adapter: remapped={load.get('remapped')} "
    f"vision LoRA {load.get('vision_lora_b_active')}/{load.get('vision_lora_b')} "
    f"fully_loaded={load.get('fully_loaded')}",
    flush=True,
)
assert load.get("fully_loaded"), "adapter NOT fully loaded - abort"
assert status["quantization"] == "nf4", "expected 4-bit NF4 for config A2 - abort"

model, processor = engine._model, engine._processor
settings = engine.settings


def infer(batch_stems):
    """Generate for one batch; returns [(stem, polygon, raw_text, seconds, tokens)]."""
    texts, images = [], []
    for stem in batch_stems:
        e = tracings[stem]
        img = load_frame(BENCH / "frames" / f"{stem}.png")
        img = img.convert("RGB") if img.mode != "RGB" else img
        prompt, _ = build_prompt("LV", view=e["view"], instant=e["instant"])
        texts.append(
            processor.apply_chat_template(
                build_messages(img, prompt), add_generation_prompt=True, tokenize=False
            )
        )
        images.append([img])

    started = time.time()
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(
        model.device
    )
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    elapsed = time.time() - started
    input_len = inputs["input_ids"].shape[1]
    results = []
    for i, stem in enumerate(batch_stems):
        text = processor.decode(out[i][input_len:], skip_special_tokens=True)
        poly = sanitize_polygon(parse_polygon(text, "LV"), settings.norm_scale)
        results.append((stem, poly, text, elapsed / len(batch_stems),
                        int(out.shape[1] - input_len), int(input_len)))
    return results


completed = len(done)
with RAW.open("a", encoding="utf-8") as sink:
    for start in range(0, len(todo), BATCH):
        chunk = todo[start : start + BATCH]
        try:
            outcomes = infer(chunk)
        except Exception as exc:  # noqa: BLE001 - record and continue
            for stem in chunk:
                sink.write(json.dumps({"stem": stem, "parsed": False,
                                       "error": f"{type(exc).__name__}: {exc}"[:400]}) + "\n")
            sink.flush()
            print(f"  BATCH FAILED ({chunk[0]}...): {type(exc).__name__}", flush=True)
            continue

        for stem, poly, text, secs, gen_tok, in_tok in outcomes:
            e = tracings[stem]
            rec = {
                "stem": stem, "patient": stem.split("_")[0], "view": e["view"],
                "instant": e["instant"], "image_h": e["image_h"], "image_w": e["image_w"],
                "spacing_h": e["spacing_h"], "spacing_w": e["spacing_w"],
                "seconds": round(secs, 1), "input_tokens": in_tok,
                "generated_tokens": gen_tok, "raw_response": text,
            }
            if poly:
                rec.update(parsed=True, polygon=poly, vertices=len(poly))
                rec["dice_vs_polygon"] = round(
                    polygon_dice(poly, e["lv_polygon"], e["image_h"], e["image_w"]), 4)
            else:
                rec.update(parsed=False, error="no parseable polygon")
            sink.write(json.dumps(rec) + "\n")
            completed += 1
        sink.flush()

        rate = (time.time() - t0) / max(completed - len(done), 1)
        remaining = (len(stems) - completed) * rate / 60
        last = outcomes[-1]
        print(
            f"  [{completed:3}/{len(stems)}] {last[0]:30} "
            f"{len(last[1]) if last[1] else 0:2} pts  {last[3]:.0f}s/frame  "
            f"~{remaining:.0f} min left",
            flush=True,
        )

engine.unload()
print(f"\ncomplete: {completed}/{len(stems)} in {(time.time()-t0)/60:.1f} min -> {RAW}")
