"""Does batching accelerate this, and does it change the answers?

Decode runs at ~3.8 tok/s. If the GPU is under-utilised at batch 1 (memory-bound),
batching should give near-linear throughput. If it is compute-saturated (plausible with
emulated bf16 on sm_75), batching buys nothing.

Correctness gate: A2 already produced exact outputs for these frames. Batched inference
must reproduce them byte-for-byte, or it is not the same experiment and must not be used.
"""

import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, "src")
from atria_echotrace.config import Settings  # noqa: E402
from atria_echotrace.data.frames import load_frame  # noqa: E402
from atria_echotrace.domain.geometry import parse_polygon, sanitize_polygon  # noqa: E402
from atria_echotrace.logging_setup import configure  # noqa: E402
from atria_echotrace.ml.engine import InferenceEngine  # noqa: E402
from atria_echotrace.ml.prompts import build_messages, build_prompt  # noqa: E402

configure("ERROR")

BENCH = pathlib.Path("outputs/benchmark/camus_test")
A2 = {
    json.loads(l)["stem"]: json.loads(l)
    for l in pathlib.Path(
        "outputs/benchmark/pilot20_fixed/raw_predictions.jsonl"
    ).read_text().splitlines()
    if l.strip()
}
tracings = json.loads((BENCH / "tracings.json").read_text())
STEMS = [s for s in A2 if A2[s].get("parsed")][:8]

engine = InferenceEngine(Settings())
engine.load("camus")
load = engine.status()["adapter_load"]
assert load["fully_loaded"], "adapter not fully loaded"
print(
    f"adapter fully_loaded={load['fully_loaded']} "
    f"({load['vision_lora_b_active']}/{load['vision_lora_b']} vision LoRA)\n",
    flush=True,
)

model, processor = engine._model, engine._processor
settings = engine.settings


def run_batch(stems):
    """Batched generation, mirroring engine.predict but over several frames at once."""
    texts, images = [], []
    for stem in stems:
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

    inputs = processor(
        text=texts, images=images, return_tensors="pt", padding=True
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]  # left padding => uniform
    results = []
    for i, stem in enumerate(stems):
        text = processor.decode(out[i][input_len:], skip_special_tokens=True)
        results.append((stem, sanitize_polygon(parse_polygon(text, "LV"), settings.norm_scale)))
    return results


print("=" * 74)
for bs in (1, 2, 4):
    subset = STEMS[:4]
    torch.cuda.synchronize()
    t0 = time.time()
    got = []
    for i in range(0, len(subset), bs):
        got.extend(run_batch(subset[i : i + bs]))
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    identical = sum(1 for stem, poly in got if poly == A2[stem]["polygon"])
    print(
        f"  batch={bs}: {elapsed:6.1f}s for {len(subset)} frames "
        f"({elapsed / len(subset):5.1f}s/frame)  identical to A2: {identical}/{len(got)}",
        flush=True,
    )

print("=" * 74)
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
engine.unload()
