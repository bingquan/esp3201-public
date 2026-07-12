"""Generate a video rollout bank for the Week 5 labs and verify T4-feasibility.

Loads a text-to-video model in fp16 with CPU offload (the free-tier T4 config),
generates short low-resolution clips, saves them as PNG frame sequences, and
reports the MEASURED peak GPU memory so feasibility is evidence, not a guess.

Usage:
    python generate_rollout_bank.py --out ../rollout_bank --clips 2

The peak-VRAM number is the portable feasibility proxy: if it fits under a
T4's ~15 GB in fp16 + offload here, it fits on Colab's free T4. Wall-clock
here is NOT the Colab time (this may run on a much faster GPU); measure real
Colab timing with one free-tier run.

Attribution: generation follows the diffusers text-to-video tutorials. Code
is original to ESP3201.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Default prompts: a mix of robotics/physical-scene rollouts for the lab and
# the mini-assignment's probe intuition (occlusion, drop, push).
DEFAULT_PROMPTS = {
    "clip01_arm_pick": "a robotic arm slowly picking up a red cube from a table, smooth motion, realistic",
    "clip02_ball_drop": "a ball falling and bouncing on the floor, physically realistic motion",
    "clip03_push_block": "a robot hand pushing a wooden block across a table to the left",
    "clip04_occlusion": "a toy car driving behind a box and reappearing on the other side",
}


def _to_pil(frame):
    """Convert a diffusers video frame (np array or PIL) to a PIL RGB image."""
    from PIL import Image
    import numpy as np
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.dtype != "uint8":
        arr = (arr.clip(0, 1) * 255).astype("uint8") if arr.max() <= 1.0 \
            else arr.clip(0, 255).astype("uint8")
    return Image.fromarray(arr).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cerspense/zeroscope_v2_576w",
                    help="text-to-video checkpoint (PIN per offering)")
    ap.add_argument("--out", default="../rollout_bank")
    ap.add_argument("--clips", type=int, default=len(DEFAULT_PROMPTS))
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=576)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from diffusers import DiffusionPipeline

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} in fp16 (T4 config: cpu offload)...")
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    torch.cuda.reset_peak_memory_stats()
    manifest = {"model": args.model, "frames": args.frames, "steps": args.steps,
                "height": args.height, "width": args.width, "clips": []}

    items = list(DEFAULT_PROMPTS.items())[: args.clips]
    for name, prompt in items:
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        t0 = time.perf_counter()
        result = pipe(prompt, num_frames=args.frames, num_inference_steps=args.steps,
                      height=args.height, width=args.width, generator=gen)
        dt = time.perf_counter() - t0
        frames = result.frames[0]  # list/array of frames (np arrays for this pipeline)

        clip_dir = out_root / name
        clip_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            _to_pil(frame).save(clip_dir / f"frame_{i:03d}.png")

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {name}: {len(frames)} frames, {dt:.1f}s (this GPU), "
              f"peak VRAM so far {peak_gb:.2f} GB")
        manifest["clips"].append({"name": name, "prompt": prompt,
                                  "n_frames": len(frames),
                                  "seconds_this_gpu": round(dt, 1)})

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    manifest["peak_vram_gb_fp16_offload"] = round(peak_gb, 2)
    manifest["note"] = ("peak_vram measured here is the T4-fit proxy; "
                        "seconds_this_gpu is NOT the Colab T4 time.")
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nMEASURED peak VRAM (fp16 + cpu offload): {peak_gb:.2f} GB")
    print(f"T4 has ~15 GB -> {'FITS' if peak_gb < 14 else 'TIGHT/NO-FIT'} on a free T4.")
    print(f"Wrote {len(items)} clips + manifest to {out_root}")


if __name__ == "__main__":
    main()
