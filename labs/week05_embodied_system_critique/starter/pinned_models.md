# Week 5 Pinned Models (per offering)

Instructors pin one exact checkpoint per lab and verify it on the free tier
before release. This file records the **verified** pin for the current offering.

## 5b — Video world model

- **Checkpoint:** `cerspense/zeroscope_v2_576w`
- **Config (mirrors free-tier T4):** `torch_dtype=torch.float16`, `enable_model_cpu_offload()`, `enable_vae_slicing()`
- **Generation settings:** 576×320, 24 frames, 25 inference steps
- **MEASURED peak VRAM:** **5.81 GB** (fp16 + CPU offload) → fits a free T4 (~15 GB) with large headroom
- **Generation speed:** ~8–10 s per clip on an H200 in this environment. **This is NOT the Colab T4 time** — a T4 is much slower; measure real T4 timing with one free-tier run (expect minutes per clip, still within a free session).
- **Output format note:** this pipeline (`TextToVideoSDPipeline`) returns **numpy** frames in `output.frames[0]`, not PIL. `video_eval.frames_from_diffusers()` and `generate_rollout_bank._to_pil()` handle both; verified against this checkpoint.
- **Status:** verified end-to-end here — generation runs, frames save, and the lab metrics score the real clips (temporal consistency ≈ 0.99). A `rollout_bank/` sample is committed for no-GPU students.

To regenerate the bank: `python generate_rollout_bank.py --out ../rollout_bank --clips 4 --frames 24`.

### 5b Part 2 — Opening the Box (verified end-to-end, H200)

All four exercises were run against this exact pinned checkpoint before release:

- **VAE round-trip:** one 320×576 frame → latent shape `(1, 4, 40, 72)` → **48.0x** compression (element count) → **0.0015** mean-abs-diff reconstruction error (0-1 scale). `pipe.vae` is a standard `AutoencoderKL` (per-frame 2D, not spatiotemporal).
- **Device pitfall (found and fixed):** `pipe.vae.device` reports `cuda` even when `enable_model_cpu_offload()` has the weights sitting on CPU, because the offload hook only fires on the pipeline's `forward()`, not on a raw `.encode()`/`.decode()` call. Fix: explicitly `.to('cuda')` (or read `next(vae.parameters()).device`) rather than trusting `.device`.
- **Diffusion step-count sweep** (5/15/25 steps): all three ran clean; `temporal_consistency` 0.951 → 0.973 → 0.972 (5→15→25) — quality gain from 5→15 is visible, 15→25 is marginal on this prompt/seed.
- **Guidance-scale sweep** (1.0/9.0/15.0): all three ran clean; flicker rises sharply at high guidance (0.0015 → 0.0066 → 0.0178).
- **Latent-space interpolation:** `TextToVideoSDPipeline` is on the **older** diffusers callback API — `callback(step, timestep, latents)`, not `callback_on_step_end(pipe, step, timestep, callback_kwargs)`. Learned latent shape via a cheap 1-step probe: `(1, 4, 24, 40, 72)` (**must** pass the same `height`/`width` to the probe call as the real generations, or it silently probes the pipeline's default resolution instead). Plain linear interpolation (lerp) between two seeds' initial noise measurably shrinks the norm (526 → 526 → **372** at the midpoint) and decodes to a flat, near-featureless frame; **slerp** (spherical interpolation) holds the norm constant (≈526 throughout) and decodes to a genuinely distinct, coherent in-between scene. The notebook uses slerp for exactly this reason.
- **Status:** verified end-to-end on an H200 (~4-8s per generation there; expect the same T4 multiplier as the Part 1 note above). Total Part 2 GPU time budget: comfortably under the ~1 hour allotted.

## 5a — VLM (object hallucination)

- **Checkpoint:** `HuggingFaceTB/SmolVLM-Instruct`
- **Config (mirrors free-tier T4):** `torch_dtype=torch.float16`, on `cuda`; standard transformers `AutoModelForImageTextToText` + chat template + `generate(max_new_tokens=8)`
- **MEASURED peak VRAM:** **~4.5 GB** (weights) / **~5.0 GB** (during the probe) in fp16 → fits a free T4 (~15 GB) with large headroom
- **Verified probe results** (synthetic shape scenes, measured here):
  - easy probe (random absent): accuracy ≈ 0.86, **hallucination_rate ≈ 0.0**, recall ≈ 0.72
  - hard probe (binding traps): accuracy ≈ 0.84, **hallucination_rate ≈ 0.0**, recall ≈ 0.72
  - prompt sensitivity: flip_rate_overall ≈ 0.06 (one flip on a present object)
- **Honest finding:** on clean synthetic shapes SmolVLM is **well-grounded** — it does not hallucinate even on binding traps, and is largely phrasing-robust. The failure that *does* reliably appear is **recall** (it misses ~28% of present objects). This is deliberately used as a lesson: a failure mode you care about (hallucination) may not surface unless your **evaluation inputs are hard enough**. The lab provides three levers to push harder — binding-trap probes (`run_probe(..., hard=True)`), prompt sensitivity, and a real-image hook (`load_labeled_images` + `run_probe_on_images`) for natural photos where hallucination is more pronounced. The worksheet asks students to reason about exactly this.
- **Free-API alternative:** the notebook's `GeminiVLM` path (Gemini Flash, free tier) — PIN the exact model id and supply a key.
- The offline `MockVLM` needs no model and is used for the smoke test.

## How a pin is verified

1. Load in fp16 with CPU offload (the T4 config).
2. Run one real generation / inference.
3. Record `torch.cuda.max_memory_allocated() / 1e9` — must sit under ~14 GB.
4. Confirm the output adapter parses the real output format.
5. Note that wall-clock here is not the Colab time; one free-tier run confirms timing.
