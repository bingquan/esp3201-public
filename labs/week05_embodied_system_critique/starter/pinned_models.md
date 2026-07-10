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
