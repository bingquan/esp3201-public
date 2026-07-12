# Week 6 Starter Package — Hands-On Video World Models

A runnable companion for the formative lab track. Best on free-tier Colab; has a path that runs without a GPU.

## Layout

- `video_eval.py` + `notebooks/week06_video_world_model_colab.ipynb` — **video world-model lab**. Generate clips with a small text-to-video model, vary prompting/controls, and score temporal consistency, motion, and flicker alongside human-judged criteria.
- `generate_rollout_bank.py` — regenerates `../rollout_bank/` on any GPU (or a Colab T4).
- `requirements.txt` — dependencies (model extras are installed inside the notebook).

The companion VLM hallucination-probe lab (`vlm_lab.py`, `notebooks/week05_vlm_grounding_colab.ipynb`) lives in `labs/week05_embodied_system_critique/starter/`.

## Quick start

**Offline checks (no GPU, no key)**
```bash
python checks/smoke_test_video.py      # video metrics on synthetic clips
python starter/video_eval.py           # metrics on static/moving/noise clips
```

**On Colab**
- GPU runtime required to generate; **or** use the precomputed `rollout_bank/` fallback. **PIN THIS**: set a small text-to-video checkpoint that fits the T4.

## Teaching intent

Treats a video generator as a world model: students learn which prompting strategies and input controls drive quality/consistency, where coherence breaks with horizon, and why no single metric captures usefulness — connecting directly to the Week 6 mini-assignment's trust-horizon judgment.

Report only what your own run (or the provided rollout bank) produced.

## Attribution

- Video lab: **VBench** (Huang et al., 2023); diffusers AnimateDiff / CogVideoX tutorials.

All code in this package is original to ESP3201.
