# Week 5 Starter Package — Hands-On Embodied AI & World Models

Two runnable companions for the formative lab track. Both are best on free-tier Colab; both have a path that runs without a GPU.

## Layout

- `vlm_lab.py` + `notebooks/week05a_vlm_grounding_colab.ipynb` — **VLM hallucination probe** (POPE-style). Synthesize scenes with exact ground truth, ask a vision-language model about present/absent objects, measure the hallucination rate.
- `video_eval.py` + `notebooks/week05b_video_world_model_colab.ipynb` — **video world-model lab**. Generate clips with a small text-to-video model, vary prompting/controls, and score temporal consistency, motion, and flicker alongside human-judged criteria.
- `requirements.txt` — dependencies for both (model/API extras are installed inside the notebooks).

## Quick start

**Offline checks (no GPU, no key)**
```bash
python checks/smoke_test_vlm.py        # VLM scoring pipeline (mock backend)
python checks/smoke_test_video.py      # video metrics on synthetic clips
python starter/vlm_lab.py              # mock hallucination demo
python starter/video_eval.py           # metrics on static/moving/noise clips
```

**On Colab**
- 5a (VLM): GPU runtime for a small local VLM, **or** the free Gemini API path, **or** the offline mock for a dry run. **PIN THIS**: set the model id you verified.
- 5b (video): GPU runtime required to generate; **or** use the precomputed `rollout_bank/` fallback. **PIN THIS**: set a small text-to-video checkpoint that fits the T4.

## Teaching intent

- **5a** teaches the difference between a *grounded* answer and a *fluent* one: a yes-biased VLM hallucinates objects that are not present, and the hallucination rate makes that measurable.
- **5b** treats a video generator as a world model: students learn which prompting strategies and input controls drive quality/consistency, where coherence breaks with horizon, and why no single metric captures usefulness — connecting directly to the Week 5 mini-assignment's trust-horizon judgment.

Report only what your own run (or the provided rollout bank) produced.

## Attribution

- VLM probe: **POPE** (Li et al., 2023, *Evaluating Object Hallucination in LVLMs*); HuggingFace VLM notebooks.
- Video lab: **VBench** (Huang et al., 2023); diffusers AnimateDiff / CogVideoX tutorials.

All code in this package is original to ESP3201.
