# Week 5 Starter Package — Hands-On Embodied AI

A runnable companion for the formative lab track. Best on free-tier Colab; has a path that runs without a GPU.

## Layout

- `vlm_lab.py` + `notebooks/week05_vlm_grounding_colab.ipynb` — **VLM hallucination probe** (POPE-style). Synthesize scenes with exact ground truth, ask a vision-language model about present/absent objects, measure the hallucination rate.
- `requirements.txt` — dependencies (model/API extras are installed inside the notebook).

The companion video world-model lab (`video_eval.py`, `notebooks/week06_video_world_model_colab.ipynb`) now lives in `labs/week06_video_world_model_study/starter/`.

## Quick start

**Offline checks (no GPU, no key)**
```bash
python checks/smoke_test_vlm.py        # VLM scoring pipeline (mock backend)
python starter/vlm_lab.py              # mock hallucination demo
```

**On Colab**
- GPU runtime for a small local VLM, **or** the free Gemini API path, **or** the offline mock for a dry run. **PIN THIS**: set the model id you verified.

## Teaching intent

Teaches the difference between a *grounded* answer and a *fluent* one: a yes-biased VLM hallucinates objects that are not present, and the hallucination rate makes that measurable.

Report only what your own run produced.

## Attribution

- VLM probe: **POPE** (Li et al., 2023, *Evaluating Object Hallucination in LVLMs*); HuggingFace VLM notebooks.

All code in this package is original to ESP3201.
