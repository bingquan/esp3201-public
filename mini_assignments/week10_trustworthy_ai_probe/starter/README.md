# Week 10 Starter Package — Hands-On Trustworthy AI

A runnable companion to the Week 10 trustworthy-AI audit. **CPU-only, no GPU, no API keys.** Runs on
free-tier Google Colab or any local Python 3.10+.

## Layout

- `trustworthy_ai_lab.py` — fairness recomputation utilities, an explainability local-vs-global
  demo, and a membership-inference privacy demo.
- `fairness_probe_data.csv` — a small illustrative per-subgroup results table (a different, smaller
  example than the graded mini-assignment's own dataset).
- `notebooks/week10_trustworthy_ai_colab.ipynb` — guided student notebook (Parts A–D + worksheet).
- `requirements.txt` — dependencies (`numpy`, `scikit-learn`; both preinstalled on Colab).

## Quick start

**On Google Colab**
1. Upload this folder, or set `COURSE_REPO_URL` in the notebook's setup cell and run it (it `git clone`s the course repo).
2. Runtime → Change runtime type → **CPU** is fine (no GPU needed).
3. Run all cells; fill the worksheet at the bottom.

**Locally**
```bash
python checks/smoke_test.py                       # from the lab root
jupyter notebook starter/notebooks/week10_trustworthy_ai_colab.ipynb
```

## Teaching intent

Three pillars, three concrete demonstrations that a report's conclusion is not evidence:

- **Fairness** — a weighted headline accuracy can look clean while one subgroup sits far below it.
- **Explainability** — a local explanation at one instance can point to a different feature than
  the model's actual global weights, because a feature can be spuriously predictive in a narrow
  region of the data without being the model's real reasoning.
- **Privacy** — a small, overfit model shows a measurable confidence gap between training and
  held-out points; that gap alone is enough to guess at training-set membership.

Report only numbers your own run produced.

## Attribution

The explainability contrast follows the local-fidelity critique behind LIME (Ribeiro et al., 2016)
and SHAP (Lundberg & Lee, 2017). The privacy probe follows the shadow-model/confidence-gap framing
of Shokri et al., *Membership Inference Attacks Against Machine Learning Models* (2017). All code in
this package is original to ESP3201.
