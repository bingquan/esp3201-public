# ESP3201 — Public Colab Notebooks

Public mirror of the student-facing Colab notebooks for ESP3201 (Machine
Learning in Robotics & Engineering). Kept separate from the main course repo
so notebooks can be opened in Colab without authentication.

This repo intentionally contains **only the runnable notebooks** — no
support scripts, smoke tests, data files, or briefs. Every notebook here is
self-contained (support code and any needed data are embedded directly in
the notebook's own setup cells), so it runs standalone from a fresh Colab
session with no other files from this repo required. The full course
materials — labs, rubrics, evaluation harness, briefs, answer keys — live in
the private course repo.

Currently mirrored:

- `mini_assignments/week04_planner_tradeoff_study/starter/notebooks/week04_planning_colab.ipynb` — Week 4, search under real constraints (CPU-only)
- `mini_assignments/week05_embodied_system_critique/starter/notebooks/week05_vlm_grounding_colab.ipynb` — Week 5, VLM hallucination probe (v1): SmolVLM (local GPU) vs a free OpenRouter model, compared. Needs a GPU runtime and/or an `OPENROUTER_API_KEY` Colab secret.
- `mini_assignments/week05_embodied_system_critique/starter/notebooks/week05_open_vocab_grounding_colab.ipynb` — Week 5, open-vocabulary grounding across OWL-ViT / SAM 3 / SAM / SmolVLM (v2; independent of v1, not a replacement). Six CC0 starter photos are embedded in the notebook; provenance is printed by the image-bank cell.
- `mini_assignments/week06_video_world_model_study/starter/notebooks/week06_video_world_model_colab.ipynb` — Week 6, video world-model probe
- `mini_assignments/week07_reward_hacking_diagnostics/starter/notebooks/week07_rl_colab.ipynb` — Week 7, Q-learning and reward hacking (CPU-only)
- `mini_assignments/week09_agent_safety_case_colab/notebooks/week09_agent_colab-v3.ipynb` — Week 9, probing an agent (current version; earlier `week09_agent_colab-v2.ipynb` and `week09_agent_safety_case_colab.ipynb` kept for reference)
- `mini_assignments/week10_trustworthy_ai_probe/starter/notebooks/week10_trustworthy_ai_colab.ipynb` — Week 10, fairness/explainability/privacy probes (CPU-only)
- `mini_assignments/week10_trustworthy_ai_probe/starter/notebooks/week10b_agentic_security_colab_v4.ipynb` — Week 10b, current version: indirect prompt injection, action permission gates, and keyword filtering. Includes runnable starter examples and recorded live attack/defence traces. Uses the Gemini API with a `GEMINI_API_KEY`; CPU runtime, no GPU required.
- `mini_assignments/week10_trustworthy_ai_probe/starter/notebooks/week10b_agentic_security_colab_v3.ipynb` — Week 10b v3, kept for reference.
- `mini_assignments/week10_trustworthy_ai_probe/starter/notebooks/week10b_agentic_security_colab_v2.ipynb` — Week 10b v2, kept for reference; uses a GPU runtime and/or an `OPENROUTER_API_KEY` Colab secret.
- `mini_assignments/week10_trustworthy_ai_probe/starter/notebooks/week10b_agentic_security_colab.ipynb` — Week 10b v1, kept for reference. Same experiments and same code cells as v2; v2 only reorders the guardrail sections into one escalating story and adds an anchor table.

Open any notebook directly in Colab from its GitHub URL.
