# Week 7 Starter Package — Hands-On RL & Reward-Hacking Lab

A runnable companion to the Week 7 reward-hacking diagnostics. **CPU-only, no GPU, no API keys.** Trains in seconds on free-tier Google Colab or any local Python 3.10+.

## Layout

- `rl_lab.py` — generic tabular `q_learning` (Gymnasium-style API), the `RewardHackGrid` environment with a toggleable mis-specified reward, and evaluation helpers.
- `notebooks/week07_rl_colab.ipynb` — guided student notebook (Parts A–C + worksheet).
- `requirements.txt` — `numpy`, `gymnasium`, `matplotlib`.

## Quick start

**On Google Colab**
1. Set `COURSE_REPO_URL` in the setup cell (it `git clone`s the repo) or upload this folder.
2. A **CPU runtime** is sufficient.
3. Run all cells; fill the worksheet.

**Locally**
```bash
pip install numpy gymnasium matplotlib
python checks/smoke_test.py            # from the lab root
python starter/rl_lab.py               # quick reward-hacking demo
```

## Teaching intent

- **Part A** establishes that a tabular agent learns a policy from reward (FrozenLake, `slippery=False`, ~solves in 6 steps).
- **Part B** is the core: under the `proxy` reward, the optimal policy **farms the polish tile** (high return, zero goal-success); under the `fixed` reward it goes straight to the goal. The gap between *reward earned* and *task success* is the reward-hacking signal.
- **Part C** hands students the hacked Q-table and the reward spec so they trace *why* the policy cheats and propose a fix that they then re-measure (not just "higher reward").

The expected contrast (your run may vary by seed): proxy mode ≈ return 25, goal-success 0.0; fixed mode ≈ return 4.8, goal-success 1.0. Report your own numbers.

## Attribution

The Q-learning workflow mirrors the **HuggingFace Deep RL Course (Unit 2)** and **Farama Gymnasium** tutorials. The reward-hacking framing follows **DeepMind's specification-gaming catalogue** and **OpenAI's *Faulty Reward Functions in the Wild*** (CoastRunners). All code in this package is original to ESP3201.
