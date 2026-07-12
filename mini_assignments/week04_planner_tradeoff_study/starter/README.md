# Week 4 Starter Package — Hands-On Search Lab

A runnable companion to the Week 4 planner-tradeoff study. **CPU-only, no GPU, no API keys.** Runs on free-tier Google Colab or any local Python 3.10+.

## Layout

- `search_lab.py` — grid world + instrumented planners (BFS, DFS, UCS, Greedy, A*, inflated/inadmissible A*, Weighted A*) and a dynamic-obstacle replanning scenario.
- `notebooks/week04_planning_colab.ipynb` — guided student notebook (Parts A–D + worksheet).
- `requirements.txt` — dependencies (none beyond the standard library; matplotlib optional for your own plots).

## Quick start

**On Google Colab**
1. Upload this folder, or set `COURSE_REPO_URL` in the notebook's setup cell and run it (it `git clone`s the course repo).
2. Runtime → Change runtime type → **CPU** is fine (no GPU needed).
3. Run all cells; fill the worksheet at the bottom.

**Locally**
```bash
python checks/../checks/smoke_test.py      # from the lab root: python checks/smoke_test.py
jupyter notebook starter/notebooks/week04_planning_colab.ipynb
```
Run directly without Jupyter to see the planner panel:
```bash
python starter/search_lab.py
```

## Teaching intent

Students should **not** implement A* from scratch. The point is to read the instrumentation — `nodes_expanded` (CPU proxy), `max_frontier` (memory proxy), `runtime_ms`, and `cost` — and reason about which planner is defensible under a stated runtime/memory/replanning constraint. The maps are chosen so that:

- `terrain` makes DFS return a valid-but-expensive path and shows an admissible A* expanding far fewer nodes than UCS;
- `open` shows an admissible heuristic failing to reduce effort because of f-contour ties;
- the replanning scenario shows partial replanning (from the agent's cell) is much cheaper than replanning from scratch.

Report only numbers your own run produced.

## Attribution

Problem framing follows the grid-search pedagogy of **Berkeley CS188** (Pacman search projects) and **Russell & Norvig, *Artificial Intelligence: A Modern Approach*** (and the `aima-python` reference implementations). All code in this package is original to ESP3201; no CS188 code is copied.
