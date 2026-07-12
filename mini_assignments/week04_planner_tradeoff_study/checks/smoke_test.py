"""Week 4 lab smoke test. CPU-only, no third-party deps.

Run from the lab root:  python checks/smoke_test.py
Asserts the search planners are wired correctly and the instrumentation is sane.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from search_lab import (  # noqa: E402
    astar, bfs, dfs, greedy, ucs, weighted_astar, inadmissible,
    load_map, replan_on_obstacle, run_all, sweep_weight,
)


def test_all_planners_find_a_path():
    for name in ("open", "maze", "terrain"):
        grid = load_map(name)
        for planner in (bfs, dfs, ucs, greedy, astar):
            res = planner(grid if planner in (greedy, astar, ucs) else load_map(name))
            # reload grid for stateful-free planners; greedy/astar/ucs don't mutate
            res = planner(load_map(name))
            assert res.found, f"{res.algorithm} failed to find a path on {name}"
            assert res.path[0] == grid.start and res.path[-1] == grid.goal
            assert res.nodes_expanded > 0
            assert res.max_frontier >= 1
            assert res.runtime_s >= 0.0


def test_ucs_and_bfs_optimal_on_unit_cost():
    # On the unit-cost 'maze', BFS, UCS and admissible A* must agree on cost.
    grid_name = "maze"
    opt = ucs(load_map(grid_name)).cost
    assert bfs(load_map(grid_name)).cost == opt
    assert astar(load_map(grid_name)).cost == opt


def test_admissible_astar_expands_no_more_than_ucs():
    # A* with an admissible heuristic should never expand more nodes than UCS.
    for name in ("open", "maze"):
        ucs_n = ucs(load_map(name)).nodes_expanded
        astar_n = astar(load_map(name)).nodes_expanded
        assert astar_n <= ucs_n, f"A* expanded more than UCS on {name}"


def test_inflated_heuristic_is_faster_but_can_cost_more():
    # On terrain, the inflated (inadmissible) heuristic should expand fewer
    # nodes than admissible A*, illustrating the speed/optimality trade.
    name = "terrain"
    admissible = astar(load_map(name))
    inflated = astar(load_map(name), inadmissible(3.0), label="A* inflated")
    assert inflated.nodes_expanded <= admissible.nodes_expanded
    assert inflated.cost >= admissible.cost  # never cheaper than optimal


def test_partial_replan_cheaper_than_full():
    grid = load_map("open")
    initial = astar(load_map("open"))
    # Pick an obstacle on the path (not start/goal).
    obstacle = initial.path[len(initial.path) // 2]
    grid = load_map("open")
    init, full, partial = replan_on_obstacle(grid, obstacle, planner=astar)
    assert init.found and full is not None and partial is not None
    assert full.found and partial.found
    assert partial.nodes_expanded <= full.nodes_expanded


def test_run_all_returns_panel():
    rows = run_all("maze")
    assert len(rows) == 7
    assert all(r.runtime_s >= 0 for r in rows)


def test_weight_sweep_reduces_effort():
    # On a large open field, inflating the heuristic weight should expand far
    # fewer nodes than plain A* (weight 1.0), with no worse cost here.
    rows = sweep_weight("open_large", weights=(1.0, 2.0))
    (w1, r1), (w2, r2) = rows
    assert r2.nodes_expanded < r1.nodes_expanded
    assert r2.cost == r1.cost


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} smoke tests passed.")
