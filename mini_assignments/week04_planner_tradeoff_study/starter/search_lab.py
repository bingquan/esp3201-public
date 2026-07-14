"""Week 4 hands-on lab: instrumented graph-search planners on a grid.

Pure-Python, CPU-only, no third-party dependencies (matplotlib is used only in
the notebook for visualization). The point of this module is NOT to be a fast
production planner; it is to make the *cost* of search visible so students can
weigh path quality against search effort, memory, and replanning.

Every planner returns a SearchResult with the same instrumentation:
- path / cost           : solution quality
- nodes_expanded        : search effort (CPU proxy)
- max_frontier          : peak open-set size (memory proxy)
- runtime_s             : wall-clock time
- found                 : whether a path was found

Attribution: problem framing follows the classic grid-search pedagogy of
Berkeley CS188 (Pacman search projects) and Russell & Norvig, AIMA. All code
here is original to this course; no CS188 code is copied.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

Cell = Tuple[int, int]  # (row, col)


# --------------------------------------------------------------------------- #
# Grid world
# --------------------------------------------------------------------------- #
class Grid:
    """A 2D grid with blocked cells and optional per-cell movement cost.

    A map is given as a list of equal-length strings:
        '.' free cell (cost 1)
        '#' blocked cell
        digit '2'..'9' free cell whose *entry* cost is that digit (terrain)
        'S' start, 'G' goal
    """

    def __init__(self, rows: List[str], diagonal: bool = False):
        self.rows = rows
        self.height = len(rows)
        self.width = len(rows[0]) if rows else 0
        self.diagonal = diagonal
        self.start: Optional[Cell] = None
        self.goal: Optional[Cell] = None
        self.blocked = set()
        self.cost: Dict[Cell, int] = {}
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                cell = (r, c)
                if ch == "#":
                    self.blocked.add(cell)
                elif ch == "S":
                    self.start = cell
                elif ch == "G":
                    self.goal = cell
                elif ch.isdigit():
                    self.cost[cell] = int(ch)

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def passable(self, cell: Cell) -> bool:
        return cell not in self.blocked

    def entry_cost(self, cell: Cell) -> int:
        """Cost to step *into* `cell` (terrain cost, default 1)."""
        return self.cost.get(cell, 1)

    def neighbors(self, cell: Cell) -> List[Cell]:
        r, c = cell
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.diagonal:
            steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        result = []
        for dr, dc in steps:
            nxt = (r + dr, c + dc)
            if self.in_bounds(nxt) and self.passable(nxt):
                result.append(nxt)
        return result

    def block(self, cell: Cell) -> None:
        """Drop a new obstacle (used by the replanning scenario)."""
        self.blocked.add(cell)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class SearchResult:
    algorithm: str
    found: bool
    path: List[Cell] = field(default_factory=list)
    cost: float = float("inf")
    nodes_expanded: int = 0
    max_frontier: int = 0
    runtime_s: float = 0.0
    # cells in the order popped off the frontier -- lets a plot show *how* the
    # search grew (BFS's ripple vs DFS's snake vs A*'s cone), not just the count
    expanded_order: List[Cell] = field(default_factory=list)

    @property
    def path_len(self) -> int:
        return len(self.path)

    def as_row(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "found": self.found,
            "cost": round(self.cost, 2) if self.found else None,
            "path_len": self.path_len if self.found else None,
            "nodes_expanded": self.nodes_expanded,
            "max_frontier": self.max_frontier,
            "runtime_ms": round(self.runtime_s * 1000, 3),
        }


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #
def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def inadmissible(weight: float) -> Callable[[Cell, Cell], float]:
    """Manhattan distance inflated by `weight` (> 1 makes it inadmissible).

    Inflated heuristics expand fewer nodes but can lose optimality. Students
    use this to see the speed/optimality trade directly.
    """

    def h(a: Cell, b: Cell) -> float:
        return weight * manhattan(a, b)

    return h


# --------------------------------------------------------------------------- #
# Path reconstruction
# --------------------------------------------------------------------------- #
def _reconstruct(came_from: Dict[Cell, Optional[Cell]], goal: Cell) -> List[Cell]:
    path = [goal]
    node = goal
    while came_from.get(node) is not None:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def _path_cost(grid: Grid, path: List[Cell]) -> float:
    return sum(grid.entry_cost(cell) for cell in path[1:])


# --------------------------------------------------------------------------- #
# Uninformed search: BFS and DFS
# --------------------------------------------------------------------------- #
def bfs(grid: Grid) -> SearchResult:
    start, goal = grid.start, grid.goal
    t0 = time.perf_counter()
    frontier = deque([start])
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    order: List[Cell] = []
    expanded = 0
    max_frontier = 1
    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        node = frontier.popleft()
        expanded += 1
        order.append(node)
        if node == goal:
            path = _reconstruct(came_from, goal)
            return SearchResult("BFS", True, path, _path_cost(grid, path),
                                expanded, max_frontier, time.perf_counter() - t0,
                                expanded_order=order)
        for nxt in grid.neighbors(node):
            if nxt not in came_from:
                came_from[nxt] = node
                frontier.append(nxt)
    return SearchResult("BFS", False, nodes_expanded=expanded,
                        max_frontier=max_frontier, runtime_s=time.perf_counter() - t0,
                        expanded_order=order)


def dfs(grid: Grid) -> SearchResult:
    start, goal = grid.start, grid.goal
    t0 = time.perf_counter()
    frontier = [start]
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    visited = set()
    order: List[Cell] = []
    expanded = 0
    max_frontier = 1
    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        node = frontier.pop()
        if node in visited:
            continue
        visited.add(node)
        expanded += 1
        order.append(node)
        if node == goal:
            path = _reconstruct(came_from, goal)
            return SearchResult("DFS", True, path, _path_cost(grid, path),
                                expanded, max_frontier, time.perf_counter() - t0,
                                expanded_order=order)
        for nxt in grid.neighbors(node):
            if nxt not in visited:
                came_from.setdefault(nxt, node)
                frontier.append(nxt)
    return SearchResult("DFS", False, nodes_expanded=expanded, expanded_order=order,
                        max_frontier=max_frontier, runtime_s=time.perf_counter() - t0)


# --------------------------------------------------------------------------- #
# Best-first family: UCS, Greedy, A*, Weighted A*
# --------------------------------------------------------------------------- #
def _best_first(grid: Grid, label: str,
                g_weight: float, h_weight: float,
                heuristic: Callable[[Cell, Cell], float]) -> SearchResult:
    """Generic priority-queue search.

    Priority f = g_weight * g + h_weight * h. Special cases:
      UCS         : g_weight=1, h_weight=0
      Greedy      : g_weight=0, h_weight=1
      A*          : g_weight=1, h_weight=1
      Weighted A* : g_weight=1, h_weight=w (w > 1)
    """
    start, goal = grid.start, grid.goal
    t0 = time.perf_counter()
    g_score: Dict[Cell, float] = {start: 0.0}
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    counter = 0  # tie-breaker to keep heap stable and avoid comparing cells
    h0 = heuristic(start, goal)
    frontier: List[Tuple[float, int, Cell]] = [(g_weight * 0 + h_weight * h0, counter, start)]
    closed = set()
    order: List[Cell] = []
    expanded = 0
    max_frontier = 1
    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        _, _, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        order.append(node)
        if node == goal:
            path = _reconstruct(came_from, goal)
            return SearchResult(label, True, path, _path_cost(grid, path),
                                expanded, max_frontier, time.perf_counter() - t0,
                                expanded_order=order)
        for nxt in grid.neighbors(node):
            tentative = g_score[node] + grid.entry_cost(nxt)
            if tentative < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative
                came_from[nxt] = node
                counter += 1
                f = g_weight * tentative + h_weight * heuristic(nxt, goal)
                heapq.heappush(frontier, (f, counter, nxt))
    return SearchResult(label, False, nodes_expanded=expanded, expanded_order=order,
                        max_frontier=max_frontier, runtime_s=time.perf_counter() - t0)


def ucs(grid: Grid) -> SearchResult:
    return _best_first(grid, "UCS", 1.0, 0.0, manhattan)


def greedy(grid: Grid, heuristic: Callable[[Cell, Cell], float] = manhattan) -> SearchResult:
    return _best_first(grid, "Greedy", 0.0, 1.0, heuristic)


def astar(grid: Grid, heuristic: Callable[[Cell, Cell], float] = manhattan,
          label: str = "A*") -> SearchResult:
    return _best_first(grid, label, 1.0, 1.0, heuristic)


def weighted_astar(grid: Grid, weight: float = 2.0) -> SearchResult:
    return _best_first(grid, f"Weighted A* (w={weight})", 1.0, weight, manhattan)


# --------------------------------------------------------------------------- #
# Replanning scenario
# --------------------------------------------------------------------------- #
def replan_on_obstacle(grid: Grid, new_obstacle: Cell,
                       planner: Callable[[Grid], SearchResult] = astar):
    """Simulate a dynamic obstacle appearing on a planned path.

    Returns (initial, full_replan, partial_replan):
      initial        : the original plan
      full_replan    : replan from the start after the obstacle appears
      partial_replan : replan only from the cell just before the blockage

    The comparison shows that replanning from the agent's current position is
    much cheaper than replanning from scratch -- a core deployment point.
    """
    initial = planner(grid)
    if not initial.found or new_obstacle not in initial.path:
        return initial, None, None

    grid.block(new_obstacle)

    full_replan = planner(grid)

    idx = initial.path.index(new_obstacle)
    resume = initial.path[idx - 1] if idx > 0 else grid.start
    sub = Grid(grid.rows)  # rebuild to reset start, then re-apply obstacle
    sub.blocked = set(grid.blocked)
    sub.cost = dict(grid.cost)
    sub.start = resume
    sub.goal = grid.goal
    partial_replan = planner(sub)
    partial_replan.algorithm = "Partial replan"
    full_replan.algorithm = "Full replan"
    return initial, full_replan, partial_replan


# --------------------------------------------------------------------------- #
# Example maps
# --------------------------------------------------------------------------- #
MAPS: Dict[str, List[str]] = {
    # Mostly open: heuristic guidance pays off, BFS wastes effort.
    "open": [
        "S........",
        ".........",
        "....#....",
        "....#....",
        "....#....",
        ".........",
        "........G",
    ],
    # Maze: many dead ends; greedy gets misled, A* still efficient.
    "maze": [
        "S.#......",
        "..#.####.",
        "..#.#..#.",
        "....#.##.",
        "#####.#..",
        "....#.#.#",
        ".##.#.#..",
        ".#....#.G",
    ],
    # Non-uniform terrain: high-cost cells make the greedy/manhattan heuristic
    # misleading, so an inflated heuristic can produce a costlier path.
    "terrain": [
        "S99......",
        ".99......",
        ".99.####.",
        ".99......",
        "....99999",
        ".....111G",
    ],
    # Large open field: with a manhattan heuristic, plain A* expands a big
    # fraction of the grid (many cells tie on the optimal f-contour), while
    # Weighted A* drives straight at the goal and expands far fewer. The weight
    # sweep on this map shows a large, satisfying effort reduction.
    "open_large": [
        "S..............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "...............",
        "..............G",
    ],
}


def load_map(name: str, diagonal: bool = False) -> Grid:
    if name not in MAPS:
        raise KeyError(f"unknown map '{name}'; choices: {sorted(MAPS)}")
    return Grid(MAPS[name], diagonal=diagonal)


def render(grid: Grid, path: Optional[List[Cell]] = None) -> str:
    """ASCII render of a grid with an optional path overlaid as '*'."""
    path_set = set(path or [])
    lines = []
    for r in range(grid.height):
        chars = []
        for c in range(grid.width):
            cell = (r, c)
            if cell == grid.start:
                chars.append("S")
            elif cell == grid.goal:
                chars.append("G")
            elif cell in grid.blocked:
                chars.append("#")
            elif cell in path_set:
                chars.append("*")
            else:
                chars.append(grid.rows[r][c] if grid.rows[r][c].isdigit() else ".")
        lines.append("".join(chars))
    return "\n".join(lines)


def sweep_weight(grid_name: str,
                 weights=(1.0, 1.5, 2.0, 3.0, 5.0)) -> List[Tuple[float, SearchResult]]:
    """Run Weighted A* across a range of heuristic weights on one map.

    weight=1.0 is plain A* (optimal). As the weight grows the search expands
    fewer nodes but the path can get more expensive: students watch the
    speed/optimality trade emerge instead of being told about it.
    """
    return [(w, weighted_astar(load_map(grid_name), weight=w)) for w in weights]


def run_all(grid_name: str) -> List[SearchResult]:
    """Run the standard planner panel on a named map and return results."""
    results = []
    results.append(bfs(load_map(grid_name)))
    results.append(dfs(load_map(grid_name)))
    results.append(ucs(load_map(grid_name)))
    results.append(greedy(load_map(grid_name)))
    results.append(astar(load_map(grid_name)))
    results.append(astar(load_map(grid_name), inadmissible(3.0), label="A* (inflated h, w=3)"))
    results.append(weighted_astar(load_map(grid_name), weight=2.0))
    return results


if __name__ == "__main__":
    for name in MAPS:
        print(f"\n=== map: {name} ===")
        for res in run_all(name):
            print(res.as_row())
