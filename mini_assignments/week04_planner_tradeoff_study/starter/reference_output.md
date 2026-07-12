# Week 4 Search Lab Reference Output

This file gives a fallback evidence source for students who cannot run the companion lab. Running `python starter/search_lab.py` or the Colab notebook is still preferred.

The values below omit wall-clock runtime because it depends on the machine. Use cost, path length, nodes expanded, and max frontier for stable comparisons.

## Static Planner Panel

### `open`

| Planner | Cost | Path length | Nodes expanded | Max frontier |
| --- | ---: | ---: | ---: | ---: |
| BFS | 14 | 15 | 60 | 7 |
| UCS | 14 | 15 | 60 | 7 |
| A* | 14 | 15 | 60 | 7 |
| Weighted A* (`w=2.0`) | 14 | 15 | 15 | 13 |

Interpretation: on this open map, many cells tie on the same optimal `f` contour. Plain A* keeps the optimal cost but does not reduce expansion versus UCS; weighted A* follows the heuristic more aggressively.

### `terrain`

| Planner | Cost | Path length | Nodes expanded | Max frontier |
| --- | ---: | ---: | ---: | ---: |
| BFS | 13 | 14 | 50 | 6 |
| DFS | 37 | 14 | 46 | 34 |
| UCS | 13 | 14 | 32 | 12 |
| A* | 13 | 14 | 17 | 11 |
| Weighted A* (`w=2.0`) | 13 | 14 | 14 | 12 |

Interpretation: path length alone is misleading because high-cost terrain changes the real objective. DFS returns a valid but expensive path; A* preserves the low cost while expanding fewer nodes than UCS.

## Dynamic Obstacle Example

On the `open` map, the initial A* path is:

```text
(0,0) -> (1,0) -> (2,0) -> (3,0) -> (4,0) -> (5,0) -> (6,0) -> (6,1) -> (6,2) -> (6,3) -> (6,4) -> (6,5) -> (6,6) -> (6,7) -> (6,8)
```

If a new obstacle blocks `(6,4)` after the robot has already moved near the lower corridor:

| Replanning mode | Cost from its start state | Path length | Nodes expanded | Max frontier |
| --- | ---: | ---: | ---: | ---: |
| Full replan from original start | 14 | 15 | 59 | 7 |
| Partial replan from just before blockage | 7 | 8 | 12 | 8 |

Interpretation: this is not proof that all D* Lite-style methods are always better. It is evidence that repeated local updates can make reuse or local replanning attractive when the robot has already progressed along the path.
