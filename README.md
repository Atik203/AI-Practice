# A\* Search - Student-Friendly Guide

This README explains the A\* implementation step by step, maps the pseudocode to your notebook, and answers common “what-if” questions so you’re ready for viva-style questions.

Files:

- Astar_search.ipynb — main notebook
- input.txt — graph definition (nodes, edges, start, goal)

## Input format (input.txt)

- First line: number of nodes V
- Next V lines: nodeId x y
- Next line: number of edges E
- Next E lines: node1 node2 cost (directed edge node1 -> node2)
- Next line: start node id
- Next line: goal node id

Example used here:

```text
3
S 1 2
A 2 2
G 4 5
3
S A 1
A G 1
S G 10
S
G
```

## Concise A\* pseudocode

- Initialize a min priority queue `minQ` ordered by `f = g + h`.
- `g ← 0`
- `h ← euclidean_distance(startnode, goalnode)`
- `f ← g + h`
- Create `startstate(startnode, g, f, parent=None)` and insert into `minQ`.
- While `minQ` is not empty:
  - `currstate ← extract_min(minQ)` (smallest f)
  - If `currstate.node` is the goal: print solution path and cost; return.
  - For each neighbor `M` of `currstate.node`:
    - `g ← currstate.g + edge_cost(currstate.node, M)`
    - `h ← euclidean_distance(M, goalnode)`
    - `f ← g + h`
    - Create `newstate(M, g, f, parent=currstate)` and insert into `minQ`.

## Detailed flow with two practical helpers

- `best_g`: dict node -> lowest g found so far; skip pushing worse paths.
- `closed`: set of nodes already expanded with finalized best g (safe if heuristic is consistent).

Loop steps:

1. Pop the state with the smallest `f` from the priority queue.
2. If it’s the goal, reconstruct the path via `parent` links and return.
3. Add this node to `closed`.
4. For each neighbor, compute `tentative_g = current.g + cost`.
   - If `tentative_g` improves on `best_g[neighbor]`, update `best_g`, compute `h`, create a new State, push it.

## Mapping to notebook cells

- Cell 1: Reads `input.txt` into `coords` (node -> (x,y)) and `adjlist` (node -> [(neighbor, cost), ...]).
- Cell 2: `heuristic(nid, goal, coords)` returns Euclidean distance from `nid` to `goal`.
- Cell 3: `class State` has `nid, g, h, f=g+h, parent`, plus:
  - `__lt__` so the priority queue can order states.
  - `path()` to rebuild the route from start to this state.
- Cell 4: `astar(...)` uses `PriorityQueue`, `best_g`, `closed`. Prints:
  - `Path: S A G`
  - `Total cost: 2`

## What the special pieces do

### 1) `__lt__` (less-than) in `State`

Why needed: PriorityQueue must compare two states to know which comes out first. We define: smaller `f` first; if `f` ties, prefer larger `g`.

In words:

- Primary: smaller `f = g + h` is better.
- Tie-break: if `f` is equal, pick the one with larger `g` (therefore smaller `h`).

In code:

```python
if self.f == other.f:
    return self.g > other.g   # prefer larger g (smaller h)
return self.f < other.f       # otherwise smaller f first
```

Why prefer larger g on ties (`self.g > other.g`)?

- With `f` fixed, larger `g` means smaller `h` (closer to goal by heuristic). This tends to drive the search forward and often reduces expansions with consistent heuristics.

What if we use `self.g < other.g` instead?

- Still correct. It favors shallower nodes (larger `h`) on ties, which can fan out near the start and usually expands more nodes. Difference is performance/order, not correctness.

What if we remove `__lt__` entirely?

- `PriorityQueue` will raise a `TypeError` when it needs to compare two `State` objects. You must either define `__lt__` or push comparable tuples like `(f, -g, counter, state)`.

Alternative without `__lt__`:

- Push tuples: `(f, -g, counter, state)` where `counter` is from `itertools.count()` to guarantee unique ordering. Then the queue compares tuples, not `State`.

### 2) `path()` method

- Starts at the current state (usually the goal), follows `parent` pointers back to the start, collects node IDs, reverses them, and returns e.g. `["S", "A", "G"]`.
- Complexity: O(length of path).

## Heuristic notes (Euclidean distance)

- Admissible: `h(n) ≤ true_cost_to_goal(n)` for all n. Guarantees optimality with standard A\*.
- Consistent: `h(n) ≤ cost(n,m) + h(m)` for each edge (n→m). With consistency, once a node is popped, its best `g` is final (safe to keep `closed` without reopening).
- In the sample graph, edges are arbitrary (e.g., A→G cost 1) while Euclidean(A,G) ≈ 3.61, so Euclidean is not admissible there. A\* still finds the optimal path in this tiny case, but non-admissible heuristics can break optimality in general.
- If your assignment expects Euclidean to be admissible, edge costs typically reflect geometric distances (or are ≥ straight-line distances).

What if `h = 0` for all nodes?

- A\* becomes Dijkstra’s algorithm: always optimal, but explores more nodes (slower).

What if `h` overestimates (not admissible)?

- A\* may return a suboptimal path. With the current no-reopen policy, you can finalize a node too early.

## Why `best_g` and `closed`?

- `best_g`: prevents pushing worse duplicate paths—saves time/memory.
- `closed`: avoids re-expanding nodes whose best `g` is final (assuming consistency)—saves time.

What if we omit them?

- No `best_g`: many duplicates in the queue; still correct with admissible `h`, but slower.
- No `closed`: nodes can be expanded multiple times; slower. With inconsistent `h`, you might need to “reopen” nodes when a better `g` is found later.

## Directed vs. undirected edges

- Current code treats edges as directed (only `n1 -> n2`). If your graph is undirected, also add the reverse edge (`n2 -> n1`) with the same cost when reading edges.

## Output format

On success:

```text
Path: S A G
Total cost: 2
```

On failure (goal unreachable):

```text
Path:
Total cost: N/A (goal not reachable)
```

## How to run

- Open `Astar_search.ipynb` and run all cells from top to bottom. It reads `input.txt`, runs A\*, and prints the path and total cost.

## Complexity

- With a heap-based priority queue, time ≈ `O((V + E) log V)`; memory ≈ `O(V)`.

## Quick glossary

- `g(n)`: cost from start to node n.
- `h(n)`: heuristic estimate from n to goal.
- `f(n) = g(n) + h(n)`: estimated total cost via n.
- `open` (priority queue): frontier of discovered states.
- `closed`: set of nodes already expanded.
- `best_g`: map of best-known g per node.

This guide matches the notebook behavior and covers the “why” behind each choice, plus what changes would do if you implemented them differently.
