import math
from queue import PriorityQueue

coords = {}  # node id is the key
adjlist = {}  # node id is the key
with open("input.txt", "r") as f:
    V = int(f.readline())
    for _ in range(V):
        strs = f.readline().split()
        nid, x, y = strs[0], int(strs[1]), int(strs[2])
        coords[nid] = (x, y)  # x, y kept as a tuple
        adjlist[nid] = []  # create empty list for each node's adjnodes

    E = int(f.readline())
    for _ in range(E):
        strs = f.readline().split()
        n1, n2, c = strs[0], strs[1], int(strs[2])
        adjlist[n1].append((n2, c))  # (n2, c) tuple
    startnid = f.readline().rstrip()
    goalnid = f.readline().rstrip()

for nid in adjlist:
    print(nid, coords[nid], "--->", adjlist[nid])
    for tup in adjlist[nid]:
        print("\t", tup[0], tup[1])
print("start", startnid, "goal", goalnid)


class State:
    def __init__(self, nid, g, h, parent=None):
        self.nid = nid
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other: "State"):
        # PriorityQueue uses < to break ties; prefer smaller f, then larger g
        if self.f == other.f:
            return self.g > other.g
        return self.f < other.f

    def __repr__(self):
        return f"State({self.nid}, g={self.g}, h={self.h:.2f}, f={self.f:.2f})"

    def path(self):
        out = []
        cur = self
        while cur:
            out.append(cur.nid)
            cur = cur.parent
        return list(reversed(out))


def heuristic(nid, goal, coords):
    (x1, y1) = coords[nid]
    (xg, yg) = coords[goal]
    return math.sqrt((x1 - xg) ** 2 + (y1 - yg) ** 2)


def astar(start_id: str, goal_id: str, adjlist: dict, coords: dict):
    minQ = PriorityQueue()
    h0 = heuristic(start_id, goal_id, coords)
    start_state = State(start_id, g=0, h=h0, parent=None)
    minQ.put(start_state)

    # Track the best known g for each node
    best_g = {start_id: 0}
    visited = set()

    while not minQ.empty():
        current = minQ.get()

        if current.nid in visited:
            continue

        if current.nid == goal_id:
            print("Path:", " -> ".join(current.path()))
            print("Total cost:", current.g)
            return

        visited.add(current.nid)

        for neighbor_id, edge_cost in adjlist[current.nid]:
            if neighbor_id in visited:
                continue
            g = current.g + edge_cost
            if g < best_g.get(neighbor_id, math.inf):
                h = heuristic(neighbor_id, goal_id, coords)
                neighbor_state = State(neighbor_id, g, h, parent=current)
                best_g[neighbor_id] = g
                minQ.put(neighbor_state)

    print("Path:")
    print("Total cost: N/A (goal not reachable)")


astar(startnid, goalnid, adjlist, coords)
