import heapq
import numpy as np


class AStarPlanner:
    """
    Grid-based A* path planner for a 2D continuous environment.

    Parameters
    ----------
    env        : Environment object
    resolution : grid cell size in metres
    robot_rad  : safety inflation radius
    """

    def __init__(self, env, resolution=0.25, robot_rad=0.40):
        self.env        = env
        self.res        = resolution
        self.robot_rad  = robot_rad
        self.nx         = int(env.width  / resolution) + 1
        self.ny         = int(env.height / resolution) + 1

        # Precompute 8-connected move costs
        self._moves = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                self._moves.append((dx, dy, np.hypot(dx, dy)))

    # ------------------------------------------------------------------
    def plan(self, start: np.ndarray, goal: np.ndarray):
        """Return smoothed path as (N,2) array, or None if not found."""
        sx, sy = self._w2g(start[0], start[1])
        gx, gy = self._w2g(goal[0],  goal[1])

        open_heap = []
        heapq.heappush(open_heap, (0.0, sx, sy))
        came_from = {}
        g = {(sx, sy): 0.0}
        in_open = {(sx, sy)}

        while open_heap:
            _, cx, cy = heapq.heappop(open_heap)
            in_open.discard((cx, cy))

            if cx == gx and cy == gy:
                return self._smooth(self._reconstruct(came_from, cx, cy, sx, sy))

            for dx, dy, cost in self._moves:
                nx_, ny_ = cx + dx, cy + dy
                if not (0 <= nx_ < self.nx and 0 <= ny_ < self.ny):
                    continue
                wx, wy = self._g2w(nx_, ny_)
                if self.env.is_occupied(wx, wy, margin=self.robot_rad):
                    continue
                ng = g[(cx, cy)] + cost
                if ng < g.get((nx_, ny_), float('inf')):
                    came_from[(nx_, ny_)] = (cx, cy)
                    g[(nx_, ny_)] = ng
                    f = ng + self._heuristic(nx_, ny_, gx, gy)
                    if (nx_, ny_) not in in_open:
                        heapq.heappush(open_heap, (f, nx_, ny_))
                        in_open.add((nx_, ny_))
        return None

    # ------------------------------------------------------------------
    def _reconstruct(self, came_from, cx, cy, sx, sy):
        path = []
        cur = (cx, cy)
        while cur in came_from:
            path.append(list(self._g2w(*cur)))
            cur = came_from[cur]
        path.append(list(self._g2w(sx, sy)))
        path.reverse()
        return np.array(path)

    def _smooth(self, path, iters=80):
        if path is None or len(path) < 3:
            return path
        s = path.tolist()
        for _ in range(iters):
            for i in range(1, len(s) - 1):
                px = (s[i-1][0] + s[i+1][0]) / 2
                py = (s[i-1][1] + s[i+1][1]) / 2
                if not self.env.is_occupied(px, py, margin=self.robot_rad * 0.8):
                    s[i] = [px, py]
        return np.array(s)

    def _heuristic(self, ax, ay, bx, by):
        return np.hypot(ax - bx, ay - by)

    def _w2g(self, x, y):
        return int(x / self.res), int(y / self.res)

    def _g2w(self, gx, gy):
        return gx * self.res, gy * self.res
