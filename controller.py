import numpy as np


class PurePursuitController:
    """
    Pure-Pursuit lateral controller for waypoint following.

    Parameters
    ----------
    lookahead : look-ahead distance [m]
    max_v     : maximum linear speed [m/s]
    max_omega : maximum angular speed [rad/s]
    goal_tol  : distance threshold to declare goal reached [m]
    """

    def __init__(self, lookahead=1.2, max_v=1.2, max_omega=1.8, goal_tol=0.4):
        self.lookahead = lookahead
        self.max_v     = max_v
        self.max_omega = max_omega
        self.goal_tol  = goal_tol

    # ------------------------------------------------------------------
    def compute(self, state: np.ndarray, path: np.ndarray):
        """Return (v, omega) command to follow *path*."""
        x, y, th = state
        target = self._lookahead_point(x, y, path)
        if target is None:
            return 0.0, 0.0

        dx, dy  = target[0] - x, target[1] - y
        alpha   = self._norm(np.arctan2(dy, dx) - th)

        v     = self.max_v * max(0.0, np.cos(alpha))
        omega = self.max_omega * (2 * np.sin(alpha) / self.lookahead)
        omega = float(np.clip(omega, -self.max_omega, self.max_omega))
        return v, omega

    def reached_goal(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return np.hypot(state[0] - goal[0], state[1] - goal[1]) < self.goal_tol

    # ------------------------------------------------------------------
    def _lookahead_point(self, x, y, path):
        closest_idx, min_d = 0, float('inf')
        for i, wp in enumerate(path):
            d = np.hypot(wp[0] - x, wp[1] - y)
            if d < min_d:
                min_d, closest_idx = d, i
        for wp in path[closest_idx:]:
            if np.hypot(wp[0] - x, wp[1] - y) >= self.lookahead:
                return wp
        return path[-1]

    @staticmethod
    def _norm(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
