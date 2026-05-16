import numpy as np


class DifferentialRobot:
    """Non-holonomic unicycle (differential-drive) robot model."""

    def __init__(self, x=1.0, y=1.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta      # heading [rad]
        self.v     = 0.0        # linear velocity [m/s]
        self.omega = 0.0        # angular velocity [rad/s]

        self.max_v     = 1.5    # m/s
        self.max_omega = np.pi  # rad/s

    # ------------------------------------------------------------------
    def update(self, v: float, omega: float, dt: float) -> np.ndarray:
        """Unicycle kinematics integration (Euler)."""
        self.v     = float(np.clip(v,     -self.max_v,     self.max_v))
        self.omega = float(np.clip(omega, -self.max_omega, self.max_omega))

        self.x     += self.v * np.cos(self.theta) * dt
        self.y     += self.v * np.sin(self.theta) * dt
        self.theta  = self._norm(self.theta + self.omega * dt)

        return self.get_state()

    def get_state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    @staticmethod
    def _norm(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi
