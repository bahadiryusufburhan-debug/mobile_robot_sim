import numpy as np


# ---------------------------------------------------------------------------
class LiDAR:
    """2D LiDAR – ray-cast scanner with Gaussian range noise."""

    def __init__(self, max_range=7.0, num_beams=90, noise_std=0.05):
        self.max_range = max_range
        self.num_beams = num_beams
        self.noise_std = noise_std
        self.angles    = np.linspace(-np.pi, np.pi, num_beams, endpoint=False)

    # ------------------------------------------------------------------
    def scan(self, robot_state: np.ndarray, env) -> np.ndarray:
        """Return array of noisy range measurements."""
        x, y, theta = robot_state
        ranges = np.empty(self.num_beams)
        for i, a in enumerate(self.angles):
            r = self._ray_cast(x, y, theta + a, env)
            ranges[i] = np.clip(r + np.random.normal(0, self.noise_std),
                                0, self.max_range)
        return ranges

    def _ray_cast(self, x, y, angle, env, step=0.08):
        dx, dy = np.cos(angle) * step, np.sin(angle) * step
        cx, cy = x, y
        for _ in range(int(self.max_range / step)):
            cx += dx; cy += dy
            if env.is_occupied(cx, cy):
                return np.hypot(cx - x, cy - y)
        return self.max_range

    def scan_to_points(self, robot_state: np.ndarray,
                       ranges: np.ndarray) -> np.ndarray:
        """Convert ranges → Cartesian hit-points in world frame."""
        x, y, theta = robot_state
        pts = []
        for a, r in zip(self.angles, ranges):
            if r < self.max_range - 0.1:
                beam = theta + a
                pts.append([x + r * np.cos(beam), y + r * np.sin(beam)])
        return np.array(pts) if pts else np.zeros((0, 2))

    def noisy_position(self, true_state: np.ndarray,
                       noise_std=0.25) -> np.ndarray:
        """Simulate LiDAR-based localisation (noisy position fix)."""
        return true_state[:2] + np.random.normal(0, noise_std, 2)


# ---------------------------------------------------------------------------
class IMU:
    """Simulated IMU – measures angular velocity + linear acceleration."""

    def __init__(self, gyro_noise=0.04, accel_noise=0.08, gyro_bias=None):
        self.gyro_noise  = gyro_noise
        self.accel_noise = accel_noise
        self.gyro_bias   = gyro_bias if gyro_bias is not None \
                           else np.random.normal(0, 0.01)

    def measure(self, robot):
        omega_m = robot.omega + self.gyro_bias \
                  + np.random.normal(0, self.gyro_noise)
        ax = robot.v * np.cos(robot.theta) + np.random.normal(0, self.accel_noise)
        ay = robot.v * np.sin(robot.theta) + np.random.normal(0, self.accel_noise)
        return omega_m, np.array([ax, ay])


# ---------------------------------------------------------------------------
class WheelEncoder:
    """Simulated wheel encoder – measures v and omega with noise."""

    def __init__(self, v_noise=0.03, omega_noise=0.02):
        self.v_noise     = v_noise
        self.omega_noise = omega_noise

    def measure(self, robot):
        v_m     = robot.v     + np.random.normal(0, self.v_noise)
        omega_m = robot.omega + np.random.normal(0, self.omega_noise)
        return v_m, omega_m
