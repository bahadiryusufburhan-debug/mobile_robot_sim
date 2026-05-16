import numpy as np


class ExtendedKalmanFilter:
    """
    EKF for 2D robot localisation.
    State  : [x, y, theta]
    Inputs : encoder (v, omega)
    Updates: (1) IMU heading, (2) LiDAR position fix
    """

    def __init__(self, init_state: np.ndarray):
        self.state = init_state.copy().astype(float)
        self.P = np.diag([0.1, 0.1, 0.05])

        # Process noise (encoder odometry uncertainty)
        self.Q = np.diag([0.04**2, 0.04**2, 0.02**2])

        # Measurement noise
        self.R_imu   = np.array([[0.04**2]])        # heading from IMU
        self.R_lidar = np.diag([0.25**2, 0.25**2])  # position fix from LiDAR

    # ------------------------------------------------------------------
    def predict(self, v: float, omega: float, dt: float) -> np.ndarray:
        """Motion-model prediction step."""
        x, y, th = self.state
        # State transition
        self.state[0] += v * np.cos(th) * dt
        self.state[1] += v * np.sin(th) * dt
        self.state[2]  = self._norm(th + omega * dt)

        # Jacobian F
        F = np.array([
            [1, 0, -v * np.sin(th) * dt],
            [0, 1,  v * np.cos(th) * dt],
            [0, 0,  1]
        ])
        self.P = F @ self.P @ F.T + self.Q
        return self.state.copy()

    # ------------------------------------------------------------------
    def update_imu(self, omega_meas: float, dt: float) -> np.ndarray:
        """Correct heading using IMU angular-velocity measurement."""
        # predicted heading after dt
        th_pred = self._norm(self.state[2] + omega_meas * dt)
        H = np.array([[0.0, 0.0, 1.0]])
        innov = self._norm(th_pred - self.state[2])
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T / S[0, 0]
        self.state    += K.flatten() * innov
        self.state[2]  = self._norm(self.state[2])
        self.P         = (np.eye(3) - np.outer(K, H)) @ self.P
        return self.state.copy()

    # ------------------------------------------------------------------
    def update_lidar(self, pos_meas: np.ndarray) -> np.ndarray:
        """Correct x,y using LiDAR position fix."""
        H = np.array([[1, 0, 0],
                      [0, 1, 0]], dtype=float)
        innov = pos_meas - H @ self.state
        S = H @ self.P @ H.T + self.R_lidar
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state   += K @ innov
        self.state[2] = self._norm(self.state[2])
        self.P        = (np.eye(3) - K @ H) @ self.P
        return self.state.copy()

    @staticmethod
    def _norm(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi
