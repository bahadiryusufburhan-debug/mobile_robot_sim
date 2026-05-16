import numpy as np


class DeadReckoning:
    """Encoder-only dead-reckoning localisation."""

    def __init__(self, init_state: np.ndarray):
        self.state = init_state.copy().astype(float)

    def update(self, v_meas: float, omega_meas: float, dt: float) -> np.ndarray:
        x, y, th = self.state
        self.state[0] += v_meas * np.cos(th) * dt
        self.state[1] += v_meas * np.sin(th) * dt
        self.state[2]  = self._norm(th + omega_meas * dt)
        return self.state.copy()

    @staticmethod
    def _norm(a):
        return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
def compute_errors(true_xy: np.ndarray,
                   est_xy:  np.ndarray) -> tuple:
    """
    Compute per-step Euclidean position error, RMSE, and MAE.

    Parameters
    ----------
    true_xy : (N, 2)
    est_xy  : (N, 2)

    Returns
    -------
    errors : (N,)   per-step distance error
    rmse   : float
    mae    : float
    """
    n   = min(len(true_xy), len(est_xy))
    err = np.hypot(true_xy[:n, 0] - est_xy[:n, 0],
                   true_xy[:n, 1] - est_xy[:n, 1])
    return err, float(np.sqrt(np.mean(err**2))), float(np.mean(err))
