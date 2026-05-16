import numpy as np


class Environment:
    """2D robot environment with rectangular obstacles."""

    def __init__(self):
        self.width = 20.0   # metres
        self.height = 20.0  # metres
        self.start = np.array([1.0, 1.0])
        self.goal  = np.array([18.5, 18.5])

        # (cx, cy, w, h)  – 14 obstacles (>= 10 required)
        self.obstacles = [
            (3.0,  3.0,  2.0, 1.5),
            (7.0,  2.0,  1.5, 2.0),
            (5.0,  6.5,  2.0, 1.5),
            (9.5,  5.0,  2.0, 2.0),
            (13.0, 3.0,  2.0, 1.5),
            (15.5, 6.5,  1.5, 2.5),
            (3.0,  10.5, 1.5, 3.0),
            (7.0,  11.0, 2.5, 1.5),
            (11.0, 9.5,  2.0, 2.0),
            (14.5, 11.0, 1.5, 2.5),
            (17.5, 14.0, 1.5, 2.0),
            (5.0,  15.5, 2.0, 1.5),
            (10.0, 16.5, 2.5, 1.5),
            (14.0, 17.5, 2.0, 1.5),
        ]

    # ------------------------------------------------------------------
    def is_occupied(self, x, y, margin=0.0):
        """Return True if (x,y) is inside an obstacle or out of bounds."""
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return True
        for ox, oy, ow, oh in self.obstacles:
            if abs(x - ox) < (ow / 2 + margin) and abs(y - oy) < (oh / 2 + margin):
                return True
        return False
