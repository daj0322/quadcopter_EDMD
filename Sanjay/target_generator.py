import numpy as np


def straight_line_target(t: float, p0: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Straight-line target trajectory.

    Args:
        t: time [s]
        p0: initial position, shape (3,)
        v: constant velocity, shape (3,)

    Returns:
        p: position at time t, shape (3,)
        v: velocity at time t, shape (3,)
    """
    p = p0 + v * t
    return p, v