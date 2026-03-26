import numpy as np
from dataclasses import dataclass
from target_generator import straight_line_target


@dataclass
class TargetConfig:
    p0: np.ndarray          # initial target position, shape (3,)
    v: np.ndarray           # constant target velocity, shape (3,)
    head_start: float       # [s]


@dataclass
class InterceptorConfig:
    x0: np.ndarray          # full 12-state initial condition


@dataclass
class ScenarioConfig:
    dt: float
    t_final: float
    capture_radius: float


def is_captured(p_i: np.ndarray, p_t: np.ndarray, capture_radius: float) -> bool:
    return np.linalg.norm(p_i - p_t) <= capture_radius


def generate_target_history(
    dt: float,
    t_final: float,
    target_cfg: TargetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate target time, position, velocity histories.

    Returns:
        t_hist: shape (N,)
        p_hist: shape (N, 3)
        v_hist: shape (N, 3)
    """
    t_hist = np.arange(0.0, t_final + dt, dt)
    p_hist = np.zeros((len(t_hist), 3))
    v_hist = np.zeros((len(t_hist), 3))

    for k, t in enumerate(t_hist):
        p, v = straight_line_target(t + target_cfg.head_start, target_cfg.p0, target_cfg.v)
        p_hist[k] = p
        v_hist[k] = v

    return t_hist, p_hist, v_hist

def simulate_open_loop_interception(
    scenario_cfg: ScenarioConfig,
    target_cfg: TargetConfig,
    interceptor_cfg: InterceptorConfig,
):
    """
    First debug version: interceptor does not move.
    This is just to verify the target history, head start, and capture logic.
    """
    t_hist, p_t_hist, v_t_hist = generate_target_history(
        scenario_cfg.dt,
        scenario_cfg.t_final,
        target_cfg,
    )

    x_i_hist = np.zeros((len(t_hist), interceptor_cfg.x0.shape[0]))
    x_i_hist[0] = interceptor_cfg.x0

    captured = False
    capture_time = None
    min_dist = np.inf

    for k in range(len(t_hist)):
        p_i = x_i_hist[k, 0:3]
        p_t = p_t_hist[k]

        dist = np.linalg.norm(p_i - p_t)
        min_dist = min(min_dist, dist)

        if is_captured(p_i, p_t, scenario_cfg.capture_radius):
            captured = True
            capture_time = t_hist[k]
            break

        # interceptor is stationary in this first debug version
        if k < len(t_hist) - 1:
            x_i_hist[k + 1] = x_i_hist[k]

    return {
        "t": t_hist,
        "x_i": x_i_hist,
        "p_t": p_t_hist,
        "v_t": v_t_hist,
        "captured": captured,
        "capture_time": capture_time,
        "min_dist": min_dist,
    }
