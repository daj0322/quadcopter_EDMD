import numpy as np


def quadcopter_pd_controller(x, x_ref, Kp, Kd, m, g):
    """
    PD controller for linearized quadcopter hover.

    Parameters
    ----------
    x     : (12,) array  - current state  [pW; vW; eta; omegaB]
    x_ref : (12,) array  - reference state
    Kp    : (4,12) array or scalar - proportional gain matrix
    Kd    : (4,12) array or scalar - derivative gain matrix
    m     : float        - mass [kg]
    g     : float        - gravitational acceleration [m/s^2]

    Returns
    -------
    u : (4,) array - total control input

    Control law:
        u = u_star + Kp @ e_p + Kd @ e_d

    Note:
        delta_u = u - u_star is computed inside quadcopter_dynamics,
        consistent with:  x_dot = A*x + B*delta_u
    """
    x     = np.asarray(x,     dtype=float).ravel()
    x_ref = np.asarray(x_ref, dtype=float).ravel()
    assert x.shape     == (12,), "x must be length 12"
    assert x_ref.shape == (12,), "x_ref must be length 12"

    # Expand scalar gains
    if np.isscalar(Kp):
        Kp = Kp * np.eye(4, 12)
    if np.isscalar(Kd):
        Kd = Kd * np.eye(4, 12)

    Kp = np.asarray(Kp, dtype=float)
    Kd = np.asarray(Kd, dtype=float)
    assert Kp.shape == (4, 12), "Kp must be (4,12) or scalar"
    assert Kd.shape == (4, 12), "Kd must be (4,12) or scalar"

    # Hover equilibrium
    u_star = np.array([m * g, 0.0, 0.0, 0.0])

    # State error
    e = x_ref - x

    # Proportional error: position (indices 0-2) + attitude (indices 6-8)
    e_p = np.concatenate([e[0:3], np.zeros(3), e[6:9], np.zeros(3)])

    # Derivative error: velocity (indices 3-5) + angular rate (indices 9-11)
    e_d = np.concatenate([np.zeros(3), e[3:6], np.zeros(3), e[9:12]])

    # Total control input
    u = u_star + Kp @ e_p + Kd @ e_d
    return u
