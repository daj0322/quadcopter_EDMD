import numpy as np


def quadcopter_dynamics(x, u, A, B, m, g):
    """
    Linearized quadcopter dynamics.

    x_dot = quadcopter_dynamics(x, u, A, B, m, g)

    Parameters
    ----------
    x : (12,) array  - current state
    u : (4,)  array  - total control input (from controller)
    A : (12,12) array - state matrix
    B : (12,4)  array - input matrix
    m : float         - mass [kg]
    g : float         - gravitational acceleration [m/s^2]

    Returns
    -------
    x_dot : (12,) array - state derivative

    delta_u = u - u_star is computed here, consistent with the paper:
        x_dot = A*x + B*delta_u
    """
    # Hover equilibrium
    u_star = np.array([m * g, 0.0, 0.0, 0.0])

    # Control perturbation  (paper: delta_u = u - u_star)
    delta_u = u - u_star

    # Linearized dynamics (Eq. 29)
    x_dot = A @ x + B @ delta_u
    return x_dot
