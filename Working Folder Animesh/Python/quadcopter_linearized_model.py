import numpy as np
from scipy import signal


def quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts=None):
    """
    Continuous & discrete linearized hover model.

    Parameters
    ----------
    m   : float  - mass [kg]
    g   : float  - gravitational acceleration [m/s^2]
    Ixx : float  - moment of inertia about x-axis [kg·m^2]
    Iyy : float  - moment of inertia about y-axis [kg·m^2]
    Izz : float  - moment of inertia about z-axis [kg·m^2]
    kv  : float  - translational drag coefficient [N·s/m]
    kw  : float  - angular drag coefficient [N·m·s/rad]
    Ts  : float or None - sample time for discretization (None to skip)

    Returns
    -------
    A     : (12,12) ndarray  - continuous-time state matrix
    B     : (12,4)  ndarray  - continuous-time input matrix
    sys_c : scipy.signal.StateSpace  - continuous-time state-space object
    sys_d : scipy.signal.dlti or None - discrete-time state-space (ZOH), None if Ts is None

    State:  x = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    Input:  delta_u = u - u_star,  u_star = [mg, 0, 0, 0]^T
    """
    # Damping rates (Eq. 28)
    dv = kv / m
    Dw = np.diag([kw / Ixx, kw / Iyy, kw / Izz])

    # Gravity-attitude coupling (Eq. 32)
    G = np.array([
        [ 0,  g, 0],
        [-g,  0, 0],
        [ 0,  0, 0]
    ])

    # Input sub-matrices (Eq. 31)
    Bv = np.array([
        [0,    0,       0,       0      ],
        [0,    0,       0,       0      ],
        [1/m,  0,       0,       0      ]
    ])

    Bw = np.array([
        [0,    1/Ixx,  0,       0      ],
        [0,    0,      1/Iyy,   0      ],
        [0,    0,      0,       1/Izz  ]
    ])

    # State matrix A (Eq. 30)
    O3  = np.zeros((3, 3))
    I3  = np.eye(3)
    O34 = np.zeros((3, 4))

    A = np.block([
        [O3,  I3,       O3,  O3 ],
        [O3, -dv * I3,   G,  O3 ],
        [O3,  O3,       O3,  I3 ],
        [O3,  O3,       O3, -Dw ]
    ])

    # Input matrix B (Eq. 31)
    B = np.vstack([O34, Bv, O34, Bw])

    # Continuous-time state-space
    C = np.eye(12)
    D = np.zeros((12, 4))
    sys_c = signal.StateSpace(A, B, C, D)

    # Discretization (ZOH)
    if Ts is not None:
        sys_d = sys_c.to_discrete(Ts, method='zoh')
    else:
        sys_d = None

    return A, B, sys_c, sys_d
