import numpy as np


class LinearQuadcopter:
    """
    Linearized quadcopter model around hover.

    State:
    [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]

    Input:
    u = [u1, u2, u3, u4]
        u1: total thrust (N)
        u2: roll torque (Nm)
        u3: pitch torque (Nm)
        u4: yaw torque (Nm)
    """

    def __init__(self, m, g, I, k_drag_linear=0.0, k_drag_angular=0.0):

        self.m = m
        self.g = g
        self.I = I

        self.k_drag_linear = k_drag_linear
        self.k_drag_angular = k_drag_angular

        self.Ixx = I[0, 0]
        self.Iyy = I[1, 1]
        self.Izz = I[2, 2]

        # Hover equilibrium
        self.u_hover = np.array([m * g, 0.0, 0.0, 0.0])

    # -------------------------------------------------
    # Continuous Linear Dynamics
    # -------------------------------------------------
    def fct_dynamics(self, t, state, u):

        delta_u = np.array(u) - self.u_hover

        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state

        m = self.m
        g = self.g

        dv = self.k_drag_linear / m
        Dwx = self.k_drag_angular / self.Ixx
        Dwy = self.k_drag_angular / self.Iyy
        Dwz = self.k_drag_angular / self.Izz

        dstate = np.zeros(12)

        # Position
        dstate[0] = vx
        dstate[1] = vy
        dstate[2] = vz

        # Linear velocity
        dstate[3] = g * theta - dv * vx
        dstate[4] = -g * phi - dv * vy
        dstate[5] = (1 / m) * delta_u[0] - dv * vz

        # Euler angles
        dstate[6] = p
        dstate[7] = q
        dstate[8] = r

        # Angular rates
        dstate[9]  = (1 / self.Ixx) * delta_u[1] - Dwx * p
        dstate[10] = (1 / self.Iyy) * delta_u[2] - Dwy * q
        dstate[11] = (1 / self.Izz) * delta_u[3] - Dwz * r

        return dstate