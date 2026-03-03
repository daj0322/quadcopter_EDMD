import numpy as np


class LinearPDController:

    def __init__(self, quad,
                 kp_pos, kd_pos,
                 kp_ang, kd_ang):

        self.quad = quad

        self.kp_pos = np.array(kp_pos)
        self.kd_pos = np.array(kd_pos)

        self.kp_ang = np.array(kp_ang)
        self.kd_ang = np.array(kd_ang)

    # -------------------------------------------------
    def fct_step(self, state, ref):

        m = self.quad.m
        g = self.quad.g

        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state

        xr, yr, zr = ref["pos"]

        # -------------------------
        # Position PD
        # -------------------------
        ex = xr - x
        ey = yr - y
        ez = zr - z

        evx = -vx
        evy = -vy
        evz = -vz

        ax_des = self.kp_pos[0] * ex + self.kd_pos[0] * evx
        ay_des = self.kp_pos[1] * ey + self.kd_pos[1] * evy
        az_des = self.kp_pos[2] * ez + self.kd_pos[2] * evz

        # -------------------------
        # Desired angles (small-angle)
        # -------------------------
        phi_des = -ay_des / g
        theta_des = ax_des / g

        # -------------------------
        # Attitude PD
        # -------------------------
        ephi = phi_des - phi
        etheta = theta_des - theta
        epsi = 0 - psi

        ep = -p
        eq = -q
        er = -r

        u2 = self.kp_ang[0] * ephi + self.kd_ang[0] * ep
        u3 = self.kp_ang[1] * etheta + self.kd_ang[1] * eq
        u4 = self.kp_ang[2] * epsi + self.kd_ang[2] * er

        # -------------------------
        # Thrust
        # -------------------------
        u1 = m * (g + az_des)

        return np.array([u1, u2, u3, u4])