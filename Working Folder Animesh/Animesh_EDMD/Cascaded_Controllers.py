import numpy as np

from PID_Trajectory_Controller import PID_trajectory_controller
from PID_Mixer import pid_mixer

class QuadPIDController6Fixed:
    def __init__(self, quad, kp_pos, ki_pos, kd_pos, kp_ang, ki_ang, kd_ang, max_speed=400.0, a_xy_max=3.0, a_z_max=5.0, tilt_max_deg=45.0, torque_roll_pitch_max=0.12, yaw_tau_max=None):
        self.quad = quad
        self.max_speed = float(max_speed)

        # Position PIDs output desired accelerations
        self.pid_x = PID_trajectory_controller(kp_pos[0], ki_pos[0], kd_pos[0], integral_limit=1.1)
        self.pid_y = PID_trajectory_controller(kp_pos[1], ki_pos[1], kd_pos[1], integral_limit=1.1)
        self.pid_z = PID_trajectory_controller(kp_pos[2], ki_pos[2], kd_pos[2], integral_limit=1.1)

        # Roll/Pitch/Yaw angle PIDs (torques)
        self.pid_phi   = PID_trajectory_controller(kp_ang[0], ki_ang[0], kd_ang[0], integral_limit=0.1)
        self.pid_theta = PID_trajectory_controller(kp_ang[1], ki_ang[1], kd_ang[1], integral_limit=0.1)
        self.pid_psi = PID_trajectory_controller(kp_ang[2], ki_ang[2], kd_ang[2], integral_limit=0.1)

        self.a_xy_max = float(a_xy_max)
        self.a_z_max = float(a_z_max)
        self.tilt_max = np.deg2rad(float(tilt_max_deg))
        self.torque_max = float(torque_roll_pitch_max)
        if yaw_tau_max is None:
            self.yaw_tau_max = 0.8 * 4.0 * self.quad.kD * self.max_speed**2
        else:
            self.yaw_tau_max = float(yaw_tau_max)

    def fct_reset(self):
        for pid in [self.pid_x, self.pid_y, self.pid_z, self.pid_phi, self.pid_theta, self.pid_psi]:
            pid.fct_reset()

    @staticmethod
    def fct_wrap_angle(angle):
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    def fct_desired_yaw(self, ref, current_psi=0.0):
        if "yaw" in ref:
            psi_des = float(ref["yaw"])
        else:
            vel = np.asarray(ref.get("vel", np.zeros(3)), dtype=float)
            if vel.shape[0] >= 2 and np.linalg.norm(vel[:2]) > 1e-6:
                psi_des = float(np.arctan2(vel[1], vel[0]))
            else:
                psi_des = float(current_psi)

        return float(current_psi + self.fct_wrap_angle(psi_des - current_psi))

    def fct_yaw_torque(self, psi, psi_des, dt):
        yaw_error = self.fct_wrap_angle(psi_des - psi)
        u4 = self.pid_psi.fct_control(0.0, yaw_error, dt)
        return float(np.clip(u4, -self.yaw_tau_max, self.yaw_tau_max))

    def fct_step(self, state, ref, dt):
        m, g = self.quad.m, self.quad.g
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state

        # Outer loop: position -> desired accel (world)
        xr, yr, zr = ref["pos"]

        # PID outputs (position correction)
        ux = self.pid_x.fct_control(x, xr, dt)
        uy = self.pid_y.fct_control(y, yr, dt)
        uz = self.pid_z.fct_control(z, zr, dt)

        ux = float(np.clip(ux, -self.a_xy_max, self.a_xy_max))
        uy = float(np.clip(uy, -self.a_xy_max, self.a_xy_max))
        uz = float(np.clip(uz, -self.a_z_max, self.a_z_max))

        # Desired yaw follows the reference heading.
        psi_des = self.fct_desired_yaw(ref, psi)

        # Yaw-invariant lateral control
        cpsi = np.cos(psi_des)
        spsi = np.sin(psi_des)

        # Compute desired roll & pitch from yaw-aligned lateral acceleration.
        phi_des   = (ux*spsi - uy*cpsi) / g
        theta_des = (ux*cpsi + uy*spsi) / g

        phi_des   = float(np.clip(phi_des,   -self.tilt_max, self.tilt_max))
        theta_des = float(np.clip(theta_des, -self.tilt_max, self.tilt_max))

        # Total thrust command, compensated for the commanded tilt.
        tilt_comp = max(0.2, np.cos(phi_des) * np.cos(theta_des))
        u1 = self.quad.m * (self.quad.g + uz) / tilt_comp
        u1 = float(max(0.0, u1))

        # Inner loop: attitude -> torques
        u2 = self.pid_phi.fct_control(phi, phi_des, dt)
        u3 = self.pid_theta.fct_control(theta, theta_des, dt)
        u2 = float(np.clip(u2, -self.torque_max, self.torque_max))
        u3 = float(np.clip(u3, -self.torque_max, self.torque_max))
        u4 = self.fct_yaw_torque(psi, psi_des, dt)
        u = [u1,u2,u3,u4]

        # 4-DOF mixer
        omega_cmd = pid_mixer.fct_mixer(u, self.quad.kT, self.quad.kD, self.quad.l, min_omega=0.0,
                                        max_omega=self.max_speed)
        u_att = np.array([u1, phi_des, theta_des, psi_des], dtype=float)
        return omega_cmd, u, u_att
