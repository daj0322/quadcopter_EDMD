import numpy as np

from PID_Trajectory_Controller import PID_trajectory_controller
from iPID_Trajectory_Controller import iPID_trajectory_controller
from Helperfcts import helperfcts
from PID_Mixer import pid_mixer

class QuadPIDController6Fixed:
    def __init__(self, quad, kp_pos, ki_pos, kd_pos, kp_ang, ki_ang, kd_ang, max_speed=400.0, a_xy_max=3.0, a_z_max=5.0, tilt_max_deg=45.0, torque_roll_pitch_max=0.12, yaw_tau_max=0.01):
        self.quad = quad
        self.max_speed = float(max_speed)

        # Position PIDs output desired accelerations
        self.pid_x = PID_trajectory_controller(kp_pos[0], ki_pos[0], kd_pos[0], integral_limit=1.1)
        self.pid_y = PID_trajectory_controller(kp_pos[1], ki_pos[1], kd_pos[1], integral_limit=1.1)
        self.pid_z = PID_trajectory_controller(kp_pos[2], ki_pos[2], kd_pos[2], integral_limit=1.1)

        # Roll/Pitch/Yaw angle PIDs (torques)
        self.pid_phi   = PID_trajectory_controller(kp_ang[0], ki_ang[0], kd_ang[0], integral_limit=0.1)
        self.pid_theta = PID_trajectory_controller(kp_ang[1], ki_ang[1], kd_ang[1], integral_limit=0.1)
        self.pid_psi = PID_trajectory_controller(kp_ang[2], ki_ang[2], kd_ang[2], integral_limit=0.1) #kp=1.0, ki=0.0, kd=0.2

        self.a_xy_max = float(a_xy_max)
        self.a_z_max = float(a_z_max)
        self.tilt_max = np.deg2rad(float(tilt_max_deg))
        self.torque_max = float(torque_roll_pitch_max)
        self.yaw_tau_max = float(yaw_tau_max)

    def fct_reset(self):
        for pid in [self.pid_x, self.pid_y, self.pid_z, self.pid_phi, self.pid_theta, self.pid_psi]:
            pid.fct_reset()

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

        # Desired yaw
        psi_des = float(0)

        # Yaw-invariant lateral control
        cpsi = np.cos(psi_des)
        spsi = np.sin(psi_des)

        # Compute desired roll & pitch from yaw-aligned force
        phi_des   = (ux*spsi - uy*cpsi)
        theta_des = (ux*cpsi + uy*spsi)

        phi_des   = float(np.clip(phi_des,   -self.tilt_max, self.tilt_max))
        theta_des = float(np.clip(theta_des, -self.tilt_max, self.tilt_max))

        # Total thrust command (magnitude of desired force)
        u1 = self.quad.m * (self.quad.g + uz)
        u1 = float(max(0.0, u1))

        # Inner loop: attitude -> torques
        u2 = self.pid_phi.fct_control(phi, phi_des, dt)
        u3 = self.pid_theta.fct_control(theta, theta_des, dt)
        u2 = float(np.clip(u2, -self.torque_max, self.torque_max))
        u3 = float(np.clip(u3, -self.torque_max, self.torque_max))
        u4 = 0
        u = [u1,u2,u3,u4]

        # 4-DOF mixer
        omega_cmd = pid_mixer.fct_mixer(u, self.quad.kT, self.quad.kD, self.quad.l, min_omega=0.0, max_omega=self.max_speed)
        return omega_cmd,u
    
class QuadIPIDController6Fixed:
    def __init__(self, quad, kp_pos, ki_pos, kd_pos, kp_ang, ki_ang, kd_ang, max_speed=400.0, a_xy_max=3.0, a_z_max=5.0, tilt_max_deg=20.0, torque_roll_pitch_max=0.12, yaw_tau_max=0.01, ipid_alpha=0.2):
        self.quad = quad
        self.max_speed = float(max_speed)

        # Position controllers iPID
        self.ipid_x = iPID_trajectory_controller(kp_pos[0], ki_pos[0], kd_pos[0], integral_limit=1.1, alpha=ipid_alpha)
        self.ipid_y = iPID_trajectory_controller(kp_pos[1], ki_pos[1], kd_pos[1], integral_limit=1.1, alpha=ipid_alpha)
        self.ipid_z = iPID_trajectory_controller(kp_pos[2], ki_pos[2], kd_pos[2], integral_limit=1.1, alpha=ipid_alpha)

        # Roll/Pitch/Yaw angle iPID
        self.ipid_phi   = iPID_trajectory_controller(kp_ang[0], ki_ang[0], kd_ang[0], integral_limit=0.1, alpha=ipid_alpha)
        self.ipid_theta = iPID_trajectory_controller(kp_ang[1], ki_ang[1], kd_ang[1], integral_limit=0.1, alpha=ipid_alpha)
        self.ipid_psi = iPID_trajectory_controller(kp_ang[2], ki_ang[2], kd_ang[2], integral_limit=0.1, alpha=ipid_alpha) #1.0, 0.0, 0.2

        self.a_xy_max = float(a_xy_max)
        self.a_z_max = float(a_z_max)
        self.tilt_max = np.deg2rad(float(tilt_max_deg))
        self.torque_max = float(torque_roll_pitch_max)
        self.yaw_tau_max = float(yaw_tau_max)

    def fct_reset(self):
        for pid in [self.ipid_x, self.ipid_y, self.ipid_z, self.ipid_phi, self.ipid_theta, self.ipid_psi]:
            pid.fct_reset()

    def fct_step(self, state, ref, dt):
        m, g = self.quad.m, self.quad.g
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state

        # Outer loop: position -> desired accel (world)
        xr, yr, zr = ref["pos"]

        # iPID outputs (position correction)
        ux = self.ipid_x.fct_control(x, xr, dt)
        uy = self.ipid_y.fct_control(y, yr, dt)
        uz = self.ipid_z.fct_control(z, zr, dt)

        ux = float(np.clip(ux, -self.a_xy_max, self.a_xy_max))
        uy = float(np.clip(uy, -self.a_xy_max, self.a_xy_max))
        uz = float(np.clip(uz, -self.a_z_max, self.a_z_max))

        # Desired yaw
        psi_des = float(0)

        # Yaw-invariant lateral control (same as PID class)
        cpsi = np.cos(psi_des)
        spsi = np.sin(psi_des)

        # Compute desired roll & pitch from yaw-aligned force
        phi_des   = (ux*spsi - uy*cpsi)
        theta_des = (ux*cpsi + uy*spsi)

        phi_des   = float(np.clip(phi_des, -self.tilt_max, self.tilt_max))
        theta_des = float(np.clip(theta_des, -self.tilt_max, self.tilt_max))

        # Total thrust command
        u1 = self.quad.m * (self.quad.g + uz)
        u1 = float(max(0.0, u1))

        # Inner loop: attitude -> torques
        u2 = self.ipid_phi.fct_control(phi, phi_des, dt)
        u3 = self.ipid_theta.fct_control(theta, theta_des, dt)
        u2 = float(np.clip(u2, -self.torque_max, self.torque_max))
        u3 = float(np.clip(u3, -self.torque_max, self.torque_max))
        u4 = 0
        u = [u1,u2,u3,u4]

        # 4-DOF mixer
        omega_cmd = pid_mixer.fct_mixer(u, self.quad.kT, self.quad.kD, self.quad.l, min_omega=0.0, max_omega=self.max_speed
        )
        return omega_cmd,u