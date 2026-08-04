import numpy as np

from PID_Trajectory_Controller import PID_trajectory_controller
from iPID_Trajectory_Controller import iPID_trajectory_controller
from Helperfcts import helperfcts
from PID_Mixer import pid_mixer


def _vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)

class QuadPIDController6Fixed:
    def __init__(self, quad, kp_pos, ki_pos, kd_pos, kp_ang, ki_ang, kd_ang, max_speed=400.0, a_xy_max=3.0, a_z_max=5.0, tilt_max_deg=45.0, torque_roll_pitch_max=0.12, yaw_tau_max=0.015):
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

        # Outer loop: position/velocity -> desired accel (world)
        xr, yr, zr = ref["pos"]
        vxr, vyr, vzr = ref.get("vel", np.zeros(3))
        axr, ayr, azr = ref.get("acc", np.zeros(3))

        drag_comp = self.quad.k_drag_linear / m
        ux = axr + drag_comp * vxr + self.pid_x.fct_control_with_error_rate(xr - x, vxr - vx, dt)
        uy = ayr + drag_comp * vyr + self.pid_y.fct_control_with_error_rate(yr - y, vyr - vy, dt)
        uz = azr + drag_comp * vzr + self.pid_z.fct_control_with_error_rate(zr - z, vzr - vz, dt)

        ux = float(np.clip(ux, -self.a_xy_max, self.a_xy_max))
        uy = float(np.clip(uy, -self.a_xy_max, self.a_xy_max))
        uz = float(np.clip(uz, -self.a_z_max, self.a_z_max))

        # Desired yaw from reference trajectory. Use a wrapped equivalent near
        # the current yaw so the PID does not chase artificial +/-pi jumps.
        psi_des = float(ref.get("yaw", 0.0))
        psi_ref = psi + helperfcts.wrap_angle(psi_des - psi)

        # Yaw-invariant lateral control
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        # Compute desired roll & pitch from yaw-aligned force
        phi_des   = (ux*spsi - uy*cpsi) / g
        theta_des = (ux*cpsi + uy*spsi) / g

        phi_des   = float(np.clip(phi_des,   -self.tilt_max, self.tilt_max))
        theta_des = float(np.clip(theta_des, -self.tilt_max, self.tilt_max))

        # Total thrust command (magnitude of desired force)
        thrust_direction_z = np.cos(phi_des) * np.cos(theta_des)
        u1 = self.quad.m * (self.quad.g + uz) / max(0.2, thrust_direction_z)
        u1 = float(max(0.0, u1))

        # Inner loop: attitude -> torques
        u2 = self.pid_phi.fct_control_with_error_rate(phi_des - phi, -p, dt)
        u3 = self.pid_theta.fct_control_with_error_rate(theta_des - theta, -q, dt)
        u2 = float(np.clip(u2, -self.torque_max, self.torque_max))
        u3 = float(np.clip(u3, -self.torque_max, self.torque_max))
        u4 = self.pid_psi.fct_control_with_error_rate(psi_ref - psi, -r, dt)
        u4 = float(np.clip(u4, -self.yaw_tau_max, self.yaw_tau_max))
        u_requested = np.array([u1, u2, u3, u4], dtype=float)
        omega_cmd, u_applied, _, _ = pid_mixer.fct_allocate_wrench(
            u_requested, self.quad.kT, self.quad.kD, self.quad.l,
            min_omega=0.0, max_omega=self.max_speed,
            prop_efficiency=self.quad.prop_efficiency,
        )
        self.last_requested_wrench = u_requested
        self.last_applied_wrench = u_applied
        return omega_cmd, u_applied

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
        u_requested = np.array([u1, u2, u3, u4], dtype=float)
        omega_cmd, u_applied, _, _ = pid_mixer.fct_allocate_wrench(
            u_requested, self.quad.kT, self.quad.kD, self.quad.l,
            min_omega=0.0, max_omega=self.max_speed,
            prop_efficiency=self.quad.prop_efficiency,
        )
        self.last_requested_wrench = u_requested
        self.last_applied_wrench = u_applied
        return omega_cmd, u_applied


class QuadPX4LikeController:
    def __init__(
        self,
        quad,
        max_speed=400.0,
        pos_p=(1.0, 1.0, 2.0),
        vel_p=(3.0, 3.0, 5.0),
        vel_i=(0.08, 0.08, 1.0),
        att_p=(7.0, 7.0, 4.0),
        rate_p=(0.075, 0.075, 0.02),
        rate_sp_max=(4.0, 4.0, 0.9),
        vel_int_limit=(1.0, 1.0, 1.0),
        vel_sp_max_xy=4.0,
        vel_sp_max_z=3.0,
        acc_max_xy=5.0,
        acc_max_z=5.0,
        tilt_max_deg=45.0,
        thrust_max=12.0,
        torque_max=(0.12, 0.12, 0.02),
    ):
        self.quad = quad
        self.max_speed = float(max_speed)

        self.pos_p = np.array(pos_p, dtype=float)
        self.vel_p = np.array(vel_p, dtype=float)
        self.vel_i = np.array(vel_i, dtype=float)
        self.att_p = np.array(att_p, dtype=float)
        self.rate_p = np.array(rate_p, dtype=float)
        self.rate_sp_max = np.array(rate_sp_max, dtype=float)
        self.vel_int_limit = np.array(vel_int_limit, dtype=float)
        self.vel_integral = np.zeros(3)

        self.vel_sp_max_xy = float(vel_sp_max_xy)
        self.vel_sp_max_z = float(vel_sp_max_z)
        self.acc_max_xy = float(acc_max_xy)
        self.acc_max_z = float(acc_max_z)
        self.tilt_max = np.deg2rad(float(tilt_max_deg))
        self.thrust_max = float(thrust_max)
        self.torque_max = np.array(torque_max, dtype=float)

    def fct_reset(self):
        self.vel_integral[:] = 0.0

    def _limit_xy(self, v, limit):
        out = np.array(v, dtype=float)
        n = np.linalg.norm(out[:2])
        if n > limit > 0.0:
            out[:2] *= limit / n
        return out

    def _limit_tilt(self, acc_sp):
        out = np.array(acc_sp, dtype=float)
        horizontal = np.linalg.norm(out[:2])
        max_horizontal = self.quad.g * np.tan(self.tilt_max)
        if horizontal > max_horizontal > 0.0:
            out[:2] *= max_horizontal / horizontal
        return out

    def fct_step(self, state, ref, dt):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state

        pos = np.array([x, y, z], dtype=float)
        vel = np.array([vx, vy, vz], dtype=float)
        rates = np.array([p, q, r], dtype=float)

        pos_ref = np.asarray(ref["pos"], dtype=float)
        vel_ref = np.asarray(ref.get("vel", np.zeros(3)), dtype=float)
        acc_ref = np.asarray(ref.get("acc", np.zeros(3)), dtype=float)
        yaw_ref = float(ref.get("yaw", 0.0))
        yaw_rate_ref = float(np.clip(ref.get("yaw_rate", 0.0), -self.rate_sp_max[2], self.rate_sp_max[2]))

        pos_error = pos_ref - pos
        vel_sp = vel_ref + self.pos_p * pos_error
        vel_sp = self._limit_xy(vel_sp, self.vel_sp_max_xy)
        vel_sp[2] = float(np.clip(vel_sp[2], -self.vel_sp_max_z, self.vel_sp_max_z))

        vel_error = vel_sp - vel
        self.vel_integral += vel_error * dt
        self.vel_integral = np.clip(self.vel_integral, -self.vel_int_limit, self.vel_int_limit)

        drag_comp = self.quad.k_drag_linear / self.quad.m
        acc_sp = acc_ref + drag_comp * vel_ref + self.vel_p * vel_error + self.vel_i * self.vel_integral
        acc_sp = self._limit_xy(acc_sp, self.acc_max_xy)
        acc_sp[2] = float(np.clip(acc_sp[2], -self.acc_max_z, self.acc_max_z))
        acc_sp = self._limit_tilt(acc_sp)

        force_world = self.quad.m * (acc_sp + np.array([0.0, 0.0, self.quad.g]))
        yaw_near = psi + helperfcts.wrap_angle(yaw_ref - psi)
        R_des = helperfcts.fct_desired_rotation_from_force_and_yaw(force_world, yaw_near)
        R = self.quad.fct_R_matrix(phi, theta, psi)

        attitude_error = 0.5 * _vee(R.T @ R_des - R_des.T @ R)
        rate_sp = self.att_p * attitude_error
        rate_sp += R_des.T @ np.array([0.0, 0.0, yaw_rate_ref])
        rate_sp = np.clip(rate_sp, -self.rate_sp_max, self.rate_sp_max)

        rate_error = rate_sp - rates
        torque = self.rate_p * rate_error
        torque = np.clip(torque, -self.torque_max, self.torque_max)

        u1 = float(np.clip(np.linalg.norm(force_world), 0.0, self.thrust_max))
        u_requested = np.array(
            [u1, float(torque[0]), float(torque[1]), float(torque[2])],
            dtype=float,
        )
        omega_cmd, u_applied, _, _ = pid_mixer.fct_allocate_wrench(
            u_requested, self.quad.kT, self.quad.kD, self.quad.l,
            min_omega=0.0, max_omega=self.max_speed,
            prop_efficiency=self.quad.prop_efficiency,
        )
        self.last_requested_wrench = u_requested
        self.last_applied_wrench = u_applied
        return omega_cmd, u_applied
