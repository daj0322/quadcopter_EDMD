import numpy as np
from scipy.integrate import solve_ivp
from PID_Mixer import pid_mixer

class ClosedLoopQuad:
    def __init__(self, quad, controller):
        self.quad = quad
        self.controller = controller

    def fct_simulate(self, time, dt, ref_traj, init_state):
        state = np.array(init_state, dtype=float)
        states         = np.zeros((len(time), len(state)))
        omegas         = np.zeros((len(time), 4))
        control_inputs = np.zeros((len(time), 4))
        u_att_log      = np.zeros((len(time), 4))

        for i, t in enumerate(time):
            states[i] = state
            omega_cmd, u, u_att = self.controller.fct_step(state, ref_traj[i], dt)
            control_inputs[i] = u
            u_att_log[i] = u_att
            omegas[i] = omega_cmd

            def ode(t_local, s_local):
                return self.quad.fct_dynamics(t_local, s_local, omega_cmd)

            sol = solve_ivp(ode, [t, t + dt], state, method="RK45")
            state = sol.y[:, -1]


        self.controller.fct_reset()
        return time, states, omegas, control_inputs, u_att_log

    def fct_step_attitude(self, state, u1, phi_des, theta_des, dt, psi_des=0.0, inner_dt=0.01):
        """
        Advance the plant by one step using direct attitude commands.
        """

        state_current = np.array(state, dtype=float)
        n_substeps = max(1, int(np.ceil(float(dt) / float(inner_dt))))
        h = float(dt) / n_substeps

        for _ in range(n_substeps):
            phi, theta, psi = state_current[6], state_current[7], state_current[8]

            # Inner-loop attitude control runs at the plant integration rate while
            # the outer MPC attitude command is held over the full MPC interval.
            u2 = self.controller.pid_phi.fct_control(phi, phi_des, h)
            u3 = self.controller.pid_theta.fct_control(theta, theta_des, h)
            u4 = self.controller.fct_yaw_torque(psi, psi_des, h)
            u2 = float(np.clip(u2, -self.controller.torque_max, self.controller.torque_max))
            u3 = float(np.clip(u3, -self.controller.torque_max, self.controller.torque_max))

            u = [u1, u2, u3, u4]
            omega_cmd = pid_mixer.fct_mixer(
                u, self.quad.kT, self.quad.kD, self.quad.l,
                min_omega=0.0, max_omega=self.controller.max_speed
            )

            def ode(t_local, s_local):
                return self.quad.fct_dynamics(t_local, s_local, omega_cmd)

            sol = solve_ivp(ode, [0, h], state_current, method="RK45")
            state_current = sol.y[:, -1]

        return state_current
