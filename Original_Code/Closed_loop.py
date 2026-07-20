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

    def fct_step_attitude(self, state, u1, phi_des, theta_des, dt):
        """
        Advance the plant by one step using direct attitude commands.
        """

        phi, theta = state[6], state[7]

        # Inner-loop attitude control.
        u2 = self.controller.pid_phi.fct_control(phi, phi_des, dt)
        u3 = self.controller.pid_theta.fct_control(theta, theta_des, dt)
        u2 = float(np.clip(u2, -self.controller.torque_max, self.controller.torque_max))
        u3 = float(np.clip(u3, -self.controller.torque_max, self.controller.torque_max))
        u4 = 0.0

        u = [u1, u2, u3, u4]
        omega_cmd = pid_mixer.fct_mixer(
            u, self.quad.kT, self.quad.kD, self.quad.l,
            min_omega=0.0, max_omega=self.controller.max_speed
        )

        def ode(t_local, s_local):
            return self.quad.fct_dynamics(t_local, s_local, omega_cmd)

        sol = solve_ivp(ode, [0, dt], state, method="RK45")
        return sol.y[:, -1]