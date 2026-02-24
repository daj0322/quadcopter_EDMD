# %% Import libraries
import numpy as np
from scipy.integrate import solve_ivp

class ClosedLoopQuad:
    def __init__(self, quad, controller):
        self.quad = quad
        self.controller = controller

    def fct_simulate(self, time, dt, ref_traj, init_state):
        state = np.array(init_state, dtype=float)
        states = np.zeros((len(time), len(state)))
        omegas = np.zeros((len(time), 4))
        control_inputs = np.zeros((len(time), 4))  # u1 u2 u3 u4

        for i, t in enumerate(time):
            omega_cmd,u = self.controller.fct_step(state, ref_traj[i], dt)
            control_inputs[i] = u

            def ode(t_local, s_local):
                return self.quad.fct_dynamics(t_local, s_local, omega_cmd)

            sol = solve_ivp(ode, [t, t + dt], state, method="RK45")
            state = sol.y[:, -1]

            states[i] = state
            omegas[i] = omega_cmd
        self.controller.fct_reset()
        return time, states, omegas, control_inputs