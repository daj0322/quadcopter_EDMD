# %% Import libraries
import numpy as np
from scipy.integrate import solve_ivp

class ClosedLoopQuad:
    def __init__(self, quad, controller):
        self.quad = quad
        self.controller = controller

    def fct_simulate(self, time, dt, ref_traj, init_state,
                     return_requested=False):
        state = np.array(init_state, dtype=float)
        states = np.zeros((len(time), len(state)))
        omegas = np.zeros((len(time), 4))
        control_inputs = np.zeros((len(time), 4))
        requested_inputs = np.zeros((len(time), 4))

        for i, t in enumerate(time):
            # Log the state before the command is applied so each training
            # tuple is aligned as (x_k, u_k, x_{k+1}).
            states[i] = state
            omega_cmd, u_applied = self.controller.fct_step(state, ref_traj[i], dt)
            control_inputs[i] = u_applied
            requested_inputs[i] = getattr(
                self.controller, "last_requested_wrench", u_applied
            )

            def ode(t_local, s_local):
                return self.quad.fct_dynamics(t_local, s_local, omega_cmd)

            sol = solve_ivp(ode, [t, t + dt], state, method="RK45")
            state = sol.y[:, -1]

            omegas[i] = omega_cmd
        self.controller.fct_reset()
        self.last_applied_inputs = control_inputs
        self.last_requested_inputs = requested_inputs
        if return_requested:
            return time, states, omegas, control_inputs, requested_inputs
        return time, states, omegas, control_inputs
