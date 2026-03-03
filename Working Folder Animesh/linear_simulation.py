import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from linear_quadcopter import LinearQuadcopter
from linear_controller import LinearPDController


# -------------------------------------------------
# Simulation parameters
# -------------------------------------------------
m = 0.5
g = 9.81

Ixx, Iyy, Izz = 5e-3, 5e-3, 9e-3
I = np.diag([Ixx, Iyy, Izz])

quad = LinearQuadcopter(
    m, g, I,
    k_drag_linear=0.3,
    k_drag_angular=0.02
)

controller = LinearPDController(
    quad,
    kp_pos=[4, 4, 8],
    kd_pos=[3, 3, 5],
    kp_ang=[8, 8, 5],
    kd_ang=[2, 2, 1]
)

# -------------------------------------------------
# Reference
# -------------------------------------------------
def reference(t):
    return {
        "pos": np.array([1.0, 1.0, 1.0])
    }


# -------------------------------------------------
# Simulation
# -------------------------------------------------
dt = 0.01
T = 10
time = np.arange(0, T, dt)

state = np.zeros(12)
states = []

for t in time:

    ref = reference(t)
    u = controller.fct_step(state, ref)

    def ode(t_local, s_local):
        return quad.fct_dynamics(t_local, s_local, u)

    sol = solve_ivp(ode, [t, t + dt], state)
    state = sol.y[:, -1]

    states.append(state.copy())

states = np.array(states)

# -------------------------------------------------
# Plots
# -------------------------------------------------
plt.figure()
plt.plot(time, states[:, 0], label="x")
plt.plot(time, states[:, 1], label="y")
plt.plot(time, states[:, 2], label="z")
plt.legend()
plt.title("Position")
plt.grid()

plt.figure()
plt.plot(time, states[:, 6], label="phi")
plt.plot(time, states[:, 7], label="theta")
plt.plot(time, states[:, 8], label="psi")
plt.legend()
plt.title("Angles")
plt.grid()

plt.show()