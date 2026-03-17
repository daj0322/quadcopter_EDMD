import numpy as np
import matplotlib.pyplot as plt

from quadcopter_linearized_model import quadcopter_linearized_model
from quadcopter_pd_controller import quadcopter_pd_controller
from quadcopter_dynamics import quadcopter_dynamics


def simulate_quadcopter():
    # ------------------------------------------------------------------ #
    #  Parameters
    # ------------------------------------------------------------------ #
    m   = 1.0;  g   = 9.81
    Ixx = 0.01; Iyy = 0.01; Izz = 0.02
    kv  = 0.1;  kw  = 0.01
    Ts  = 0.01

    # ------------------------------------------------------------------ #
    #  Build linearized model
    # ------------------------------------------------------------------ #
    A, B, _, _ = quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts)

    # ------------------------------------------------------------------ #
    #  PD Gains  (4 x 12 matrices, all zeros initially)
    # ------------------------------------------------------------------ #
    Kp = np.zeros((4, 12))
    Kd = np.zeros((4, 12))

    # z position -> thrust
    Kp[0, 2]  =  2.0;  Kd[0, 5]  =  1.5

    # y error -> roll (phi) -> vy  (y_dot ~ -g*phi, negative sign)
    Kp[1, 1]  = -0.5;  Kd[1, 4]  = -0.3   # y pos + vy  -> u2 (roll moment)
    Kp[1, 6]  =  3.0;  Kd[1, 9]  =  0.5   # phi + p     -> u2

    # x error -> pitch (theta) -> vx  (x_dot ~ g*theta, positive sign)
    Kp[2, 0]  =  0.5;  Kd[2, 3]  =  0.3   # x pos + vx  -> u3 (pitch moment)
    Kp[2, 7]  =  3.0;  Kd[2, 10] =  0.5   # theta + q   -> u3

    # psi -> yaw
    Kp[3, 8]  =  1.0;  Kd[3, 11] =  0.3

    # ------------------------------------------------------------------ #
    #  Initial and reference states
    # ------------------------------------------------------------------ #
    x     = np.zeros(12)
    x_ref = np.zeros(12)
    x_ref[0:3] = [1.0, 1.0, 1.0]    # target position x=1, y=1, z=1

    # ------------------------------------------------------------------ #
    #  Simulation loop (Euler integration)
    # ------------------------------------------------------------------ #
    t_end = 10.0
    t     = np.arange(0, t_end + Ts, Ts)
    nstep = len(t)

    X = np.zeros((12, nstep))
    U = np.zeros((4,  nstep))

    for k in range(nstep):
        X[:, k] = x
        u        = quadcopter_pd_controller(x, x_ref, Kp, Kd, m, g)
        U[:, k] = u
        x_dot    = quadcopter_dynamics(x, u, A, B, m, g)
        x        = x + Ts * x_dot

    # ------------------------------------------------------------------ #
    #  Plot states
    # ------------------------------------------------------------------ #
    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz',
                   'phi', 'theta', 'psi', 'p', 'q', 'r']
    ref_vals    = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    fig, axes = plt.subplots(4, 3, figsize=(14, 10))
    fig.suptitle('Quadcopter PD Controller — States (ref = [1,1,1])')

    for i, ax in enumerate(axes.flat):
        ax.plot(t, X[i, :], 'b', linewidth=1.2, label='state')
        ax.axhline(ref_vals[i], color='r', linestyle='--', linewidth=1.0, label='reference')
        ax.set_xlabel('t [s]'); ax.set_ylabel(state_names[i])
        ax.set_title(state_names[i]); ax.grid(True)
        ax.legend(loc='best', fontsize=7)

    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  Plot control inputs
    # ------------------------------------------------------------------ #
    input_names = ['u1 (thrust)', 'u2 (roll)', 'u3 (pitch)', 'u4 (yaw)']

    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 6))
    fig2.suptitle('Quadcopter PD Controller — Control Inputs')

    for i, ax in enumerate(axes2.flat):
        ax.plot(t, U[i, :], 'k', linewidth=1.2)
        ax.set_xlabel('t [s]'); ax.set_ylabel(input_names[i])
        ax.set_title(input_names[i]); ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    simulate_quadcopter()
