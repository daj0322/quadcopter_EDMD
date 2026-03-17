"""
simulate_leader_follower.py
===========================
Leader–follower quadcopter simulation.

  • Leader  : PD controller flying to [1, 1, 1]
  • Follower: Linear MPC intercepting the leader's trajectory

The follower first aims at the predicted intercept point; once it gets
within 0.05 m of the leader it locks on and tracks it continuously.
"""

import numpy as np
import matplotlib.pyplot as plt

from quadcopter_linearized_model import quadcopter_linearized_model
from quadcopter_pd_controller import quadcopter_pd_controller
from quadcopter_dynamics import quadcopter_dynamics
from simulate_quadcopter_mpc import LinearMPC     # re-use the MPC class


def simulate_leader_follower():

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
    A, B, _, sys_d = quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts)
    Ad = sys_d.A
    Bd = sys_d.B

    nx = 12; nu = 4

    # ------------------------------------------------------------------ #
    #  PD Gains (leader)
    # ------------------------------------------------------------------ #
    Kp = np.zeros((4, 12))
    Kd = np.zeros((4, 12))

    Kp[0, 2]  =  2.0;  Kd[0, 5]  =  1.5
    Kp[1, 1]  = -0.5;  Kd[1, 4]  = -0.3
    Kp[1, 6]  =  3.0;  Kd[1, 9]  =  0.5
    Kp[2, 0]  =  0.5;  Kd[2, 3]  =  0.3
    Kp[2, 7]  =  3.0;  Kd[2, 10] =  0.5
    Kp[3, 8]  =  1.0;  Kd[3, 11] =  0.3

    # ------------------------------------------------------------------ #
    #  MPC setup (follower) — same weights & constraints as MPC script
    # ------------------------------------------------------------------ #
    Q  = np.diag([10, 10, 10, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1])
    R  = np.diag([0.1, 0.1, 0.1, 0.1])
    Rd = np.diag([0.05, 0.05, 0.05, 0.05])

    y_min = np.array([-5, -5, -5,  -3, -3, -3, -0.3, -0.3, -np.pi, -2, -2, -2])
    y_max = np.array([ 5,  5,  5,   3,  3,  3,  0.3,  0.3,  np.pi,  2,  2,  2])

    u_nominal = np.array([m * g, 0.0, 0.0, 0.0])
    mpc = LinearMPC(Ad, Bd, N=10, Nc=3, Q=Q, R=R, Rd=Rd,
                    u_nominal=u_nominal, y_min=y_min, y_max=y_max)

    # ------------------------------------------------------------------ #
    #  Initial states
    # ------------------------------------------------------------------ #
    x_leader_0   = np.zeros(nx);  x_leader_0[0:3]   = [0.0, 0.0, 0.0]
    x_follower_0 = np.zeros(nx);  x_follower_0[0:3]  = [1.0, 0.0, 0.0]
    leader_ref   = np.zeros(nx);  leader_ref[0:3]    = [1.0, 1.0, 1.0]

    t_end = 15.0
    t     = np.arange(0, t_end + Ts, Ts)
    nstep = len(t)

    # ------------------------------------------------------------------ #
    #  Pre-simulate leader trajectory to find intercept point
    # ------------------------------------------------------------------ #
    X_leader_full = np.zeros((nx, nstep))
    x_sim = x_leader_0.copy()
    for k in range(nstep):
        X_leader_full[:, k] = x_sim
        u_sim  = quadcopter_pd_controller(x_sim, leader_ref, Kp, Kd, m, g)
        x_dot  = quadcopter_dynamics(x_sim, u_sim, A, B, m, g)
        x_sim  = x_sim + Ts * x_dot

    max_follower_speed = 1.5   # approximate [m/s]
    intercept_k = nstep - 1   # default to end

    for k in range(nstep):
        dist        = np.linalg.norm(X_leader_full[0:3, k] - x_follower_0[0:3])
        time_needed = dist / max_follower_speed
        time_avail  = k * Ts
        if time_avail >= time_needed:
            intercept_k = k
            break

    intercept_pos  = X_leader_full[0:3, intercept_k]
    intercept_t    = intercept_k * Ts
    print(f"Interception point : [{intercept_pos[0]:.2f}, "
          f"{intercept_pos[1]:.2f}, {intercept_pos[2]:.2f}]")
    print(f"Interception time  : {intercept_t:.2f} s")

    # ------------------------------------------------------------------ #
    #  Main simulation loop
    # ------------------------------------------------------------------ #
    X_leader   = np.zeros((nx, nstep))
    X_follower = np.zeros((nx, nstep))
    U_leader   = np.zeros((nu, nstep))
    U_follower = np.zeros((nu, nstep))

    x_leader   = x_leader_0.copy()
    x_follower = x_follower_0.copy()
    intercepted = False

    print("Running leader–follower simulation … (this may take a while)")
    for k in range(nstep):
        if k % 100 == 0:
            print(f"  step {k}/{nstep}")

        X_leader[:, k]   = x_leader
        X_follower[:, k] = x_follower

        # --- Leader: PD flies to [1,1,1] ---
        u_leader      = quadcopter_pd_controller(x_leader, leader_ref, Kp, Kd, m, g)
        U_leader[:, k] = u_leader
        x_dot_leader  = quadcopter_dynamics(x_leader, u_leader, A, B, m, g)
        x_leader       = x_leader + Ts * x_dot_leader

        # --- Follower: aim at intercept, then lock on ---
        if np.linalg.norm(x_follower[0:3] - x_leader[0:3]) < 0.05:
            intercepted = True

        if not intercepted:
            follower_ref       = np.zeros(nx)
            follower_ref[0:3]  = intercept_pos
        else:
            follower_ref = x_leader.copy()

        u_follower        = mpc.compute(x_follower, follower_ref)
        U_follower[:, k]  = u_follower
        x_dot_follower    = quadcopter_dynamics(x_follower, u_follower, A, B, m, g)
        x_follower         = x_follower + Ts * x_dot_follower

    # ------------------------------------------------------------------ #
    #  Plot 1: Full 3D trajectories
    # ------------------------------------------------------------------ #
    fig1 = plt.figure(figsize=(9, 7))
    ax1  = fig1.add_subplot(111, projection='3d')
    ax1.plot(X_leader[0, :],   X_leader[1, :],   X_leader[2, :],   'b',  lw=1.5, label='Leader (PD)')
    ax1.plot(X_follower[0, :], X_follower[1, :], X_follower[2, :], 'r--', lw=1.5, label='Follower (MPC)')
    ax1.scatter([0], [0], [0], c='b', marker='o', s=80, label='Leader start')
    ax1.scatter([1], [0], [0], c='r', marker='o', s=80, label='Follower start')
    ax1.scatter(*intercept_pos, c='k', marker='*', s=140, label='Intercept point')
    ax1.scatter([1], [1], [1], c='g', marker='*', s=120, label='Leader target')
    ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]'); ax1.set_zlabel('z [m]')
    ax1.set_title('3D Full Trajectories'); ax1.legend(loc='best'); ax1.grid(True)
    ax1.view_init(elev=30, azim=45)
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  Plot 2: 3D trajectory — leader clipped at interception
    # ------------------------------------------------------------------ #
    fig2 = plt.figure(figsize=(9, 7))
    ax2  = fig2.add_subplot(111, projection='3d')
    ax2.plot(X_leader[0, :intercept_k+1],
             X_leader[1, :intercept_k+1],
             X_leader[2, :intercept_k+1], 'b', lw=1.5, label='Leader (clipped)')
    ax2.plot(X_follower[0, :], X_follower[1, :], X_follower[2, :], 'r--', lw=1.5, label='Follower (MPC)')
    ax2.scatter([0], [0], [0], c='b', marker='o', s=80, label='Leader start')
    ax2.scatter([1], [0], [0], c='r', marker='o', s=80, label='Follower start')
    ax2.scatter(*X_leader[0:3, intercept_k], c='k', marker='*', s=140, label='Intercept point')
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]'); ax2.set_zlabel('z [m]')
    ax2.set_title('3D Intercept — Leader hidden after interception')
    ax2.legend(loc='best'); ax2.grid(True)
    ax2.view_init(elev=30, azim=45)
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  Plot 3: Position over time — leader clipped
    # ------------------------------------------------------------------ #
    pos_labels = ['x [m]', 'y [m]', 'z [m]']
    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 8))
    fig3.suptitle('Position — Leader clipped at interception')
    for i in range(3):
        axes3[i].plot(t[:intercept_k+1], X_leader[i, :intercept_k+1],
                      'b', lw=1.5, label='Leader (clipped)')
        axes3[i].plot(t, X_follower[i, :], 'r--', lw=1.5, label='Follower (MPC)')
        axes3[i].axvline(intercept_t, color='k', linestyle='--', lw=1.0, label='Intercept')
        axes3[i].set_xlabel('t [s]'); axes3[i].set_ylabel(pos_labels[i])
        axes3[i].set_title(pos_labels[i]); axes3[i].grid(True)
        axes3[i].legend(loc='best')
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  Plot 4: Separation distance
    # ------------------------------------------------------------------ #
    dist_vec = np.linalg.norm(X_leader[0:3, :] - X_follower[0:3, :], axis=0)
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(t, dist_vec, 'g', lw=1.5)
    ax4.axhline(0.05, color='r', linestyle='--', lw=1.0, label='Intercept threshold (0.05 m)')
    ax4.axvline(intercept_t, color='b', linestyle='--', lw=1.0,
                label=f't = {intercept_t:.2f} s')
    ax4.set_xlabel('t [s]'); ax4.set_ylabel('Distance [m]')
    ax4.set_title('Separation Distance Between Leader and Follower')
    ax4.legend(loc='best'); ax4.grid(True)
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  Plot 5: Control inputs
    # ------------------------------------------------------------------ #
    input_names = ['u1 (thrust)', 'u2 (roll)', 'u3 (pitch)', 'u4 (yaw)']
    fig5, axes5 = plt.subplots(2, 2, figsize=(11, 7))
    fig5.suptitle('Control Inputs — Leader vs Follower')
    for i, ax in enumerate(axes5.flat):
        ax.plot(t, U_leader[i, :],   'b',  lw=1.2, label='Leader')
        ax.plot(t, U_follower[i, :], 'r--', lw=1.2, label='Follower')
        ax.set_xlabel('t [s]'); ax.set_ylabel(input_names[i])
        ax.set_title(input_names[i]); ax.grid(True); ax.legend(loc='best')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    simulate_leader_follower()
