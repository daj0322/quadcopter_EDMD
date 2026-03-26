"""
simulate_quadcopter_mpc.py
==========================
Linear MPC for a quadcopter hover model.

Replaces MATLAB's mpc() toolbox with a hand-rolled receding-horizon QP
solved at each time step via scipy.optimize.minimize (SLSQP).

The MPC minimises:
    J = sum_{i=0}^{N-1}  (y_{k+i} - r)' Q (y_{k+i} - r)
      + sum_{i=0}^{Nc-1} (delta_u_{k+i})' R (delta_u_{k+i})
      + sum_{i=0}^{Nc-1} (delta_u_{k+i} - delta_u_{k+i-1})' Rd (...)

subject to output / state constraints.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from quadcopter_linearized_model import quadcopter_linearized_model
from quadcopter_dynamics import quadcopter_dynamics


def straight_line_target(t, p0, v):
    p = p0 + v * t
    return p, v

# ======================================================================
#  Tiny MPC helper
# ======================================================================
class LinearMPC:
    """
    Receding-horizon linear MPC for  x_{k+1} = Ad x_k + Bd u_k
    """

    def __init__(self, Ad, Bd, N, Nc, Q, R, Rd,
                 u_nominal,
                 y_min=None, y_max=None):
        self.Ad = Ad
        self.Bd = Bd
        self.nx = Ad.shape[0]
        self.nu = Bd.shape[1]
        self.N  = N
        self.Nc = Nc
        self.Q  = Q
        self.R  = R
        self.Rd = Rd
        self.u_nominal = u_nominal          # equilibrium input
        self.y_min = y_min if y_min is not None else -np.inf * np.ones(self.nx)
        self.y_max = y_max if y_max is not None else  np.inf * np.ones(self.nx)

        # warm-start storage
        self._u_prev = np.zeros(self.nu * Nc)

    def _predict(self, x0, du_flat):
        """Roll out N-step prediction; du_flat holds Nc*nu decision variables."""
        Ad, Bd = self.Ad, self.Bd
        nx, nu, N, Nc = self.nx, self.nu, self.N, self.Nc

        # Build full u sequence (hold last du after Nc steps)
        du = du_flat.reshape(Nc, nu)
        X  = np.zeros((N + 1, nx))
        X[0] = x0
        u_k = self.u_nominal.copy()

        for i in range(N):
            if i < Nc:
                u_k = u_k + du[i]
            X[i + 1] = Ad @ X[i] + Bd @ (u_k - self.u_nominal)
        return X   # (N+1, nx)

    def _cost(self, du_flat, x0, x_ref):
        X  = self._predict(x0, du_flat)
        Q, R, Rd = self.Q, self.R, self.Rd
        nu, Nc   = self.nu, self.Nc
        du = du_flat.reshape(Nc, nu)

        J = 0.0
        for i in range(1, self.N + 1):
            e  = X[i] - x_ref
            J += e @ Q @ e

        for i in range(Nc):
            J += du[i] @ R @ du[i]

        for i in range(1, Nc):
            diff  = du[i] - du[i - 1]
            J    += diff @ Rd @ diff

        return J

    def _constraints(self, du_flat, x0):
        X = self._predict(x0, du_flat)
        cons = []
        for i in range(1, self.N + 1):
            cons.append(X[i] - self.y_min)   # >= 0
            cons.append(self.y_max - X[i])    # >= 0
        return np.concatenate(cons)

    def compute(self, x0, x_ref):
        """
        Solve one MPC step and return the first optimal u.
        """
        x_ref = np.asarray(x_ref, dtype=float).ravel()

        constraints = {'type': 'ineq',
                       'fun': self._constraints,
                       'args': (x0,)}

        res = minimize(self._cost,
                       self._u_prev,
                       args=(x0, x_ref),
                       method='SLSQP',
                       constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-6})

        du_opt        = res.x.reshape(self.Nc, self.nu)
        self._u_prev  = res.x          # warm-start next step
        u_opt         = self.u_nominal + du_opt[0]
        return u_opt


# ======================================================================
#  Main simulation
# ======================================================================
def simulate_quadcopter_mpc():

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
    #  MPC weights  (match MATLAB mpcobj.Weights)
    # ------------------------------------------------------------------ #
    Q_diag  = np.array([10, 10, 10, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1])
    R_diag  = np.array([0.1, 0.1, 0.1, 0.1])
    Rd_diag = np.array([0.05, 0.05, 0.05, 0.05])

    Q  = np.diag(Q_diag)
    R  = np.diag(R_diag)
    Rd = np.diag(Rd_diag)

    # ------------------------------------------------------------------ #
    #  Output (state) constraints
    # ------------------------------------------------------------------ #
    y_min = np.array([-5, -5, -0.5, -3, -3, -3, -0.3, -0.3, -np.pi, -2, -2, -2])
    y_max = np.array([ 5,  5,  5.0,  3,  3,  3,  0.3,  0.3,  np.pi,  2,  2,  2])

    # ------------------------------------------------------------------ #
    #  Create MPC object
    # ------------------------------------------------------------------ #
    u_nominal = np.array([m * g, 0.0, 0.0, 0.0])
    N  = 100   # prediction horizon
    Nc = 3    # control horizon

    mpc = LinearMPC(Ad, Bd, N, Nc, Q, R, Rd, u_nominal, y_min, y_max)

    print("\n========== MPC DEBUG ==========")
    print("Ad shape:", Ad.shape)
    print("Bd shape:", Bd.shape)
    print("nx:", nx)
    print("nu:", nu)
    print("Horizon N:", N)
    print("Control horizon Nc:", Nc)
    print("Q diag:", np.diag(Q))
    print("R diag:", np.diag(R))
    print("Rd diag:", np.diag(Rd))
    print("y_min:", y_min)
    print("y_max:", y_max)
    print("================================")

    '''
    # ------------------------------------------------------------------ #
    #  Reference
    # ------------------------------------------------------------------ #
    x_ref       = np.zeros(nx)
    x_ref[0:3]  = [1.0, 1.0, 1.0]
    # ------------------------------------------------------------------ #
    #  Simulation
    # ------------------------------------------------------------------ #
    t_end = 10.0
    t     = np.arange(0, t_end + Ts, Ts)
    nstep = len(t)
    
    X_mpc    = np.zeros((nx, nstep))   # MPC internal model trajectory
    X_actual = np.zeros((nx, nstep))   # ground-truth linear model
    U_actual = np.zeros((nu, nstep))

    x_mpc    = np.zeros(nx)
    x_actual = np.zeros(nx)


    print("Running MPC simulation … (this may take a minute)")
    for k in range(nstep):
        if k % 100 == 0:
            print(f"  step {k}/{nstep}")

        X_mpc[:, k]    = x_mpc
        X_actual[:, k] = x_actual

        # MPC computes u based on its internal model
        u              = mpc.compute(x_mpc, x_ref)
        U_actual[:, k] = u

        # MPC internal model step
        x_mpc = Ad @ x_mpc + Bd @ (u - u_nominal)

        # Ground-truth propagation
        x_dot    = quadcopter_dynamics(x_actual, u, A, B, m, g)
        x_actual = x_actual + Ts * x_dot
    '''
    # ------------------------------------------------------------------ #
    #  Straight-line target setup
    # ------------------------------------------------------------------ #
    target_p0 = np.array([2.0, 0.0, 1.0])  # target initial position
    target_v = np.array([0.5, 0.0, 0.0])  # constant target velocity
    head_start = 2.0  # [s]
    capture_radius = 0.5  # [m]

    # ------------------------------------------------------------------ #
    #  Simulation
    # ------------------------------------------------------------------ #
    t_end = 20.0
    t = np.arange(0, t_end + Ts, Ts)
    nstep = len(t)


    print("\n========== SCENARIO DEBUG ==========")
    print("Ts:", Ts)
    print("t_end:", t_end)
    print("target_p0:", target_p0)
    print("target_v:", target_v)
    print("head_start:", head_start)
    print("capture_radius:", capture_radius)
    print("u_nominal:", u_nominal)
    print("====================================")

    X_mpc = np.zeros((nx, nstep))  # MPC internal model trajectory
    X_actual = np.zeros((nx, nstep))  # ground-truth linear model
    U_actual = np.zeros((nu, nstep))
    X_target = np.zeros((nx, nstep))  # target state history

    x_mpc = np.zeros(nx)
    x_actual = np.zeros(nx)
    captured = False
    capture_time = None
    min_dist = np.inf
    stop_idx = nstep - 1

    sep_hist = np.zeros(nstep)
    model_mismatch_hist = np.zeros(nstep)
    u_norm_hist = np.zeros(nstep)
    speed_i_hist = np.zeros(nstep)
    speed_t_hist = np.zeros(nstep)

    print("Running straight-line target interception with linear MPC ...")
    for k in range(nstep):
        if k % 100 == 0:
            print(f"  step {k}/{nstep}")

        tk = t[k]

        # target has already moved for 'head_start' seconds
        p_t, v_t = straight_line_target(tk + head_start, target_p0, target_v)

        x_target = np.zeros(nx)
        x_target[0:3] = p_t
        x_target[3:6] = v_t

        X_target[:, k] = x_target
        X_mpc[:, k] = x_mpc
        X_actual[:, k] = x_actual

        # follower MPC tracks current target state
        u = mpc.compute(x_mpc, x_target)
        U_actual[:, k] = u

        # MPC internal model step
        x_mpc = Ad @ x_mpc + Bd @ (u - u_nominal)

        # Ground-truth propagation
        x_dot = quadcopter_dynamics(x_actual, u, A, B, m, g)
        x_actual = x_actual + Ts * x_dot

        p_i = x_actual[0:3]
        v_i = x_actual[3:6]

        sep = np.linalg.norm(p_i - p_t)
        sep_hist[k] = sep
        model_mismatch_hist[k] = np.linalg.norm(x_actual - x_mpc)
        u_norm_hist[k] = np.linalg.norm(u)
        speed_i_hist[k] = np.linalg.norm(v_i)
        speed_t_hist[k] = np.linalg.norm(v_t)

        if k in [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700, 900, nstep - 1]:
            print(f"\n--- STEP {k}/{nstep - 1} ---")
            print("t =", tk)
            print("target pos =", p_t)
            print("target vel =", v_t)
            print("interceptor pos =", p_i)
            print("interceptor vel =", v_i)
            print("reference pos =", x_target[0:3])
            print("reference vel =", x_target[3:6])
            print("control u =", u)
            print("||u|| =", np.linalg.norm(u))
            print("separation =", sep)
            print("model mismatch ||x_actual - x_mpc|| =", model_mismatch_hist[k])
            print("u relative to nominal =", u - u_nominal)

        # capture check
        dist = np.linalg.norm(x_actual[0:3] - p_t)
        min_dist = min(min_dist, dist)

        if (not captured) and (dist <= capture_radius):
            captured = True
            capture_time = tk
            stop_idx = k
            print(f"Capture achieved at t = {tk:.2f} s")
            print("Capture separation:", dist)
            break


    t_used = t[:stop_idx + 1]
    X_actual_used = X_actual[:, :stop_idx + 1]
    X_target_used = X_target[:, :stop_idx + 1]
    U_actual_used = U_actual[:, :stop_idx + 1]
    sep_hist_used = sep_hist[:stop_idx + 1]
    model_mismatch_used = model_mismatch_hist[:stop_idx + 1]
    u_norm_used = u_norm_hist[:stop_idx + 1]
    speed_i_used = speed_i_hist[:stop_idx + 1]
    speed_t_used = speed_t_hist[:stop_idx + 1]

    print("\n========== CONTROL SUMMARY ==========")
    print("u min per channel:", np.min(U_actual_used, axis=1))
    print("u max per channel:", np.max(U_actual_used, axis=1))
    print("max ||u||:", np.max(u_norm_used))
    print("=====================================")
    print("Captured:", captured)
    print("Capture time:", capture_time)
    print("Minimum distance:", min_dist)
    print("Final interceptor position:", X_actual_used[0:3, -1])
    print("Final target position:", X_target_used[0:3, -1])
    print("\n========== SEPARATION SUMMARY ==========")
    print("initial separation:", sep_hist_used[0])
    print("final separation:", sep_hist_used[-1])
    print("minimum separation:", np.min(sep_hist_used))
    print("time of minimum separation:", t_used[np.argmin(sep_hist_used)])
    print("========================================")
    print("\n========== MODEL MISMATCH SUMMARY ==========")
    print("initial mismatch:", model_mismatch_used[0])
    print("final mismatch:", model_mismatch_used[-1])
    print("max mismatch:", np.max(model_mismatch_used))
    print("time of max mismatch:", t_used[np.argmax(model_mismatch_used)])
    print("============================================")

    # ------------------------------------------------------------------ #
    #  Plot 1: 3D trajectories
    # ------------------------------------------------------------------ #
    fig1 = plt.figure(figsize=(9, 7))
    ax1 = fig1.add_subplot(111, projection='3d')

    ax1.plot(X_target_used[0, :], X_target_used[1, :], X_target_used[2, :],
             'b', lw=1.5, label='Target')
    ax1.plot(X_actual_used[0, :], X_actual_used[1, :], X_actual_used[2, :],
             'r--', lw=1.5, label='Interceptor (linear MPC)')

    # Interceptor start
    ax1.scatter([0], [0], [0],
                c='r', marker='o', s=80, label='Interceptor start')

    # Target initial position before head start
    ax1.scatter([target_p0[0]], [target_p0[1]], [target_p0[2]],
                c='c', marker='^', s=90, label='Target initial position')

    # Target position when chase begins
    ax1.scatter([X_target_used[0, 0]], [X_target_used[1, 0]], [X_target_used[2, 0]],
                c='b', marker='o', s=90, label='Target position at chase start')

    # Interceptor position at capture
    if captured:
        ax1.scatter([X_actual_used[0, -1]], [X_actual_used[1, -1]], [X_actual_used[2, -1]],
                    c='g', marker='*', s=160, label='Intercept position')

    # Target position at capture
    if captured:
        ax1.scatter([X_target_used[0, -1]], [X_target_used[1, -1]], [X_target_used[2, -1]],
                    c='k', marker='x', s=100, label='Target at intercept')

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_zlabel('z [m]')
    ax1.set_title('Straight-line target interception')
    ax1.legend(loc='best')
    ax1.grid(True)
    ax1.view_init(elev=30, azim=45)
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  Plot 2: separation distance
    # ------------------------------------------------------------------ #
    sep = np.linalg.norm(X_actual_used[0:3, :] - X_target_used[0:3, :], axis=0)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t_used, sep, 'k', lw=1.5, label='Separation distance')
    ax2.axhline(capture_radius, color='r', linestyle='--', lw=1.2, label='Capture radius')
    if capture_time is not None:
        ax2.axvline(capture_time, color='g', linestyle='--', lw=1.2, label='Capture time')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('Distance [m]')
    ax2.set_title('Interceptor-target separation')
    ax2.grid(True)
    ax2.legend(loc='best')
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  Plot 3: control inputs
    # ------------------------------------------------------------------ #
    input_names = ['u1 (thrust)', 'u2 (roll)', 'u3 (pitch)', 'u4 (yaw)']
    fig3, axes3 = plt.subplots(2, 2, figsize=(10, 6))
    fig3.suptitle('Linear MPC Control Inputs')

    for i, ax in enumerate(axes3.flat):
        ax.plot(t_used, U_actual_used[i, :], 'g', linewidth=1.2)
        ax.set_xlabel('t [s]')
        ax.set_ylabel(input_names[i])
        ax.set_title(input_names[i])
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(t_used, sep_hist_used, 'b', lw=1.5, label='Separation')
    ax4.axhline(capture_radius, color='r', linestyle='--', label='Capture radius')
    if capture_time is not None:
        ax4.axvline(capture_time, color='g', linestyle='--', label='Capture time')
    ax4.set_xlabel('t [s]')
    ax4.set_ylabel('Distance [m]')
    ax4.set_title('Separation vs time')
    ax4.grid(True)
    ax4.legend()
    plt.tight_layout()
    plt.show()

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(t_used, model_mismatch_used, 'm', lw=1.5)
    ax5.set_xlabel('t [s]')
    ax5.set_ylabel(r'$||x_{actual} - x_{mpc}||$')
    ax5.set_title('Internal MPC state vs actual state mismatch')
    ax5.grid(True)
    plt.tight_layout()
    plt.show()

    fig6, ax6 = plt.subplots(figsize=(10, 4))
    ax6.plot(t_used, speed_i_used, label='Interceptor speed')
    ax6.plot(t_used, speed_t_used, label='Target speed')
    ax6.set_xlabel('t [s]')
    ax6.set_ylabel('Speed [m/s]')
    ax6.set_title('Interceptor and target speeds')
    ax6.grid(True)
    ax6.legend()
    plt.tight_layout()

    plt.show()

    fig7, ax7 = plt.subplots(figsize=(8, 5))

    ax7.plot(X_target_used[0, :], X_target_used[2, :], 'b', lw=1.5, label='Target')
    ax7.plot(X_actual_used[0, :], X_actual_used[2, :], 'r--', lw=1.5, label='Interceptor')

    # Interceptor start
    ax7.scatter([0], [0], c='r', marker='o', s=80, label='Interceptor start')

    # Target initial position
    ax7.scatter([target_p0[0]], [target_p0[2]], c='c', marker='^', s=90, label='Target initial position')

    # Target at chase start
    ax7.scatter([X_target_used[0, 0]], [X_target_used[2, 0]],
                c='b', marker='o', s=90, label='Target position at chase start')

    # Interceptor at capture
    if captured:
        ax7.scatter([X_actual_used[0, -1]], [X_actual_used[2, -1]],
                    c='g', marker='*', s=160, label='Intercept position')

    # Target at capture
    if captured:
        ax7.scatter([X_target_used[0, -1]], [X_target_used[2, -1]],
                    c='k', marker='x', s=100, label='Target at intercept')

    ax7.set_xlabel('x [m]')
    ax7.set_ylabel('z [m]')
    ax7.set_title('Straight-line interception (x-z view)')
    ax7.grid(True)
    ax7.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    simulate_quadcopter_mpc()
