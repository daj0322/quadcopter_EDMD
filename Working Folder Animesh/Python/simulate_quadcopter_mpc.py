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
    N  = 10   # prediction horizon
    Nc = 3    # control horizon

    mpc = LinearMPC(Ad, Bd, N, Nc, Q, R, Rd, u_nominal, y_min, y_max)

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

    # ------------------------------------------------------------------ #
    #  Plot 1: MPC vs Actual — Position
    # ------------------------------------------------------------------ #
    ref_vals    = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pos_labels  = ['x [m]', 'y [m]', 'z [m]']

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('Linear MPC vs Actual Linear Model — Position Tracking')
    for i in range(3):
        axes[i].plot(t, X_mpc[i, :],    'b',  linewidth=1.5, label='MPC internal')
        axes[i].plot(t, X_actual[i, :], 'r--', linewidth=1.5, label='Actual linear model')
        axes[i].axhline(ref_vals[i], color='g', linestyle=':', linewidth=1.2, label='Reference')
        axes[i].set_xlabel('t [s]'); axes[i].set_ylabel(pos_labels[i])
        axes[i].set_title(pos_labels[i]); axes[i].grid(True)
        axes[i].legend(loc='best')
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  Plot 2: All 12 states
    # ------------------------------------------------------------------ #
    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz',
                   'phi', 'theta', 'psi', 'p', 'q', 'r']

    fig2, axes2 = plt.subplots(4, 3, figsize=(14, 10))
    fig2.suptitle('Linear MPC vs Actual Linear Model — All States')
    for i, ax in enumerate(axes2.flat):
        ax.plot(t, X_mpc[i, :],    'b',  linewidth=1.2, label='MPC internal')
        ax.plot(t, X_actual[i, :], 'r--', linewidth=1.2, label='Actual')
        ax.axhline(ref_vals[i], color='g', linestyle=':', linewidth=1.0, label='Reference')
        ax.set_xlabel('t [s]'); ax.set_ylabel(state_names[i])
        ax.set_title(state_names[i]); ax.grid(True)
        ax.legend(loc='best', fontsize=6)
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  Plot 3: Control inputs
    # ------------------------------------------------------------------ #
    input_names = ['u1 (thrust)', 'u2 (roll)', 'u3 (pitch)', 'u4 (yaw)']
    fig3, axes3 = plt.subplots(2, 2, figsize=(10, 6))
    fig3.suptitle('Quadcopter Linear MPC — Control Inputs')
    for i, ax in enumerate(axes3.flat):
        ax.plot(t, U_actual[i, :], 'g', linewidth=1.2)
        ax.set_xlabel('t [s]'); ax.set_ylabel(input_names[i])
        ax.set_title(input_names[i]); ax.grid(True)
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    #  Plot 4: 3D trajectory
    # ------------------------------------------------------------------ #
    fig4 = plt.figure(figsize=(8, 6))
    ax4  = fig4.add_subplot(111, projection='3d')
    ax4.plot(X_mpc[0, :],    X_mpc[1, :],    X_mpc[2, :],    'b',  linewidth=1.5, label='MPC internal')
    ax4.plot(X_actual[0, :], X_actual[1, :], X_actual[2, :], 'r--', linewidth=1.5, label='Actual linear model')
    ax4.scatter([0], [0], [0], c='g', marker='o', s=80, label='Start')
    ax4.scatter([1], [1], [1], c='r', marker='*', s=120, label='Reference [1,1,1]')
    ax4.set_xlabel('x [m]'); ax4.set_ylabel('y [m]'); ax4.set_zlabel('z [m]')
    ax4.set_title('Quadcopter MPC — 3D Trajectory')
    ax4.legend(loc='best')
    ax4.view_init(elev=30, azim=45)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    simulate_quadcopter_mpc()
