import itertools
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import osqp
from scipy.linalg import expm

from Simulation import quad_sim
from Closed_loop import ClosedLoopQuad

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR         = Path(__file__).resolve().parent
EDMDC_MODEL_FILE   = "edmdc_model.pkl"
DATA_FILE          = "runs_mixed_n300.pkl"
TEST_RUN_IDX       = 59

# EDMD-MPC horizon
N_EDMD  = 50
NC_EDMD = 25

# Linear MPC horizon
N_LIN   = 50
NC_LIN  = 25

# Q penalizes physical states in SCALED space for EDMD
# [x, y, z, vx, vy, vz, phi, theta, p, q]
Q_DIAG_EDMD = np.array([
    200000.0, 200000.0, 200000.0,
       500.0,    500.0,    500.0,
         0.0,      0.0,
         0.0,      0.0,
], dtype=float)

R_DIAG_EDMD  = np.array([0.0001, 0.1, 0.1], dtype=float)
RD_DIAG_EDMD = np.array([1e-05, 0.01, 0.01], dtype=float)

DU_MIN_EDMD = np.array([-5.0, -3.5, -3.5], dtype=float)
DU_MAX_EDMD = np.array([ 5.0,  3.5,  3.5], dtype=float)

# Linear MPC weights in physical coordinates
# [x, y, z, vx, vy, vz, phi, theta, p, q]
Q_DIAG_LIN = np.array([
    200.0, 200.0, 200.0,
     10.0,  10.0,  10.0,
      0.0,   0.0,
      0.0,   0.0,
], dtype=float)

R_DIAG_LIN  = np.array([0.01, 0.1, 0.1], dtype=float)
RD_DIAG_LIN = np.array([0.001, 0.01, 0.01], dtype=float)

DU_MIN_LIN = np.array([-3.0, -0.4, -0.4], dtype=float)
DU_MAX_LIN = np.array([ 3.0,  0.4,  0.4], dtype=float)


# ============================================================
# LOAD HELPERS
# ============================================================
def load_edmdc_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["t"], data["states"], data["U"], data["ref_traj_list"]


# ============================================================
# OBSERVABLES — must match EDMDc_training.py EXACTLY
# ============================================================
# State indices (after trimming 12→10):
#   0:x  1:y  2:z  3:vx  4:vy  5:vz  6:phi  7:theta  8:p  9:q
#
# Observables (21 total):
#   [0-9]   10 linear states
#   [10-13] sin(phi), cos(phi), sin(theta), cos(theta)
#   [14-17] phi*p, theta*q, vx*phi, vy*theta
#   [18]    v_sq = vx^2 + vy^2 + vz^2
#   [19]    omega_sq = p^2 + q^2
#   [20]    bias

def observables(x_std, scaler):
    x = np.asarray(x_std).flatten()
    assert len(x) == 10, f"Expected 10-state vector, got {len(x)}"

    obs = list(x)

    phi_rad   = x[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_rad = x[7] * scaler.scale_[7] + scaler.mean_[7]

    obs.append(np.sin(phi_rad))
    obs.append(np.cos(phi_rad))
    obs.append(np.sin(theta_rad))
    obs.append(np.cos(theta_rad))

    obs.append(x[6] * x[8])   # phi * p
    obs.append(x[7] * x[9])   # theta * q
    obs.append(x[3] * x[6])   # vx * phi
    obs.append(x[4] * x[7])   # vy * theta

    obs.append(x[3]**2 + x[4]**2 + x[5]**2)
    obs.append(x[8]**2 + x[9]**2)

    obs.append(1.0)
    return np.asarray(obs, dtype=float)

def lifted_state_from_x(x10, scaler):
    x_std = scaler.transform(x10.reshape(1, -1)).flatten()
    return observables(x_std, scaler)

def drop_to_10state(x12):
    return x12[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]


# ============================================================
# LINEARIZED OUTER-LEVEL MODEL
# Inputs: [thrust, phi_des, theta_des]
# State:  [x, y, z, vx, vy, vz, phi, theta, p, q]
# ============================================================
def build_linear_outer_model(dt, m, g, tau_att=0.15, tau_rate=0.08):
    """
    Simple outer-level commanded-response model:
      xdot  = vx
      ydot  = vy
      zdot  = vz
      vxdot = g * theta
      vydot = -g * phi
      vzdot = (u1 - mg)/m
      phidot = p
      thetadot = q
      pdot = -(1/tau_rate)p + (1/(tau_rate*tau_att))(phi_des - phi)
      qdot = -(1/tau_rate)q + (1/(tau_rate*tau_att))(theta_des - theta)

    Control is delta-u around hover [delta_thrust, phi_des, theta_des]
    """
    nx, nu = 10, 3
    Ac = np.zeros((nx, nx))
    Bc = np.zeros((nx, nu))

    # kinematics
    Ac[0, 3] = 1.0
    Ac[1, 4] = 1.0
    Ac[2, 5] = 1.0

    # translational accel
    Ac[3, 7] = g         # vxdot from theta
    Ac[4, 6] = -g        # vydot from phi
    Bc[5, 0] = 1.0 / m   # vzdot from delta thrust

    # attitude/rate chain
    Ac[6, 8] = 1.0
    Ac[7, 9] = 1.0

    Ac[8, 6] = -(1.0 / (tau_rate * tau_att))
    Ac[8, 8] = -(1.0 / tau_rate)
    Bc[8, 1] =  (1.0 / (tau_rate * tau_att))

    Ac[9, 7] = -(1.0 / (tau_rate * tau_att))
    Ac[9, 9] = -(1.0 / tau_rate)
    Bc[9, 2] =  (1.0 / (tau_rate * tau_att))

    # discretize
    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = Ac
    M[:nx, nx:] = Bc
    Md = expm(M * dt)
    Ad = Md[:nx, :nx]
    Bd = Md[:nx, nx:]

    return Ad, Bd


# ============================================================
# EDMD-MPC QP
# ============================================================
class EDMDcMPC_QP:
    def __init__(self, A, B, Cz, N, NC, Q, R, Rd,
                 u_scaler, du_min, du_max, u_nominal_raw):
        self.A  = np.asarray(A, dtype=float)
        self.B  = np.asarray(B, dtype=float)
        self.Cz = np.asarray(Cz, dtype=float)
        self.N  = int(N)
        self.NC = int(NC)
        self.Q  = np.asarray(Q,  dtype=float)
        self.R  = np.asarray(R,  dtype=float)
        self.Rd = np.asarray(Rd, dtype=float)

        self.u_scaler      = u_scaler
        self.u_nom_raw     = np.asarray(u_nominal_raw, dtype=float)
        self.u_nom_scaled  = u_scaler.transform(
            self.u_nom_raw.reshape(1, -1)).flatten()

        self.du_min = np.asarray(du_min, dtype=float)
        self.du_max = np.asarray(du_max, dtype=float)

        self.nz   = self.A.shape[0]
        self.nu   = self.B.shape[1]
        self.nx   = self.Cz.shape[0]
        self.nvar = self.NC * self.nu

        self._du_prev = np.zeros(self.nvar)

        self.Sz, self.Su = self._build_prediction_matrices()

        Su_dense = self.Su.toarray()
        Su_phys  = np.zeros((self.N * self.nx, self.nvar))
        for i in range(self.N):
            for j in range(self.NC):
                Su_phys[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = \
                    self.Cz @ Su_dense[i*self.nz:(i+1)*self.nz, j*self.nu:(j+1)*self.nu]
        self.Su_phys = sp.csc_matrix(Su_phys)

        self.Qbar = sp.block_diag(
            [sp.csc_matrix(self.Q) for _ in range(self.N)], format="csc")
        self.Rbar = sp.block_diag(
            [sp.csc_matrix(self.R) for _ in range(self.NC)], format="csc")
        self.D    = self._build_difference_matrix()
        self.Rdbar = (
            sp.block_diag([sp.csc_matrix(self.Rd)
                           for _ in range(self.NC - 1)], format="csc")
            if self.NC > 1 else None
        )

        P     = self._build_hessian()
        Aineq = sp.eye(self.nvar, format="csc")
        l     = np.tile(self.du_min, self.NC)
        u     = np.tile(self.du_max, self.NC)

        self.prob = osqp.OSQP()
        self.prob.setup(P=P, q=np.zeros(self.nvar),
                        A=Aineq, l=l, u=u,
                        warm_start=True, verbose=False, polish=False)

    def _build_prediction_matrices(self):
        Sz = np.zeros((self.N * self.nz, self.nz))
        Su = np.zeros((self.N * self.nz, self.NC * self.nu))
        A_pow = [np.eye(self.nz)]
        for _ in range(self.N):
            A_pow.append(A_pow[-1] @ self.A)
        for i in range(self.N):
            Sz[i*self.nz:(i+1)*self.nz, :] = A_pow[i+1]
            for j in range(min(i+1, self.NC)):
                Su[i*self.nz:(i+1)*self.nz, j*self.nu:(j+1)*self.nu] = \
                    A_pow[i-j] @ self.B
        return sp.csc_matrix(Sz), sp.csc_matrix(Su)

    def _build_difference_matrix(self):
        if self.NC <= 1:
            return None
        rows, cols, vals = [], [], []
        for k in range(self.NC - 1):
            for j in range(self.nu):
                r = k * self.nu + j
                rows.extend([r, r])
                cols.extend([k*self.nu+j, (k+1)*self.nu+j])
                vals.extend([-1.0, 1.0])
        return sp.coo_matrix(
            (vals, (rows, cols)),
            shape=((self.NC-1)*self.nu, self.NC*self.nu)).tocsc()

    def _build_hessian(self):
        P = self.Su_phys.T @ self.Qbar @ self.Su_phys + self.Rbar
        if self.D is not None and self.Rdbar is not None:
            P = P + self.D.T @ self.Rdbar @ self.D
        return (0.5 * (P + P.T)).tocsc()

    def _build_q(self, z0, x_ref_std_horizon):
        z_free = self.Sz @ z0
        x_free = np.array([
            self.Cz @ z_free[i*self.nz:(i+1)*self.nz]
            for i in range(self.N)
        ]).reshape(-1)
        x_ref = x_ref_std_horizon.reshape(-1)
        return np.asarray(
            self.Su_phys.T @ (self.Qbar @ (x_free - x_ref))
        ).reshape(-1)

    def compute(self, z0, x_ref_std_horizon):
        q = self._build_q(z0, x_ref_std_horizon)
        self.prob.update(q=q)
        self.prob.warm_start(x=self._du_prev)
        res = self.prob.solve()

        if res.info.status not in ("solved", "solved inaccurate"):
            print(f"Warning OSQP EDMD: {res.info.status}")
            du0 = self._du_prev[:self.nu]
        else:
            du_opt = np.asarray(res.x).reshape(-1)
            self._du_prev = du_opt.copy()
            du0 = du_opt[:self.nu]

        u0_scaled = self.u_nom_scaled + du0
        u0_raw    = self.u_scaler.inverse_transform(
            u0_scaled.reshape(1, -1)).flatten()
        return u0_raw


# ============================================================
# LINEAR MPC QP
# ============================================================
class LinearMPC_QP:
    def __init__(self, A, B, N, NC, Q, R, Rd, du_min, du_max):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.N = int(N)
        self.NC = int(NC)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.Rd = np.asarray(Rd, dtype=float)

        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.nvar = self.NC * self.nu

        self.du_min = np.asarray(du_min, dtype=float)
        self.du_max = np.asarray(du_max, dtype=float)

        self._du_prev = np.zeros(self.nvar)

        self.Sx, self.Su = self._build_prediction_matrices()

        self.Qbar = sp.block_diag(
            [sp.csc_matrix(self.Q) for _ in range(self.N)], format="csc")
        self.Rbar = sp.block_diag(
            [sp.csc_matrix(self.R) for _ in range(self.NC)], format="csc")
        self.D = self._build_difference_matrix()
        self.Rdbar = (
            sp.block_diag([sp.csc_matrix(self.Rd)
                           for _ in range(self.NC - 1)], format="csc")
            if self.NC > 1 else None
        )

        P = self._build_hessian()
        Aineq = sp.eye(self.nvar, format="csc")
        l = np.tile(self.du_min, self.NC)
        u = np.tile(self.du_max, self.NC)

        self.prob = osqp.OSQP()
        self.prob.setup(P=P, q=np.zeros(self.nvar),
                        A=Aineq, l=l, u=u,
                        warm_start=True, verbose=False, polish=False)

    def _build_prediction_matrices(self):
        Sx = np.zeros((self.N * self.nx, self.nx))
        Su = np.zeros((self.N * self.nx, self.NC * self.nu))
        A_pow = [np.eye(self.nx)]
        for _ in range(self.N):
            A_pow.append(A_pow[-1] @ self.A)
        for i in range(self.N):
            Sx[i*self.nx:(i+1)*self.nx, :] = A_pow[i+1]
            for j in range(min(i+1, self.NC)):
                Su[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = \
                    A_pow[i-j] @ self.B
        return sp.csc_matrix(Sx), sp.csc_matrix(Su)

    def _build_difference_matrix(self):
        if self.NC <= 1:
            return None
        rows, cols, vals = [], [], []
        for k in range(self.NC - 1):
            for j in range(self.nu):
                r = k * self.nu + j
                rows.extend([r, r])
                cols.extend([k*self.nu+j, (k+1)*self.nu+j])
                vals.extend([-1.0, 1.0])
        return sp.coo_matrix(
            (vals, (rows, cols)),
            shape=((self.NC-1)*self.nu, self.NC*self.nu)).tocsc()

    def _build_hessian(self):
        P = self.Su.T @ self.Qbar @ self.Su + self.Rbar
        if self.D is not None and self.Rdbar is not None:
            P = P + self.D.T @ self.Rdbar @ self.D
        return (0.5 * (P + P.T)).tocsc()

    def _build_q(self, x0, x_ref_horizon):
        x_free = self.Sx @ x0
        x_ref = x_ref_horizon.reshape(-1)
        return np.asarray(
            self.Su.T @ (self.Qbar @ (x_free - x_ref))
        ).reshape(-1)

    def compute(self, x0, x_ref_horizon):
        q = self._build_q(x0, x_ref_horizon)
        self.prob.update(q=q)
        self.prob.warm_start(x=self._du_prev)
        res = self.prob.solve()

        if res.info.status not in ("solved", "solved inaccurate"):
            print(f"Warning OSQP LIN: {res.info.status}")
            du0 = self._du_prev[:self.nu]
        else:
            du_opt = np.asarray(res.x).reshape(-1)
            self._du_prev = du_opt.copy()
            du0 = du_opt[:self.nu]

        return du0  # delta-u


# ============================================================
# REFERENCE HELPERS
# ============================================================
def extract_ref_xyz(ref_traj):
    return np.array([wp["pos"][:3] for wp in ref_traj], dtype=float)

def precompute_ref_std(ref_traj, scaler, n_states=10):
    X_ref = np.zeros((len(ref_traj), n_states))
    for k in range(len(ref_traj)):
        X_ref[k, 0:3] = ref_traj[k]["pos"][:3]
        X_ref[k, 3:6] = ref_traj[k]["vel"][:3]
    return scaler.transform(X_ref)

def precompute_ref_lin(ref_traj, n_states=10):
    X_ref = np.zeros((len(ref_traj), n_states))
    for k in range(len(ref_traj)):
        X_ref[k, 0:3] = ref_traj[k]["pos"][:3]
        X_ref[k, 3:6] = ref_traj[k]["vel"][:3]
    return X_ref

def build_ref_horizon(ref_array, k, N):
    T = ref_array.shape[0]
    h = np.zeros((N, ref_array.shape[1]))
    for i in range(N):
        h[i] = ref_array[min(k + i, T - 1)]
    return h


# ============================================================
# METRICS
# ============================================================
def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))


# ============================================================
# MAIN
# ============================================================
def main():
    # --- load model ---
    model    = load_edmdc_model(SCRIPT_DIR / EDMDC_MODEL_FILE)
    A_edmd   = model["A"]
    B_edmd   = model["B"]
    scaler   = model["scaler"]
    u_scaler = model["u_scaler"]
    dt       = model["dt"]
    n_obs    = model["n_obs"]

    print(f"Model dt: {dt}")
    print(f"A: {A_edmd.shape}  B: {B_edmd.shape}  n_obs: {n_obs}")
    print(f"u_scaler mean: {u_scaler.mean_}")
    print(f"u_scaler scale: {u_scaler.scale_}")

    labels10 = ['x','y','z','vx','vy','vz','phi','theta','p','q']
    print("\nB row norms (physical states):")
    for i, lbl in enumerate(labels10):
        print(f"  {lbl:>6s}: {np.linalg.norm(B_edmd[i,:]):.6f}")

    # --- load test run ---
    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(
        SCRIPT_DIR / DATA_FILE)

    if states_all.shape[2] == 12:
        states_all = states_all[:, :, [0,1,2,3,4,5,6,7,9,10]]

    if U_all.shape[2] == 4:
        U_all = U_all[:, :, :3]

    sim_dt = t_all[0, 1] - t_all[0, 0]
    step   = int(round(dt / sim_dt))
    idx    = np.arange(0, t_all.shape[1], step)
    t_all      = t_all[:, idx]
    states_all = states_all[:, idx, :]
    U_all      = U_all[:, idx, :]
    ref_traj_list = [r[::step] for r in ref_traj_list]

    run_idx = TEST_RUN_IDX % states_all.shape[0]
    t_ref   = t_all[run_idx]
    X_true  = states_all[run_idx]
    U_saved = U_all[run_idx]
    ref_xyz = extract_ref_xyz(ref_traj_list[run_idx])
    T       = min(len(t_ref), X_true.shape[0], ref_xyz.shape[0])
    t_ref   = t_ref[:T]
    X_true  = X_true[:T]
    U_saved = U_saved[:T]
    ref_xyz = ref_xyz[:T]

    print(f"\nTest run: {run_idx}  T={T}  duration={t_ref[-1]:.1f}s")

    # --- sim + nominal input ---
    sim = quad_sim()
    u_nominal = np.array([sim.q_mass * sim.g, 0.0, 0.0], dtype=float)

    # --- Cz: extract 10 physical states from 21-dim observable ---
    Cz = np.zeros((10, n_obs))
    Cz[:10, :10] = np.eye(10)

    # --- references ---
    ref_std = precompute_ref_std(ref_traj_list[run_idx][:T], scaler, n_states=10)
    ref_lin = precompute_ref_lin(ref_traj_list[run_idx][:T], n_states=10)

    # --- build EDMD MPC ---
    Q_edmd  = np.diag(Q_DIAG_EDMD)
    R_edmd  = np.diag(R_DIAG_EDMD)
    Rd_edmd = np.diag(RD_DIAG_EDMD)

    mpc_edmd = EDMDcMPC_QP(
        A=A_edmd, B=B_edmd, Cz=Cz,
        N=N_EDMD, NC=NC_EDMD,
        Q=Q_edmd, R=R_edmd, Rd=Rd_edmd,
        u_scaler=u_scaler,
        du_min=DU_MIN_EDMD,
        du_max=DU_MAX_EDMD,
        u_nominal_raw=u_nominal,
    )

    # --- build linear MPC ---
    A_lin, B_lin = build_linear_outer_model(
        dt=dt, m=sim.q_mass, g=sim.g, tau_att=0.15, tau_rate=0.08
    )

    Q_lin  = np.diag(Q_DIAG_LIN)
    R_lin  = np.diag(R_DIAG_LIN)
    Rd_lin = np.diag(RD_DIAG_LIN)

    mpc_lin = LinearMPC_QP(
        A=A_lin, B=B_lin,
        N=N_LIN, NC=NC_LIN,
        Q=Q_lin, R=R_lin, Rd=Rd_lin,
        du_min=DU_MIN_LIN, du_max=DU_MAX_LIN
    )

    # --------------------------------------------------------
    # EDMD-MPC closed-loop
    # --------------------------------------------------------
    X_mpc_edmd = np.zeros((T, 10))
    U_mpc_edmd = np.zeros((T, 3))

    x_current_12 = np.zeros(12)
    x10_init     = X_true[0]
    x_current_12[0:6]  = x10_init[0:6]
    x_current_12[6:8]  = x10_init[6:8]
    x_current_12[9:11] = x10_init[8:10]

    X_mpc_edmd[0] = drop_to_10state(x_current_12)
    solve_times_edmd = []

    print("\nRunning EDMD-MPC closed-loop...")
    for k in range(T - 1):
        if k % 50 == 0:
            print(f"  EDMD step {k}/{T-1}  pos=({X_mpc_edmd[k,0]:.2f}, "
                  f"{X_mpc_edmd[k,1]:.2f}, {X_mpc_edmd[k,2]:.2f})")

        x10 = drop_to_10state(x_current_12)
        z_k = lifted_state_from_x(x10, scaler)
        x_ref_h = build_ref_horizon(ref_std, k, N_EDMD)

        t0    = time.perf_counter()
        u_cmd = mpc_edmd.compute(z_k, x_ref_h)
        solve_times_edmd.append(time.perf_counter() - t0)

        u_cmd[0] = np.clip(u_cmd[0], 0.5 * sim.q_mass * sim.g,
                                      2.0 * sim.q_mass * sim.g)
        u_cmd[1] = np.clip(u_cmd[1], -sim.controller_PID.tilt_max,
                                       sim.controller_PID.tilt_max)
        u_cmd[2] = np.clip(u_cmd[2], -sim.controller_PID.tilt_max,
                                       sim.controller_PID.tilt_max)

        U_mpc_edmd[k] = u_cmd

        x_next_12 = sim.sim_PID.fct_step_attitude(
            x_current_12,
            u1        = u_cmd[0],
            phi_des   = u_cmd[1],
            theta_des = u_cmd[2],
            dt        = dt
        )

        x_current_12 = x_next_12
        X_mpc_edmd[k + 1] = drop_to_10state(x_next_12)

    U_mpc_edmd[-1] = U_mpc_edmd[-2]

    # --------------------------------------------------------
    # Linear MPC closed-loop
    # --------------------------------------------------------
    X_mpc_lin = np.zeros((T, 10))
    U_mpc_lin = np.zeros((T, 3))

    x_current_12_lin = np.zeros(12)
    x_current_12_lin[0:6]  = x10_init[0:6]
    x_current_12_lin[6:8]  = x10_init[6:8]
    x_current_12_lin[9:11] = x10_init[8:10]

    X_mpc_lin[0] = drop_to_10state(x_current_12_lin)
    solve_times_lin = []

    print("\nRunning Linear MPC closed-loop...")
    for k in range(T - 1):
        if k % 50 == 0:
            print(f"  LIN  step {k}/{T-1}  pos=({X_mpc_lin[k,0]:.2f}, "
                  f"{X_mpc_lin[k,1]:.2f}, {X_mpc_lin[k,2]:.2f})")

        x10_lin = drop_to_10state(x_current_12_lin)
        x_ref_h = build_ref_horizon(ref_lin, k, N_LIN)

        t0 = time.perf_counter()
        du_cmd = mpc_lin.compute(x10_lin, x_ref_h)
        solve_times_lin.append(time.perf_counter() - t0)

        u_cmd = u_nominal + du_cmd
        u_cmd[0] = np.clip(u_cmd[0], 0.5 * sim.q_mass * sim.g,
                                      2.0 * sim.q_mass * sim.g)
        u_cmd[1] = np.clip(u_cmd[1], -sim.controller_PID.tilt_max,
                                       sim.controller_PID.tilt_max)
        u_cmd[2] = np.clip(u_cmd[2], -sim.controller_PID.tilt_max,
                                       sim.controller_PID.tilt_max)

        U_mpc_lin[k] = u_cmd

        x_next_12_lin = sim.sim_PID.fct_step_attitude(
            x_current_12_lin,
            u1        = u_cmd[0],
            phi_des   = u_cmd[1],
            theta_des = u_cmd[2],
            dt        = dt
        )

        x_current_12_lin = x_next_12_lin
        X_mpc_lin[k + 1] = drop_to_10state(x_next_12_lin)

    U_mpc_lin[-1] = U_mpc_lin[-2]

    print(f"\nSolve time EDMD — avg: {1e3*np.mean(solve_times_edmd):.2f} ms  max: {1e3*np.max(solve_times_edmd):.2f} ms")
    print(f"Solve time LIN  — avg: {1e3*np.mean(solve_times_lin):.2f} ms  max: {1e3*np.max(solve_times_lin):.2f} ms")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    pos_err_edmd = np.linalg.norm(X_mpc_edmd[:, 0:3] - ref_xyz, axis=1)
    pos_err_lin  = np.linalg.norm(X_mpc_lin[:, 0:3]  - ref_xyz, axis=1)
    pos_err_true = np.linalg.norm(X_true[:, 0:3]     - ref_xyz, axis=1)

    print(f"\nPosition RMSE — EDMD-MPC: {rmse(X_mpc_edmd[:,0:3], ref_xyz):.4f} m")
    print(f"Position RMSE — LIN-MPC : {rmse(X_mpc_lin[:,0:3],  ref_xyz):.4f} m")
    print(f"Position RMSE — Saved PID: {rmse(X_true[:,0:3],    ref_xyz):.4f} m")

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------
    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(ref_xyz[:,0],       ref_xyz[:,1],       ref_xyz[:,2],       "k",    lw=2,   label="Reference")
    ax.plot(X_true[:,0],        X_true[:,1],        X_true[:,2],        "gray", lw=1.5, label="Saved PID")
    ax.plot(X_mpc_lin[:,0],     X_mpc_lin[:,1],     X_mpc_lin[:,2],     "b-",   lw=1.5, label="Linear MPC")
    ax.plot(X_mpc_edmd[:,0],    X_mpc_edmd[:,1],    X_mpc_edmd[:,2],    "g-",   lw=1.5, label="EDMD-MPC")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Trajectory tracking comparison")
    ax.legend()
    ax.grid(True)

    units = ['m','m','m','m/s','m/s','m/s','rad','rad','rad/s','rad/s']
    labels10 = ['x','y','z','vx','vy','vz','phi','theta','p','q']
    fig2, axs = plt.subplots(2, 5, figsize=(20, 7))
    for i in range(10):
        row, col = divmod(i, 5)
        ax = axs[row, col]
        ax.plot(t_ref, X_true[:, i],     "gray", lw=1.2, label="Saved PID")
        ax.plot(t_ref, X_mpc_lin[:, i],  "b-",   lw=1.2, label="Linear MPC")
        ax.plot(t_ref, X_mpc_edmd[:, i], "g-",   lw=1.2, label="EDMD-MPC")
        ax.set_title(f"{labels10[i]} [{units[i]}]")
        ax.set_xlabel("t [s]")
        ax.grid(True)
        if i == 0:
            ax.legend()
    fig2.tight_layout()

    u_labels = ['thrust [N]', 'phi_des [rad]', 'theta_des [rad]']
    fig3, axs3 = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        axs3[i].plot(t_ref, U_saved[:, i],    "gray", lw=1.2, label="Saved PID")
        axs3[i].plot(t_ref, U_mpc_lin[:, i],  "b-",   lw=1.2, label="Linear MPC")
        axs3[i].plot(t_ref, U_mpc_edmd[:, i], "g-",   lw=1.2, label="EDMD-MPC")
        axs3[i].set_title(u_labels[i])
        axs3[i].set_xlabel("t [s]")
        axs3[i].grid(True)
        if i == 0:
            axs3[i].legend()
    fig3.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(t_ref, pos_err_true, "gray", label="Saved PID")
    plt.plot(t_ref, pos_err_lin,  "b",    label="Linear MPC")
    plt.plot(t_ref, pos_err_edmd, "g",    label="EDMD-MPC")
    plt.xlabel("t [s]")
    plt.ylabel("Position error to reference [m]")
    plt.title("Trajectory tracking error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()