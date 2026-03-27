import itertools
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

import scipy.sparse as sp
import osqp


# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_FILE = "runs_traj2_plus_hover_n300.pkl"
EDMDC_MODEL_FILE = "edmdc_model_traj2_plus_hover_n300.pkl"

TEST_RUN_IDX = 199
DT_OVERRIDE = None

# MPC horizon
N = 50
NC = 12

N_EDMD = 5
NC_EDMD = 2

# Linearized hover model parameters
G = 9.81
IXX = 0.01
IYY = 0.01
IZZ = 0.02
KV = 0.1
KW = 0.01

# Cost weights
Q_DIAG = np.array([10, 10, 15, 2, 2, 2, 80, 80, 5, 10, 10, 2], dtype=float)
R_DIAG = np.array([8.0, 4.0, 4.0, 3.0], dtype=float)
RD_DIAG = np.array([4.0, 2.0, 2.0, 1.5], dtype=float)

# Linear MPC bounds
DU_MIN = np.array([-0.6, -0.05, -0.05, -0.03], dtype=float)
DU_MAX = np.array([0.6, 0.05, 0.05, 0.03], dtype=float)

# Cost weights for edmdc
Q_DIAG_EDMD = np.array([20, 20, 25, 3, 3, 3, 120, 120, 8, 20, 20, 4], dtype=float)
R_DIAG_EDMD = np.array([0,0,0,0], dtype=float)
RD_DIAG_EDMD = np.array([0,0,0,0], dtype=float)

# EDMD-MPC trust-region bounds (start tighter)
DU_MIN_EDMD = np.array([-0.12, -0.008, -0.008, -0.004], dtype=float)
DU_MAX_EDMD = np.array([ 0.12,  0.008,  0.008,  0.004], dtype=float)


# ============================================================
# LOADERS
# ============================================================
def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["t"], data["states"], data["U"], data["ref_traj_list"]


def load_edmdc_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


# ============================================================
# REFERENCE EXTRACTION
# ============================================================
def extract_ref_xyz(ref_item):
    if isinstance(ref_item, list) and len(ref_item) > 0 and isinstance(ref_item[0], dict):
        if "pos" in ref_item[0]:
            pts = [np.asarray(d["pos"]).reshape(-1)[:3] for d in ref_item]
            return np.asarray(pts, dtype=float)

    try:
        arr = np.asarray(ref_item, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]
        if arr.ndim == 2 and arr.shape[0] >= 3:
            return arr[:3, :].T
    except Exception:
        pass

    if isinstance(ref_item, (list, tuple)) and len(ref_item) >= 3:
        x = np.asarray(ref_item[0]).reshape(-1)
        y = np.asarray(ref_item[1]).reshape(-1)
        z = np.asarray(ref_item[2]).reshape(-1)
        T = min(len(x), len(y), len(z))
        return np.column_stack((x[:T], y[:T], z[:T])).astype(float)

    raise ValueError(f"Unsupported reference format: {type(ref_item)}")


# ============================================================
# LINEARIZED HOVER MODEL
# ============================================================
def quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts):
    nx = 12
    nu = 4

    Ac = np.zeros((nx, nx))
    Bc = np.zeros((nx, nu))

    # position dynamics
    Ac[0, 3] = 1.0
    Ac[1, 4] = 1.0
    Ac[2, 5] = 1.0

    # translational dynamics
    Ac[3, 3] = -kv
    Ac[3, 7] = g
    Ac[4, 4] = -kv
    Ac[4, 6] = -g
    Ac[5, 5] = -kv
    Bc[5, 0] = 1.0 / m

    # attitude kinematics
    Ac[6, 9] = 1.0
    Ac[7, 10] = 1.0
    Ac[8, 11] = 1.0

    # angular rate dynamics
    Ac[9, 9] = -kw
    Ac[10, 10] = -kw
    Ac[11, 11] = -kw

    Bc[9, 1] = 1.0 / Ixx
    Bc[10, 2] = 1.0 / Iyy
    Bc[11, 3] = 1.0 / Izz

    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = Ac
    M[:nx, nx:] = Bc

    Md = expm(M * Ts)
    Ad = Md[:nx, :nx]
    Bd = Md[:nx, nx:]

    return Ad, Bd


def linearized_step(x, u, Ad, Bd, u_nominal):
    du = u - u_nominal
    return Ad @ x + Bd @ du


# ============================================================
# EDMD OBSERVABLES
# EXACTLY MATCH TRAINING: input must already be standardized
# ============================================================
def observables_edmd_standardized(x_std, scaler):
    x = np.asarray(x_std).flatten()
    n = len(x)
    if n != 12:
        raise ValueError(f"Expected 12-state standardized vector, got {n}")

    obs = list(x)

    for i, j in itertools.combinations_with_replacement(range(n), 2):
        obs.append(x[i] * x[j])

    pos_vel_indices = [0, 1, 2, 3, 4, 5]
    for i in pos_vel_indices:
        obs.append(x[i] ** 3)

    vx, vy, vz = x[3], x[4], x[5]
    p, q, r = x[9], x[10], x[11]
    obs.append(vx**2 + vy**2 + vz**2)
    obs.append(p**2 + q**2 + r**2)

    phi_raw = x[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_raw = x[7] * scaler.scale_[7] + scaler.mean_[7]
    yaw_raw = x[8] * scaler.scale_[8] + scaler.mean_[8]

    phi_rad = phi_raw
    theta_rad = theta_raw
    yaw_rad = yaw_raw

    obs += [
        np.sin(yaw_rad), np.cos(yaw_rad),
        np.sin(phi_rad), np.cos(phi_rad),
        np.sin(theta_rad), np.cos(theta_rad),
    ]

    obs.append(1.0)
    return np.asarray(obs, dtype=float)


def lifted_state_from_x(x, scaler_edmd):
    x_std = scaler_edmd.transform(x.reshape(1, -1)).flatten()
    return observables_edmd_standardized(x_std, scaler_edmd)


def rollout_edmd_from_controls_training_style(
    x0, U_seq, A_edmd, B_edmd, scaler_edmd, u_scaler_edmd, clip_value=1e6
):
    """
    Match the training script exactly:
      Psi_pred[:,0] = observables( standardized(x0) )
      Psi_pred[:,k] = A Psi_pred[:,k-1] + B u_scaled(k-1)
      x_pred = inverse_transform(Psi_pred[:12])
    """
    M = U_seq.shape[0] + 1
    n_obs = A_edmd.shape[0]

    Psi_pred = np.zeros((n_obs, M))

    x0_std = scaler_edmd.transform(x0.reshape(1, -1)).flatten()
    Psi_pred[:, 0] = observables_edmd_standardized(x0_std, scaler_edmd)

    for k in range(1, M):
        u_k = U_seq[k - 1].reshape(1, -1)
        u_k_s = u_scaler_edmd.transform(u_k).flatten()
        Psi_pred[:, k] = A_edmd @ Psi_pred[:, k - 1] + B_edmd @ u_k_s
        Psi_pred[:, k] = np.clip(Psi_pred[:, k], -clip_value, clip_value)

    x_pred = scaler_edmd.inverse_transform(Psi_pred[:12, :].T)
    return x_pred


# ============================================================
# LINEAR MPC (QP)
# ============================================================
class LinearMPC_QP:
    def __init__(self, Ad, Bd, N, NC, Q, R, Rd, u_nominal, du_min, du_max):
        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float)
        self.N = int(N)
        self.NC = int(NC)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.Rd = np.asarray(Rd, dtype=float)
        self.u_nominal = np.asarray(u_nominal, dtype=float)
        self.du_min = np.asarray(du_min, dtype=float)
        self.du_max = np.asarray(du_max, dtype=float)

        self.nx = self.Ad.shape[0]
        self.nu = self.Bd.shape[1]
        self.nvar = self.NC * self.nu

        self._du_prev = np.zeros(self.nvar)

        self.Sx, self.Su = self._build_prediction_matrices()
        self.Qbar = sp.block_diag([sp.csc_matrix(self.Q) for _ in range(self.N)], format="csc")
        self.Rbar = sp.block_diag([sp.csc_matrix(self.R) for _ in range(self.NC)], format="csc")

        self.D = self._build_difference_matrix()
        self.Rdbar = sp.block_diag([sp.csc_matrix(self.Rd) for _ in range(self.NC - 1)], format="csc") if self.NC > 1 else None

        P = self._build_hessian()
        A = sp.eye(self.nvar, format="csc")

        l = np.tile(self.du_min, self.NC)
        u = np.tile(self.du_max, self.NC)

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=P,
            q=np.zeros(self.nvar),
            A=A,
            l=l,
            u=u,
            warm_start=True,
            verbose=False,
            polish=False,
        )

    def _build_prediction_matrices(self):
        Sx = np.zeros((self.N * self.nx, self.nx))
        Su = np.zeros((self.N * self.nx, self.NC * self.nu))

        A_powers = [np.eye(self.nx)]
        for _ in range(1, self.N + 1):
            A_powers.append(A_powers[-1] @ self.Ad)

        for i in range(self.N):
            Sx[i*self.nx:(i+1)*self.nx, :] = A_powers[i + 1]

            for j in range(min(i + 1, self.NC)):
                Aij = A_powers[i - j]
                Su[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = Aij @ self.Bd

        return sp.csc_matrix(Sx), sp.csc_matrix(Su)

    def _build_difference_matrix(self):
        if self.NC <= 1:
            return None

        rows = []
        cols = []
        vals = []

        for k in range(self.NC - 1):
            for j in range(self.nu):
                r = k * self.nu + j
                c1 = k * self.nu + j
                c2 = (k + 1) * self.nu + j
                rows.extend([r, r])
                cols.extend([c1, c2])
                vals.extend([-1.0, 1.0])

        D = sp.coo_matrix(
            (vals, (rows, cols)),
            shape=((self.NC - 1) * self.nu, self.NC * self.nu)
        )
        return D.tocsc()

    def _build_hessian(self):
        P_track = self.Su.T @ self.Qbar @ self.Su
        P = P_track + self.Rbar

        if self.D is not None:
            P = P + self.D.T @ self.Rdbar @ self.D

        P = 0.5 * (P + P.T)
        return P.tocsc()

    def _build_q(self, x0, x_ref_horizon):
        x_ref_stack = x_ref_horizon.reshape(-1)
        x_free = self.Sx @ x0
        q = self.Su.T @ (self.Qbar @ (x_free - x_ref_stack))
        return np.asarray(q).reshape(-1)

    def compute(self, x0, x_ref_horizon):
        q = self._build_q(x0, x_ref_horizon)

        self.prob.update(q=q)
        self.prob.warm_start(x=self._du_prev)

        res = self.prob.solve()

        if res.info.status not in ("solved", "solved inaccurate"):
            print(f"Warning: OSQP did not solve linear MPC QP cleanly: {res.info.status}")
            du0 = self._du_prev[:self.nu]
            return self.u_nominal + du0

        du_opt = np.asarray(res.x).reshape(-1)
        self._du_prev = du_opt.copy()

        du0 = du_opt[:self.nu]
        return self.u_nominal + du0


# ============================================================
# EDMD-MPC (QP in lifted space)
# ============================================================
class EDMDcMPC_QP:
    def __init__(self, A_edmd, B_edmd, Cz, N, NC, Q, R, Rd,
                 u_scaler, du_min_raw, du_max_raw, u_nominal_raw):
        self.A = np.asarray(A_edmd, dtype=float)
        self.B = np.asarray(B_edmd, dtype=float)
        self.Cz = np.asarray(Cz, dtype=float)

        self.N = int(N)
        self.NC = int(NC)

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.Rd = np.asarray(Rd, dtype=float)

        self.u_scaler = u_scaler
        self.u_nominal_raw = np.asarray(u_nominal_raw, dtype=float)

        self.u_nominal_scaled, self.du_min_scaled, self.du_max_scaled = compute_scaled_du_bounds(
            self.u_nominal_raw, du_min_raw, du_max_raw, self.u_scaler
        )

        self.nz = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.nx = self.Cz.shape[0]
        self.nvar = self.NC * self.nu

        self._du_prev = np.zeros(self.nvar)

        self.Sz, self.Su = self._build_prediction_matrices()

        Qz = self.Cz.T @ self.Q @ self.Cz
        self.Qbar = sp.block_diag([sp.csc_matrix(Qz) for _ in range(self.N)], format="csc")
        self.Rbar = sp.block_diag([sp.csc_matrix(self.R) for _ in range(self.NC)], format="csc")

        self.D = self._build_difference_matrix()
        self.Rdbar = (
            sp.block_diag([sp.csc_matrix(self.Rd) for _ in range(self.NC - 1)], format="csc")
            if self.NC > 1 else None
        )

        self.nominal_stack = np.tile(self.u_nominal_scaled, self.NC)

        P = self._build_hessian()
        Aineq = sp.eye(self.nvar, format="csc")
        l = np.tile(self.du_min_scaled, self.NC)
        u = np.tile(self.du_max_scaled, self.NC)

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=P,
            q=np.zeros(self.nvar),
            A=Aineq,
            l=l,
            u=u,
            warm_start=True,
            verbose=False,
            polish=False,
        )

    def _build_prediction_matrices(self):
        Sz = np.zeros((self.N * self.nz, self.nz))
        Su = np.zeros((self.N * self.nz, self.NC * self.nu))

        A_powers = [np.eye(self.nz)]
        for _ in range(1, self.N + 1):
            A_powers.append(A_powers[-1] @ self.A)

        for i in range(self.N):
            Sz[i*self.nz:(i+1)*self.nz, :] = A_powers[i + 1]
            for j in range(min(i + 1, self.NC)):
                Aij = A_powers[i - j]
                Su[i*self.nz:(i+1)*self.nz, j*self.nu:(j+1)*self.nu] = Aij @ self.B

        return sp.csc_matrix(Sz), sp.csc_matrix(Su)

    def _build_difference_matrix(self):
        if self.NC <= 1:
            return None

        rows, cols, vals = [], [], []
        for k in range(self.NC - 1):
            for j in range(self.nu):
                r = k * self.nu + j
                c1 = k * self.nu + j
                c2 = (k + 1) * self.nu + j
                rows.extend([r, r])
                cols.extend([c1, c2])
                vals.extend([-1.0, 1.0])

        D = sp.coo_matrix(
            (vals, (rows, cols)),
            shape=((self.NC - 1) * self.nu, self.NC * self.nu)
        )
        return D.tocsc()

    def _build_hessian(self):
        P = self.Su.T @ self.Qbar @ self.Su + self.Rbar
        if self.D is not None:
            P = P + self.D.T @ self.Rdbar @ self.D
        return (0.5 * (P + P.T)).tocsc()

    def _build_q(self, z0, x_ref_std_horizon):
        z_free = self.Sz @ z0

        z_ref = np.zeros((self.N, self.nz))
        for i in range(self.N):
            z_ref[i, :12] = x_ref_std_horizon[i]
        z_ref_stack = z_ref.reshape(-1)

        # predicted lifted trajectory offset from constant nominal scaled input
        du_nom_stack = self.nominal_stack
        z_nom = z_free + self.Su @ du_nom_stack

        q_track = self.Su.T @ (self.Qbar @ (z_nom - z_ref_stack))
        q_input = self.Rbar @ du_nom_stack

        if self.D is not None:
            q_slew = self.D.T @ (self.Rdbar @ (self.D @ du_nom_stack))
        else:
            q_slew = 0.0

        q = q_track + q_input + q_slew
        return np.asarray(q).reshape(-1)

    def compute(self, z0, x_ref_std_horizon):
        q = self._build_q(z0, x_ref_std_horizon)

        self.prob.update(q=q)
        self.prob.warm_start(x=self._du_prev)

        res = self.prob.solve()
        if res.info.status not in ("solved", "solved inaccurate"):
            print(f"Warning: OSQP EDMD-MPC status: {res.info.status}")
            du0_scaled = self._du_prev[:self.nu]
        else:
            du_opt_scaled = np.asarray(res.x).reshape(-1)
            self._du_prev = du_opt_scaled.copy()
            du0_scaled = du_opt_scaled[:self.nu]

        u0_scaled = self.u_nominal_scaled + du0_scaled
        u0_raw = self.u_scaler.inverse_transform(u0_scaled.reshape(1, -1)).flatten()
        return u0_raw
# ============================================================
# HELPERS
# ============================================================
def build_reference_horizon(ref_xyz, k, N):
    T = ref_xyz.shape[0]
    x_ref_h = np.zeros((N, 12))
    for i in range(N):
        idx = min(k + i, T - 1)
        x_ref_h[i, 0:3] = ref_xyz[idx, :]
    return x_ref_h


def build_reference_horizon_standardized(ref_xyz, k, N, scaler_edmd):
    T = ref_xyz.shape[0]
    X_ref = np.zeros((N, 12))

    for i in range(N):
        idx = min(k + i, T - 1)
        x_ref = np.zeros(12)
        x_ref[0:3] = ref_xyz[idx, :]
        # leave vx, vy, vz, phi, theta, psi, p, q, r at zero
        X_ref[i] = x_ref

    return scaler_edmd.transform(X_ref)


def rmse_per_state(X_true, X_pred):
    err = X_true - X_pred
    each = np.sqrt(np.mean(err**2, axis=0))
    total = np.sqrt(np.mean(err**2))
    return each, total


def rmse_xyz(ref_xyz, xyz_pred):
    err = ref_xyz - xyz_pred
    each = np.sqrt(np.mean(err**2, axis=0))
    total = np.sqrt(np.mean(err**2))
    return each, total


def rmse_over_window(X_true, X_pred, start_idx=0, end_idx=500):
    err = X_true[start_idx:end_idx] - X_pred[start_idx:end_idx]
    each = np.sqrt(np.mean(err**2, axis=0))
    total = np.sqrt(np.mean(err**2))
    return each, total

def compute_scaled_du_bounds(u_nominal_raw, du_min_raw, du_max_raw, u_scaler):
    u_nom = np.asarray(u_nominal_raw, dtype=float).reshape(1, -1)
    u_min = (u_nom + np.asarray(du_min_raw, dtype=float).reshape(1, -1))
    u_max = (u_nom + np.asarray(du_max_raw, dtype=float).reshape(1, -1))

    u_nom_s = u_scaler.transform(u_nom).flatten()
    u_min_s = u_scaler.transform(u_min).flatten()
    u_max_s = u_scaler.transform(u_max).flatten()

    du_min_s = u_min_s - u_nom_s
    du_max_s = u_max_s - u_nom_s
    return u_nom_s, du_min_s, du_max_s


# ============================================================
# MAIN
# ============================================================
def main():
    data_path = SCRIPT_DIR / DATA_FILE
    model_path = SCRIPT_DIR / EDMDC_MODEL_FILE

    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(data_path)
    model = load_edmdc_model(model_path)

    A_edmd = model["A"]
    B_edmd = model["B"]
    scaler_edmd = model["scaler"]
    u_scaler_edmd = model["u_scaler"]
    dt_model = model["dt"]

    dt = dt_model if DT_OVERRIDE is None else DT_OVERRIDE

    print("Loaded data:", data_path.name)
    print("states_all shape:", states_all.shape)
    print("U_all shape:", U_all.shape)
    print("Loaded EDMDc model:", model_path.name)
    print("A_edmd shape:", A_edmd.shape)
    print("B_edmd shape:", B_edmd.shape)
    print("dt used:", dt)

    sim_dt = t_all[0, 1] - t_all[0, 0]
    ratio = dt / sim_dt
    step = int(round(ratio))

    if not np.isclose(ratio, step, rtol=1e-6, atol=1e-8):
        raise ValueError(f"dt={dt} must be integer multiple of sim dt={sim_dt}")

    idx = np.arange(0, t_all.shape[1], step)
    t_all = t_all[:, idx]
    states_all = states_all[:, idx, :]
    U_all = U_all[:, idx, :]
    ref_traj_list = [ref[::step] for ref in ref_traj_list]

    run_idx = states_all.shape[0] - 1 if TEST_RUN_IDX < 0 else min(TEST_RUN_IDX, states_all.shape[0] - 1)

    t_ref = t_all[run_idx]
    X_true = states_all[run_idx]
    U_saved = U_all[run_idx]
    ref_xyz = extract_ref_xyz(ref_traj_list[run_idx])

    T = min(len(t_ref), X_true.shape[0], U_saved.shape[0], ref_xyz.shape[0])
    t_ref = t_ref[:T]
    X_true = X_true[:T]
    U_saved = U_saved[:T]
    ref_xyz = ref_xyz[:T]

    print("Using test run:", run_idx)
    print("Reference xyz shape:", ref_xyz.shape)
    print("Saved control shape:", U_saved.shape)

    # Hover estimate from data
    u1_hover_est = float(np.mean(U_saved[:, 0]))
    mass_est = u1_hover_est / G
    u_nominal = np.array([u1_hover_est, 0.0, 0.0, 0.0], dtype=float)

    print("Estimated hover thrust from saved controls:", u1_hover_est)
    print("Estimated mass for linear model:", mass_est)

    Ad, Bd = quadcopter_linearized_model(mass_est, G, IXX, IYY, IZZ, KV, KW, dt)

    Q = np.diag(Q_DIAG)
    R = np.diag(R_DIAG)
    Rd = np.diag(RD_DIAG)

    Q_edmd = np.diag(Q_DIAG_EDMD)
    R_edmd = np.diag(R_DIAG_EDMD)
    Rd_edmd = np.diag(RD_DIAG_EDMD)

    # selector from lifted state to standardized physical state
    Cz = np.zeros((12, A_edmd.shape[0]))
    Cz[:, :12] = np.eye(12)

    lin_mpc = LinearMPC_QP(
        Ad=Ad,
        Bd=Bd,
        N=N,
        NC=NC,
        Q=Q,
        R=R,
        Rd=Rd,
        u_nominal=u_nominal,
        du_min=DU_MIN,
        du_max=DU_MAX,
    )

    edmd_mpc = EDMDcMPC_QP(
        A_edmd=A_edmd,
        B_edmd=B_edmd,
        Cz=Cz,
        N=N_EDMD,
        NC=NC_EDMD,
        Q=Q_edmd,
        R=R_edmd,
        Rd=Rd_edmd,
        u_scaler=u_scaler_edmd,
        du_min_raw=DU_MIN_EDMD,
        du_max_raw=DU_MAX_EDMD,
        u_nominal_raw=u_nominal,
    )

    x0 = X_true[0].copy()

    # --------------------------------------------------------
    # 1) Linear MPC tracking on linearized plant
    # --------------------------------------------------------
    X_lin_mpc = np.zeros((T, 12))
    U_lin_mpc = np.zeros((T, 4))
    X_lin_mpc[0] = x0.copy()
    x_lin = x0.copy()

    solve_times_lin = []

    print("Running linear MPC tracking on linearized plant...")
    for k in range(T - 1):
        if k % 100 == 0:
            print(f" linear step {k}/{T - 1}")

        x_ref_h = build_reference_horizon(ref_xyz, k, N)

        t0 = time.perf_counter()
        u_lin = lin_mpc.compute(x_lin, x_ref_h)
        solve_times_lin.append(time.perf_counter() - t0)

        U_lin_mpc[k] = u_lin
        x_lin = linearized_step(x_lin, u_lin, Ad, Bd, u_nominal)
        X_lin_mpc[k + 1] = x_lin

    U_lin_mpc[-1] = U_lin_mpc[-2]

    print(f"Average linear MPC solve time: {1e3 * np.mean(solve_times_lin):.3f} ms")
    print(f"Max linear MPC solve time: {1e3 * np.max(solve_times_lin):.3f} ms")
    print("\nLinear MPC max abs state per coordinate:")
    print(np.max(np.abs(X_lin_mpc), axis=0))
    bad_idx_lin = np.where(np.any(np.abs(X_lin_mpc) > 100, axis=1))[0]
    print("First bad linear index (>100 abs in any state):", bad_idx_lin[0] if len(bad_idx_lin) else None)
    print("Max |phi|, |theta|:",
          np.max(np.abs(X_lin_mpc[:, 6])),
          np.max(np.abs(X_lin_mpc[:, 7])))

    # --------------------------------------------------------
    # 2) EDMDc open-loop rollout EXACTLY like training
    # --------------------------------------------------------
    print("Running EDMDc open-loop rollout (training-style)...")
    X_edmd_ol = rollout_edmd_from_controls_training_style(
        X_true[0],
        U_saved[:-1],
        A_edmd,
        B_edmd,
        scaler_edmd,
        u_scaler_edmd,
        clip_value=1e6,
    )

    # --------------------------------------------------------
    # 3) EDMDc-MPC tracking on EDMD predictor
    # --------------------------------------------------------
    X_edmd_mpc = np.zeros((T, 12))
    U_edmd_mpc = np.zeros((T, 4))
    X_edmd_mpc[0] = x0.copy()

    x_edmd_mpc = x0.copy()
    z_edmd_mpc = lifted_state_from_x(x_edmd_mpc, scaler_edmd)

    solve_times_edmd = []

    print("Running EDMDc-MPC tracking on EDMD predictor...")
    for k in range(T - 1):
        if k % 100 == 0:
            print(f" edmd-mpc step {k}/{T - 1}")

        x_ref_std_h = build_reference_horizon_standardized(ref_xyz, k, N_EDMD, scaler_edmd)

        t0 = time.perf_counter()
        u_cmd = edmd_mpc.compute(z_edmd_mpc, x_ref_std_h)
        solve_times_edmd.append(time.perf_counter() - t0)

        U_edmd_mpc[k] = u_cmd

        u_cmd_s = u_scaler_edmd.transform(u_cmd.reshape(1, -1)).flatten()
        z_edmd_mpc = A_edmd @ z_edmd_mpc + B_edmd @ u_cmd_s
        z_edmd_mpc = np.clip(z_edmd_mpc, -1e6, 1e6)

        x_edmd_mpc = scaler_edmd.inverse_transform(z_edmd_mpc[:12].reshape(1, -1)).flatten()
        X_edmd_mpc[k + 1] = x_edmd_mpc

    U_edmd_mpc[-1] = U_edmd_mpc[-2]

    print(f"Average EDMD-MPC solve time: {1e3 * np.mean(solve_times_edmd):.3f} ms")
    print(f"Max EDMD-MPC solve time: {1e3 * np.max(solve_times_edmd):.3f} ms")
    print("\nEDMD-MPC max abs state per coordinate:")
    print(np.max(np.abs(X_edmd_mpc), axis=0))
    bad_idx_edmd_mpc = np.where(np.any(np.abs(X_edmd_mpc) > 100, axis=1))[0]
    print("First bad EDMD-MPC index (>100 abs in any state):", bad_idx_edmd_mpc[0] if len(bad_idx_edmd_mpc) else None)

    # --------------------------------------------------------
    # diagnostics
    # --------------------------------------------------------
    print("\nDiagnostic control ranges:")
    print("u_nominal:", u_nominal)
    print("U_saved min:", U_saved.min(axis=0))
    print("U_saved max:", U_saved.max(axis=0))
    print("U_lin_mpc min:", U_lin_mpc.min(axis=0))
    print("U_lin_mpc max:", U_lin_mpc.max(axis=0))
    print("U_edmd_mpc min:", U_edmd_mpc.min(axis=0))
    print("U_edmd_mpc max:", U_edmd_mpc.max(axis=0))
    u_nominal_scaled_dbg, du_min_scaled_dbg, du_max_scaled_dbg = compute_scaled_du_bounds(
        u_nominal, DU_MIN_EDMD, DU_MAX_EDMD, u_scaler_edmd
    )
    print("EDMD scaled nominal input:", u_nominal_scaled_dbg)
    print("EDMD scaled du min:", du_min_scaled_dbg)
    print("EDMD scaled du max:", du_max_scaled_dbg)

    print("\nFirst 5 xyz samples:")
    print("True:\n", X_true[:5, :3])
    print("Linear MPC:\n", X_lin_mpc[:5, :3])
    print("EDMD open-loop:\n", X_edmd_ol[:5, :3])
    print("EDMD-MPC:\n", X_edmd_mpc[:5, :3])

    print("\nEDMD open-loop max abs state per coordinate:")
    print(np.max(np.abs(X_edmd_ol), axis=0))
    bad_idx = np.where(np.any(np.abs(X_edmd_ol) > 100, axis=1))[0]
    print("First bad rollout index (>100 abs in any state):", bad_idx[0] if len(bad_idx) else None)

    # --------------------------------------------------------
    # metrics
    # --------------------------------------------------------
    rmse_lin_each, rmse_lin_total = rmse_per_state(X_true, X_lin_mpc)
    rmse_edmd_each, rmse_edmd_total = rmse_per_state(X_true, X_edmd_ol)
    rmse_edmd_mpc_each, rmse_edmd_mpc_total = rmse_per_state(X_true, X_edmd_mpc)

    rmse_lin_xyz_each, rmse_lin_xyz_total = rmse_xyz(ref_xyz, X_lin_mpc[:, 0:3])
    rmse_edmd_xyz_each, rmse_edmd_xyz_total = rmse_xyz(ref_xyz, X_edmd_ol[:, 0:3])
    rmse_edmd_mpc_xyz_each, rmse_edmd_mpc_xyz_total = rmse_xyz(ref_xyz, X_edmd_mpc[:, 0:3])

    rmse_edmd_5s_each, rmse_edmd_5s_total = rmse_over_window(X_true, X_edmd_ol, 0, min(500, T))
    rmse_edmd_mpc_5s_each, rmse_edmd_mpc_5s_total = rmse_over_window(X_true, X_edmd_mpc, 0, min(500, T))

    print("\n========== RMSE TO SAVED HELD-OUT TRAJECTORY ==========")
    print(f"Linear MPC state RMSE vs saved trajectory:      {rmse_lin_total:.6f}")
    print(f"EDMDc open-loop state RMSE vs saved trajectory: {rmse_edmd_total:.6f}")
    print(f"EDMDc-MPC state RMSE vs saved trajectory:       {rmse_edmd_mpc_total:.6f}")

    print("\n========== POSITION RMSE TO REFERENCE PATH ==========")
    print(f"Linear MPC xyz RMSE vs reference path:      {rmse_lin_xyz_total:.6f}")
    print(f"EDMDc open-loop xyz RMSE vs reference path: {rmse_edmd_xyz_total:.6f}")
    print(f"EDMDc-MPC xyz RMSE vs reference path:       {rmse_edmd_mpc_xyz_total:.6f}")

    print(f"\nEDMDc open-loop RMSE over first 5 s: {rmse_edmd_5s_total:.6f}")
    print(f"EDMDc-MPC RMSE over first 5 s:       {rmse_edmd_mpc_5s_total:.6f}")

    labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"]
    for i, lbl in enumerate(labels):
        print(
            f"{lbl:>6s} | Linear MPC: {rmse_lin_each[i]:.6f} | "
            f"EDMD open: {rmse_edmd_each[i]:.6f} | "
            f"EDMD-MPC: {rmse_edmd_mpc_each[i]:.6f}"
        )

    # --------------------------------------------------------
    # plots
    # --------------------------------------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], "k", lw=2.0, label="Reference")
    ax.plot(X_true[:, 0], X_true[:, 1], X_true[:, 2], color="gray", lw=1.4, label="Saved true run")
    ax.plot(X_lin_mpc[:, 0], X_lin_mpc[:, 1], X_lin_mpc[:, 2], "r-.", lw=1.5, label="Linear MPC")
    ax.plot(X_edmd_ol[:, 0], X_edmd_ol[:, 1], X_edmd_ol[:, 2], "b--", lw=1.5, label="EDMDc open-loop")
    ax.plot(X_edmd_mpc[:, 0], X_edmd_mpc[:, 1], X_edmd_mpc[:, 2], "g-", lw=1.5, label="EDMDc-MPC")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Linear MPC vs EDMDc open-loop vs EDMDc-MPC")
    ax.legend()
    ax.grid(True)

    units = ["m", "m", "m", "m/s", "m/s", "m/s", "rad", "rad", "rad", "rad/s", "rad/s", "rad/s"]
    fig2, axs = plt.subplots(4, 3, figsize=(16, 10))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.plot(t_ref, X_true[:, i], color="gray", lw=1.2, label="Saved true run")
        ax.plot(t_ref, X_lin_mpc[:, i], "r-.", lw=1.1, label="Linear MPC")
        ax.plot(t_ref, X_edmd_ol[:, i], "b--", lw=1.1, label="EDMDc open-loop")
        ax.plot(t_ref, X_edmd_mpc[:, i], "g-", lw=1.1, label="EDMDc-MPC")
        ax.set_title(
            f"{labels[i]} | Lin {rmse_lin_each[i]:.3f} | "
            f"Open {rmse_edmd_each[i]:.3f} | "
            f"EMPC {rmse_edmd_mpc_each[i]:.3f}"
        )
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f"{labels[i]} [{units[i]}]")
        ax.grid(True)
        if i == 0:
            ax.legend()

    pos_err_lin = np.linalg.norm(X_lin_mpc[:, 0:3] - ref_xyz, axis=1)
    pos_err_edmd = np.linalg.norm(X_edmd_ol[:, 0:3] - ref_xyz, axis=1)
    pos_err_edmd_mpc = np.linalg.norm(X_edmd_mpc[:, 0:3] - ref_xyz, axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(t_ref, pos_err_lin, label="Linear MPC -> reference")
    plt.plot(t_ref, pos_err_edmd, label="EDMDc open-loop -> reference")
    plt.plot(t_ref, pos_err_edmd_mpc, label="EDMDc-MPC -> reference")
    plt.xlabel("t [s]")
    plt.ylabel("Position error norm [m]")
    plt.title("Position error to reference")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    import pickle

    with open("linear_mpc_results.pkl", "wb") as f:
        pickle.dump({
            "X": X_lin_mpc,
            "U": U_lin_mpc,
        }, f)

    with open("edmd_mpc_results.pkl", "wb") as f:
        pickle.dump({
            "X": X_edmd_mpc,
            "U": U_edmd_mpc,
        }, f)


if __name__ == "__main__":
    main()