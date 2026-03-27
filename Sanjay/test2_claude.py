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

DATA_FILE = "runs_traj2_n200.pkl"
EDMDC_MODEL_FILE = "edmdc_model_traj2_n200.pkl"

TEST_RUN_IDX = -1
DT_OVERRIDE = None

# MPC horizon (hover-linearized)
N  = 50
NC = 12

# EDMD-MPC horizon
# Position response from attitude changes takes ~1-2s at dt=0.01.
# N=150 (1.5s) gives the cascade: input→p,q→phi,theta→vx,vy→x,y time to appear.
N_EDMD  = 200
NC_EDMD = 20

# Linearized hover model parameters
G   = 9.81
IXX = 0.01
IYY = 0.01
IZZ = 0.02
KV  = 0.1
KW  = 0.01

# Cost weights — hover-linearized MPC (raw state space)
Q_DIAG  = np.array([10, 10, 15, 2, 2, 2, 80, 80, 5, 10, 10, 2], dtype=float)
R_DIAG  = np.array([8.0, 4.0, 4.0, 3.0], dtype=float)
RD_DIAG = np.array([4.0, 2.0, 2.0, 1.5], dtype=float)

# Linear MPC bounds (raw control increments)
DU_MIN = np.array([-0.6, -0.05, -0.05, -0.03], dtype=float)
DU_MAX = np.array([ 0.6,  0.05,  0.05,  0.03], dtype=float)

# Cost weights — EDMD-MPC (standardized state space via Cz)
#
# B_edmd authority (row norms, standardized space):
#   x,y,z:         ~2e-7, 4e-7, 2e-5   (negligible)
#   vx,vy,vz:      ~3e-4, 3e-4, 0.012  (very small)
#   phi,theta,psi: ~0.027, 0.022, 2e-4  (small)
#   p,q,r:         ~1.21, 1.10, 0.003   (strong — primary control channel)
#
# Strategy: heavily penalize attitude (phi,theta) and angular rates (p,q)
# since those are reachable. Weight vz moderately (B norm 0.012).
# Keep R very small for thrust so the optimizer uses it freely for z.
# Moments have strong B authority so R can be moderate.
Q_DIAG_EDMD = np.array([
    10.0,  10.0,  15.0,    # x, y, z        (position — visible at long horizon)
    2.0,   2.0,   5.0,     # vx, vy, vz
    200.0, 200.0,  0.0,    # phi, theta, psi (attitude drives position; psi degenerate)
    200.0, 200.0,  0.0,    # p, q, r         (B norm ~1.2; r degenerate)
], dtype=float)

R_DIAG_EDMD  = np.array([0.05, 0.5, 0.5, 0.0], dtype=float)
RD_DIAG_EDMD = np.array([0.02, 0.2, 0.2, 0.0], dtype=float)

# EDMD-MPC bounds in SCALED control space:
# raw ±2.0 N  → scaled ±10.9  thrust  (matches saved data authority)
# raw ±0.1 Nm → scaled ±1.0   moments
DU_MIN_EDMD_SCALED = np.array([-10.9, -1.0, -1.0, 0.0], dtype=float)
DU_MAX_EDMD_SCALED = np.array([ 10.9,  1.0,  1.0, 0.0], dtype=float)


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
    nx, nu = 12, 4
    Ac = np.zeros((nx, nx))
    Bc = np.zeros((nx, nu))

    Ac[0, 3] = 1.0;  Ac[1, 4] = 1.0;  Ac[2, 5] = 1.0
    Ac[3, 3] = -kv;  Ac[3, 7] =  g
    Ac[4, 4] = -kv;  Ac[4, 6] = -g
    Ac[5, 5] = -kv;  Bc[5, 0] = 1.0 / m
    Ac[6, 9]  = 1.0; Ac[7, 10] = 1.0; Ac[8, 11] = 1.0
    Ac[9,  9]  = -kw; Bc[9,  1] = 1.0 / Ixx
    Ac[10, 10] = -kw; Bc[10, 2] = 1.0 / Iyy
    Ac[11, 11] = -kw; Bc[11, 3] = 1.0 / Izz

    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = Ac
    M[:nx, nx:] = Bc
    Md = expm(M * Ts)
    return Md[:nx, :nx], Md[:nx, nx:]


def linearized_step(x, u, Ad, Bd, u_nominal):
    return Ad @ x + Bd @ (u - u_nominal)


# ============================================================
# EDMD OBSERVABLES  (must match training exactly)
# ============================================================
def observables_edmd_standardized(x_std, scaler):
    x = np.asarray(x_std).flatten()
    if len(x) != 12:
        raise ValueError(f"Expected 12-state standardized vector, got {len(x)}")

    obs = list(x)
    for i, j in itertools.combinations_with_replacement(range(12), 2):
        obs.append(x[i] * x[j])
    for i in [0, 1, 2, 3, 4, 5]:
        obs.append(x[i] ** 3)

    vx, vy, vz = x[3], x[4], x[5]
    p,  q,  r  = x[9], x[10], x[11]
    obs.append(vx**2 + vy**2 + vz**2)
    obs.append(p**2  + q**2  + r**2)

    phi_rad   = x[6]  * scaler.scale_[6]  + scaler.mean_[6]
    theta_rad = x[7]  * scaler.scale_[7]  + scaler.mean_[7]
    yaw_rad   = x[8]  * scaler.scale_[8]  + scaler.mean_[8]
    obs += [np.sin(yaw_rad), np.cos(yaw_rad),
            np.sin(phi_rad), np.cos(phi_rad),
            np.sin(theta_rad), np.cos(theta_rad)]
    obs.append(1.0)
    return np.asarray(obs, dtype=float)


def lifted_state_from_x(x, scaler_edmd):
    x_std = scaler_edmd.transform(x.reshape(1, -1)).flatten()
    return observables_edmd_standardized(x_std, scaler_edmd)


def rollout_edmd_from_controls_training_style(
        x0, U_seq, A_edmd, B_edmd, scaler_edmd, u_scaler_edmd, clip_value=1e6):
    M     = U_seq.shape[0] + 1
    n_obs = A_edmd.shape[0]
    Psi   = np.zeros((n_obs, M))

    x0_std    = scaler_edmd.transform(x0.reshape(1, -1)).flatten()
    Psi[:, 0] = observables_edmd_standardized(x0_std, scaler_edmd)

    for k in range(1, M):
        u_k_s     = u_scaler_edmd.transform(U_seq[k-1].reshape(1, -1)).flatten()
        Psi[:, k] = np.clip(A_edmd @ Psi[:, k-1] + B_edmd @ u_k_s, -clip_value, clip_value)

    return scaler_edmd.inverse_transform(Psi[:12, :].T)


# ============================================================
# LINEAR MPC (QP) — raw physical state space
# Decision variable: du[0..NC-1]  (raw control increments)
# ============================================================
class LinearMPC_QP:
    def __init__(self, Ad, Bd, N, NC, Q, R, Rd, u_nominal, du_min, du_max):
        self.Ad        = np.asarray(Ad, dtype=float)
        self.Bd        = np.asarray(Bd, dtype=float)
        self.N         = int(N)
        self.NC        = int(NC)
        self.Q         = np.asarray(Q,  dtype=float)
        self.R         = np.asarray(R,  dtype=float)
        self.Rd        = np.asarray(Rd, dtype=float)
        self.u_nominal = np.asarray(u_nominal, dtype=float)
        self.du_min    = np.asarray(du_min, dtype=float)
        self.du_max    = np.asarray(du_max, dtype=float)

        self.nx   = self.Ad.shape[0]
        self.nu   = self.Bd.shape[1]
        self.nvar = self.NC * self.nu
        self._du_prev = np.zeros(self.nvar)

        self.Sx, self.Su = self._build_prediction_matrices()
        self.Qbar  = sp.block_diag([sp.csc_matrix(self.Q)  for _ in range(self.N)],  format="csc")
        self.Rbar  = sp.block_diag([sp.csc_matrix(self.R)  for _ in range(self.NC)], format="csc")
        self.D     = self._build_difference_matrix()
        self.Rdbar = (
            sp.block_diag([sp.csc_matrix(self.Rd) for _ in range(self.NC - 1)], format="csc")
            if self.NC > 1 else None
        )

        P = self._build_hessian()
        A = sp.eye(self.nvar, format="csc")
        l = np.tile(self.du_min, self.NC)
        u = np.tile(self.du_max, self.NC)

        self.prob = osqp.OSQP()
        self.prob.setup(P=P, q=np.zeros(self.nvar), A=A, l=l, u=u,
                        warm_start=True, verbose=False, polish=False)

    def _build_prediction_matrices(self):
        Sx = np.zeros((self.N * self.nx, self.nx))
        Su = np.zeros((self.N * self.nx, self.NC * self.nu))
        A_pow = [np.eye(self.nx)]
        for _ in range(self.N):
            A_pow.append(A_pow[-1] @ self.Ad)
        for i in range(self.N):
            Sx[i*self.nx:(i+1)*self.nx, :] = A_pow[i+1]
            for j in range(min(i+1, self.NC)):
                Su[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = A_pow[i-j] @ self.Bd
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
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=((self.NC-1)*self.nu, self.NC*self.nu)).tocsc()

    def _build_hessian(self):
        P = self.Su.T @ self.Qbar @ self.Su + self.Rbar
        if self.D is not None:
            P = P + self.D.T @ self.Rdbar @ self.D
        return (0.5 * (P + P.T)).tocsc()

    def _build_q(self, x0, x_ref_horizon):
        x_free      = self.Sx @ x0
        x_ref_stack = x_ref_horizon.reshape(-1)
        return np.asarray(self.Su.T @ (self.Qbar @ (x_free - x_ref_stack))).reshape(-1)

    def compute(self, x0, x_ref_horizon):
        q = self._build_q(x0, x_ref_horizon)
        self.prob.update(q=q)
        self.prob.warm_start(x=self._du_prev)
        res = self.prob.solve()

        if res.info.status not in ("solved", "solved inaccurate"):
            print(f"Warning: OSQP: {res.info.status}")
            du0 = self._du_prev[:self.nu]
        else:
            du_opt = np.asarray(res.x).reshape(-1)
            self._du_prev = du_opt.copy()
            du0 = du_opt[:self.nu]

        return self.u_nominal + du0


# ============================================================
# EDMD-MPC (QP in full lifted space)
#
# Decision variable: du_scaled[0..NC-1]
#   du_scaled = u_scaled - u_nominal_scaled
#
# Bounds supplied directly in scaled space (pre-computed from diagnostics).
# Yaw column (index 3) zeroed in B_edmd_clean: training had zero yaw
# variance → degenerate u_scaler → meaningless B column.
#
# Cost weights Q_DIAG_EDMD heavily penalize angular rates (p,q rows 9,10)
# because those are the states B_edmd can directly influence (norm ~1.2).
# Position tracks indirectly via attitude — this mirrors the real
# quadcopter cascade: thrust → angular rate → attitude → position.
# ============================================================
class EDMDcMPC_QP:
    def __init__(self, A_edmd, B_edmd, Cz, N, NC, Q, R, Rd,
                 u_scaler, du_min_scaled, du_max_scaled, u_nominal_raw):
        self.A   = np.asarray(A_edmd, dtype=float)
        self.B   = np.asarray(B_edmd, dtype=float)
        self.Cz  = np.asarray(Cz,     dtype=float)

        self.N  = int(N)
        self.NC = int(NC)

        self.Q  = np.asarray(Q,  dtype=float)
        self.R  = np.asarray(R,  dtype=float)
        self.Rd = np.asarray(Rd, dtype=float)

        self.u_scaler         = u_scaler
        self.u_nominal_raw    = np.asarray(u_nominal_raw, dtype=float)
        self.u_nominal_scaled = u_scaler.transform(
            self.u_nominal_raw.reshape(1, -1)).flatten()

        self.du_min_scaled = np.asarray(du_min_scaled, dtype=float)
        self.du_max_scaled = np.asarray(du_max_scaled, dtype=float)

        self.nz   = self.A.shape[0]
        self.nu   = self.B.shape[1]
        self.nx   = self.Cz.shape[0]
        self.nvar = self.NC * self.nu
        self._du_prev = np.zeros(self.nvar)

        self.Sz, self.Su = self._build_prediction_matrices()

        # Build Su_phys: project Su from lifted space to physical state space via Cz.
        # This eliminates spurious gradient contributions from nonlinear observable
        # dimensions that have nothing to do with the position tracking objective.
        # Su_phys[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = Cz @ Su[i*nz:(i+1)*nz, j*nu:(j+1)*nu]
        Su_dense = self.Su.toarray()
        Su_phys  = np.zeros((self.N * self.nx, self.nvar))
        for i in range(self.N):
            for j in range(self.NC):
                Su_phys[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = \
                    self.Cz @ Su_dense[i*self.nz:(i+1)*self.nz, j*self.nu:(j+1)*self.nu]
        self.Su_phys   = sp.csc_matrix(Su_phys)
        self.Qbar_phys = sp.block_diag([sp.csc_matrix(self.Q) for _ in range(self.N)], format="csc")

        self.Rbar  = sp.block_diag([sp.csc_matrix(self.R) for _ in range(self.NC)], format="csc")
        self.D     = self._build_difference_matrix()
        self.Rdbar = (
            sp.block_diag([sp.csc_matrix(self.Rd) for _ in range(self.NC - 1)], format="csc")
            if self.NC > 1 else None
        )

        P     = self._build_hessian()
        Aineq = sp.eye(self.nvar, format="csc")
        l     = np.tile(self.du_min_scaled, self.NC)
        u     = np.tile(self.du_max_scaled, self.NC)

        self.prob = osqp.OSQP()
        self.prob.setup(P=P, q=np.zeros(self.nvar), A=Aineq, l=l, u=u,
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
                Su[i*self.nz:(i+1)*self.nz, j*self.nu:(j+1)*self.nu] = A_pow[i-j] @ self.B
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
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=((self.NC-1)*self.nu, self.NC*self.nu)).tocsc()

    def _build_hessian(self):
        # Use Su_phys so the Hessian reflects only physical-state tracking cost,
        # not spurious costs from nonlinear observable dimensions.
        P = self.Su_phys.T @ self.Qbar_phys @ self.Su_phys + self.Rbar
        if self.D is not None:
            P = P + self.D.T @ self.Rdbar @ self.D
        return (0.5 * (P + P.T)).tocsc()

    def _build_q(self, z0, x_ref_std_horizon):
        """
        Project the lifted free response through Cz to physical state space
        before forming the tracking error. This avoids the spurious gradient
        that would arise from nonlinear observable dimensions (e.g. x^2, sin(phi))
        which have large values in z_free but are not meaningful tracking targets.

        q = Su_phys' Q_phys (x_free - x_ref)
        where x_free[i] = Cz @ z_free[i*nz:(i+1)*nz]  (standardized physical state)
        """
        z_free = self.Sz @ z0

        # Project each horizon step to standardized physical state
        x_free = np.array([
            self.Cz @ z_free[i*self.nz:(i+1)*self.nz]
            for i in range(self.N)
        ]).reshape(-1)                                    # (N*nx,)

        x_ref = x_ref_std_horizon.reshape(-1)            # (N*nx,)

        return np.asarray(
            self.Su_phys.T @ (self.Qbar_phys @ (x_free - x_ref))
        ).reshape(-1)

    def compute(self, z0, x_ref_std_horizon):
        q = self._build_q(z0, x_ref_std_horizon)
        self.prob.update(q=q)
        self.prob.warm_start(x=self._du_prev)
        res = self.prob.solve()

        if res.info.status not in ("solved", "solved inaccurate"):
            print(f"Warning: OSQP EDMD-MPC: {res.info.status}")
            du0_scaled = self._du_prev[:self.nu]
        else:
            du_opt_scaled = np.asarray(res.x).reshape(-1)
            self._du_prev = du_opt_scaled.copy()
            du0_scaled    = du_opt_scaled[:self.nu]

        u0_scaled = self.u_nominal_scaled + du0_scaled
        u0_raw    = self.u_scaler.inverse_transform(u0_scaled.reshape(1, -1)).flatten()
        return u0_raw


# ============================================================
# REFERENCE HORIZON BUILDERS
# ============================================================
def build_reference_horizon(ref_xyz, k, N):
    T = ref_xyz.shape[0]
    x_ref_h = np.zeros((N, 12))
    for i in range(N):
        x_ref_h[i, 0:3] = ref_xyz[min(k+i, T-1)]
    return x_ref_h


def precompute_ref_standardized(ref_xyz, scaler_edmd):
    """Pre-transform full reference trajectory to standardized space (done once)."""
    T = ref_xyz.shape[0]
    X_ref_raw = np.zeros((T, 12))
    X_ref_raw[:, 0:3] = ref_xyz
    return scaler_edmd.transform(X_ref_raw)   # (T, 12)


def build_reference_horizon_standardized(ref_std, k, N):
    T = ref_std.shape[0]
    horizon = np.zeros((N, 12))
    for i in range(N):
        horizon[i] = ref_std[min(k+i, T-1)]
    return horizon


# ============================================================
# METRICS
# ============================================================
def rmse_per_state(X_true, X_pred):
    err = X_true - X_pred
    return np.sqrt(np.mean(err**2, axis=0)), np.sqrt(np.mean(err**2))

def rmse_xyz(ref_xyz, xyz_pred):
    err = ref_xyz - xyz_pred
    return np.sqrt(np.mean(err**2, axis=0)), np.sqrt(np.mean(err**2))

def rmse_over_window(X_true, X_pred, start_idx=0, end_idx=500):
    err = X_true[start_idx:end_idx] - X_pred[start_idx:end_idx]
    return np.sqrt(np.mean(err**2, axis=0)), np.sqrt(np.mean(err**2))


# ============================================================
# MAIN
# ============================================================
def main():
    data_path  = SCRIPT_DIR / DATA_FILE
    model_path = SCRIPT_DIR / EDMDC_MODEL_FILE

    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(data_path)
    model = load_edmdc_model(model_path)

    A_edmd        = model["A"]
    B_edmd        = model["B"]
    scaler_edmd   = model["scaler"]
    u_scaler_edmd = model["u_scaler"]
    dt_model      = model["dt"]
    dt = dt_model if DT_OVERRIDE is None else DT_OVERRIDE

    # ----------------------------------------------------------
    # Sanitize B_edmd: zero out yaw column (index 3).
    # Training data had zero yaw variance:
    #   u_scaler.scale_[3] = 1.0  (no normalization applied)
    #   x_scaler.scale_[11] ≈ 2e-19  (yaw rate never moved)
    # This makes B_edmd[:,3] meaningless — zero it to prevent
    # the optimizer from commanding nonsensical yaw torques.
    # ----------------------------------------------------------
    B_edmd_clean = B_edmd.copy()
    B_edmd_clean[:, 3] = 0.0

    print("Loaded data:", data_path.name)
    print("states_all shape:", states_all.shape)
    print("U_all shape:", U_all.shape)
    print("A_edmd shape:", A_edmd.shape, " B_edmd shape:", B_edmd.shape)
    print("dt used:", dt)
    print("\nB_edmd[:12,:] row norms:", np.linalg.norm(B_edmd[:12, :], axis=1))
    print("u_scaler scale:", u_scaler_edmd.scale_)
    print("x_scaler scale:", scaler_edmd.scale_)

    # --- subsample to model dt ---
    sim_dt = t_all[0, 1] - t_all[0, 0]
    ratio  = dt / sim_dt
    step   = int(round(ratio))
    if not np.isclose(ratio, step, rtol=1e-6, atol=1e-8):
        raise ValueError(f"dt={dt} must be integer multiple of sim_dt={sim_dt}")

    idx           = np.arange(0, t_all.shape[1], step)
    t_all         = t_all[:, idx]
    states_all    = states_all[:, idx, :]
    U_all         = U_all[:, idx, :]
    ref_traj_list = [ref[::step] for ref in ref_traj_list]

    run_idx = (states_all.shape[0] - 1) if TEST_RUN_IDX < 0 \
              else min(TEST_RUN_IDX, states_all.shape[0] - 1)

    t_ref   = t_all[run_idx]
    X_true  = states_all[run_idx]
    U_saved = U_all[run_idx]
    ref_xyz = extract_ref_xyz(ref_traj_list[run_idx])

    T = min(len(t_ref), X_true.shape[0], U_saved.shape[0], ref_xyz.shape[0])
    t_ref   = t_ref[:T];   X_true  = X_true[:T]
    U_saved = U_saved[:T]; ref_xyz = ref_xyz[:T]

    print(f"\nUsing test run: {run_idx}  |  T={T}")

    # --- hover estimate ---
    u1_hover_est  = float(np.mean(U_saved[:, 0]))
    mass_est      = u1_hover_est / G
    u_nominal     = np.array([u1_hover_est, 0.0, 0.0, 0.0], dtype=float)
    print(f"Hover thrust: {u1_hover_est:.4f}  |  Mass: {mass_est:.4f} kg")

    # --- hover-linearized model ---
    Ad, Bd = quadcopter_linearized_model(mass_est, G, IXX, IYY, IZZ, KV, KW, dt)

    # --- cost matrices ---
    Q       = np.diag(Q_DIAG)
    R       = np.diag(R_DIAG)
    Rd      = np.diag(RD_DIAG)
    Q_edmd  = np.diag(Q_DIAG_EDMD)
    R_edmd  = np.diag(R_DIAG_EDMD)
    Rd_edmd = np.diag(RD_DIAG_EDMD)

    # Cz: select standardized physical state (first 12 dims) from lifted vector
    Cz = np.zeros((12, A_edmd.shape[0]))
    Cz[:, :12] = np.eye(12)

    # Pre-standardize reference trajectory once (avoids per-step scaler calls)
    ref_std = precompute_ref_standardized(ref_xyz, scaler_edmd)

    # --- build controllers ---
    lin_mpc = LinearMPC_QP(
        Ad=Ad, Bd=Bd, N=N, NC=NC, Q=Q, R=R, Rd=Rd,
        u_nominal=u_nominal, du_min=DU_MIN, du_max=DU_MAX,
    )

    edmd_mpc = EDMDcMPC_QP(
        A_edmd=A_edmd,
        B_edmd=B_edmd_clean,               # yaw column zeroed
        Cz=Cz,
        N=N_EDMD, NC=NC_EDMD,
        Q=Q_edmd, R=R_edmd, Rd=Rd_edmd,
        u_scaler=u_scaler_edmd,
        du_min_scaled=DU_MIN_EDMD_SCALED,  # pre-computed in scaled space
        du_max_scaled=DU_MAX_EDMD_SCALED,
        u_nominal_raw=u_nominal,
    )

    x0 = X_true[0].copy()

    # --------------------------------------------------------
    # 1) Linear MPC on hover-linearized plant
    # --------------------------------------------------------
    X_lin_mpc = np.zeros((T, 12));  U_lin_mpc = np.zeros((T, 4))
    X_lin_mpc[0] = x0.copy();       x_lin = x0.copy()
    solve_times_lin = []

    print("\nRunning Linear MPC (hover-linearized plant)...")
    for k in range(T - 1):
        if k % 100 == 0:
            print(f"  step {k}/{T-1}")
        x_ref_h = build_reference_horizon(ref_xyz, k, N)
        t0 = time.perf_counter()
        u_lin = lin_mpc.compute(x_lin, x_ref_h)
        solve_times_lin.append(time.perf_counter() - t0)
        U_lin_mpc[k] = u_lin
        x_lin = linearized_step(x_lin, u_lin, Ad, Bd, u_nominal)
        X_lin_mpc[k+1] = x_lin
    U_lin_mpc[-1] = U_lin_mpc[-2]
    print(f"  Avg: {1e3*np.mean(solve_times_lin):.3f} ms  "
          f"Max: {1e3*np.max(solve_times_lin):.3f} ms")
    bad = np.where(np.any(np.abs(X_lin_mpc) > 100, axis=1))[0]
    print(f"  Divergence index: {bad[0] if len(bad) else 'None'}")

    # --------------------------------------------------------
    # 2) EDMDc open-loop rollout (training-style, for reference)
    # --------------------------------------------------------
    print("\nRunning EDMDc open-loop rollout...")
    X_edmd_ol = rollout_edmd_from_controls_training_style(
        X_true[0], U_saved[:-1], A_edmd, B_edmd, scaler_edmd, u_scaler_edmd,
    )

    # --------------------------------------------------------
    # 3) EDMD-MPC in full lifted space
    #    - QP solved in 105-dim lifted space (stable by construction)
    #    - State propagated in full lifted space using B_edmd_clean
    #    - Physical state recovered each step for feedback and logging
    # --------------------------------------------------------
    X_edmd_mpc = np.zeros((T, 12));  U_edmd_mpc = np.zeros((T, 4))
    X_edmd_mpc[0] = x0.copy()

    z_edmd_mpc = lifted_state_from_x(x0, scaler_edmd)
    solve_times_edmd = []

    print("\nRunning EDMD-MPC (full lifted-space QP)...")
    for k in range(T - 1):
        if k % 100 == 0:
            print(f"  step {k}/{T-1}")

        x_ref_std_h = build_reference_horizon_standardized(ref_std, k, N_EDMD)

        t0    = time.perf_counter()
        u_cmd = edmd_mpc.compute(z_edmd_mpc, x_ref_std_h)
        solve_times_edmd.append(time.perf_counter() - t0)

        U_edmd_mpc[k] = u_cmd

        # Propagate in full lifted space using sanitized B
        u_cmd_s    = u_scaler_edmd.transform(u_cmd.reshape(1, -1)).flatten()
        z_edmd_mpc = A_edmd @ z_edmd_mpc + B_edmd_clean @ u_cmd_s
        z_edmd_mpc = np.clip(z_edmd_mpc, -1e6, 1e6)

        X_edmd_mpc[k+1] = scaler_edmd.inverse_transform(
            z_edmd_mpc[:12].reshape(1, -1)).flatten()

    U_edmd_mpc[-1] = U_edmd_mpc[-2]
    print(f"  Avg: {1e3*np.mean(solve_times_edmd):.3f} ms  "
          f"Max: {1e3*np.max(solve_times_edmd):.3f} ms")
    bad = np.where(np.any(np.abs(X_edmd_mpc) > 100, axis=1))[0]
    print(f"  Divergence index: {bad[0] if len(bad) else 'None'}")

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    print("\n--- Control ranges ---")
    for name, U in [("Saved",      U_saved),
                    ("Linear MPC", U_lin_mpc),
                    ("EDMD-MPC",   U_edmd_mpc)]:
        print(f"  {name:12s}  min: {U.min(axis=0)}  max: {U.max(axis=0)}")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    rmse_lin_each,      rmse_lin_total      = rmse_per_state(X_true, X_lin_mpc)
    rmse_edmd_each,     rmse_edmd_total     = rmse_per_state(X_true, X_edmd_ol)
    rmse_edmd_mpc_each, rmse_edmd_mpc_total = rmse_per_state(X_true, X_edmd_mpc)

    rmse_lin_xyz_each,      rmse_lin_xyz_total      = rmse_xyz(ref_xyz, X_lin_mpc[:, 0:3])
    rmse_edmd_xyz_each,     rmse_edmd_xyz_total     = rmse_xyz(ref_xyz, X_edmd_ol[:, 0:3])
    rmse_edmd_mpc_xyz_each, rmse_edmd_mpc_xyz_total = rmse_xyz(ref_xyz, X_edmd_mpc[:, 0:3])

    _, rmse_edmd_5s_total     = rmse_over_window(X_true, X_edmd_ol,  0, min(500, T))
    _, rmse_edmd_mpc_5s_total = rmse_over_window(X_true, X_edmd_mpc, 0, min(500, T))

    print("\n========== RMSE vs SAVED TRAJECTORY ==========")
    print(f"  Linear MPC : {rmse_lin_total:.6f}")
    print(f"  EDMD OL    : {rmse_edmd_total:.6f}")
    print(f"  EDMD-MPC   : {rmse_edmd_mpc_total:.6f}")

    print("\n========== POSITION RMSE vs REFERENCE PATH ==========")
    print(f"  Linear MPC : {rmse_lin_xyz_total:.6f}")
    print(f"  EDMD OL    : {rmse_edmd_xyz_total:.6f}")
    print(f"  EDMD-MPC   : {rmse_edmd_mpc_xyz_total:.6f}")

    print(f"\nEDMD OL  RMSE first 5 s: {rmse_edmd_5s_total:.6f}")
    print(f"EDMD-MPC RMSE first 5 s: {rmse_edmd_mpc_5s_total:.6f}")

    labels = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"]
    print()
    for i, lbl in enumerate(labels):
        print(f"  {lbl:>6s} | Lin {rmse_lin_each[i]:.6f} | "
              f"OL {rmse_edmd_each[i]:.6f} | MPC {rmse_edmd_mpc_each[i]:.6f}")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(ref_xyz[:, 0],   ref_xyz[:, 1],   ref_xyz[:, 2],   "k",
            lw=2.0, label="Reference")
    ax.plot(X_true[:, 0],    X_true[:, 1],    X_true[:, 2],
            color="gray", lw=1.4, label="Saved run")
    ax.plot(X_lin_mpc[:, 0], X_lin_mpc[:, 1], X_lin_mpc[:, 2], "r-.",
            lw=1.5, label="Linear MPC")
    ax.plot(X_edmd_ol[:, 0], X_edmd_ol[:, 1], X_edmd_ol[:, 2], "b--",
            lw=1.5, label="EDMDc OL")
    ax.plot(X_edmd_mpc[:,0], X_edmd_mpc[:,1], X_edmd_mpc[:,2], "g-",
            lw=1.5, label="EDMD-MPC")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Trajectory comparison"); ax.legend(); ax.grid(True)

    units = ["m","m","m","m/s","m/s","m/s","rad","rad","rad","rad/s","rad/s","rad/s"]
    fig2, axs = plt.subplots(4, 3, figsize=(16, 10))
    for i, ax in enumerate(axs.flatten()):
        ax.plot(t_ref, X_true[:, i],    color="gray", lw=1.2, label="Saved run")
        ax.plot(t_ref, X_lin_mpc[:, i], "r-.", lw=1.1, label="Linear MPC")
        ax.plot(t_ref, X_edmd_ol[:, i], "b--", lw=1.1, label="EDMDc OL")
        ax.plot(t_ref, X_edmd_mpc[:,i], "g-",  lw=1.1, label="EDMD-MPC")
        ax.set_title(f"{labels[i]} | Lin {rmse_lin_each[i]:.3f} | "
                     f"OL {rmse_edmd_each[i]:.3f} | MPC {rmse_edmd_mpc_each[i]:.3f}")
        ax.set_xlabel("t [s]"); ax.set_ylabel(f"{labels[i]} [{units[i]}]")
        ax.grid(True)
        if i == 0:
            ax.legend()
    fig2.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(t_ref, np.linalg.norm(X_lin_mpc[:, 0:3]  - ref_xyz, axis=1), label="Linear MPC")
    plt.plot(t_ref, np.linalg.norm(X_edmd_ol[:, 0:3]  - ref_xyz, axis=1), label="EDMDc OL")
    plt.plot(t_ref, np.linalg.norm(X_edmd_mpc[:, 0:3] - ref_xyz, axis=1), label="EDMD-MPC")
    plt.xlabel("t [s]"); plt.ylabel("Position error [m]")
    plt.title("Position error to reference"); plt.grid(True); plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()