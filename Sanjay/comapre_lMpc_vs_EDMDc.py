import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

# Must exist in your project
from quadcopter_dynamics import quadcopter_dynamics


# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Change these two as needed
DATA_FILE = "runs_traj2_n200.pkl"
EDMDC_MODEL_FILE = "edmdc_model_traj2_n200.pkl"

# Pick which saved run to use as the moving reference.
# -1 means last run in the file.
TEST_RUN_IDX = -1

# If None, use the dt stored in the EDMDc model file.
# If you want to force a dt, set e.g. 0.01 or 0.1.
DT_OVERRIDE = None

# If True, both controllers are applied to the same nonlinear plant.
# If False, they are applied to the linear predictor.
USE_NONLINEAR_PLANT = True

# If the run is too slow, reduce these first.
N = 5          # prediction horizon
NC = 2         # control horizon

# physical parameters
MASS = 1.0
G = 9.81
IXX = 0.01
IYY = 0.01
IZZ = 0.02
KV = 0.1
KW = 0.01

# cost weights
Q_DIAG = np.array([30, 30, 30,   2, 2, 2,   1, 1, 1,   0.2, 0.2, 0.2], dtype=float)
R_DIAG = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
RD_DIAG = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)

# simple state constraints
Y_MIN = np.array([-10, -10, -1,  -5, -5, -5,  -0.8, -0.8, -np.pi, -4, -4, -4], dtype=float)
Y_MAX = np.array([ 10,  10, 12,   5,  5,  5,   0.8,  0.8,  np.pi,  4,  4,  4], dtype=float)

# input increments around nominal
DU_MIN = np.array([-0.5, -0.2, -0.2, -0.1], dtype=float)
DU_MAX = np.array([ 0.5,  0.2,  0.2,  0.1], dtype=float)

DU_MIN_EDMD = np.array([-0.1, -0.05, -0.05, -0.05], dtype=float)
DU_MAX_EDMD = np.array([ 0.1,  0.05,  0.05,  0.05], dtype=float)
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
# HELPER: robust reference extraction
# ref_traj_list may be:
# - array shape (T, 3+) or (3+, T)
# - list/tuple of x,y,z arrays
# - list of dicts with "pos"
# ============================================================
def extract_ref_xyz(ref_item):
    # Case 1: list of dicts like ref_traj[k]["pos"]
    if isinstance(ref_item, list) and len(ref_item) > 0 and isinstance(ref_item[0], dict):
        if "pos" in ref_item[0]:
            pts = [np.asarray(d["pos"]).reshape(-1)[:3] for d in ref_item]
            return np.asarray(pts, dtype=float)

    arr = np.asarray(ref_item, dtype=object if isinstance(ref_item, list) else None)

    # Case 2: already numeric (T, 3+) or (T, 3)
    try:
        arr_num = np.asarray(ref_item, dtype=float)
        if arr_num.ndim == 2 and arr_num.shape[1] >= 3:
            return arr_num[:, :3]
        if arr_num.ndim == 2 and arr_num.shape[0] >= 3:
            return arr_num[:3, :].T
    except Exception:
        pass

    # Case 3: tuple/list like [x_ref, y_ref, z_ref]
    if isinstance(ref_item, (list, tuple)) and len(ref_item) >= 3:
        x = np.asarray(ref_item[0]).reshape(-1)
        y = np.asarray(ref_item[1]).reshape(-1)
        z = np.asarray(ref_item[2]).reshape(-1)
        T = min(len(x), len(y), len(z))
        return np.column_stack((x[:T], y[:T], z[:T])).astype(float)

    raise ValueError(
        f"Unsupported ref trajectory format. "
        f"type={type(ref_item)}, shape={getattr(np.asarray(ref_item), 'shape', None)}"
    )


# ============================================================
# LINEAR MODEL
# ============================================================
def quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts):
    nx = 12
    nu = 4

    Ac = np.zeros((nx, nx))
    Bc = np.zeros((nx, nu))

    # position kinematics
    Ac[0, 3] = 1.0
    Ac[1, 4] = 1.0
    Ac[2, 5] = 1.0

    # translational dynamics about hover
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

    # angular dynamics
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
    return Ad, Bd, Ac, Bc


# ============================================================
# EDMDC OBSERVABLES
# This must match your EDMDc training script structure.
# Your trained model shown in logs had A shape (105, 105), B shape (105, 4),
# consistent with 12 linear + quadratic terms + selected cubic + energy +
# trig + bias. :contentReference[oaicite:1]{index=1}
# ============================================================
def observables_edmd(x, scaler):
    x = np.asarray(x).flatten()
    x_s = scaler.transform(x.reshape(1, -1)).flatten()

    obs = list(x_s)
    n = len(x_s)

    # quadratic terms
    for i, j in itertools.combinations_with_replacement(range(n), 2):
        obs.append(x_s[i] * x_s[j])

    # selected cubic on position and velocity only
    pos_vel_indices = [0, 1, 2, 3, 4, 5]
    for i in pos_vel_indices:
        obs.append(x_s[i] ** 3)

    # energy-like features
    vx, vy, vz = x_s[3], x_s[4], x_s[5]
    p, q, r = x_s[9], x_s[10], x_s[11]
    obs.append(vx**2 + vy**2 + vz**2)
    obs.append(p**2 + q**2 + r**2)

    # trig terms on angles
    # Keep consistent with your current training code assumption.
    phi_raw = x_s[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_raw = x_s[7] * scaler.scale_[7] + scaler.mean_[7]
    psi_raw = x_s[8] * scaler.scale_[8] + scaler.mean_[8]

    phi_rad = phi_raw
    theta_rad = theta_raw
    psi_rad = psi_raw

    obs += [
        np.sin(psi_rad), np.cos(psi_rad),
        np.sin(phi_rad), np.cos(phi_rad),
        np.sin(theta_rad), np.cos(theta_rad),
    ]

    obs.append(1.0)
    return np.asarray(obs, dtype=float)


def edmd_step(x, u, A_edmd, B_edmd, scaler_edmd, u_scaler_edmd, clip_value=1e6):
    z = observables_edmd(x, scaler_edmd)
    u_s = u_scaler_edmd.transform(u.reshape(1, -1)).flatten()
    z_next = A_edmd @ z + B_edmd @ u_s
    z_next = np.clip(z_next, -clip_value, clip_value)
    x_next = scaler_edmd.inverse_transform(z_next[:12].reshape(1, -1)).flatten()
    return x_next


# ============================================================
# GENERIC MPC
# ============================================================
class GenericMPC:
    """
    Decision variable: delta-u over first NC steps.
    After NC, last input is held.
    """
    def __init__(self, predictor, nx, nu, N, NC, Q, R, Rd, u_nominal,
                 y_min=None, y_max=None, du_min=None, du_max=None):
        self.predictor = predictor
        self.nx = nx
        self.nu = nu
        self.N = N
        self.NC = NC
        self.Q = Q
        self.R = R
        self.Rd = Rd
        self.u_nominal = u_nominal.copy()

        self.y_min = y_min if y_min is not None else -np.inf * np.ones(nx)
        self.y_max = y_max if y_max is not None else  np.inf * np.ones(nx)

        self.du_min = du_min if du_min is not None else -np.inf * np.ones(nu)
        self.du_max = du_max if du_max is not None else  np.inf * np.ones(nu)

        self._u_prev = np.zeros(self.nu * self.NC)

    def _build_u_seq(self, du_flat):
        du = du_flat.reshape(self.NC, self.nu)
        u_seq = []
        u_k = self.u_nominal.copy()

        for i in range(self.N):
            if i < self.NC:
                u_k = u_k + du[i]
            u_seq.append(u_k.copy())

        return np.asarray(u_seq), du

    def _predict(self, x0, du_flat):
        u_seq, _ = self._build_u_seq(du_flat)
        X = np.zeros((self.N + 1, self.nx))
        X[0] = x0.copy()

        x = x0.copy()
        for i in range(self.N):
            x = self.predictor(x, u_seq[i])
            X[i + 1] = x

        return X, u_seq

    def _cost(self, du_flat, x0, x_ref_horizon):
        X, _ = self._predict(x0, du_flat)
        _, du = self._build_u_seq(du_flat)

        J = 0.0

        for i in range(1, self.N + 1):
            e = X[i] - x_ref_horizon[i - 1]
            J += e @ self.Q @ e
            J += 1e-3 * (X[i] @ X[i])  # state magnitude penalty

        for i in range(self.NC):
            du_i = du[i]
            J += du_i @ self.R @ du_i

        for i in range(1, self.NC):
            diff = du[i] - du[i - 1]
            J += diff @ self.Rd @ diff

        return J

    def _state_constraints(self, du_flat, x0):
        X, _ = self._predict(x0, du_flat)
        cons = []
        for i in range(1, self.N + 1):
            cons.append(X[i] - self.y_min)
            cons.append(self.y_max - X[i])
        return np.concatenate(cons)

    def _input_bounds(self):
        bounds = []
        for _ in range(self.NC):
            for j in range(self.nu):
                bounds.append((self.du_min[j], self.du_max[j]))
        return bounds

    def compute(self, x0, x_ref_horizon):
        res = minimize(
            self._cost,
            self._u_prev,
            args=(x0, x_ref_horizon),
            method="SLSQP",
            bounds=self._input_bounds(),
            #constraints=constraints,
            options={"maxiter": 120, "ftol": 1e-4, "disp": False},
        )

        if not res.success:
            print("Warning: optimizer did not fully converge:", res.message)
            du_opt = self._u_prev.reshape(self.NC, self.nu)
            u_opt = self.u_nominal + du_opt[0]
            return u_opt

        self._u_prev = res.x
        du_opt = res.x.reshape(self.NC, self.nu)
        u_opt = self.u_nominal + du_opt[0]
        return u_opt


# ============================================================
# HELPERS
# ============================================================
def build_reference_horizon(ref_xyz, k, N):
    """
    ref_xyz shape (T, 3)
    Build N-step 12-state reference with xyz filled and others zero.
    """
    T = ref_xyz.shape[0]
    x_ref_h = np.zeros((N, 12))
    for i in range(N):
        idx = min(k + i, T - 1)
        x_ref_h[i, 0:3] = ref_xyz[idx, :3]
    return x_ref_h


def rmse_per_state(X_true, X_pred):
    err = X_true - X_pred
    return np.sqrt(np.mean(err**2, axis=0)), np.sqrt(np.mean(err**2))


# ============================================================
# MAIN
# ============================================================
def main():
    data_path = SCRIPT_DIR / DATA_FILE
    model_path = SCRIPT_DIR / EDMDC_MODEL_FILE

    # --------------------------------------------------------
    # Load dataset and model
    # --------------------------------------------------------
    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(data_path)
    edmd_model = load_edmdc_model(model_path)

    n_runs = states_all.shape[0]

    A_edmd = edmd_model["A"]
    B_edmd = edmd_model["B"]
    scaler_edmd = edmd_model["scaler"]
    u_scaler_edmd = edmd_model["u_scaler"]
    dt_edmd = edmd_model["dt"]

    dt = dt_edmd if DT_OVERRIDE is None else DT_OVERRIDE

    print("Loaded data:", data_path.name)
    print("Detected n_runs:", n_runs)
    print("states_all shape:", states_all.shape)
    print("U_all shape:", U_all.shape)
    print("Loaded EDMDc model:", model_path.name)
    print("A_edmd shape:", A_edmd.shape)
    print("B_edmd shape:", B_edmd.shape)
    print("dt used:", dt)

    # --------------------------------------------------------
    # Downsample dataset to dt if needed
    # --------------------------------------------------------
    sim_dt = t_all[0, 1] - t_all[0, 0]
    ratio = dt / sim_dt
    step = int(round(ratio))

    if not np.isclose(ratio, step, rtol=1e-6, atol=1e-8):
        raise ValueError(f"dt={dt} must be integer multiple of sim dt={sim_dt}")

    idx = np.arange(0, t_all.shape[1], step)
    t_all_ds = t_all[:, idx]
    states_all_ds = states_all[:, idx, :]
    ref_traj_list_ds = [ref[::step] for ref in ref_traj_list]

    print("Downsampled states shape:", states_all_ds.shape)
    print("Downsample step:", step)

    # --------------------------------------------------------
    # Select test run
    # --------------------------------------------------------
    n_runs_ds = states_all_ds.shape[0]
    if TEST_RUN_IDX is None or TEST_RUN_IDX < 0:
        run_idx = n_runs_ds - 1
    else:
        run_idx = min(TEST_RUN_IDX, n_runs_ds - 1)

    t_ref = t_all_ds[run_idx]
    X_ref_true = states_all_ds[run_idx]

    ref_item = ref_traj_list_ds[run_idx]
    print("Raw ref item type:", type(ref_item))
    print("Raw ref item shape:", getattr(np.asarray(ref_item), "shape", None))

    ref_xyz = extract_ref_xyz(ref_item)

    print("Using test run:", run_idx)
    print("Reference length:", len(t_ref))
    print("Reference xyz shape:", ref_xyz.shape)

    # Align lengths if needed
    T = min(len(t_ref), len(ref_xyz), X_ref_true.shape[0])
    t_ref = t_ref[:T]
    X_ref_true = X_ref_true[:T]
    ref_xyz = ref_xyz[:T]

    # --------------------------------------------------------
    # Build predictors
    # --------------------------------------------------------
    Ad, Bd, Ac, Bc = quadcopter_linearized_model(
        MASS, G, IXX, IYY, IZZ, KV, KW, dt
    )

    u_nominal = np.array([MASS * G, 0.0, 0.0, 0.0], dtype=float)

    def linear_predictor(x, u):
        du = u - u_nominal
        return Ad @ x + Bd @ du

    def edmd_predictor(x, u):
        return edmd_step(x, u, A_edmd, B_edmd, scaler_edmd, u_scaler_edmd)

    x_test0 = np.zeros(12)
    u_hover = np.array([MASS * G, 0.0, 0.0, 0.0], dtype=float)
    u_edmd_bad = np.array([9.31, 0.2, -0.2, 0.0], dtype=float)

    x_lin_hover = linear_predictor(x_test0, u_hover)
    x_edmd_hover = edmd_predictor(x_test0, u_hover)

    x_lin_bad = linear_predictor(x_test0, u_edmd_bad)
    x_edmd_bad = edmd_predictor(x_test0, u_edmd_bad)

    print("\nOne-step from x=0 under hover:")
    print("Linear:", x_lin_hover)
    print("EDMDc :", x_edmd_hover)

    print("\nOne-step from x=0 under saturated EDMD-like input:")
    print("Linear:", x_lin_bad)
    print("EDMDc :", x_edmd_bad)

    Q = np.diag(Q_DIAG)
    R = np.diag(R_DIAG)
    Rd = np.diag(RD_DIAG)

    # --------------------------------------------------------
    # One-step EDMD sanity check on real samples from saved run
    # --------------------------------------------------------
    print("\nOne-step EDMD sanity check on real data samples:")

    sample_indices = [0, 1, 2, 5, 10, 20, 50]
    for k in sample_indices:
        xk = X_ref_true[k]
        uk = U_all[run_idx, k]  # use original saved control at same step
        x_true_next = X_ref_true[min(k + 1, len(X_ref_true) - 1)]

        x_edmd_next = edmd_predictor(xk, uk)
        x_lin_next = linear_predictor(xk, uk)

        err_edmd = np.linalg.norm(x_true_next - x_edmd_next)
        err_lin = np.linalg.norm(x_true_next - x_lin_next)

        print(f"k={k}")
        print("  ||true - EDMD|| =", err_edmd)
        print("  ||true - Linear|| =", err_lin)
        print("  true next xyz   =", x_true_next[:3])
        print("  edmd next xyz   =", x_edmd_next[:3])
        print("  linear next xyz =", x_lin_next[:3])

    mpc_lin = GenericMPC(
        predictor=linear_predictor,
        nx=12, nu=4, N=N, NC=NC,
        Q=Q, R=R, Rd=Rd,
        u_nominal=u_nominal,
        y_min=Y_MIN, y_max=Y_MAX,
        du_min=DU_MIN, du_max=DU_MAX,
    )

    mpc_edmd = GenericMPC(
        predictor=edmd_predictor,
        nx=12, nu=4, N=N, NC=NC,
        Q=Q, R=R, Rd=Rd,
        u_nominal=u_nominal,
        y_min=Y_MIN, y_max=Y_MAX,
        du_min=DU_MIN_EDMD, du_max=DU_MAX_EDMD,
    )

    # --------------------------------------------------------
    # Closed-loop simulation on same plant
    # --------------------------------------------------------
    X_lin = np.zeros((T, 12))
    X_edmd = np.zeros((T, 12))
    U_lin = np.zeros((T, 4))
    U_edmd = np.zeros((T, 4))

    x0 = np.zeros(12)
    x_lin = x0.copy()
    x_edmd = x0.copy()

    X_lin[0] = x_lin
    X_edmd[0] = x_edmd

    print("Running controller comparison...")
    for k in range(T - 1):
        if k % 25 == 0:
            print(f" step {k}/{T-1}")

        x_ref_h = build_reference_horizon(ref_xyz, k, N)

        u_lin = mpc_lin.compute(x_lin, x_ref_h)
        u_edmd = mpc_edmd.compute(x_edmd, x_ref_h)

        U_lin[k] = u_lin
        U_edmd[k] = u_edmd

        if USE_NONLINEAR_PLANT:
            xdot_lin = quadcopter_dynamics(x_lin, u_lin, Ac, Bc, MASS, G)
            xdot_edmd = quadcopter_dynamics(x_edmd, u_edmd, Ac, Bc, MASS, G)

            x_lin = x_lin + dt * xdot_lin
            x_edmd = x_edmd + dt * xdot_edmd
        else:
            x_lin = linear_predictor(x_lin, u_lin)
            x_edmd = linear_predictor(x_edmd, u_edmd)

        X_lin[k + 1] = x_lin
        X_edmd[k + 1] = x_edmd


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    print("First 10 linear controls:\n", U_lin[:10])
    print("First 10 EDMD controls:\n", U_edmd[:10])
    rmse_lin_each, rmse_lin_total = rmse_per_state(X_ref_true, X_lin)
    rmse_edmd_each, rmse_edmd_total = rmse_per_state(X_ref_true, X_edmd)

    print("\n========== TRACKING RMSE ==========")
    print(f"Linear MPC total RMSE: {rmse_lin_total:.6f}")
    print(f"EDMDc  MPC total RMSE: {rmse_edmd_total:.6f}")
    print("===================================")

    labels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']
    for i, lbl in enumerate(labels):
        print(f"{lbl:>6s} | Linear MPC: {rmse_lin_each[i]:.6f} | EDMDc MPC: {rmse_edmd_each[i]:.6f}")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], 'k', lw=2.0, label='Reference')
    ax.plot(X_lin[:, 0], X_lin[:, 1], X_lin[:, 2], 'r-.', lw=1.5, label='Linear MPC')
    ax.plot(X_edmd[:, 0], X_edmd[:, 1], X_edmd[:, 2], 'b--', lw=1.5, label='EDMDc MPC')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Linear MPC vs EDMDc MPC')
    ax.legend()
    ax.grid(True)

    units = ['m', 'm', 'm', 'm/s', 'm/s', 'm/s', 'deg', 'deg', 'deg', 'rad/s', 'rad/s', 'rad/s']
    fig2, axs = plt.subplots(4, 3, figsize=(16, 10))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.plot(t_ref, X_ref_true[:, i], 'k', lw=1.2, label='Saved run')
        ax.plot(t_ref, X_lin[:, i], 'r-.', lw=1.1, label='Linear MPC')
        ax.plot(t_ref, X_edmd[:, i], 'b--', lw=1.1, label='EDMDc MPC')
        ax.set_title(f"{labels[i]} | Lin {rmse_lin_each[i]:.3f} | EDMD {rmse_edmd_each[i]:.3f}")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f"{labels[i]} [{units[i]}]")
        ax.grid(True)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()