"""
real_data_edmdc_mpc_transfer_yaw.py
===================================
Train EDMDc from real flight log data (including yaw), then run the
real-data-trained controller on the simulation plant as a transfer test.

Data sources (from ArduPilot .xlsx log):
  - AHR2: Roll, Pitch, Yaw, Lat, Lng, Alt
  - ATT:  DesRoll, DesPitch, DesYaw
  - CTUN: ThO

State vector (12):
  [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]

Control input (4):
  [thrust, des_roll, des_pitch, des_yaw]

Experiment:
  1. Load multiple real flight logs
  2. Extract 12-state trajectories + 4-input control
  3. Train EDMDc on real logs
  4. Evaluate rollout on held-out real flight
  5. Build MPC from the real-data-trained EDMDc model
  6. Run that controller on the simulation plant (transfer test)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv
from pyproj import Transformer
import matplotlib.pyplot as plt
import cvxpy as cp


# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "flight_data"   # folder with Data_*.xlsx logs

dt_edmd = 0.10
ROLLOUT_HORIZON = 20

SAVGOL_WINDOW = 11
SAVGOL_POLY = 3

TEST_FLIGHT_IDX = -1   # last flight
LAMBDA_CANDIDATES = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)


# ============================================================
# HELPERS
# ============================================================
def find_column(df, keywords, sheet_name, file_path):
    """Find a column matching any of the keywords (case-insensitive)."""
    for col in df.columns:
        col_lower = str(col).lower()
        if any(k.lower() in col_lower for k in keywords):
            return col
    raise KeyError(f"[{file_path.name}:{sheet_name}] no column matches {keywords}")


def downsample_to_dt(t, states, U, dt_target):
    """Downsample approximately to target dt."""
    dt_actual = np.mean(np.diff(t))
    step = max(1, int(round(dt_target / dt_actual)))
    idx = np.arange(0, len(t), step)
    return t[idx], states[idx], U[idx]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def wrap_angle_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ============================================================
# DATA LOADING
# ============================================================
def load_flight(file_path):
    """
    Load a single flight log and extract states + control inputs.

    Returns
    -------
    t : (T,)
    states : (T, 12)
      [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
    U : (T, 4)
      [thrust, des_roll, des_pitch, des_yaw]
    """
    fp = Path(file_path)

    df_ahr2 = pd.read_excel(fp, sheet_name="AHR2", engine="openpyxl")
    df_att = pd.read_excel(fp, sheet_name="ATT", engine="openpyxl")
    df_ctun = pd.read_excel(fp, sheet_name="CTUN", engine="openpyxl")

    # --- time from AHR2 ---
    t_col = find_column(df_ahr2, ["timeus", "time"], "AHR2", fp)
    t_ahr2 = df_ahr2[t_col].astype(float).values
    if "us" in t_col.lower() or t_ahr2[0] > 1e9:
        t_ahr2 = t_ahr2 * 1e-6
    t_ahr2 -= t_ahr2[0]

    # --- attitude ---
    roll_deg = df_ahr2[find_column(df_ahr2, ["roll"], "AHR2", fp)].values.astype(float)
    pitch_deg = df_ahr2[find_column(df_ahr2, ["pitch"], "AHR2", fp)].values.astype(float)
    yaw_deg = df_ahr2[find_column(df_ahr2, ["yaw"], "AHR2", fp)].values.astype(float)

    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.unwrap(np.deg2rad(yaw_deg))

    # --- position from lat/lon/alt ---
    lat = df_ahr2[find_column(df_ahr2, ["lat"], "AHR2", fp)].values.astype(float)
    lon = df_ahr2[find_column(df_ahr2, ["lng", "lon"], "AHR2", fp)].values.astype(float)
    alt = df_ahr2[find_column(df_ahr2, ["alt"], "AHR2", fp)].values.astype(float)

    east, north = transformer.transform(lon, lat)
    x = north - north[0]
    y = east - east[0]
    z = alt - alt[0]

    # --- smooth positions before differentiation ---
    if len(x) > SAVGOL_WINDOW:
        x = savgol_filter(x, SAVGOL_WINDOW, SAVGOL_POLY)
        y = savgol_filter(y, SAVGOL_WINDOW, SAVGOL_POLY)
        z = savgol_filter(z, SAVGOL_WINDOW, SAVGOL_POLY)

    dt_actual = np.mean(np.diff(t_ahr2))

    # --- translational velocities ---
    vx = np.gradient(x, dt_actual)
    vy = np.gradient(y, dt_actual)
    vz = np.gradient(z, dt_actual)

    # --- angular rates ---
    p = np.gradient(roll, dt_actual)
    q = np.gradient(pitch, dt_actual)
    r = np.gradient(yaw, dt_actual)

    if len(vx) > SAVGOL_WINDOW:
        vx = savgol_filter(vx, SAVGOL_WINDOW, SAVGOL_POLY)
        vy = savgol_filter(vy, SAVGOL_WINDOW, SAVGOL_POLY)
        vz = savgol_filter(vz, SAVGOL_WINDOW, SAVGOL_POLY)
        p = savgol_filter(p, SAVGOL_WINDOW, SAVGOL_POLY)
        q = savgol_filter(q, SAVGOL_WINDOW, SAVGOL_POLY)
        r = savgol_filter(r, SAVGOL_WINDOW, SAVGOL_POLY)

    # --- controls from ATT + CTUN ---
    # ATT times
    t_att_col = find_column(df_att, ["timeus", "time"], "ATT", fp)
    t_att = df_att[t_att_col].astype(float).values
    if "us" in t_att_col.lower() or t_att[0] > 1e9:
        t_att = t_att * 1e-6
    t_att -= t_att[0]

    des_roll_deg = df_att[find_column(df_att, ["desroll"], "ATT", fp)].values.astype(float)
    des_pitch_deg = df_att[find_column(df_att, ["despitch"], "ATT", fp)].values.astype(float)
    des_yaw_deg = df_att[find_column(df_att, ["desyaw"], "ATT", fp)].values.astype(float)

    des_roll = np.deg2rad(des_roll_deg)
    des_pitch = np.deg2rad(des_pitch_deg)
    des_yaw = np.unwrap(np.deg2rad(des_yaw_deg))

    # CTUN times
    t_ctun_col = find_column(df_ctun, ["timeus", "time"], "CTUN", fp)
    t_ctun = df_ctun[t_ctun_col].astype(float).values
    if "us" in t_ctun_col.lower() or t_ctun[0] > 1e9:
        t_ctun = t_ctun * 1e-6
    t_ctun -= t_ctun[0]

    tho_col = find_column(df_ctun, ["tho"], "CTUN", fp)
    thrust_raw = df_ctun[tho_col].values.astype(float)

    # crude thrust normalization into Newton-ish range
    if np.max(thrust_raw) > 10:
        thrust_raw = thrust_raw / np.max(thrust_raw)
    thrust_N = thrust_raw * 9.81

    # interpolate controls to AHR2 time base
    thrust_interp = np.interp(t_ahr2, t_ctun, thrust_N)
    des_roll_interp = np.interp(t_ahr2, t_att, des_roll)
    des_pitch_interp = np.interp(t_ahr2, t_att, des_pitch)
    des_yaw_interp = np.interp(t_ahr2, t_att, des_yaw)

    states = np.column_stack([
        x, y, z,
        vx, vy, vz,
        roll, pitch, yaw,
        p, q, r
    ])

    U = np.column_stack([
        thrust_interp,
        des_roll_interp,
        des_pitch_interp,
        des_yaw_interp
    ])

    print(f"  Loaded {fp.name}: T={len(t_ahr2)}, dt={dt_actual:.4f}s, duration={t_ahr2[-1]:.1f}s")
    print(f"    pos range: x=[{x.min():.2f},{x.max():.2f}] y=[{y.min():.2f},{y.max():.2f}] z=[{z.min():.2f},{z.max():.2f}]")
    print(f"    yaw range (deg, unwrapped): [{np.rad2deg(yaw.min()):.1f}, {np.rad2deg(yaw.max()):.1f}]")

    return t_ahr2, states, U


# ============================================================
# OBSERVABLES
# First 12 entries are always the standardized physical states.
# ============================================================
def observables(x_std, scaler):
    """
    x_std: standardized 12-state vector
    returns lifted observable vector whose first 12 entries are x_std
    """
    x = np.asarray(x_std).flatten()
    assert len(x) == 12, f"Expected 12-state, got {len(x)}"

    obs = list(x)

    # unscale angles to physical radians
    roll = x[6] * scaler.scale_[6] + scaler.mean_[6]
    pitch = x[7] * scaler.scale_[7] + scaler.mean_[7]
    yaw = x[8] * scaler.scale_[8] + scaler.mean_[8]

    # trig
    obs.extend([
        np.sin(roll), np.cos(roll),
        np.sin(pitch), np.cos(pitch),
        np.sin(yaw), np.cos(yaw),
    ])

    # cross terms
    obs.extend([
        x[6] * x[9],     # roll * p
        x[7] * x[10],    # pitch * q
        x[8] * x[11],    # yaw * r
        x[3] * x[6],     # vx * roll
        x[4] * x[7],     # vy * pitch
        x[3] * x[8],     # vx * yaw
        x[4] * x[8],     # vy * yaw
        x[5] * x[7],     # vz * pitch
    ])

    # energy-like
    obs.extend([
        x[3] ** 2 + x[4] ** 2 + x[5] ** 2,       # translational speed^2
        x[9] ** 2 + x[10] ** 2 + x[11] ** 2,     # rotational rate^2
    ])

    # additional quadratics
    obs.extend([
        x[5] ** 2,
        x[6] ** 2,
        x[7] ** 2,
        x[8] ** 2,
        x[9] * x[10],
        x[10] * x[11],
        x[9] * x[11],
    ])

    # bias
    obs.append(1.0)

    return np.array(obs, dtype=float)


# ============================================================
# EDMDc TRAINING
# ============================================================
def train_edmdc(Psi, Phi, U_std, lam):
    Omega = np.vstack([Psi, U_std])

    if lam == 0:
        AB = Phi @ pinv(Omega)
    else:
        G = Omega @ Omega.T
        AB = Phi @ Omega.T @ np.linalg.inv(G + lam * np.eye(G.shape[0]))

    n_obs = Psi.shape[0]
    A = AB[:, :n_obs]
    B = AB[:, n_obs:]

    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho > 1.0:
        A = A / rho
        print(f"  Stabilized A with rho={rho:.6f}")

    return A, B


def rolling_horizon_rmse(states, inputs, A, B, scaler, u_scaler, horizon):
    n_total = states.shape[0]
    n_windows = n_total - horizon
    if n_windows <= 0:
        return float("inf"), float("inf"), float("inf")

    pos_list = []
    vel_list = []
    yaw_list = []

    stride = max(1, n_windows // 100)

    for start in range(0, n_windows, stride):
        seg_s = states[start:start + horizon + 1]
        seg_u = inputs[start:start + horizon]

        psi = np.zeros((A.shape[0], horizon + 1))
        x0_std = scaler.transform(seg_s[0].reshape(1, -1)).flatten()
        psi[:, 0] = observables(x0_std, scaler)

        for k in range(horizon):
            u_std = u_scaler.transform(seg_u[k].reshape(1, -1)).flatten()
            psi[:, k + 1] = A @ psi[:, k] + B @ u_std

        x_pred = scaler.inverse_transform(psi[:12, :].T)
        err = seg_s - x_pred

        pos_list.append(np.sqrt(np.mean(err[:, 0:3] ** 2)))
        vel_list.append(np.sqrt(np.mean(err[:, 3:6] ** 2)))

        yaw_err = wrap_angle_pi(err[:, 8])
        yaw_list.append(np.sqrt(np.mean(yaw_err ** 2)))

    return float(np.mean(pos_list)), float(np.mean(vel_list)), float(np.mean(yaw_list))


# ============================================================
# SELF-CONTAINED MPC
# ============================================================
class EDMDcMPC_QP:
    """
    MPC in lifted coordinates.
    Optimization variable is u in standardized control coordinates.
    """
    def __init__(
        self,
        A, B, Cz,
        N,
        Q, R, Rd,
        u_scaler,
        u_nominal_raw,
        u_min_raw,
        u_max_raw,
        du_min_raw,
        du_max_raw,
        solver="OSQP",
    ):
        self.A = A
        self.B = B
        self.Cz = Cz
        self.N = N
        self.Q = Q
        self.R = R
        self.Rd = Rd
        self.u_scaler = u_scaler
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.ny = Cz.shape[0]
        self.solver = solver

        # convert raw control quantities to standardized coordinates
        self.u_nominal_std = self._to_std(u_nominal_raw)
        self.u_min_std = self._to_std(u_min_raw)
        self.u_max_std = self._to_std(u_max_raw)

        # delta-u transforms linearly with scaling
        self.du_min_std = np.asarray(du_min_raw, dtype=float) / self.u_scaler.scale_
        self.du_max_std = np.asarray(du_max_raw, dtype=float) / self.u_scaler.scale_

    def _to_std(self, u_raw):
        u_raw = np.asarray(u_raw, dtype=float).reshape(1, -1)
        return self.u_scaler.transform(u_raw).flatten()

    def _to_raw(self, u_std):
        u_std = np.asarray(u_std, dtype=float).reshape(1, -1)
        return self.u_scaler.inverse_transform(u_std).flatten()

    def compute(self, z0, x_ref_h_std, u_prev_raw):
        """
        z0 : lifted state
        x_ref_h_std : (ny, N) standardized physical-state reference
        u_prev_raw : previous control in raw units
        """
        u_prev_std = self._to_std(u_prev_raw)

        z = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.N))

        cost = 0
        constr = [z[:, 0] == z0]

        for k in range(self.N):
            yk = self.Cz @ z[:, k]
            rk = x_ref_h_std[:, k]

            cost += cp.quad_form(yk - rk, self.Q)
            cost += cp.quad_form(u[:, k] - self.u_nominal_std, self.R)

            if k == 0:
                duk = u[:, k] - u_prev_std
            else:
                duk = u[:, k] - u[:, k - 1]

            cost += cp.quad_form(duk, self.Rd)

            constr += [
                z[:, k + 1] == self.A @ z[:, k] + self.B @ u[:, k],
                self.u_min_std <= u[:, k],
                u[:, k] <= self.u_max_std,
                self.du_min_std <= duk,
                duk <= self.du_max_std,
            ]

        cost += cp.quad_form(self.Cz @ z[:, self.N] - x_ref_h_std[:, self.N - 1], self.Q)

        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=self.solver, warm_start=True, verbose=False)

        if u.value is None:
            return np.asarray(u_prev_raw, dtype=float)

        return self._to_raw(u[:, 0].value)


# ============================================================
# SIM HELPERS
# Assumes sim state ordering:
# [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
# ============================================================
def sim_to_state12(x_sim):
    x_sim = np.asarray(x_sim).flatten()
    if len(x_sim) < 12:
        raise ValueError(f"Simulation state needs at least 12 entries, got {len(x_sim)}")
    return x_sim[:12].copy()


def lifted_state_from_x12(x12, scaler):
    x_std = scaler.transform(np.asarray(x12).reshape(1, -1)).flatten()
    return observables(x_std, scaler)


def build_ref_horizon(ref_std, k, N):
    ny, T = ref_std.shape
    out = np.zeros((ny, N))
    for j in range(N):
        idx = min(k + j, T - 1)
        out[:, j] = ref_std[:, idx]
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    # --------------------------------------------------------
    # find logs
    # --------------------------------------------------------
    all_files = sorted(DATA_DIR.glob("Data_*.xlsx"),
                       key=lambda p: int(p.stem.split("_")[1]))

    if not all_files:
        print(f"No Data_*.xlsx files found in {DATA_DIR}")
        return

    print(f"Found {len(all_files)} flight files in {DATA_DIR}")

    # --------------------------------------------------------
    # load all flights
    # --------------------------------------------------------
    all_flights = []
    for fp in all_files:
        try:
            t, states, U = load_flight(fp)
            t_ds, states_ds, U_ds = downsample_to_dt(t, states, U, dt_edmd)
            all_flights.append({
                "file": fp.name,
                "t": t_ds,
                "states": states_ds,
                "U": U_ds,
            })
        except Exception as e:
            print(f"  SKIPPED {fp.name}: {e}")

    if len(all_flights) < 2:
        print("Need at least 2 valid flights.")
        return

    test_idx = TEST_FLIGHT_IDX % len(all_flights)
    train_flights = [f for i, f in enumerate(all_flights) if i != test_idx]
    test_flight = all_flights[test_idx]

    print(f"\nTraining on {len(train_flights)} flights")
    print(f"Testing on {test_flight['file']}")

    # --------------------------------------------------------
    # snapshots
    # --------------------------------------------------------
    Xc_list, Xn_list, U_list = [], [], []
    for flight in train_flights:
        s = flight["states"]
        u = flight["U"]
        if s.shape[0] < 2:
            continue
        Xc_list.append(s[:-1, :])
        Xn_list.append(s[1:, :])
        U_list.append(u[:-1, :].T)

    Xc = np.vstack(Xc_list).T
    Xn = np.vstack(Xn_list).T
    U_train = np.hstack(U_list)

    print(f"\nSnapshot shapes:")
    print(f"  Xc: {Xc.shape}")
    print(f"  Xn: {Xn.shape}")
    print(f"  U : {U_train.shape}")

    # --------------------------------------------------------
    # scalers
    # --------------------------------------------------------
    all_states = np.vstack([f["states"] for f in all_flights])
    all_U = np.vstack([f["U"] for f in all_flights])

    scaler = StandardScaler()
    scaler.fit(all_states)

    u_scaler = StandardScaler()
    u_scaler.fit(all_U)

    Xc_std = scaler.transform(Xc.T).T
    Xn_std = scaler.transform(Xn.T).T
    U_std = u_scaler.transform(U_train.T).T

    # --------------------------------------------------------
    # lifting
    # --------------------------------------------------------
    n_obs = len(observables(np.zeros(12), scaler))
    Psi = np.column_stack([observables(Xc_std[:, k], scaler) for k in range(Xc_std.shape[1])])
    Phi = np.column_stack([observables(Xn_std[:, k], scaler) for k in range(Xn_std.shape[1])])

    print(f"\nObservable dimension: {n_obs}")
    print(f"Psi: {Psi.shape}, Phi: {Phi.shape}")

    # --------------------------------------------------------
    # regularization sweep on held-out real flight
    # --------------------------------------------------------
    best_lam = None
    best_pos_rmse = float("inf")

    print("\n" + "=" * 60)
    print("REGULARIZATION SWEEP")
    print("=" * 60)

    for lam in LAMBDA_CANDIDATES:
        A_try, B_try = train_edmdc(Psi, Phi, U_std, lam)
        pos_r, vel_r, yaw_r = rolling_horizon_rmse(
            test_flight["states"], test_flight["U"],
            A_try, B_try, scaler, u_scaler, ROLLOUT_HORIZON
        )
        print(f"  lam={lam:.0e}  pos_RMSE={pos_r:.4f}  vel_RMSE={vel_r:.4f}  yaw_RMSE={yaw_r:.4f}")
        if pos_r < best_pos_rmse:
            best_pos_rmse = pos_r
            best_lam = lam

    print(f"\nBest lambda: {best_lam:.0e}")

    A, B = train_edmdc(Psi, Phi, U_std, best_lam)

    labels = ["x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "p", "q", "r"]
    units = ["m", "m", "m", "m/s", "m/s", "m/s", "rad", "rad", "rad", "rad/s", "rad/s", "rad/s"]

    print("\n" + "=" * 40)
    print("FINAL MODEL")
    print("=" * 40)
    print(f"A: {A.shape}, B: {B.shape}")
    print(f"Max |eig(A)|: {np.max(np.abs(np.linalg.eigvals(A))):.6f}")

    # --------------------------------------------------------
    # rollout on held-out real flight
    # --------------------------------------------------------
    t_test = test_flight["t"]
    states_test = test_flight["states"]
    U_test = test_flight["U"]

    h = min(ROLLOUT_HORIZON, len(states_test) - 1)
    M = h + 1

    states_short = states_test[:M]
    U_short = U_test[:M]
    t_short = t_test[:M]

    Psi_pred = np.zeros((n_obs, M))
    x0_std = scaler.transform(states_short[0].reshape(1, -1)).flatten()
    Psi_pred[:, 0] = observables(x0_std, scaler)

    for k in range(M - 1):
        u_std_k = u_scaler.transform(U_short[k].reshape(1, -1)).flatten()
        Psi_pred[:, k + 1] = A @ Psi_pred[:, k] + B @ u_std_k

    x_pred = scaler.inverse_transform(Psi_pred[:12, :].T).T

    fig, axs = plt.subplots(3, 4, figsize=(22, 12))
    for i in range(12):
        r, c = divmod(i, 4)
        ax = axs[r, c]
        err_i = states_short[:, i] - x_pred[i]
        if i == 8:
            err_i = wrap_angle_pi(err_i)
        rmse_i = np.sqrt(np.mean(err_i ** 2))
        ax.plot(t_short, states_short[:, i], label="True")
        ax.plot(t_short, x_pred[i], "--", label="EDMDc")
        ax.set_title(f"{labels[i]} (RMSE {rmse_i:.4f})")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f"{labels[i]} [{units[i]}]")
        ax.grid(True)
        if i == 0:
            ax.legend()
    plt.suptitle(f"Held-out real flight rollout: {test_flight['file']}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.plot(states_short[:, 0], states_short[:, 1], states_short[:, 2], lw=2, label="True")
    ax3d.plot(x_pred[0], x_pred[1], x_pred[2], "--", lw=2, label="EDMDc")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("Held-out rollout prediction")
    ax3d.legend()
    ax3d.grid(True)

    plt.show()

    # --------------------------------------------------------
    # save learned model
    # --------------------------------------------------------
    model_data = {
        "A": A,
        "B": B,
        "scaler": scaler,
        "u_scaler": u_scaler,
        "dt": dt_edmd,
        "lambda": best_lam,
        "n_obs": n_obs,
        "state_labels": labels,
        "u_labels": ["thrust", "des_roll", "des_pitch", "des_yaw"],
        "source": "real_flight_data_yaw_transfer",
        "n_train_flights": len(train_flights),
        "test_flight": test_flight["file"],
    }

    out_file = SCRIPT_DIR / "edmdc_model_real_yaw_transfer.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\nSaved model to {out_file}")

    # --------------------------------------------------------
    # MPC TRANSFER TEST: real-data-trained controller on sim plant
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRANSFER TEST: real-data-trained EDMDc MPC on simulation plant")
    print("=" * 60)

    from Simulation import quad_sim

    sim = quad_sim()

    # --- MPC setup ---
    N_MPC = 20
    Cz = np.zeros((12, n_obs))
    Cz[:12, :12] = np.eye(12)

    Q_diag = np.array([
        200.0, 200.0, 200.0,   # position
        10.0, 10.0, 10.0,      # velocity
        5.0, 5.0, 5.0,         # roll, pitch, yaw
        0.5, 0.5, 0.5          # p, q, r
    ])
    R_diag = np.array([0.02, 0.05, 0.05, 0.05])
    Rd_diag = np.array([0.002, 0.01, 0.01, 0.01])

    # raw control nominal / bounds
    # des_yaw nominal is set to initial yaw to avoid biasing to zero if flight heading is offset
    yaw0 = states_test[0, 8]
    u_nominal_raw = np.array([sim.q_mass * sim.g, 0.0, 0.0, yaw0])

    tilt_max = float(sim.controller_PID.tilt_max)
    yaw_cmd_max = np.pi   # adjust if your sim expects a narrower yaw-angle command
    thrust_min = 0.5 * sim.q_mass * sim.g
    thrust_max = 2.0 * sim.q_mass * sim.g

    u_min_raw = np.array([thrust_min, -tilt_max, -tilt_max, -yaw_cmd_max])
    u_max_raw = np.array([thrust_max,  tilt_max,  tilt_max,  yaw_cmd_max])

    du_min_raw = np.array([-5.0, -0.25, -0.25, -0.35])
    du_max_raw = np.array([ 5.0,  0.25,  0.25,  0.35])

    mpc = EDMDcMPC_QP(
        A=A,
        B=B,
        Cz=Cz,
        N=N_MPC,
        Q=np.diag(Q_diag),
        R=np.diag(R_diag),
        Rd=np.diag(Rd_diag),
        u_scaler=u_scaler,
        u_nominal_raw=u_nominal_raw,
        u_min_raw=u_min_raw,
        u_max_raw=u_max_raw,
        du_min_raw=du_min_raw,
        du_max_raw=du_max_raw,
        solver="OSQP",
    )

    # --- reference trajectory from held-out real flight ---
    T_mpc = min(len(states_test), 500)
    ref_states = states_test[:T_mpc].T
    ref_std = scaler.transform(ref_states.T).T   # shape (12, T_mpc)

    # --- initialize sim state from first real state ---
    x_current_sim = np.zeros(12)
    x_current_sim[:12] = states_test[0, :12]

    X_mpc = np.zeros((T_mpc, 12))
    U_mpc = np.zeros((T_mpc, 4))
    X_mpc[0] = sim_to_state12(x_current_sim)

    u_prev_raw = u_nominal_raw.copy()
    solve_times = []

    print(f"Running transfer test for {T_mpc} steps...")

    import time as time_module

    for k in range(T_mpc - 1):
        if k % 50 == 0:
            print(f"  step {k}/{T_mpc - 1}")

        x12 = sim_to_state12(x_current_sim)
        z_k = lifted_state_from_x12(x12, scaler)
        x_ref_h = build_ref_horizon(ref_std, k, N_MPC)

        t0 = time_module.perf_counter()
        u_cmd = mpc.compute(z_k, x_ref_h, u_prev_raw)
        solve_times.append(time_module.perf_counter() - t0)

        # safety clipping in raw units
        u_cmd[0] = np.clip(u_cmd[0], thrust_min, thrust_max)
        u_cmd[1] = np.clip(u_cmd[1], -tilt_max, tilt_max)
        u_cmd[2] = np.clip(u_cmd[2], -tilt_max, tilt_max)
        u_cmd[3] = np.clip(u_cmd[3], -yaw_cmd_max, yaw_cmd_max)

        U_mpc[k] = u_cmd

        # IMPORTANT:
        # This assumes your sim accepts psi_des as desired yaw angle.
        # If your sim uses a different keyword, change psi_des below.
        x_next = sim.sim_PID.fct_step_attitude(
            x_current_sim,
            u1=u_cmd[0],
            phi_des=u_cmd[1],
            theta_des=u_cmd[2],
            psi_des=u_cmd[3],
            dt=dt_edmd,
        )

        x_current_sim = np.asarray(x_next).flatten()
        X_mpc[k + 1] = sim_to_state12(x_current_sim)
        u_prev_raw = u_cmd.copy()

    U_mpc[-1] = U_mpc[-2]

    # --------------------------------------------------------
    # transfer-test metrics
    # --------------------------------------------------------
    ref_eval = states_test[:T_mpc, :]
    pos_rmse = np.sqrt(np.mean((X_mpc[:, 0:3] - ref_eval[:, 0:3]) ** 2, axis=0))
    vel_rmse = np.sqrt(np.mean((X_mpc[:, 3:6] - ref_eval[:, 3:6]) ** 2, axis=0))
    yaw_err = wrap_angle_pi(X_mpc[:, 8] - ref_eval[:, 8])
    yaw_rmse = np.sqrt(np.mean(yaw_err ** 2))

    print("\nTransfer-test RMSE:")
    print(f"  Position x,y,z : {pos_rmse[0]:.3f}, {pos_rmse[1]:.3f}, {pos_rmse[2]:.3f} m")
    print(f"  Velocity vx,vy,vz : {vel_rmse[0]:.3f}, {vel_rmse[1]:.3f}, {vel_rmse[2]:.3f} m/s")
    print(f"  Yaw : {yaw_rmse:.3f} rad  ({np.rad2deg(yaw_rmse):.2f} deg)")
    print(f"  Solve time avg : {1e3 * np.mean(solve_times):.2f} ms")
    print(f"  Solve time max : {1e3 * np.max(solve_times):.2f} ms")

    # --------------------------------------------------------
    # transfer-test plots
    # --------------------------------------------------------
    t_mpc = t_test[:T_mpc]

    # 3D position
    fig_tr = plt.figure(figsize=(10, 7))
    ax = fig_tr.add_subplot(111, projection="3d")
    ax.plot(ref_eval[:, 0], ref_eval[:, 1], ref_eval[:, 2], "k", lw=2.5, label="Held-out real flight")
    ax.plot(X_mpc[:, 0], X_mpc[:, 1], X_mpc[:, 2], "g-", lw=1.5, label="Real-trained EDMDc MPC on sim plant")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Transfer test: real-data-trained controller on sim plant")
    ax.legend()
    ax.grid(True)

    # x/y/z
    fig_xyz, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    xyz_labels = ["X [m]", "Y [m]", "Z [m]"]
    for j in range(3):
        axes[j].plot(t_mpc, ref_eval[:, j], "k", lw=2, label="Held-out real flight")
        axes[j].plot(t_mpc, X_mpc[:, j], "g-", lw=1.5, label="Transfer test")
        axes[j].set_ylabel(xyz_labels[j])
        axes[j].grid(True, alpha=0.3)
        if j == 0:
            axes[j].legend()
    axes[-1].set_xlabel("Time [s]")
    plt.suptitle("Position tracking in transfer test", fontsize=13)
    plt.tight_layout()

    # yaw
    plt.figure(figsize=(12, 4))
    plt.plot(t_mpc, ref_eval[:, 8], "k", lw=2, label="Held-out real yaw")
    plt.plot(t_mpc, X_mpc[:, 8], "g-", lw=1.5, label="Transfer test yaw")
    plt.xlabel("Time [s]")
    plt.ylabel("Yaw [rad]")
    plt.title(f"Yaw tracking (RMSE {yaw_rmse:.4f} rad)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # controls
    fig_u, axes_u = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    u_labels = ["thrust [N]", "des_roll [rad]", "des_pitch [rad]", "des_yaw [rad]"]
    for i in range(4):
        axes_u[i].plot(t_mpc, U_mpc[:, i], lw=1.8)
        axes_u[i].set_ylabel(u_labels[i])
        axes_u[i].grid(True, alpha=0.3)
    axes_u[-1].set_xlabel("Time [s]")
    plt.suptitle("Transfer-test controls", fontsize=13)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()