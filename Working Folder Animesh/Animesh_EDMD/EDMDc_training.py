import numpy as np
import pickle
from scipy.linalg import pinv
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt



# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
dt = 0.01               # EDMD time step (s), 100 Hz
SHORT_HORIZON_SECONDS = 2.0
MPC_HORIZON = int(round(SHORT_HORIZON_SECONDS / dt))
ROLLING_WINDOW_STRIDE_SECONDS = 0.1
SWEEP_ROLLING_WINDOW_STRIDE_SECONDS = 1.0
STATE_DIM = 12
STATE_LABELS = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
RAW_INPUT_DIM = 4
RAW_INPUT_LABELS = ["thrust", "phi_des", "theta_des", "psi_des"]
INPUT_LIFT_TYPE = "thrust_direction"
INPUT_LIFT_LABELS = RAW_INPUT_LABELS + ["thrust_x", "thrust_y", "thrust_z"]

# Train/evaluate only on the trajectory families we care about:
# runs 0-49 helix, 50-99 figure-8, 100-149 lissajous.
TARGET_FAMILY_RANGES = {
    "helix": range(0, 50),
    "figure-8": range(50, 100),
    "lissajous": range(100, 150),
}
target_indices = sorted(
    idx for family_range in TARGET_FAMILY_RANGES.values()
    for idx in family_range
)

# Held-out test indices, one per target family.
test_indices = [39, 59, 129]

# The screenshots inspect the first 2 seconds, so give that transient more
# influence in the least-squares fit.
EARLY_TRANSIENT_SECONDS = 2.0
EARLY_TRANSIENT_WEIGHT = 8.0
EARLY_TRANSIENT_STEPS = int(round(EARLY_TRANSIENT_SECONDS / dt))

ENFORCE_KINEMATIC_ROWS = True

# Tikhonov regularization candidates
LAMBDA_CANDIDATES = [1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

# The figures show a single free rollout from the start of each held-out
# trajectory. Include that exact plot error in lambda selection so the chosen
# model looks good in the diagnostic plots, not only in averaged rolling RMSE.
FIRST_ROLLOUT_SCORE_WEIGHT = 2.0
PLOT_STATE_SCORE_WEIGHTS = np.array([
    1.5, 1.5, 0.3,   # x, y, z
    1.2, 1.2, 0.3,   # vx, vy, vz
    0.2, 0.2, 0.7,   # phi, theta, psi
    0.2, 0.2, 0.7,   # p, q, r
], dtype=float)

# These are suggested "move forward" limits for the short-horizon training check.
# Adjust them if your project has stricter or looser tracking requirements.
SHORT_HORIZON_LIMITS = {
    "rolling_pos": 0.20,
    "y": 0.20,
    "vy": 0.20,
    "yaw": 0.25,
    "r": 0.25,
}


# Load simulation data
def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["t"], data["states"], data["U"], data["ref_traj_list"]


def thrust_direction_from_state_phys(states_phys):
    """Return the world-frame thrust direction from physical 12-state samples."""
    states = np.asarray(states_phys, dtype=float)
    scalar = states.ndim == 1
    states_2d = np.atleast_2d(states)

    if states_2d.shape[1] < STATE_DIM:
        raise ValueError(
            f"Expected at least {STATE_DIM} state entries, got {states_2d.shape[1]}"
        )

    phi = states_2d[:, 6]
    theta = states_2d[:, 7]
    psi = states_2d[:, 8]

    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_theta, c_theta = np.sin(theta), np.cos(theta)
    s_psi, c_psi = np.sin(psi), np.cos(psi)

    dirs = np.column_stack([
        c_psi*s_theta*c_phi + s_psi*s_phi,
        s_psi*s_theta*c_phi - c_psi*s_phi,
        c_theta*c_phi,
    ])
    return dirs[0] if scalar else dirs


def lift_inputs_from_phys(states_phys, raw_inputs):
    """
    Keep the logged 4 commands, then add thrust projected through current attitude.

    The plant/controller still uses raw commands. The EDMDc regression sees these
    extra channels because lateral acceleration is mainly thrust times attitude.
    """
    states = np.asarray(states_phys, dtype=float)
    raw = np.asarray(raw_inputs, dtype=float)
    scalar = states.ndim == 1 and raw.ndim == 1

    states_2d = np.atleast_2d(states)
    raw_2d = np.atleast_2d(raw)

    if raw_2d.shape[1] < RAW_INPUT_DIM:
        raise ValueError(
            f"Expected {RAW_INPUT_DIM} raw input channels, got {raw_2d.shape[1]}"
        )

    if states_2d.shape[0] == 1 and raw_2d.shape[0] > 1:
        states_2d = np.repeat(states_2d, raw_2d.shape[0], axis=0)
    elif raw_2d.shape[0] == 1 and states_2d.shape[0] > 1:
        raw_2d = np.repeat(raw_2d, states_2d.shape[0], axis=0)

    if states_2d.shape[0] != raw_2d.shape[0]:
        raise ValueError(
            f"State/input sample mismatch: {states_2d.shape[0]} vs {raw_2d.shape[0]}"
        )

    raw_4 = raw_2d[:, :RAW_INPUT_DIM]
    thrust = raw_4[:, :1]
    thrust_dir = thrust_direction_from_state_phys(states_2d)
    lifted = np.hstack([raw_4, thrust * thrust_dir])
    return lifted[0] if scalar else lifted


def scaled_lifted_input_from_phys(state_phys, raw_input, u_scaler):
    lifted = lift_inputs_from_phys(state_phys, raw_input)
    return u_scaler.transform(np.atleast_2d(lifted)).flatten()

t_all, states_all, U_all, ref_traj_list = load_simulation_runs("runs_mixed_n300.pkl")

if U_all.shape[2] < RAW_INPUT_DIM:
    raise ValueError(
        f"Expected logged inputs to include yaw command ({RAW_INPUT_DIM} channels), "
        f"but got {U_all.shape[2]}."
    )

n_runs   = t_all.shape[0]
train_indices = [
    i for i in target_indices
    if i < n_runs and i not in test_indices
]
test_indices = [i for i in test_indices if i < n_runs]

print("Target families:", ", ".join(TARGET_FAMILY_RANGES.keys()))
print("Held-out test indices:", test_indices)
print("Loaded file: runs_mixed_n300.pkl")
print("Total runs:", n_runs)
print("Target runs:", len(target_indices))
print("Training runs:", len(train_indices))
print("t shape:", t_all.shape)
print("states shape (raw 12):", states_all.shape)
print("U shape (raw):", U_all.shape)
print("ref count:", len(ref_traj_list))

# Downsample to the training time step
sim_dt = t_all[0, 1] - t_all[0, 0]
ratio = dt / sim_dt
step = int(round(ratio))

if not np.isclose(ratio, step, rtol=1e-6, atol=1e-8):
    raise ValueError(
        f"EDMD dt={dt} must be an integer multiple of simulation dt={sim_dt}"
    )

print(f"\nDownsampling: sim_dt={sim_dt}, edmd_dt={dt}, step={step}")

idx = np.arange(0, t_all.shape[1], step)
U_raw = U_all

t_all = t_all[:, idx]
states_all = states_all[:, idx, :]
ref_traj_list = [ref_traj[::step] for ref_traj in ref_traj_list]

# The state transition from t[k] to t[k+1] spans several 0.01 s simulator
# control updates. Use the interval-average command as the EDMD input for that
# 0.1 s transition instead of just the first 0.01 s command sample.
U_interval = np.zeros((U_raw.shape[0], len(idx), U_raw.shape[2]))
for k in range(len(idx) - 1):
    U_interval[:, k, :] = np.mean(U_raw[:, idx[k]:idx[k + 1], :], axis=1)
U_interval[:, -1, :] = U_interval[:, -2, :]
U_all = U_interval

if states_all.shape[2] != STATE_DIM:
    raise ValueError(
        f"Expected simulation states with {STATE_DIM} entries "
        f"[x,y,z,vx,vy,vz,phi,theta,psi,p,q,r], got {states_all.shape[2]}"
    )

if U_all.shape[2] != RAW_INPUT_DIM:
    raise ValueError(
        f"Expected {RAW_INPUT_DIM} attitude-command inputs "
        f"[thrust, phi_des, theta_des, psi_des], got {U_all.shape[2]}"
    )

print(f"Downsampled shape: states={states_all.shape}, U={U_all.shape}")

# Build training snapshots
Xc_list, Xn_list, U_list, W_list = [], [], [], []

for run in train_indices:
    states_run = states_all[run]
    U_run = U_all[run]

    if states_run.shape[0] < 2:
        continue

    n_transitions = states_run.shape[0] - 1
    sample_weights = np.ones(n_transitions, dtype=float)
    early_steps = min(EARLY_TRANSIENT_STEPS, n_transitions)
    sample_weights[:early_steps] = EARLY_TRANSIENT_WEIGHT

    Xc_list.append(states_run[:-1, :])
    Xn_list.append(states_run[1:, :])
    U_list.append(U_run[:-1, :].T)
    W_list.append(sample_weights)

Xc = np.vstack(Xc_list).T          # (12, K)
Xn = np.vstack(Xn_list).T          # (12, K)
U_train = np.hstack(U_list)        # (4, K)
sample_weights = np.concatenate(W_list)

print("\n========== SNAPSHOT DEBUG ==========")
print("Xc shape:", Xc.shape)
print("Xn shape:", Xn.shape)
print("U_train shape:", U_train.shape)
print("Sample weights shape:", sample_weights.shape)
print("Early transient weight:", EARLY_TRANSIENT_WEIGHT)
print("Number of transitions per run:", states_all.shape[1] - 1)
print("Expected total transitions:", len(train_indices) * (states_all.shape[1] - 1))
print("====================================")

# Scale lifted model inputs
X_all_for_u_scaler = states_all[train_indices].reshape(-1, states_all.shape[2])
U_all_raw_flat = U_all[train_indices].reshape(-1, U_all.shape[2])
U_all_lifted_flat = lift_inputs_from_phys(X_all_for_u_scaler, U_all_raw_flat)
U_train_lifted = lift_inputs_from_phys(Xc.T, U_train.T)

u_scaler = StandardScaler()
u_scaler.fit(U_all_lifted_flat)
U_norm = u_scaler.transform(U_train_lifted).T

print("\n========== INPUT SCALER DEBUG ==========")
print("Raw input labels:", RAW_INPUT_LABELS)
print("Lifted input labels:", INPUT_LIFT_LABELS)
print("Input lift type:", INPUT_LIFT_TYPE)
print("Lifted input shape:", U_train_lifted.shape)
print("Input scaler mean:", u_scaler.mean_)
print("Input scaler scale:", u_scaler.scale_)
print("Scaled U_train mean (approx):", np.mean(U_norm, axis=1))
print("Scaled U_train std  (approx):", np.std(U_norm, axis=1))
print("========================================")

# Scale states
X_all_flat = states_all[train_indices].reshape(-1, states_all.shape[2])
scaler = StandardScaler()
scaler.fit(X_all_flat)
Xc_s = scaler.transform(Xc.T).T
Xn_s = scaler.transform(Xn.T).T

print("\n========== STATE SCALER DEBUG ==========")
print("State scaler mean:", scaler.mean_)
print("State scaler scale:", scaler.scale_)
print("Scaled Xc mean (approx):", np.mean(Xc_s, axis=1))
print("Scaled Xc std  (approx):", np.std(Xc_s, axis=1))
print("Scaled Xn mean (approx):", np.mean(Xn_s, axis=1))
print("Scaled Xn std  (approx):", np.std(Xn_s, axis=1))
print("========================================")

# Legacy 10-state lifting kept only for reference.
# The active training lift is the 12-state observables() below.
#
# [ 0- 9] 10 linear states
# [10-13] sin(phi), cos(phi), sin(theta), cos(theta)
# [14-17] phi*p, theta*q, vx*phi, vy*theta
# [18-19] v_sq, omega_sq
# [20-25] vx*theta, vy*phi, vz², phi², theta², p*q
# [26]    bias

def observables_legacy_10state(x, scaler):
    """
    Return the lifted observable vector for a standardized 10-state input.
    """
    x = np.asarray(x).flatten()
    assert len(x) == 10, f"Expected 10-state vector, got {len(x)}"

    obs = list(x)  # 10 linear terms

    # ----- Trig terms (unscale to radians first) -----
    phi_rad   = x[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_rad = x[7] * scaler.scale_[7] + scaler.mean_[7]

    s_phi   = np.sin(phi_rad)
    c_phi   = np.cos(phi_rad)
    s_theta = np.sin(theta_rad)
    c_theta = np.cos(theta_rad)

    obs += [s_phi, c_phi, s_theta, c_theta]

    # ----- Cross terms (angle × rate, velocity × angle) -----
    obs.append(x[6] * x[8])   # phi * p
    obs.append(x[7] * x[9])   # theta * q
    obs.append(x[3] * x[6])   # vx * phi
    obs.append(x[4] * x[7])   # vy * theta

    # ----- Energy-like terms -----
    v_sq = x[3]**2 + x[4]**2 + x[5]**2
    omega_sq = x[8]**2 + x[9]**2
    obs.append(v_sq)
    obs.append(omega_sq)

    # ----- Targeted quadratic (velocity-angle, angle², gyroscopic) -----
    obs.append(x[3] * x[7])   # vx * theta
    obs.append(x[4] * x[6])   # vy * phi
    obs.append(x[5] * x[5])   # vz²
    obs.append(x[6] * x[6])   # phi²
    obs.append(x[7] * x[7])   # theta²
    obs.append(x[8] * x[9])   # p * q

    # ----- Bias -----
    obs.append(1.0)

    return np.array(obs, dtype=float)


def observables(x, scaler):
    """
    Return the lifted observable vector for a standardized 12-state input.
    Must match edmdc_mpc.py exactly.
    """
    x = np.asarray(x).flatten()
    assert len(x) == STATE_DIM, f"Expected 12-state vector, got {len(x)}"

    obs = list(x)  # 12 linear terms

    phi_rad = x[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_rad = x[7] * scaler.scale_[7] + scaler.mean_[7]
    psi_rad = x[8] * scaler.scale_[8] + scaler.mean_[8]

    s_phi, c_phi = np.sin(phi_rad), np.cos(phi_rad)
    s_theta, c_theta = np.sin(theta_rad), np.cos(theta_rad)
    s_psi, c_psi = np.sin(psi_rad), np.cos(psi_rad)

    obs += [
        s_phi, c_phi,
        s_theta, c_theta,
        s_psi, c_psi,
    ]

    obs += [
        x[6] * x[9],     # phi * p
        x[7] * x[10],    # theta * q
        x[8] * x[11],    # psi * r
        x[3] * x[6],     # vx * phi
        x[4] * x[7],     # vy * theta
        x[3] * x[8],     # vx * psi
        x[4] * x[8],     # vy * psi
        x[5] * x[7],     # vz * theta
    ]

    obs += [
        x[3]**2 + x[4]**2 + x[5]**2,
        x[9]**2 + x[10]**2 + x[11]**2,
    ]

    obs += [
        x[5] * x[5],     # vz^2
        x[6] * x[6],     # phi^2
        x[7] * x[7],     # theta^2
        x[8] * x[8],     # psi^2
        x[9] * x[10],    # p * q
        x[10] * x[11],   # q * r
        x[9] * x[11],    # p * r
    ]

    # Lateral trajectory-shape terms for x/y/vx/vy. These help the 100 Hz
    # model capture figure-8 and lissajous curvature without changing inputs.
    obs += [
        x[0] * x[1],     # x * y
        x[0] * x[3],     # x * vx
        x[1] * x[4],     # y * vy
        x[0] * x[4],     # x * vy
        x[1] * x[3],     # y * vx
        x[3] * x[4],     # vx * vy
        x[0] * x[0],     # x^2
        x[1] * x[1],     # y^2
        x[3] * x[3],     # vx^2
        x[4] * x[4],     # vy^2
        x[0] * x[7],     # x * theta
        x[1] * x[6],     # y * phi
        x[3] * x[7],     # vx * theta
        x[4] * x[6],     # vy * phi
    ]

    vx, vy, vz = x[3], x[4], x[5]
    body_vx = c_theta*c_psi*vx + c_theta*s_psi*vy - s_theta*vz
    body_vy = (s_phi*s_theta*c_psi - c_phi*s_psi)*vx + \
              (s_phi*s_theta*s_psi + c_phi*c_psi)*vy + \
              s_phi*c_theta*vz
    body_vz = (c_phi*s_theta*c_psi + s_phi*s_psi)*vx + \
              (c_phi*s_theta*s_psi - s_phi*c_psi)*vy + \
              c_phi*c_theta*vz

    thrust_dir_x = c_psi*s_theta*c_phi + s_psi*s_phi
    thrust_dir_y = s_psi*s_theta*c_phi - c_psi*s_phi
    thrust_dir_z = c_theta*c_phi

    obs += [
        body_vx, body_vy, body_vz,
        thrust_dir_x, thrust_dir_y, thrust_dir_z,
    ]

    obs.append(1.0)

    return np.array(obs, dtype=float)


# Test observable dimension
n_obs_test = len(observables(np.zeros(STATE_DIM), scaler))
print(f"\nObservable dimension: {n_obs_test}")

# Lifted snapshot matrices
Psi = np.column_stack([observables(Xc_s[:, k], scaler) for k in range(Xc_s.shape[1])])
Phi = np.column_stack([observables(Xn_s[:, k], scaler) for k in range(Xn_s.shape[1])])

print("\n========== LIFTING DEBUG ==========")
print("Psi shape:", Psi.shape)
print("Phi shape:", Phi.shape)

Omega = np.vstack([Psi, U_norm])
print("Omega shape:", Omega.shape)

try:
    svals = np.linalg.svd(Omega, compute_uv=False)
    print("Omega singular values (first 10):", svals[:10])
    print("Omega singular values (last 10):", svals[-10:])
    if svals[-1] == 0:
        print("Omega condition number: inf (smallest singular value is zero)")
    else:
        print("Omega condition number:", svals[0] / svals[-1])
except Exception as e:
    print("SVD failed:", e)

print("===================================")

# ============================================================
# REGULARIZATION SWEEP
# ============================================================
def enforce_kinematic_rows(A, B, scaler, dt):
    """Enforce exact standardized integrator rows for positions and angles."""
    if not ENFORCE_KINEMATIC_ROWS:
        return A, B

    A = A.copy()
    B = B.copy()
    bias_idx = A.shape[1] - 1

    for state_idx, rate_idx in [(0, 3), (1, 4), (2, 5),
                                (6, 9), (7, 10), (8, 11)]:
        A[state_idx, :] = 0.0
        B[state_idx, :] = 0.0
        A[state_idx, state_idx] = 1.0
        A[state_idx, rate_idx] = dt * scaler.scale_[rate_idx] / scaler.scale_[state_idx]
        A[state_idx, bias_idx] = dt * scaler.mean_[rate_idx] / scaler.scale_[state_idx]

    return A, B


def train_edmdc(Psi, Phi, U_norm, n_obs, lam, sample_weights=None):
    """Fit EDMDc via weighted Tikhonov-regularized least squares."""
    Omega = np.vstack([Psi, U_norm])
    Phi_fit = Phi

    if sample_weights is not None:
        sqrt_w = np.sqrt(np.asarray(sample_weights, dtype=float)).reshape(1, -1)
        Omega = Omega * sqrt_w
        Phi_fit = Phi * sqrt_w

    G = Omega @ Omega.T
    Y = Phi_fit @ Omega.T
    if lam != 0:
        G = G + lam * np.eye(G.shape[0])
    AB = Y @ pinv(G)

    A = AB[:, :n_obs]
    B = AB[:, n_obs:]

    return enforce_kinematic_rows(A, B, scaler, dt)


def rolling_horizon_rmse(states, inputs, A, B, scaler, u_scaler,
                         horizon, observables_fn, stride=1):
    n_total = states.shape[0]
    n_windows = n_total - horizon
    if n_windows <= 0:
        raise ValueError("Trajectory is shorter than the evaluation horizon.")
    stride = max(1, int(stride))

    pos_rmse_list = []
    vel_rmse_list = []
    full_rmse_list = []
    per_state_rmse_list = []

    start_indices = list(range(0, n_windows, stride))
    if start_indices[-1] != n_windows - 1:
        start_indices.append(n_windows - 1)

    for start in start_indices:
        states_seg = states[start:start + horizon + 1]
        inputs_seg = inputs[start:start + horizon]

        psi_pred = np.zeros((A.shape[0], horizon + 1))
        psi_pred[:, 0] = observables_fn(
            scaler.transform(states_seg[0].reshape(1, -1)).flatten(),
            scaler
        )

        for k in range(1, horizon + 1):
            x_prev_phys = scaler.inverse_transform(
                psi_pred[:STATE_DIM, k - 1].reshape(1, -1)
            ).flatten()
            u_k_s = scaled_lifted_input_from_phys(
                x_prev_phys, inputs_seg[k - 1], u_scaler
            )
            psi_pred[:, k] = A @ psi_pred[:, k - 1] + B @ u_k_s

        x_pred = scaler.inverse_transform(psi_pred[:STATE_DIM, :].T)

        err = states_seg - x_pred
        pos_err = err[:, 0:3]
        vel_err = err[:, 3:6]

        pos_rmse_list.append(np.sqrt(np.mean(pos_err**2)))
        vel_rmse_list.append(np.sqrt(np.mean(vel_err**2)))
        full_rmse_list.append(np.sqrt(np.mean(err**2)))
        per_state_rmse_list.append(np.sqrt(np.mean(err**2, axis=0)))

    return (
        float(np.mean(pos_rmse_list)),
        float(np.mean(vel_rmse_list)),
        float(np.mean(full_rmse_list)),
        np.mean(np.asarray(per_state_rmse_list), axis=0),
    )


def single_rollout_metrics(states, inputs, A, B, scaler, u_scaler,
                           horizon, observables_fn):
    """Evaluate the same initial free rollout shown in the diagnostic plots."""
    M = min(horizon + 1, states.shape[0])
    states_short = states[:M]
    inputs_short = inputs[:M]

    psi_pred = np.zeros((A.shape[0], M))
    psi_pred[:, 0] = observables_fn(
        scaler.transform(states_short[0, :].reshape(1, -1)).flatten(),
        scaler
    )

    for k in range(1, M):
        x_prev_phys = scaler.inverse_transform(
            psi_pred[:STATE_DIM, k - 1].reshape(1, -1)
        ).flatten()
        u_k_s = scaled_lifted_input_from_phys(
            x_prev_phys, inputs_short[k - 1, :], u_scaler
        )
        psi_pred[:, k] = A @ psi_pred[:, k - 1] + B @ u_k_s

    x_pred = scaler.inverse_transform(psi_pred[:STATE_DIM, :].T).T
    err = states_short.T - x_pred
    rmse_each = np.sqrt(np.mean(err**2, axis=1))
    pos_rmse = np.sqrt(np.mean(err[0:3, :]**2))
    vel_rmse = np.sqrt(np.mean(err[3:6, :]**2))
    weighted_state_rmse = np.sqrt(
        np.mean(PLOT_STATE_SCORE_WEIGHTS * rmse_each**2)
    )
    return (
        float(pos_rmse),
        float(vel_rmse),
        float(np.sqrt(np.mean(err**2))),
        rmse_each,
        float(weighted_state_rmse),
    )


family_names = {
    39: "helix",
    59: "figure-8",
    129: "lissajous",
}

n_obs = Psi.shape[0]
h = MPC_HORIZON
rolling_stride = max(1, int(round(ROLLING_WINDOW_STRIDE_SECONDS / dt)))
sweep_rolling_stride = max(1, int(round(SWEEP_ROLLING_WINDOW_STRIDE_SECONDS / dt)))

print(f"\n{'='*60}")
print(f"REGULARIZATION SWEEP ({len(LAMBDA_CANDIDATES)} candidates)")
print(f"Evaluating rolling {h}-step RMSE on held-out trajectories")
print(
    f"Sweep rolling-window stride: {sweep_rolling_stride} steps "
    f"({sweep_rolling_stride * dt:.2f} s)"
)
print(f"{'='*60}")

best_lam = 0
best_score = float("inf")
best_avg_roll_pos = float("inf")
best_avg_plot_score = float("inf")
sweep_rows = []

for lam in LAMBDA_CANDIDATES:
    A_try, B_try = train_edmdc(
        Psi, Phi, U_norm, n_obs, lam,
        sample_weights=sample_weights,
    )

    per_traj = {}
    total_roll_pos = 0.0
    total_plot_score = 0.0
    for tidx in test_indices:
        name = family_names.get(tidx, str(tidx))
        try:
            pos_r, _, _, _ = rolling_horizon_rmse(
                states_all[tidx], U_all[tidx],
                A_try, B_try, scaler, u_scaler, h,
                observables, stride=sweep_rolling_stride
            )
            _, _, _, _, plot_score = single_rollout_metrics(
                states_all[tidx], U_all[tidx],
                A_try, B_try, scaler, u_scaler, h,
                observables
            )
            per_traj[name] = pos_r
            total_roll_pos += pos_r
            total_plot_score += plot_score
        except Exception:
            per_traj[name] = float("inf")
            total_roll_pos = float("inf")
            total_plot_score = float("inf")

    avg_roll_pos = total_roll_pos / len(test_indices)
    avg_plot_score = total_plot_score / len(test_indices)
    score = avg_roll_pos + FIRST_ROLLOUT_SCORE_WEIGHT * avg_plot_score
    detail = "  ".join(f"{n}={v:.4f}" for n, v in per_traj.items())
    print(
        f"  lam={lam:.0e}  score={score:.4f}  "
        f"roll_pos={avg_roll_pos:.4f}  plot_score={avg_plot_score:.4f}  "
        f"{detail}"
    )
    sweep_rows.append({
        "lambda": lam,
        "score": score,
        "rolling_pos": avg_roll_pos,
        "plot_score": avg_plot_score,
    })

    if score < best_score:
        best_score = score
        best_avg_roll_pos = avg_roll_pos
        best_avg_plot_score = avg_plot_score
        best_lam = lam

print(
    f"\nBest lambda: {best_lam:.0e} "
    f"(score={best_score:.4f}, rolling pos={best_avg_roll_pos:.4f}, "
    f"plot score={best_avg_plot_score:.4f})"
)

finite_sweep_rows = [
    row for row in sweep_rows
    if np.isfinite(row["score"])
]
if finite_sweep_rows:
    lam_values = np.array([row["lambda"] for row in finite_sweep_rows], dtype=float)
    score_values = np.array([row["score"] for row in finite_sweep_rows], dtype=float)
    rolling_values = np.array([row["rolling_pos"] for row in finite_sweep_rows], dtype=float)
    plot_values = np.array([row["plot_score"] for row in finite_sweep_rows], dtype=float)

    fig_sweep, ax_sweep = plt.subplots(figsize=(9, 5))
    ax_sweep.semilogx(lam_values, score_values, marker="o", linewidth=2.0, label="selection score")
    ax_sweep.semilogx(lam_values, rolling_values, marker="s", linewidth=1.6, label="rolling position RMSE")
    ax_sweep.semilogx(lam_values, plot_values, marker="^", linewidth=1.6, label="first-rollout plot score")
    ax_sweep.axvline(best_lam, color="black", linestyle="--", linewidth=1.2, label=f"chosen lambda={best_lam:.0e}")
    ax_sweep.set_xlabel("lambda")
    ax_sweep.set_ylabel("RMSE / score")
    ax_sweep.set_title("EDMDc Lambda Selection")
    ax_sweep.grid(True, which="both", alpha=0.3)
    ax_sweep.legend()
    fig_sweep.tight_layout()

# ============================================================
# FINAL MODEL WITH BEST LAMBDA
# ============================================================
A, B = train_edmdc(
    Psi, Phi, U_norm, n_obs, best_lam,
    sample_weights=sample_weights,
)

rho = np.max(np.abs(np.linalg.eigvals(A)))

print(f"\n========== FINAL MODEL (lambda={best_lam:.0e}) ==========")
print("A shape:", A.shape)
print("B shape:", B.shape)

eigvals = np.linalg.eigvals(A)
abs_eigs = np.sort(np.abs(eigvals))

print("Max abs eigenvalue of A:", np.max(np.abs(eigvals)))
print("Top 10 abs eigenvalues:", abs_eigs[-10:])
print("Any NaN in A?", np.isnan(A).any())
print("Any Inf in A?", np.isinf(A).any())
print("Any NaN in B?", np.isnan(B).any())
print("Any Inf in B?", np.isinf(B).any())
print("=================================")

print("\n========== B ROW NORMS ==========")
labels_10 = STATE_LABELS
for i, lbl in enumerate(labels_10):
    print(f"  {lbl:>6s}: {np.linalg.norm(B[i,:]):.6f}")
print("  --- lifted rows ---")
lifted_labels = ['sin_phi','cos_phi','sin_theta','cos_theta',
                 'phi*p','theta*q','vx*phi','vy*theta',
                 'v_sq','omega_sq',
                 'vx*theta','vy*phi','vz²','phi²','theta²','p*q',
                 'bias']
lifted_labels = ['sin_phi','cos_phi','sin_theta','cos_theta','sin_psi','cos_psi',
                 'phi*p','theta*q','psi*r','vx*phi','vy*theta','vx*psi','vy*psi','vz*theta',
                 'v_sq','omega_sq',
                 'vz^2','phi^2','theta^2','psi^2','p*q','q*r','p*r',
                 'x*y','x*vx','y*vy','x*vy','y*vx','vx*vy',
                 'x^2','y^2','vx^2','vy^2','x*theta','y*phi','vx*theta','vy*phi',
                 'body_vx','body_vy','body_vz','thrust_dir_x','thrust_dir_y','thrust_dir_z',
                 'bias']
for i, lbl in enumerate(lifted_labels):
    print(f"  {lbl:>12s}: {np.linalg.norm(B[STATE_DIM+i,:]):.6f}")
print("==================================")


# ============================================================
# Held-out short-horizon evaluation
# ============================================================
labels = STATE_LABELS
units  = ['m','m','m','m/s','m/s','m/s','rad','rad','rad','rad/s','rad/s','rad/s']

print("\n========== SHORT-HORIZON EVALUATION ==========")
print(f"Evaluation horizon: {h} steps ({h * dt:.2f} s)")
print(f"Rolling-window stride: {rolling_stride} steps ({rolling_stride * dt:.2f} s)")
print("==============================================")

summary_rows = []

for test_idx in test_indices:
    t_test = t_all[test_idx]
    states_test = states_all[test_idx].copy()
    U_test = U_all[test_idx]
    ref_test = ref_traj_list[test_idx]

    name = family_names.get(test_idx, f"idx {test_idx}")

    M = min(h + 1, states_test.shape[0])
    t_short = t_test[:M]
    states_short = states_test[:M]
    U_short = U_test[:M]

    Psi_pred = np.zeros((n_obs, M))
    Psi_pred[:, 0] = observables(
        scaler.transform(states_short[0, :].reshape(1, -1)).flatten(),
        scaler
    )

    for k in range(1, M):
        x_prev_phys = scaler.inverse_transform(
            Psi_pred[:STATE_DIM, k - 1].reshape(1, -1)
        ).flatten()
        u_k_s = scaled_lifted_input_from_phys(
            x_prev_phys, U_short[k - 1, :], u_scaler
        )
        Psi_pred[:, k] = A @ Psi_pred[:, k - 1] + B @ u_k_s

    x_pred = scaler.inverse_transform(Psi_pred[:STATE_DIM, :].T).T
    err = states_short.T - x_pred

    rmse_each = np.sqrt(np.mean(err**2, axis=1))
    rmse_total = np.sqrt(np.mean(err**2))

    X_test_s = scaler.transform(states_short)
    one_step_pred = np.zeros_like(states_short.T)
    one_step_pred[:, 0] = states_short[0]

    for k in range(states_short.shape[0] - 1):
        psi_k = observables(X_test_s[k], scaler)
        u_k_s = scaled_lifted_input_from_phys(
            states_short[k], U_short[k], u_scaler
        )
        psi_next = A @ psi_k + B @ u_k_s
        x_next_pred = scaler.inverse_transform(
            psi_next[:STATE_DIM].reshape(1, -1)
        ).flatten()
        one_step_pred[:, k + 1] = x_next_pred

    err_one = states_short.T - one_step_pred
    rmse_one = np.sqrt(np.mean(err_one**2))
    pos_rmse_roll, vel_rmse_roll, full_rmse_roll, per_state_rmse_roll = rolling_horizon_rmse(
        states_test, U_test, A, B, scaler, u_scaler, h,
        observables, stride=rolling_stride
    )

    print(f"\n--- {name} (idx={test_idx}) ---")
    print(f"One-step total RMSE:              {rmse_one:.4f}")
    print(f"Single {h}-step rollout RMSE:       {rmse_total:.4f}")
    print(f"Rolling {h}-step position RMSE:     {pos_rmse_roll:.4f}")
    print(f"Rolling {h}-step velocity RMSE:     {vel_rmse_roll:.4f}")
    print(f"Rolling {h}-step full-state RMSE:   {full_rmse_roll:.4f}")
    print(
        f"Rolling weak states: y={per_state_rmse_roll[1]:.4f}m  "
        f"vy={per_state_rmse_roll[4]:.4f}m/s  "
        f"yaw={per_state_rmse_roll[8]:.4f}rad  "
        f"r={per_state_rmse_roll[11]:.4f}rad/s"
    )
    print("Per-state short-horizon RMSE (single rollout from initial state):")
    for lbl, val in zip(labels, rmse_each):
        print(f"  {lbl}: {val:.4f}")

    summary_rows.append((
        name, test_idx, rmse_one, rmse_total,
        pos_rmse_roll, vel_rmse_roll, full_rmse_roll,
        per_state_rmse_roll, rmse_each,
    ))

    # 3D short-horizon trajectory plot
    x_sim = states_short[:, 0]
    y_sim = states_short[:, 1]
    z_sim = states_short[:, 2]

    x_edmd = x_pred[0, :]
    y_edmd = x_pred[1, :]
    z_edmd = x_pred[2, :]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_sim, y_sim, z_sim, linewidth=2, label="True")
    ax.plot(x_edmd, y_edmd, z_edmd, '--', linewidth=2, label="EDMDc")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"{name}: short-horizon rollout ({h} steps)")
    ax.legend()
    ax.grid(True)

    # Per-state time-series plots
    n_states = len(labels)
    n_cols = 4
    n_rows = int(np.ceil(n_states / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 7))
    axs = np.asarray(axs).reshape(n_rows, n_cols)
    for i in range(n_states):
        row, col = divmod(i, n_cols)
        ax = axs[row, col]
        ax.plot(t_short, states_short[:, i], label='True')
        ax.plot(t_short, x_pred[i], '--', label='EDMDc')
        ax.set_title(f"{labels[i]} (RMSE {rmse_each[i]:.3f})")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f"{labels[i]} [{units[i]}]")
        ax.grid(True)
        if i == 0:
            ax.legend()
    for j in range(n_states, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axs[row, col].axis("off")
    fig.suptitle(f"{name}: short-horizon state prediction", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

print("\n========== SHORT-HORIZON SUMMARY ==========")
for name, idx, rmse_one, rmse_roll, pos_rmse_roll, vel_rmse_roll, full_rmse_roll, per_state_rmse_roll, rmse_each in summary_rows:
    print(
        f"{name:<18s} idx={idx:<4d} "
        f"one-step={rmse_one:.4f}  "
        f"single-rollout={rmse_roll:.4f}  "
        f"rolling-pos={pos_rmse_roll:.4f}  "
        f"rolling-vel={vel_rmse_roll:.4f}  "
        f"rolling-full={full_rmse_roll:.4f}"
    )
print("===========================================")

print("\n========== SINGLE-ROLLOUT PLOT CHECK ==========")
print("These are the subplot RMSE values from the first 2-second free rollout.")
for name, idx, _, _, _, _, _, _, rmse_each in summary_rows:
    print(
        f"{name:<18s} "
        f"x={rmse_each[0]:.4f}  y={rmse_each[1]:.4f}  "
        f"vx={rmse_each[3]:.4f}  vy={rmse_each[4]:.4f}  "
        f"psi={rmse_each[8]:.4f}  r={rmse_each[11]:.4f}"
    )
print("==============================================")

print("\n========== SHORT-HORIZON GATE ==========")
print(
    "Limits: "
    f"pos<{SHORT_HORIZON_LIMITS['rolling_pos']:.2f}m, "
    f"y<{SHORT_HORIZON_LIMITS['y']:.2f}m, "
    f"vy<{SHORT_HORIZON_LIMITS['vy']:.2f}m/s, "
    f"yaw<{SHORT_HORIZON_LIMITS['yaw']:.2f}rad, "
    f"r<{SHORT_HORIZON_LIMITS['r']:.2f}rad/s"
)
gate_all_pass = True
for name, idx, _, _, pos_rmse_roll, _, _, per_state_rmse_roll, _ in summary_rows:
    checks = {
        "rolling_pos": pos_rmse_roll,
        "y": per_state_rmse_roll[1],
        "vy": per_state_rmse_roll[4],
        "yaw": per_state_rmse_roll[8],
        "r": per_state_rmse_roll[11],
    }
    passing = all(value <= SHORT_HORIZON_LIMITS[key] for key, value in checks.items())
    gate_all_pass = gate_all_pass and passing
    status = "PASS" if passing else "REVIEW"
    print(
        f"{status:>6s}  {name:<18s} "
        f"pos={checks['rolling_pos']:.4f}  "
        f"y={checks['y']:.4f}  "
        f"vy={checks['vy']:.4f}  "
        f"yaw={checks['yaw']:.4f}  "
        f"r={checks['r']:.4f}"
    )
print("Decision:", "GOOD ENOUGH TO MOVE FORWARD" if gate_all_pass else "DO NOT TUNE YET")
print("========================================")

# Save model
model_data = {
    "A": A,
    "B": B,
    "scaler": scaler,
    "u_scaler": u_scaler,
    "dt": dt,
    "n_obs": n_obs,
    "lambda": best_lam,
    "lambda_selection_score": best_score,
    "lambda_selection_rolling_pos": best_avg_roll_pos,
    "lambda_selection_plot_score": best_avg_plot_score,
    "first_rollout_score_weight": FIRST_ROLLOUT_SCORE_WEIGHT,
    "plot_state_score_weights": PLOT_STATE_SCORE_WEIGHTS,
    "short_horizon_seconds": SHORT_HORIZON_SECONDS,
    "short_horizon_steps": h,
    "rolling_window_stride_seconds": rolling_stride * dt,
    "sweep_rolling_window_stride_seconds": sweep_rolling_stride * dt,
    "target_families": list(TARGET_FAMILY_RANGES.keys()),
    "target_indices": target_indices,
    "train_indices": train_indices,
    "early_transient_seconds": EARLY_TRANSIENT_SECONDS,
    "early_transient_weight": EARLY_TRANSIENT_WEIGHT,
    "enforce_kinematic_rows": ENFORCE_KINEMATIC_ROWS,
    "state_labels": labels,
    "raw_input_dim": RAW_INPUT_DIM,
    "raw_u_labels": RAW_INPUT_LABELS,
    "u_labels": INPUT_LIFT_LABELS,
    "input_lift_type": INPUT_LIFT_TYPE,
    "input_lift_labels": INPUT_LIFT_LABELS,
    "observable_labels": STATE_LABELS + lifted_labels,
    "source_file": "runs_mixed_n300.pkl",
    "test_indices": test_indices,
    "u_type": "attitude_cmd",
}

with open("edmdc_model_300.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\nSaved model to edmdc_model_300.pkl")
print(f"A: {A.shape}, B: {B.shape}, n_obs: {n_obs}, lambda: {best_lam:.0e}")

plt.show()
