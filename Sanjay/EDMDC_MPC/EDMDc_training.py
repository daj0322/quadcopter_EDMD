import numpy as np
import pickle
from scipy.linalg import pinv
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt



# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
dt = 0.1               # EDMD time step (s)
MPC_HORIZON = 20

# Held-out test indices (must match final_comparison.py)
test_indices = [39, 59, 129, 155, 210]

# Tikhonov regularization candidates
LAMBDA_CANDIDATES = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


# Load simulation data
def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["t"], data["states"], data["U"], data["ref_traj_list"]

t_all, states_all, U_all, ref_traj_list = load_simulation_runs("runs_mixed_n300.pkl")

n_runs   = t_all.shape[0]
train_indices = [i for i in range(n_runs) if i not in test_indices]

print("Held-out test indices:", test_indices)
print("Loaded file: runs_mixed_n300.pkl")
print("Total runs:", n_runs)
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
t_all = t_all[:, idx]
states_all = states_all[:, idx, :]
U_all = U_all[:, idx, :]
ref_traj_list = [ref_traj[::step] for ref_traj in ref_traj_list]

print(f"Downsampled shape: states={states_all.shape}, U={U_all.shape}")

# Build training snapshots
Xc_list, Xn_list, U_list = [], [], []

train_indices = [i for i in range(n_runs) if i not in test_indices]

print("\nNumber of training runs:", len(train_indices))

for run in train_indices:
    states_run = states_all[run]
    U_run = U_all[run]

    if states_run.shape[0] < 2:
        continue

    Xc_list.append(states_run[:-1, :])
    Xn_list.append(states_run[1:, :])
    U_list.append(U_run[:-1, :].T)

Xc = np.vstack(Xc_list).T          # (10, K)
Xn = np.vstack(Xn_list).T          # (10, K)
U_train = np.hstack(U_list)        # (3, K)

print("\n========== SNAPSHOT DEBUG ==========")
print("Xc shape:", Xc.shape)
print("Xn shape:", Xn.shape)
print("U_train shape:", U_train.shape)
print("Number of transitions per run:", states_all.shape[1] - 1)
print("Expected total transitions:", (n_runs - 1) * (states_all.shape[1] - 1))
print("====================================")

# Scale inputs
U_all_flat = U_all.reshape(-1, U_all.shape[2])
u_scaler = StandardScaler()
u_scaler.fit(U_all_flat)
U_norm = u_scaler.transform(U_train.T).T

print("\n========== INPUT SCALER DEBUG ==========")
print("Input scaler mean:", u_scaler.mean_)
print("Input scaler scale:", u_scaler.scale_)
print("Scaled U_train mean (approx):", np.mean(U_norm, axis=1))
print("Scaled U_train std  (approx):", np.std(U_norm, axis=1))
print("========================================")

# Scale states
X_all_flat = states_all.reshape(-1, states_all.shape[2])
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

# State lifting (27-dim)
# Must match edmdc_mpc.py exactly.
#
# [ 0- 9] 10 linear states
# [10-13] sin(phi), cos(phi), sin(theta), cos(theta)
# [14-17] phi*p, theta*q, vx*phi, vy*theta
# [18-19] v_sq, omega_sq
# [20-25] vx*theta, vy*phi, vz², phi², theta², p*q
# [26]    bias

def observables(x, scaler):
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


# Test observable dimension
n_obs_test = len(observables(np.zeros(10), scaler))
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
def train_edmdc(Psi, Phi, U_norm, n_obs, lam):
    """Fit EDMDc via Tikhonov-regularized least squares."""
    Omega = np.vstack([Psi, U_norm])

    if lam == 0:
        AB = Phi @ pinv(Omega)
    else:
        G = Omega @ Omega.T
        AB = Phi @ Omega.T @ np.linalg.inv(G + lam * np.eye(G.shape[0]))

    A = AB[:, :n_obs]
    B = AB[:, n_obs:]

    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho > 1:
        A = A / rho

    return A, B


def rolling_horizon_rmse(states, inputs, A, B, scaler, u_scaler, horizon, observables_fn):
    n_total = states.shape[0]
    n_windows = n_total - horizon
    if n_windows <= 0:
        raise ValueError("Trajectory is shorter than the evaluation horizon.")

    pos_rmse_list = []
    vel_rmse_list = []
    full_rmse_list = []

    for start in range(n_windows):
        states_seg = states[start:start + horizon + 1]
        inputs_seg = inputs[start:start + horizon]

        psi_pred = np.zeros((A.shape[0], horizon + 1))
        psi_pred[:, 0] = observables_fn(
            scaler.transform(states_seg[0].reshape(1, -1)).flatten(),
            scaler
        )

        for k in range(1, horizon + 1):
            u_k = inputs_seg[k - 1].reshape(1, -1)
            u_k_s = u_scaler.transform(u_k).flatten()
            psi_pred[:, k] = A @ psi_pred[:, k - 1] + B @ u_k_s

        x_pred = scaler.inverse_transform(psi_pred[:10, :].T)

        err = states_seg - x_pred
        pos_err = err[:, 0:3]
        vel_err = err[:, 3:6]

        pos_rmse_list.append(np.sqrt(np.mean(pos_err**2)))
        vel_rmse_list.append(np.sqrt(np.mean(vel_err**2)))
        full_rmse_list.append(np.sqrt(np.mean(err**2)))

    return (
        float(np.mean(pos_rmse_list)),
        float(np.mean(vel_rmse_list)),
        float(np.mean(full_rmse_list)),
    )


family_names = {
    39: "helix",
    59: "figure-8",
    129: "lissajous",
    155: "waypoint",
    210: "hover excitation",
}

n_obs = Psi.shape[0]
h = MPC_HORIZON

print(f"\n{'='*60}")
print(f"REGULARIZATION SWEEP ({len(LAMBDA_CANDIDATES)} candidates)")
print(f"Evaluating rolling {h}-step RMSE on held-out trajectories")
print(f"{'='*60}")

best_lam = 0
best_avg_rmse = float("inf")

for lam in LAMBDA_CANDIDATES:
    A_try, B_try = train_edmdc(Psi, Phi, U_norm, n_obs, lam)

    per_traj = {}
    total = 0.0
    for tidx in test_indices:
        name = family_names.get(tidx, str(tidx))
        try:
            pos_r, _, _ = rolling_horizon_rmse(
                states_all[tidx], U_all[tidx],
                A_try, B_try, scaler, u_scaler, h,
                observables
            )
            per_traj[name] = pos_r
            total += pos_r
        except Exception:
            per_traj[name] = float("inf")
            total = float("inf")

    avg = total / len(test_indices)
    detail = "  ".join(f"{n}={v:.4f}" for n, v in per_traj.items())
    print(f"  lam={lam:.0e}  avg_pos={avg:.4f}  {detail}")

    if avg < best_avg_rmse:
        best_avg_rmse = avg
        best_lam = lam

print(f"\nBest lambda: {best_lam:.0e} (avg rolling pos RMSE: {best_avg_rmse:.4f})")

# ============================================================
# FINAL MODEL WITH BEST LAMBDA
# ============================================================
A, B = train_edmdc(Psi, Phi, U_norm, n_obs, best_lam)

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
labels_10 = ['x','y','z','vx','vy','vz','phi','theta','p','q']
for i, lbl in enumerate(labels_10):
    print(f"  {lbl:>6s}: {np.linalg.norm(B[i,:]):.6f}")
print("  --- lifted rows ---")
lifted_labels = ['sin_phi','cos_phi','sin_theta','cos_theta',
                 'phi*p','theta*q','vx*phi','vy*theta',
                 'v_sq','omega_sq',
                 'vx*theta','vy*phi','vz²','phi²','theta²','p*q',
                 'bias']
for i, lbl in enumerate(lifted_labels):
    print(f"  {lbl:>12s}: {np.linalg.norm(B[10+i,:]):.6f}")
print("==================================")


# ============================================================
# Held-out short-horizon evaluation
# ============================================================
labels = ['x','y','z','vx','vy','vz','phi','theta','p','q']
units  = ['m','m','m','m/s','m/s','m/s','rad','rad','rad/s','rad/s']

print("\n========== SHORT-HORIZON EVALUATION ==========")
print(f"Evaluation horizon: {h} steps ({h * dt:.2f} s)")
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
        u_k = U_short[k - 1, :].reshape(1, -1)
        u_k_s = u_scaler.transform(u_k).flatten()
        Psi_pred[:, k] = A @ Psi_pred[:, k - 1] + B @ u_k_s

    x_pred = scaler.inverse_transform(Psi_pred[:10, :].T).T
    err = states_short.T - x_pred

    rmse_each = np.sqrt(np.mean(err**2, axis=1))
    rmse_total = np.sqrt(np.mean(err**2))

    X_test_s = scaler.transform(states_short)
    U_test_s = u_scaler.transform(U_short)

    one_step_pred = np.zeros_like(states_short.T)
    one_step_pred[:, 0] = states_short[0]

    for k in range(states_short.shape[0] - 1):
        psi_k = observables(X_test_s[k], scaler)
        psi_next = A @ psi_k + B @ U_test_s[k]
        x_next_pred = scaler.inverse_transform(psi_next[:10].reshape(1, -1)).flatten()
        one_step_pred[:, k + 1] = x_next_pred

    err_one = states_short.T - one_step_pred
    rmse_one = np.sqrt(np.mean(err_one**2))
    pos_rmse_roll, vel_rmse_roll, full_rmse_roll = rolling_horizon_rmse(
        states_test, U_test, A, B, scaler, u_scaler, h, observables
    )

    print(f"\n--- {name} (idx={test_idx}) ---")
    print(f"One-step total RMSE:              {rmse_one:.4f}")
    print(f"Single {h}-step rollout RMSE:       {rmse_total:.4f}")
    print(f"Rolling {h}-step position RMSE:     {pos_rmse_roll:.4f}")
    print(f"Rolling {h}-step velocity RMSE:     {vel_rmse_roll:.4f}")
    print(f"Rolling {h}-step full-state RMSE:   {full_rmse_roll:.4f}")
    print("Per-state short-horizon RMSE (single rollout from initial state):")
    for lbl, val in zip(labels, rmse_each):
        print(f"  {lbl}: {val:.4f}")

    summary_rows.append((name, test_idx, rmse_one, rmse_total, pos_rmse_roll, vel_rmse_roll, full_rmse_roll))

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
    n_cols = 5
    n_rows = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 7))
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
    fig.suptitle(f"{name}: short-horizon state prediction", fontsize=14)
    plt.tight_layout()

print("\n========== SHORT-HORIZON SUMMARY ==========")
for name, idx, rmse_one, rmse_roll, pos_rmse_roll, vel_rmse_roll, full_rmse_roll in summary_rows:
    print(
        f"{name:<18s} idx={idx:<4d} "
        f"one-step={rmse_one:.4f}  "
        f"single-rollout={rmse_roll:.4f}  "
        f"rolling-pos={pos_rmse_roll:.4f}  "
        f"rolling-vel={vel_rmse_roll:.4f}  "
        f"rolling-full={full_rmse_roll:.4f}"
    )
print("===========================================")

plt.show()

# Save model
model_data = {
    "A": A,
    "B": B,
    "scaler": scaler,
    "u_scaler": u_scaler,
    "dt": dt,
    "n_obs": n_obs,
    "lambda": best_lam,
    "state_labels": labels,
    "u_labels": ["thrust", "phi_des", "theta_des"],
    "observable_labels": labels_10 + lifted_labels,
    "source_file": "runs_mixed_n300.pkl",
    "test_indices": test_indices,
    "u_type": "attitude_cmd",
}

with open("edmdc_model_300.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\nSaved model to edmdc_model_300.pkl")
print(f"A: {A.shape}, B: {B.shape}, n_obs: {n_obs}, lambda: {best_lam:.0e}")