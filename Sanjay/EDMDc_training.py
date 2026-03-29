# %%
import numpy as np
import pandas as pd
from scipy.linalg import pinv
from scipy.signal import savgol_filter, butter, filtfilt
from sklearn.preprocessing import StandardScaler
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
from Simulation import quad_sim


# %%
# ====================================================
# CONFIG
# ====================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
noise_std = 0.00
dt = 0.1               # EDMD time step (s)

enable_filter = True
filter_type = 'savgol'
savgol_window = 11
savgol_poly = 3
butter_order = 2
butter_cutoff = 2.0

# ====================================================
# Load simulation data
# ====================================================

def load_simulation_runs(filename):
    import pickle
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["t"], data["states"], data["U"], data["ref_traj_list"]

t_all, states_all, U_all, ref_traj_list = load_simulation_runs("runs_mixed_n300.pkl")

n_runs   = t_all.shape[0]
test_indices = [39, 59, 99, 129, 155, 210]
train_indices = [i for i in range(n_runs) if i not in test_indices]
test_idx = test_indices[0]  # use first one for the training script's own plots

print("Test indices (held out):", test_indices)
print("Number of training runs:", len(train_indices))

print("Loaded file: runs_mixedu_n350.pkl")
print("Total runs:", n_runs)
print("Test index:", test_idx)
print("t shape:", t_all.shape)
print("states shape (raw 12):", states_all.shape)
print("U shape (raw):", U_all.shape)
print("ref count:", len(ref_traj_list))


print(f"\nTrimmed states to 10: {states_all.shape}")

# ====================================================
# Drop psi_des(3) — keep 3 inputs
# [thrust, phi_des, theta_des]
# ====================================================
U_all = U_all[:, :, :3]
print(f"Trimmed U to 3: {U_all.shape}")

print("\nAngle/unit sanity check:")
print("phi min/max:", np.min(states_all[:, :, 6]), np.max(states_all[:, :, 6]))
print("theta min/max:", np.min(states_all[:, :, 7]), np.max(states_all[:, :, 7]))
print("p min/max:", np.min(states_all[:, :, 8]), np.max(states_all[:, :, 8]))
print("q min/max:", np.min(states_all[:, :, 9]), np.max(states_all[:, :, 9]))

print("\nU sanity check:")
print("thrust min/max:", np.min(U_all[:, :, 0]), np.max(U_all[:, :, 0]))
print("phi_des min/max:", np.min(U_all[:, :, 1]), np.max(U_all[:, :, 1]))
print("theta_des min/max:", np.min(U_all[:, :, 2]), np.max(U_all[:, :, 2]))

# ====================================================
# Downsample simulation data to match EDMD time step
# ====================================================
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

# ====================================================
# Build snapshots for EDMDc
# ====================================================
Xc_list, Xn_list, U_list = [], [], []

train_indices = [i for i in range(n_runs) if i != test_idx]

print("\nNumber of training runs:", len(train_indices))
print("Chosen test index:", test_idx)

for run in train_indices:
    t_run = t_all[run]
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

# ====================================================
# Scale control inputs
# ====================================================
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

# ====================================================
# Scale states
# ====================================================
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

# ====================================================
# Observables
# ====================================================
# State indices (after trimming):
#   0:x  1:y  2:z  3:vx  4:vy  5:vz  6:phi  7:theta  8:p  9:q
#
# Observables:
#   [0-9]   10 linear states
#   [10-13] sin(phi), cos(phi), sin(theta), cos(theta)
#   [14-17] phi*p, theta*q, vx*phi, vy*theta  (cross terms)
#   [18]    v_sq = vx^2 + vy^2 + vz^2
#   [19]    omega_sq = p^2 + q^2
#   [20]    bias
#
# Total: 21 observables — small enough for MPC QP

def observables(x):
    """
    Lifted observables for EDMDc.

    x: standardized 10-state vector
       [x, y, z, vx, vy, vz, phi, theta, p, q]
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
    # These capture gyroscopic coupling and velocity-tilt interaction
    obs.append(x[6] * x[8])   # phi * p
    obs.append(x[7] * x[9])   # theta * q
    obs.append(x[3] * x[6])   # vx * phi
    obs.append(x[4] * x[7])   # vy * theta

    # ----- Energy-like terms -----
    v_sq = x[3]**2 + x[4]**2 + x[5]**2
    omega_sq = x[8]**2 + x[9]**2
    obs.append(v_sq)
    obs.append(omega_sq)

    # ----- Bias -----
    obs.append(1.0)

    return np.array(obs, dtype=float)


# Test observable dimension
n_obs_test = len(observables(np.zeros(10)))
print(f"\nObservable dimension: {n_obs_test}")

# ====================================================
# Lift snapshots
# ====================================================
Psi = np.column_stack([observables(Xc_s[:, k]) for k in range(Xc_s.shape[1])])
Phi = np.column_stack([observables(Xn_s[:, k]) for k in range(Xn_s.shape[1])])

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

# ====================================================
# EDMDc via pseudoinverse
# ====================================================
lam = 1e-4
G = Omega @ Omega.T
AB = Phi @ pinv(Omega)
n_obs = Psi.shape[0]
A = AB[:, :n_obs]
B = AB[:, n_obs:]

rho = np.max(np.abs(np.linalg.eigvals(A)))
if rho > 1:
    A = A / rho
    print(f"Stabilized A: scaled by 1/{rho:.8f}")

print("\n========== MODEL DEBUG ==========")
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
                 'v_sq','omega_sq','bias']
for i, lbl in enumerate(lifted_labels):
    print(f"  {lbl:>12s}: {np.linalg.norm(B[10+i,:]):.6f}")
print("==================================")

# ====================================================
# Test on held-out run
# ====================================================
t_test = t_all[test_idx]
states_test = states_all[test_idx].copy()    # (M, 10)
U_test = U_all[test_idx]                     # (M, 3)
ref_test = ref_traj_list[test_idx]

print("\n========== RAW RANGE DEBUG ==========")
print("Xc min per state:", np.min(Xc, axis=1))
print("Xc max per state:", np.max(Xc, axis=1))
print("Xn min per state:", np.min(Xn, axis=1))
print("Xn max per state:", np.max(Xn, axis=1))
print("states_test min per state:", np.min(states_test, axis=0))
print("states_test max per state:", np.max(states_test, axis=0))
print("U_train min per input:", np.min(U_train, axis=1))
print("U_train max per input:", np.max(U_train, axis=1))
print("U_test min per input:", np.min(U_test, axis=0))
print("U_test max per input:", np.max(U_test, axis=0))
print("=====================================")

# ====================================================
# Plot simulation response vs reference trajectory
# ====================================================
x_sim = states_test[:, 0]
y_sim = states_test[:, 1]
z_sim = states_test[:, 2]

xr = np.array([r["pos"][0] for r in ref_test], dtype=float)
yr = np.array([r["pos"][1] for r in ref_test], dtype=float)
zr = np.array([r["pos"][2] for r in ref_test], dtype=float)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(xr, yr, zr, '--', linewidth=2, label="Reference trajectory")
ax.plot(x_sim, y_sim, z_sim, linewidth=2, label="Simulation response")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Simulation response vs reference (world frame)")
ax.legend()
ax.grid(True)

# ====================================================
# Rollout prediction
# ====================================================
M = states_test.shape[0]

Psi_pred = np.zeros((n_obs, M))
Psi_pred[:, 0] = observables(
    scaler.transform(states_test[0, :].reshape(1, -1)).flatten()
)

clip_value = 1e6
for k in range(1, M):
    u_k = U_test[k-1, :].reshape(1, -1)
    u_k_s = u_scaler.transform(u_k).flatten()

    Psi_pred[:, k] = A @ Psi_pred[:, k - 1] + B @ u_k_s
    Psi_pred[:, k] = np.clip(Psi_pred[:, k], -clip_value, clip_value)

    if k in [1, 5, 10, 25, 50, 100, 150, 200, 250, 300, M-1]:
        print(f"\n--- Rollout step {k} ---")
        print("||Psi_pred[:,k]||2 =", np.linalg.norm(Psi_pred[:, k]))
        print("max |Psi_pred[:,k]| =", np.max(np.abs(Psi_pred[:, k])))
        print("u_k_s =", u_k_s)

# Inverse transform only the first 10 rows (linear state rows)
x_pred = scaler.inverse_transform(Psi_pred[:10, :].T).T

# ====================================================
# Plot EDMD predicted trajectory (3D)
# ====================================================
x_edmd = x_pred[0, :]
y_edmd = x_pred[1, :]
z_edmd = x_pred[2, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_sim, y_sim, z_sim, linewidth=2, label="Simulation response")
ax.plot(x_edmd, y_edmd, z_edmd, '--', linewidth=2, label="EDMD predicted trajectory")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Simulation response and EDMD prediction (world frame)")
ax.legend()
ax.grid(True)
plt.show()

# ====================================================
# One-step vs rollout comparison
# ====================================================
labels = ['x','y','z','vx','vy','vz','phi','theta','p','q']
units  = ['m','m','m','m/s','m/s','m/s','rad','rad','rad/s','rad/s']

err = states_test.T - x_pred

print("\n========== ONE-STEP DEBUG ==========")
X_test_s = scaler.transform(states_test)
U_test_s = u_scaler.transform(U_test)

one_step_pred = np.zeros_like(states_test.T)
one_step_pred[:, 0] = states_test[0]

for k in range(states_test.shape[0] - 1):
    psi_k = observables(X_test_s[k])
    psi_next = A @ psi_k + B @ U_test_s[k]
    x_next_pred = scaler.inverse_transform(psi_next[:10].reshape(1, -1)).flatten()
    one_step_pred[:, k + 1] = x_next_pred

err_one = states_test.T - one_step_pred
print("One-step total RMSE:", np.sqrt(np.mean(err_one**2)))
print("Rollout total RMSE:", np.sqrt(np.mean(err**2)))
print("====================================")

rmse_each = np.sqrt(np.mean(err**2, axis=1))
rmse_total = np.sqrt(np.mean(err**2))

print("\nPer-state RMSE:")
for lbl, val in zip(labels, rmse_each):
    print(f"{lbl}: {val:.4f}")
print(f"Total RMSE: {rmse_total:.4f}")

n_steps = err.shape[1]

quarters = {
    "first_25%": slice(0, n_steps//4),
    "first_50%": slice(0, n_steps//2),
    "full_100%": slice(0, n_steps),
}

print("\n========== SEGMENT RMSE DEBUG ==========")
for name, sl in quarters.items():
    rmse_seg_total = np.sqrt(np.mean(err[:, sl]**2))
    rmse_seg_each = np.sqrt(np.mean(err[:, sl]**2, axis=1))
    print(f"\n{name} total RMSE: {rmse_seg_total:.4f}")
    for lbl, val in zip(labels, rmse_seg_each):
        print(f"  {lbl}: {val:.4f}")
print("========================================")

# ====================================================
# Per-state time series plots
# ====================================================
n_states = len(labels)
n_cols = 5
n_rows = 2
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 7))
for i in range(n_states):
    row, col = divmod(i, n_cols)
    ax = axs[row, col]
    rmse = np.sqrt(np.mean((states_test[:, i] - x_pred[i])**2))
    ax.plot(t_test, states_test[:, i], label='True')
    ax.plot(t_test, x_pred[i], '--', label='EDMDc')
    ax.set_title(f"{labels[i]} (RMSE {rmse:.3f})")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f"{labels[i]} [{units[i]}]")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# ====================================================
# Save model
# ====================================================
import pickle

model_data = {
    "A": A,
    "B": B,
    "scaler": scaler,
    "u_scaler": u_scaler,
    "dt": dt,
    "n_obs": n_obs,
    "state_labels": labels,
    "u_labels": ["thrust", "phi_des", "theta_des"],
    "source_file": "runs_mixed_n300.pkl",
    "u_type": "attitude_cmd",
}

with open("edmdc_model_0.1.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\nSaved model to edmdc_model_300.pkl")
print(f"A: {A.shape}, B: {B.shape}, n_obs: {n_obs}")