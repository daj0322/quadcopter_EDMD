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
DATA_DIR = SCRIPT_DIR  # change if CSVs are in a subfolder
noise_std = 0.00    # Gaussian noise std; 0 to disable
dt = 0.01               # time step (s)

enable_filter = True
filter_type = 'savgol'  # 'savgol' or 'butter'
savgol_window = 11      # must be odd
savgol_poly = 3
butter_order = 2
butter_cutoff = 2.0

# ====================================================
# Collect all CSVs   (REPLACED BY QUAD SIM TRAJECTORIES)
# ====================================================

# Instead of reading CSV files, generate trajectories with quad_sim.
# We keep the structure: "all_files" is replaced by multiple simulated runs.

n_runs = 300          # number of simulated trajectories (train on n_runs-1, test on last)
test_idx = 199
traj_id = 2        # which trajectory type to use from quad_sim (1: helical or 2: figure eight)

def load_simulation_runs(filename="saved_runs.pkl"):
    import pickle

    with open(filename, "rb") as f:
        data = pickle.load(f)

    return data["t"], data["states"], data["U"], data["ref_traj_list"]

t_all, states_all, U_all, ref_traj_list = load_simulation_runs("runs_traj2_plus_hover_n300.pkl")

print("Loaded trajectory type:", traj_id)
print("Configured n_runs:", n_runs)
print("Loaded t shape:", t_all.shape)
print("Loaded states shape:", states_all.shape)
print("Loaded U shape:", U_all.shape)
print("Loaded ref count:", len(ref_traj_list))
print("\nAngle/unit sanity check:")
print("phi min/max:", np.min(states_all[:, :, 6]), np.max(states_all[:, :, 6]))
print("theta min/max:", np.min(states_all[:, :, 7]), np.max(states_all[:, :, 7]))
print("psi min/max:", np.min(states_all[:, :, 8]), np.max(states_all[:, :, 8]))

# ====================================================
# Downsample simulation data to match EDMD time step
# ====================================================

# Get simulation dt directly from time vector (works even when loading from file)
sim_dt = t_all[0, 1] - t_all[0, 0]

ratio = dt / sim_dt
step = int(round(ratio))

if not np.isclose(ratio, step, rtol=1e-6, atol=1e-8):
    raise ValueError(
        f"EDMD dt={dt} must be an integer multiple of simulation dt={sim_dt}"
    )

print(f"Downsampling: sim_dt={sim_dt}, edmd_dt={dt}, step={step}")

# Use explicit indices (safer and consistent)
idx = np.arange(0, t_all.shape[1], step)

t_all = t_all[:, idx]
states_all = states_all[:, idx, :]
U_all = U_all[:, idx, :]

# Downsample reference trajectories
ref_traj_list = [ref_traj[::step] for ref_traj in ref_traj_list]

print(f"Downsampled shape: {states_all.shape}")

# ====================================================
# Build snapshots for EDMDc
# ====================================================
Xc_list, Xn_list, U_list = [], [], []

train_indices = [i for i in range(n_runs) if i != test_idx]

print("Number of training runs:", len(train_indices))
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

# Stack all runs
Xc = np.vstack(Xc_list).T          # (12, K)
Xn = np.vstack(Xn_list).T          # (12, K)

# Stack control inputs
U_train = np.hstack(U_list)        # (n_inputs, K)

print("\n========== SNAPSHOT DEBUG ==========")
print("Xc shape:", Xc.shape)
print("Xn shape:", Xn.shape)
print("U_train shape:", U_train.shape)
print("Number of transitions per run:", states_all.shape[1] - 1)
print("Expected total transitions:", (n_runs - 1) * (states_all.shape[1] - 1))
print("====================================")

# Scale control inputs
u_scaler = StandardScaler()
U_norm = u_scaler.fit_transform(U_train.T).T   # (n_inputs, K)
print("\n========== INPUT SCALER DEBUG ==========")
print("Input scaler mean:", u_scaler.mean_)
print("Input scaler scale:", u_scaler.scale_)
print("Scaled U_train mean (approx):", np.mean(U_norm, axis=1))
print("Scaled U_train std  (approx):", np.std(U_norm, axis=1))
print("========================================")


# ====================================================
# Scale states
# ====================================================
scaler = StandardScaler()

Xc_s = scaler.fit_transform(Xc.T).T
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
# Extended observables (reduced to avoid overflow)
# ====================================================
def observables(x):
    """
    Lifted observables for EDMD/EDMDc.

    x is expected to be a *standardized* state vector of length 12:
        [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    """
    x = np.asarray(x).flatten()
    n = len(x)
    assert n == 12, f"Expected 12-state vector, got {n}"

    # ----------------------------------------------------
    # 1) Linear terms (same as before)
    # ----------------------------------------------------
    obs = list(x)

    # ----------------------------------------------------
    # 2) Quadratic terms (same as before)
    #    all x_i * x_j with i <= j
    # ----------------------------------------------------
    for i, j in itertools.combinations_with_replacement(range(n), 2):
        obs.append(x[i] * x[j])

    # ----------------------------------------------------
    # 3) Selected cubic terms
    #    - positions: x,y,z (0,1,2)
    #    - velocities: vx,vy,vz (3,4,5)
    #    These are often the most important for trajectories.
    # ----------------------------------------------------
    pos_vel_indices = [0, 1, 2, 3, 4, 5]
    for i in pos_vel_indices:
        xi = x[i]
        obs.append(xi**3)

    # ----------------------------------------------------
    # 4) "Energy-like" features
    #    - translational kinetic-like: |v|^2
    #    - angular rate "energy": p^2 + q^2 + r^2
    # ----------------------------------------------------
    vx, vy, vz = x[3], x[4], x[5]
    p, q, r = x[9], x[10], x[11]

    v_sq = vx**2 + vy**2 + vz**2
    omega_sq = p**2 + q**2 + r**2
    obs.append(v_sq)
    obs.append(omega_sq)

    # ----------------------------------------------------
    # 5) Extra trig terms for angles
    #    - you already had sin/cos(yaw)
    #    - add sin/cos for roll (phi) and pitch (theta) too
    # ----------------------------------------------------
    # Un-standardize angles and use them directly in radians
    yaw_std = x[8]
    yaw_raw = yaw_std * scaler.scale_[8] + scaler.mean_[8]
    yaw_rad = yaw_raw

    phi_std = x[6]
    theta_std = x[7]
    phi_raw = phi_std * scaler.scale_[6] + scaler.mean_[6]
    theta_raw = theta_std * scaler.scale_[7] + scaler.mean_[7]
    phi_rad = phi_raw
    theta_rad = theta_raw

    s_yaw, c_yaw = np.sin(yaw_rad), np.cos(yaw_rad)
    s_phi, c_phi = np.sin(phi_rad), np.cos(phi_rad)
    s_theta, c_theta = np.sin(theta_rad), np.cos(theta_rad)

    obs += [s_yaw, c_yaw, s_phi, c_phi, s_theta, c_theta]

    # ----------------------------------------------------
    # 6) Constant term (bias)
    # ----------------------------------------------------
    obs.append(1.0)

    return np.array(obs, dtype=float)

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
#AB = Phi @ Omega.T @ np.linalg.inv(G + lam * np.eye(G.shape[0]))
AB = Phi @ pinv(Omega)
n_obs = Psi.shape[0]
A = AB[:, :n_obs]
B = AB[:, n_obs:]


#rho = np.max(np.abs(np.linalg.eigvals(A)))
#if rho > 1:
#    A = A / rho

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

# ====================================================
# Simple prediction test (use last simulated run, unseen in training)
# ====================================================
t_test = t_all[test_idx]                    # (M,)
states_test = states_all[test_idx].copy()   # (M, 12)
U_test = U_all[test_idx]                    # (M, n_inputs)
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
# Plot simulation response vs reference trajectory (WORLD FRAME)
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
# plt.show()

M = states_test.shape[0]

Psi_pred = np.zeros((n_obs, M))
Psi_pred[:, 0] = observables(
    scaler.transform(states_test[0, :].reshape(1, -1)).flatten()
)

# Predict while clipping to prevent overflow
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

x_pred = scaler.inverse_transform(Psi_pred[:12, :].T).T

# ====================================================
# Plot simulation response and EDMD predicted trajectory (3D)
# ====================================================
x_edmd = x_pred[0, :]
y_edmd = x_pred[1, :]
z_edmd = x_pred[2, :]

x_sim = states_test[:, 0]
y_sim = states_test[:, 1]
z_sim = states_test[:, 2]

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
# plt.show()



# ====================================================
# Plot results
# ====================================================
labels = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
units  = ['m','m','m','m/s','m/s','m/s','deg','deg','deg','rad/s','rad/s','rad/s']

err = states_test.T - x_pred
print("\n========== ONE-STEP DEBUG ==========")
X_test_s = scaler.transform(states_test)
U_test_s = u_scaler.transform(U_test)

one_step_pred = np.zeros_like(states_test.T)
one_step_pred[:, 0] = states_test[0]

for k in range(states_test.shape[0] - 1):
    psi_k = observables(X_test_s[k])
    psi_next = A @ psi_k + B @ U_test_s[k]
    x_next_pred = scaler.inverse_transform(psi_next[:12].reshape(1, -1)).flatten()
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

n = err.shape[1]

quarters = {
    "first_25%": slice(0, n//4),
    "first_50%": slice(0, n//2),
    "full_100%": slice(0, n),
}

print("\n========== SEGMENT RMSE DEBUG ==========")
for name, sl in quarters.items():
    rmse_seg_total = np.sqrt(np.mean(err[:, sl]**2))
    rmse_seg_each = np.sqrt(np.mean(err[:, sl]**2, axis=1))
    print(f"\n{name} total RMSE: {rmse_seg_total:.4f}")
    for lbl, val in zip(labels, rmse_seg_each):
        print(f"  {lbl}: {val:.4f}")
print("========================================")

fig, axs = plt.subplots(3, 4, figsize=(16, 9))
for i, ax in enumerate(axs.flatten()):
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


import pickle

model_data = {
    "A": A,
    "B": B,
    "scaler": scaler,
    "u_scaler": u_scaler,
    "dt": dt,
    "source_file": "runs_traj2_plus_hover_n300.pkl",
}

with open("edmdc_model_traj2_plus_hover_n300.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Saved model to edmdc_model_traj2_plus_hover_n300.pkl")