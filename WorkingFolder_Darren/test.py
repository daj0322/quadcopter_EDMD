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
noise_std = 0.0    # Gaussian noise std; 0 to disable
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

n_runs = 5          # number of simulated trajectories (train on n_runs-1, test on last)
traj_id = 1         # which trajectory type to use from quad_sim (1: helical or 2: figure eight)

quad = quad_sim()
t_all, states_all, U_all, ref_traj_list = quad.fct_run_simulation(traj_id, n_runs)
# t_all.shape      = (n_runs, T)
# states_all.shape = (n_runs, T, 12)
# U_all.shape      = (n_runs, T, n_inputs)

print(f"Generated {n_runs} simulated trajectories from quad_sim.")

# ====================================================
# Build snapshots for EDMDc
# ====================================================
Xc_list, Xn_list, U_list = [], [], []

# Use the first n_runs - 1 trajectories for training
for run in range(n_runs - 1):
    t_run = t_all[run]               # (T,)
    states_run = states_all[run]     # (T, 12)
    U_run = U_all[run]               # (T, n_inputs)

    if states_run.shape[0] < 2:
        continue

    # current / next state snapshots
    Xc_list.append(states_run[:-1, :])   # (T-1, 12)
    Xn_list.append(states_run[1:, :])    # (T-1, 12)

    # control snapshots aligned with Xc
    U_list.append(U_run[:-1, :].T)       # (n_inputs, T-1)

# Stack all runs
Xc = np.vstack(Xc_list).T          # (12, K)
Xn = np.vstack(Xn_list).T          # (12, K)

# Stack control inputs
U_train = np.hstack(U_list)        # (n_inputs, K)

# Scale control inputs
u_scaler = StandardScaler()
U_norm = u_scaler.fit_transform(U_train.T).T   # (n_inputs, K)

# ====================================================
# Scale states
# ====================================================
scaler = StandardScaler()
Xc_s = scaler.fit_transform(Xc.T).T
Xn_s = scaler.transform(Xn.T).T

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
    # Un-standardize yaw to get back to degrees for trigs,
    # same as your original code. (psi index = 8)
    yaw_std = x[8]
    yaw_raw = yaw_std * scaler.scale_[8] + scaler.mean_[8]  # deg
    yaw_rad = np.deg2rad(yaw_raw)

    # For phi, theta you likely stored them in degrees originally,
    # standardized by 'scaler', so we unscale similarly:
    phi_std = x[6]
    theta_std = x[7]
    phi_raw = phi_std * scaler.scale_[6] + scaler.mean_[6]
    theta_raw = theta_std * scaler.scale_[7] + scaler.mean_[7]
    phi_rad = np.deg2rad(phi_raw)
    theta_rad = np.deg2rad(theta_raw)

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

# ====================================================
# EDMDc via pseudoinverse
# ====================================================
Omega = np.vstack([Psi, U_norm])
AB = Phi @ pinv(Omega)
n_obs = Psi.shape[0]
A = AB[:, :n_obs]
B = AB[:, n_obs:]  # (n_obs, n_inputs)

# ====================================================
# Simple prediction test (use last simulated run, unseen in training)
# ====================================================
t_test = t_all[n_runs - 1]                    # (M,)
states_test = states_all[n_runs - 1].copy()   # (M, 12)
U_test = U_all[n_runs - 1]                    # (M, n_inputs)
ref_test = ref_traj_list[n_runs - 1]

# ====================================================
# Plot reference trajectory for the test run (WORLD FRAME)
# ====================================================
xr = np.array([r["pos"][0] for r in ref_test], dtype=float)
yr = np.array([r["pos"][1] for r in ref_test], dtype=float)
zr = np.array([r["pos"][2] for r in ref_test], dtype=float)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(xr, yr, zr, '--', linewidth=2, label="Reference trajectory")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Reference trajectory used for test run (world frame)")
ax.legend()
ax.grid(True)
plt.show()

# ====================================================
# Plot simulation response vs reference trajectory (WORLD FRAME)
# ====================================================
x_sim = states_test[:, 0]
y_sim = states_test[:, 1]
z_sim = states_test[:, 2]

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
plt.show()

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

x_pred = scaler.inverse_transform(Psi_pred[:12, :].T).T

# ====================================================
# Plot EDMD predicted trajectory (3D) for the test run (WORLD FRAME)
# ====================================================
x_edmd = x_pred[0, :]
y_edmd = x_pred[1, :]
z_edmd = x_pred[2, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_edmd, y_edmd, z_edmd, linewidth=2, label="EDMD predicted trajectory")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("EDMD predicted trajectory (test run, world frame)")
ax.legend()
ax.grid(True)
plt.show()

# ====================================================
# Plot results
# ====================================================
labels = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
units  = ['m','m','m','m/s','m/s','m/s','deg','deg','deg','rad/s','rad/s','rad/s']

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