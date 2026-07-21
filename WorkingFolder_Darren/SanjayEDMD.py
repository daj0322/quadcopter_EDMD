import numpy as np
import pandas as pd
from scipy.linalg import pinv
from scipy.signal import savgol_filter, butter, filtfilt
from sklearn.preprocessing import StandardScaler
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
from Simulation import quad_sim

# ====================================================
# CONFIG
# ====================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR  # change if CSVs are in a subfolder
noise_std = 0.02    # Gaussian noise std; 0 to disable
dt = 0.1               # time step (s)  # (used only for CSV case)

enable_filter = True
filter_type = 'savgol'  # 'savgol' or 'butter'
savgol_window = 11      # must be odd
savgol_poly = 3
butter_order = 2
butter_cutoff = 2.0

# ====================================================
# Load CSV and preprocess
# ====================================================
def load_trajectory(fp: Path, noise_std=0.0):
    df = pd.read_csv(fp)
    states = df.iloc[:, 0:12].values.astype(float)

    # Add Gaussian noise
    if noise_std > 0:
        states += np.random.normal(0, noise_std, states.shape)

    # Time vector
    N = states.shape[0]
    t = np.arange(N) * dt

    # Optional filtering
    if enable_filter:
        fs = 1.0 / dt
        for i in range(states.shape[1]):
            if filter_type == 'savgol':
                states[:, i] = savgol_filter(states[:, i], savgol_window, savgol_poly)
            else:
                nyq = 0.5 * fs
                Wn = butter_cutoff / nyq
                b, a = butter(butter_order, Wn, btype='low')
                states[:, i] = filtfilt(b, a, states[:, i])

    # Center positions only
    states[:, 0:3] -= states[0, 0:3]

    U = None  # no control inputs in CSV
    return t, states, U

# ====================================================
# Collect all CSVs   (REPLACED BY QUAD SIM TRAJECTORIES)
# ====================================================

# Instead of reading CSV files, generate trajectories with quad_sim.
# We keep the structure: "all_files" is replaced by multiple simulated runs.

n_runs = 5          # number of trajectories for training
traj_id = 2         # which trajectory type to use from quad_sim (1: helical or 2: figure eight)

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
    # Extract this run's time and states
    t_run = t_all[run]              # (T,)
    states_run = states_all[run]    # (T, 12)

    # Center positions only (like load_trajectory did)
    states_run = states_run.copy()
    states_run[:, 0:3] -= states_run[0, 0:3]

    if states_run.shape[0] < 2:
        continue

    # Current and next state snapshots
    Xc_list.append(states_run[:-1, :])   # (T-1, 12)
    Xn_list.append(states_run[1:, :])    # (T-1, 12)

    # For now, ignore control inputs so behavior matches original CSV EDMD
    U_list.append(np.zeros((0, states_run.shape[0]-1)))

# Stack all runs
Xc = np.vstack(Xc_list).T    # (12, K)
Xn = np.vstack(Xn_list).T    # (12, K)

# U is empty (no controls)
U_train = np.hstack(U_list)  # (0, K)
U_norm = U_train             # no scaling needed

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
    # Un-standardize angles back to radians for trigs.
    yaw_std = x[8]
    yaw_rad = yaw_std * scaler.scale_[8] + scaler.mean_[8]

    phi_std = x[6]
    theta_std = x[7]
    phi_rad = phi_std * scaler.scale_[6] + scaler.mean_[6]
    theta_rad = theta_std * scaler.scale_[7] + scaler.mean_[7]

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
Omega = np.vstack([Psi, U_norm])   # U_norm is empty, so this is just Psi
AB = Phi @ pinv(Omega)
n_obs = Psi.shape[0]
A = AB[:, :n_obs]
B = AB[:, n_obs:]  # empty if no controls (will be (n_obs, 0))

# ====================================================
# Simple prediction test (use last simulated run, unseen in training)
# ====================================================
t_test = t_all[n_runs - 1]                   # (T,)
states_test = states_all[n_runs - 1].copy()  # (T, 12)

# Center positions the same way as in training
states_test[:, 0:3] -= states_test[0, 0:3]

M = states_test.shape[0]

Psi_pred = np.zeros((n_obs, M))
Psi_pred[:, 0] = observables(scaler.transform(states_test[0, :].reshape(1, -1)).flatten())

# Predict while clipping to prevent overflow
clip_value = 1e6
for k in range(1, M):
    Psi_pred[:, k] = A @ Psi_pred[:, k - 1]
    Psi_pred[:, k] = np.clip(Psi_pred[:, k], -clip_value, clip_value)

x_pred = scaler.inverse_transform(Psi_pred[:12, :].T).T

# ====================================================
# Plot results
# ====================================================
labels = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
units  = ['m','m','m','m/s','m/s','m/s','rad','rad','rad','rad/s','rad/s','rad/s']

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
