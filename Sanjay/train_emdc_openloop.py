"""
train_edmdc_openloop.py

Train an EDMDc model on OPEN-LOOP hover excitation data.

Why this works better for MPC:
  - Closed-loop PID data: A encodes closed-loop dynamics,
    B is nearly zero (controller already handles everything).
  - Open-loop excitation data: A encodes free quadcopter dynamics,
    B correctly maps raw inputs to state changes → MPC can use it.

The hover excitation trajectories are small sinusoidal references
around hover, giving rich input-output data across the full
quadcopter state space near the operating point.
"""

# %% Imports
import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import pinv
from sklearn.preprocessing import StandardScaler

from Simulation import quad_sim

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

N_RUNS    = 200       # total runs (last one held out for test)
TEST_IDX  = 199
DT        = 0.01      # EDMD time step (must divide sim dt evenly)

OUTPUT_DATA_FILE  = "runs_hover_excitation_n200.pkl"
OUTPUT_MODEL_FILE = "edmdc_model_hover_excitation_n200.pkl"

REGENERATE_DATA   = True   # set False to reload existing pkl

# ============================================================
# STEP 1 — Generate or load hover excitation data
# ============================================================
sim = quad_sim()

if REGENERATE_DATA:
    print(f"Generating {N_RUNS} hover excitation runs...")
    sim.fct_save_hover_excitation_runs(N_RUNS, filename=OUTPUT_DATA_FILE)
    print("Done.")

print(f"\nLoading data from {OUTPUT_DATA_FILE}...")
with open(OUTPUT_DATA_FILE, "rb") as f:
    data = pickle.load(f)

t_all         = data["t"]
states_all    = data["states"]
U_all         = data["U"]
ref_traj_list = data["ref_traj_list"]

print("t shape:      ", t_all.shape)
print("states shape: ", states_all.shape)
print("U shape:      ", U_all.shape)

# ============================================================
# STEP 2 — Downsample to EDMD dt
# ============================================================
sim_dt = t_all[0, 1] - t_all[0, 0]
ratio  = DT / sim_dt
step   = int(round(ratio))

if not np.isclose(ratio, step, rtol=1e-6, atol=1e-8):
    raise ValueError(f"DT={DT} must be integer multiple of sim_dt={sim_dt}")

idx           = np.arange(0, t_all.shape[1], step)
t_all         = t_all[:, idx]
states_all    = states_all[:, idx, :]
U_all         = U_all[:, idx, :]
ref_traj_list = [ref[::step] for ref in ref_traj_list]

print(f"Downsampled: sim_dt={sim_dt}, edmd_dt={DT}, step={step}")
print("Downsampled states shape:", states_all.shape)

# ============================================================
# STEP 3 — Build snapshot matrices
# ============================================================
train_indices = [i for i in range(N_RUNS) if i != TEST_IDX]
print(f"\nTraining runs: {len(train_indices)}  |  Test run: {TEST_IDX}")

Xc_list, Xn_list, U_list = [], [], []

for run in train_indices:
    s = states_all[run]    # (T, 12)
    u = U_all[run]         # (T, 4)

    if s.shape[0] < 2:
        continue

    Xc_list.append(s[:-1, :])
    Xn_list.append(s[1:,  :])
    U_list.append(u[:-1, :].T)   # (4, T-1)

Xc = np.vstack(Xc_list).T    # (12, K)
Xn = np.vstack(Xn_list).T    # (12, K)
U_train = np.hstack(U_list)  # (4,  K)

print(f"Xc shape: {Xc.shape}  Xn shape: {Xn.shape}  U_train shape: {U_train.shape}")

# ============================================================
# STEP 4 — Scale states and controls
# ============================================================
scaler   = StandardScaler()
u_scaler = StandardScaler()

Xc_s = scaler.fit_transform(Xc.T).T    # (12, K) standardized
Xn_s = scaler.transform(Xn.T).T        # (12, K)
U_s  = u_scaler.fit_transform(U_train.T).T  # (4, K)

print("\nState scaler mean: ", scaler.mean_)
print("State scaler scale:", scaler.scale_)
print("Input scaler mean: ", u_scaler.mean_)
print("Input scaler scale:", u_scaler.scale_)

# Check for degenerate yaw scaler
if scaler.scale_[11] < 1e-10:
    print("\nWARNING: yaw rate (r) scaler is degenerate — zero variance in training data.")
    print("  This is expected if yaw torque (u4) is always 0.")
    print("  B_edmd[:, 3] will be meaningless; zero it out before MPC.")

if u_scaler.scale_[3] < 1e-3 or abs(u_scaler.scale_[3] - 1.0) < 1e-6:
    print("WARNING: u4 (yaw torque) scaler is degenerate — always 0 in training.")

# ============================================================
# STEP 5 — Observable function (must match MPC evaluation code)
# ============================================================
def observables(x_std, scaler_ref):
    """
    Lifted observables. Input x_std is the STANDARDIZED 12-state vector.
    scaler_ref is the fitted StandardScaler (needed to un-standardize angles).
    """
    x = np.asarray(x_std).flatten()
    assert len(x) == 12, f"Expected 12 states, got {len(x)}"

    obs = list(x)

    # Degree-2 monomials
    for i, j in itertools.combinations_with_replacement(range(12), 2):
        obs.append(x[i] * x[j])

    # Degree-3 for position and velocity
    for i in [0, 1, 2, 3, 4, 5]:
        obs.append(x[i] ** 3)

    # Speed and angular rate energy
    vx, vy, vz = x[3], x[4], x[5]
    p,  q,  r  = x[9], x[10], x[11]
    obs.append(vx**2 + vy**2 + vz**2)
    obs.append(p**2  + q**2  + r**2)

    # Trig observables (un-standardize angles before sin/cos)
    phi_rad   = x[6]  * scaler_ref.scale_[6]  + scaler_ref.mean_[6]
    theta_rad = x[7]  * scaler_ref.scale_[7]  + scaler_ref.mean_[7]
    yaw_rad   = x[8]  * scaler_ref.scale_[8]  + scaler_ref.mean_[8]

    obs += [
        np.sin(yaw_rad),   np.cos(yaw_rad),
        np.sin(phi_rad),   np.cos(phi_rad),
        np.sin(theta_rad), np.cos(theta_rad),
    ]

    obs.append(1.0)  # bias term
    return np.asarray(obs, dtype=float)

# ============================================================
# STEP 6 — Lift snapshots
# ============================================================
print("\nLifting snapshots...")
Psi = np.column_stack([observables(Xc_s[:, k], scaler) for k in range(Xc_s.shape[1])])
Phi = np.column_stack([observables(Xn_s[:, k], scaler) for k in range(Xn_s.shape[1])])

print(f"Psi shape: {Psi.shape}  Phi shape: {Phi.shape}")

n_obs = Psi.shape[0]

Omega = np.vstack([Psi, U_s])
print(f"Omega shape: {Omega.shape}")

# Condition number check
svals = np.linalg.svd(Omega, compute_uv=False)
cond  = svals[0] / svals[-1] if svals[-1] > 0 else np.inf
print(f"Omega condition number: {cond:.3e}")
print(f"Top 5 singular values: {svals[:5]}")
print(f"Bottom 5 singular values: {svals[-5:]}")

# ============================================================
# STEP 7 — Solve EDMDc via pseudoinverse
#   [A | B] = Phi @ pinv(Omega)
# ============================================================
print("\nSolving EDMDc...")
AB = Phi @ pinv(Omega)
A_edmd = AB[:, :n_obs]
B_edmd = AB[:, n_obs:]

print(f"A_edmd shape: {A_edmd.shape}")
print(f"B_edmd shape: {B_edmd.shape}")

eigvals = np.linalg.eigvals(A_edmd)
print(f"Max |eigenvalue| of A_edmd: {np.max(np.abs(eigvals)):.4f}")
print(f"Top 5 |eigenvalues|: {np.sort(np.abs(eigvals))[-5:]}")

print("\nB_edmd[:12, :] row norms (physical state authority):")
print(np.linalg.norm(B_edmd[:12, :], axis=1))
print("B_edmd col norms:", np.linalg.norm(B_edmd, axis=0))

# ============================================================
# STEP 8 — Open-loop prediction on held-out test run
# ============================================================
t_test      = t_all[TEST_IDX]
states_test = states_all[TEST_IDX].copy()    # (T, 12)
U_test      = U_all[TEST_IDX]               # (T, 4)

T = states_test.shape[0]
Psi_pred = np.zeros((n_obs, T))

x0_std       = scaler.transform(states_test[0].reshape(1, -1)).flatten()
Psi_pred[:, 0] = observables(x0_std, scaler)

clip_value = 1e6
for k in range(1, T):
    u_k_s         = u_scaler.transform(U_test[k-1].reshape(1, -1)).flatten()
    Psi_pred[:, k] = np.clip(A_edmd @ Psi_pred[:, k-1] + B_edmd @ u_k_s,
                              -clip_value, clip_value)

x_pred = scaler.inverse_transform(Psi_pred[:12, :].T)  # (T, 12)

err        = states_test - x_pred
rmse_each  = np.sqrt(np.mean(err**2, axis=0))
rmse_total = np.sqrt(np.mean(err**2))

labels = ["x","y","z","vx","vy","vz","phi","theta","psi","p","q","r"]
print("\n========== OPEN-LOOP PREDICTION RMSE (test run) ==========")
print(f"Total RMSE: {rmse_total:.6f}")
for lbl, val in zip(labels, rmse_each):
    print(f"  {lbl:>8s}: {val:.6f}")

# ============================================================
# STEP 9 — Save model
# ============================================================
model_data = {
    "A":        A_edmd,
    "B":        B_edmd,
    "scaler":   scaler,
    "u_scaler": u_scaler,
    "dt":       DT,
    "source_file": OUTPUT_DATA_FILE,
    "n_obs":    n_obs,
    "training_note": (
        "Trained on open-loop hover excitation data (PID tracking small "
        "sinusoidal references). B matrix should have meaningful authority "
        "on all physical states, suitable for Koopman MPC."
    ),
}

with open(OUTPUT_MODEL_FILE, "wb") as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to {OUTPUT_MODEL_FILE}")

# ============================================================
# STEP 10 — Plots
# ============================================================
units = ["m","m","m","m/s","m/s","m/s","rad","rad","rad","rad/s","rad/s","rad/s"]

fig, axs = plt.subplots(4, 3, figsize=(16, 10))
for i, ax in enumerate(axs.flatten()):
    rmse_i = np.sqrt(np.mean((states_test[:, i] - x_pred[:, i])**2))
    ax.plot(t_test, states_test[:, i], color="gray", lw=1.2, label="True")
    ax.plot(t_test, x_pred[:, i],      "r--",        lw=1.1, label="EDMDc OL")
    ax.set_title(f"{labels[i]}  RMSE={rmse_i:.4f}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(f"{labels[i]} [{units[i]}]")
    ax.grid(True)
    if i == 0:
        ax.legend()
fig.suptitle("EDMDc open-loop prediction — hover excitation model", fontsize=13)
fig.tight_layout()

# 3D trajectory plot
fig3 = plt.figure(figsize=(9, 7))
ax3  = fig3.add_subplot(111, projection="3d")
ax3.plot(states_test[:, 0], states_test[:, 1], states_test[:, 2],
         color="gray", lw=1.5, label="True")
ax3.plot(x_pred[:, 0], x_pred[:, 1], x_pred[:, 2],
         "r--", lw=1.5, label="EDMDc OL")
ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_zlabel("z [m]")
ax3.set_title("3D trajectory — hover excitation model")
ax3.legend(); ax3.grid(True)

plt.show()