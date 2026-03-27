import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


# ============================================================
# CONFIG
# ============================================================
DATA_FILE = "runs_traj2_n200.pkl"
EDMDC_MODEL_FILE = "edmdc_model_traj2_n200.pkl"

# Pick whichever runs you want to test
TEST_RUNS = [150, 175, 190, 199]

MASS = 1.0
G = 9.81
IXX = 0.01
IYY = 0.01
IZZ = 0.02
KV = 0.1
KW = 0.01


# ============================================================
# LOADERS
# ============================================================
def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["t"], data["states"], data["U"], data["ref_traj_list"]


def load_edmdc_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


# ============================================================
# LINEAR MODEL
# ============================================================
def quadcopter_linearized_model(m, g, Ixx, Iyy, Izz, kv, kw, Ts):
    nx = 12
    nu = 4

    Ac = np.zeros((nx, nx))
    Bc = np.zeros((nx, nu))

    # positions
    Ac[0, 3] = 1.0
    Ac[1, 4] = 1.0
    Ac[2, 5] = 1.0

    # translational dynamics near hover
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
# ============================================================
def observables_edmd(x, scaler):
    x = np.asarray(x).flatten()
    x_s = scaler.transform(x.reshape(1, -1)).flatten()

    obs = list(x_s)
    n = len(x_s)

    for i, j in itertools.combinations_with_replacement(range(n), 2):
        obs.append(x_s[i] * x_s[j])

    pos_vel_indices = [0, 1, 2, 3, 4, 5]
    for i in pos_vel_indices:
        obs.append(x_s[i] ** 3)

    vx, vy, vz = x_s[3], x_s[4], x_s[5]
    p, q, r = x_s[9], x_s[10], x_s[11]
    obs.append(vx**2 + vy**2 + vz**2)
    obs.append(p**2 + q**2 + r**2)

    phi_raw = x_s[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_raw = x_s[7] * scaler.scale_[7] + scaler.mean_[7]
    psi_raw = x_s[8] * scaler.scale_[8] + scaler.mean_[8]

    phi_rad = np.deg2rad(phi_raw)
    theta_rad = np.deg2rad(theta_raw)
    psi_rad = np.deg2rad(psi_raw)

    obs += [
        np.sin(psi_rad), np.cos(psi_rad),
        np.sin(phi_rad), np.cos(phi_rad),
        np.sin(theta_rad), np.cos(theta_rad),
    ]

    obs.append(1.0)

    return np.array(obs, dtype=float)


# ============================================================
# ROLLOUTS
# ============================================================
def rollout_edmdc(x0, U_seq, A_edmd, B_edmd, scaler_edmd, u_scaler_edmd):
    T = U_seq.shape[0]
    X_pred = np.zeros((T, 12))
    X_pred[0] = x0

    z = observables_edmd(x0, scaler_edmd)

    for k in range(1, T):
        u = U_seq[k - 1]
        u_s = u_scaler_edmd.transform(u.reshape(1, -1)).flatten()

        z = A_edmd @ z + B_edmd @ u_s
        x = scaler_edmd.inverse_transform(z[:12].reshape(1, -1)).flatten()
        X_pred[k] = x

    return X_pred


def rollout_linear(x0, U_seq, Ad, Bd, u_nominal):
    T = U_seq.shape[0]
    X_pred = np.zeros((T, 12))
    X_pred[0] = x0

    x = x0.copy()

    for k in range(1, T):
        u = U_seq[k - 1]
        du = u - u_nominal
        x = Ad @ x + Bd @ du
        X_pred[k] = x

    return X_pred


# ============================================================
# METRICS
# ============================================================
def rmse_per_state(X_true, X_pred):
    err = X_true - X_pred
    return np.sqrt(np.mean(err**2, axis=0)), np.sqrt(np.mean(err**2))


def main():
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / DATA_FILE
    model_path = script_dir / EDMDC_MODEL_FILE

    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(data_path)
    edmd_model = load_edmdc_model(model_path)

    A_edmd = edmd_model["A"]
    B_edmd = edmd_model["B"]
    scaler_edmd = edmd_model["scaler"]
    u_scaler_edmd = edmd_model["u_scaler"]
    dt_edmd = edmd_model["dt"]

    sim_dt = t_all[0, 1] - t_all[0, 0]
    ratio = dt_edmd / sim_dt
    step = int(round(ratio))

    if not np.isclose(ratio, step, rtol=1e-6, atol=1e-8):
        raise ValueError(f"EDMD dt={dt_edmd} must be integer multiple of sim dt={sim_dt}")

    idx = np.arange(0, t_all.shape[1], step)

    t_all_ds = t_all[:, idx]
    states_all_ds = states_all[:, idx, :]
    U_all_ds = U_all[:, idx, :]

    Ad, Bd, _, _ = quadcopter_linearized_model(
        MASS, G, IXX, IYY, IZZ, KV, KW, dt_edmd
    )
    u_nominal = np.array([MASS * G, 0.0, 0.0, 0.0])

    labels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'p', 'q', 'r']

    results = []

    print("\n========== BATCH RMSE COMPARISON ==========")
    for run_idx in TEST_RUNS:
        t_test = t_all_ds[run_idx]
        X_true = states_all_ds[run_idx]
        U_test = U_all_ds[run_idx]

        x0 = X_true[0].copy()

        X_pred_edmd = rollout_edmdc(
            x0=x0,
            U_seq=U_test,
            A_edmd=A_edmd,
            B_edmd=B_edmd,
            scaler_edmd=scaler_edmd,
            u_scaler_edmd=u_scaler_edmd,
        )

        X_pred_lin = rollout_linear(
            x0=x0,
            U_seq=U_test,
            Ad=Ad,
            Bd=Bd,
            u_nominal=u_nominal,
        )

        rmse_edmd_each, rmse_edmd_total = rmse_per_state(X_true, X_pred_edmd)
        rmse_lin_each, rmse_lin_total = rmse_per_state(X_true, X_pred_lin)

        results.append({
            "run_idx": run_idx,
            "edmd_total": rmse_edmd_total,
            "lin_total": rmse_lin_total,
            "edmd_each": rmse_edmd_each,
            "lin_each": rmse_lin_each,
        })

        print(f"Run {run_idx:3d} | EDMDc total RMSE: {rmse_edmd_total:10.6f} | Linear total RMSE: {rmse_lin_total:10.6f}")

    print("===========================================")

    # Summary stats
    edmd_totals = np.array([r["edmd_total"] for r in results])
    lin_totals = np.array([r["lin_total"] for r in results])

    print("\n========== BATCH SUMMARY ==========")
    print(f"EDMDc mean total RMSE: {np.mean(edmd_totals):.6f}")
    print(f"EDMDc std  total RMSE: {np.std(edmd_totals):.6f}")
    print(f"Linear mean total RMSE: {np.mean(lin_totals):.6f}")
    print(f"Linear std  total RMSE: {np.std(lin_totals):.6f}")
    print("===================================")

    # Per-state average RMSE
    edmd_each_mean = np.mean(np.vstack([r["edmd_each"] for r in results]), axis=0)
    lin_each_mean = np.mean(np.vstack([r["lin_each"] for r in results]), axis=0)

    print("\n===== MEAN PER-STATE RMSE ACROSS TEST RUNS =====")
    for i, lbl in enumerate(labels):
        print(f"{lbl:>6s} | EDMDc: {edmd_each_mean[i]:.6f} | Linear: {lin_each_mean[i]:.6f}")
    print("===============================================")

    # Bar plot of total RMSE
    run_labels = [str(r["run_idx"]) for r in results]
    x = np.arange(len(results))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, edmd_totals, width, label="EDMDc")
    ax.bar(x + width/2, lin_totals, width, label="Linear")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.set_xlabel("Test run index")
    ax.set_ylabel("Total RMSE")
    ax.set_title("Batch comparison: EDMDc vs Linear prediction")
    ax.legend()
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.bar(x, edmd_totals, width=0.6, label="EDMDc")
    ax1.set_ylabel("EDMDc Total RMSE")
    ax1.set_title("EDMDc prediction error across test runs")
    ax1.grid(True, axis='y')

    ax2.bar(x, lin_totals, width=0.6, label="Linear")
    ax2.set_ylabel("Linear Total RMSE")
    ax2.set_xlabel("Test run index")
    ax2.set_title("Linear prediction error across test runs")
    ax2.grid(True, axis='y')

    ax2.set_xticks(x)
    ax2.set_xticklabels(run_labels)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()