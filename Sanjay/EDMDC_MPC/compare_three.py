"""
Compare current simulator logs against EDMDc and a linear baseline.

The "PID" trace is the trajectory produced by Darren's current PX4-like
controller in Simulation.py. Both learned models are rolled out with the same
logged wrench inputs:

    [thrust, tau_roll, tau_pitch, tau_yaw]
"""

import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from edmdc_mpc import (
    STATE_DIM,
    build_ref_horizon,
    extract_ref_xyz,
    lifted_state_from_x,
    load_edmdc_model,
    load_simulation_runs,
    precompute_ref_std,
    rmse,
    scaled_lifted_input_from_phys,
    wrap_angle_pi,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.environ.get("EDMDC_DATA_FILE", SCRIPT_DIR / "runs_mixed_n150.pkl"))
MODEL_FILE = Path(os.environ.get("EDMDC_MODEL_FILE", SCRIPT_DIR / "edmdc_model_yaw_wrench.pkl"))
DEFAULT_TEST_INDICES = (39, 59, 129)
ROLL_STEPS = None
PREDICTION_WINDOW_STEPS = 100
LINEAR_RIDGE_LAMBDAS = [0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0]


def fit_linear_baseline(t_all, states_all, U_all, train_indices, validation_indices):
    """Fit x[k+1] = A*x[k] + B*u[k] + c in standardized coordinates."""
    Xc = np.vstack([states_all[i, :-1, :] for i in train_indices])
    Xn = np.vstack([states_all[i, 1:, :] for i in train_indices])
    U = np.vstack([U_all[i, :-1, :] for i in train_indices])

    x_scaler = StandardScaler().fit(np.vstack([Xc, Xn]))
    u_scaler = StandardScaler().fit(U)

    Xc_s = x_scaler.transform(Xc)
    Xn_s = x_scaler.transform(Xn)
    U_s = u_scaler.transform(U)
    Omega = np.hstack([Xc_s, U_s, np.ones((Xc_s.shape[0], 1))])

    A, B, c, best_lam = tune_linear_ridge(
        Omega, Xn_s, x_scaler, u_scaler, states_all, U_all, validation_indices
    )
    print(f"Linear baseline: logged-data ridge fit, lambda={best_lam:g}")
    return A, B, c, x_scaler, u_scaler


def tune_linear_ridge(Omega, Xn_s, x_scaler, u_scaler, states_all, U_all,
                      validation_indices):
    valid_indices = [idx for idx in validation_indices if idx < states_all.shape[0]]
    if not valid_indices:
        raise ValueError("At least one validation index is required for the linear baseline.")

    best = None
    for lam in LINEAR_RIDGE_LAMBDAS:
        G = Omega.T @ Omega + float(lam) * np.eye(Omega.shape[1])
        W = np.linalg.solve(G, Omega.T @ Xn_s)
        A = W[:STATE_DIM, :].T
        B = W[STATE_DIM:STATE_DIM + 4, :].T
        c = W[-1, :]

        scores = []
        for idx in valid_indices:
            X_lin = rollout_linear_windowed(
                A, B, c, x_scaler, u_scaler,
                states_all[idx], U_all[idx], PREDICTION_WINDOW_STEPS
            )
            scores.append(rmse(X_lin[:, 0:3], states_all[idx, :, 0:3]))
        score = float(np.mean(scores)) if scores else 0.0
        if best is None or score < best[0]:
            best = (score, lam, A, B, c)

    _, best_lam, A_best, B_best, c_best = best
    return A_best, B_best, c_best, best_lam


def rollout_edmd(model, x0, U, steps):
    A = model["A"]
    B = model["B"]
    scaler = model["scaler"]
    u_scaler = model["u_scaler"]

    n_obs = A.shape[0]
    X = np.zeros((steps, STATE_DIM))
    z = lifted_state_from_x(x0, scaler)
    X[0] = x0

    for k in range(steps - 1):
        x_phys = scaler.inverse_transform(z[:STATE_DIM].reshape(1, -1)).flatten()
        u_s = scaled_lifted_input_from_phys(x_phys, U[k], u_scaler)
        z = A @ z + B @ u_s
        X[k + 1] = scaler.inverse_transform(z[:STATE_DIM].reshape(1, -1)).flatten()

    return X


def rollout_edmd_windowed(model, X_true, U, window_steps):
    steps = X_true.shape[0]
    X = np.zeros((steps, STATE_DIM))
    X[0] = X_true[0]
    for start in range(0, steps - 1, window_steps):
        end = min(start + window_steps + 1, steps)
        X[start:end] = rollout_edmd(model, X_true[start], U[start:end], end - start)
    return X


def rollout_linear(A, B, c, x_scaler, u_scaler, x0, U, steps):
    X = np.zeros((steps, STATE_DIM))
    x_s = x_scaler.transform(x0.reshape(1, -1)).flatten()
    X[0] = x0

    for k in range(steps - 1):
        u_s = u_scaler.transform(U[k].reshape(1, -1)).flatten()
        x_s = A @ x_s + B @ u_s + c
        X[k + 1] = x_scaler.inverse_transform(x_s.reshape(1, -1)).flatten()

    return X


def rollout_linear_windowed(A, B, c, x_scaler, u_scaler, X_true, U, window_steps):
    steps = X_true.shape[0]
    X = np.zeros((steps, STATE_DIM))
    X[0] = X_true[0]
    for start in range(0, steps - 1, window_steps):
        end = min(start + window_steps + 1, steps)
        X[start:end] = rollout_linear(
            A, B, c, x_scaler, u_scaler, X_true[start], U[start:end], end - start
        )
    return X


def set_axes_equal_3d(ax, xyz):
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.55 * max(np.max(maxs - mins), 1.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def set_readable_axes_3d(ax, xyz):
    xyz = np.asarray(xyz, dtype=float)
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    spans = np.maximum(maxs - mins, 0.25)
    pad = 0.12 * spans + 0.15
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])


def parse_test_cases(raw_indices, n_runs):
    try:
        indices = [int(value.strip()) for value in raw_indices.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--test-indices must be a comma-separated list of integers") from exc
    if not indices:
        raise ValueError("--test-indices must contain at least one index")
    invalid = [idx for idx in indices if idx < 0 or idx >= n_runs]
    if invalid:
        raise ValueError(f"Requested test indices outside [0, {n_runs - 1}]: {invalid}")
    return [(idx, f"run-{idx}") for idx in indices]


def main():
    parser = argparse.ArgumentParser(
        description="Compare held-out yaw-aware EDMDc and linear open-loop predictions."
    )
    parser.add_argument(
        "--test-indices",
        default=os.environ.get("EDMDC_TEST_INDICES", ",".join(map(str, DEFAULT_TEST_INDICES))),
        help="Comma-separated held-out dataset indices to visualize.",
    )
    args = parser.parse_args()

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE.name}. Run: python parallel_sim.py && python mix_traj.py"
        )
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_FILE.name}. Run: python EDMDc_training.py"
        )

    model = load_edmdc_model(MODEL_FILE)
    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(DATA_FILE)

    n_runs = states_all.shape[0]
    test_cases = parse_test_cases(args.test_indices, n_runs)
    held_out = {idx for idx, _ in test_cases}
    train_indices = [
        int(idx) for idx in model.get("train_indices", [])
        if 0 <= int(idx) < n_runs
    ]
    if not train_indices:
        train_indices = [i for i in range(n_runs) if i not in held_out]
    if held_out & set(train_indices):
        raise ValueError("The EDMDc model was trained on a requested test run.")
    validation_indices = [
        int(idx) for idx in model.get("validation_indices", [])
        if 0 <= int(idx) < n_runs and int(idx) not in held_out
    ]
    if not validation_indices:
        validation_indices = train_indices[:1]
    A_lin, B_lin, c_lin, x_lin_scaler, u_lin_scaler = fit_linear_baseline(
        t_all, states_all, U_all, train_indices, validation_indices
    )

    rows = []
    fig = plt.figure(figsize=(6.5 * len(test_cases), 6))
    fig_err, err_axes = plt.subplots(
        len(test_cases), 1, figsize=(12, 3.2 * len(test_cases)), sharex=False
    )
    err_axes = np.atleast_1d(err_axes)

    for plot_idx, (run_idx, name) in enumerate(test_cases, start=1):

        steps = states_all.shape[1] if ROLL_STEPS is None else min(ROLL_STEPS, states_all.shape[1])
        t = t_all[run_idx, :steps]
        X_pid = states_all[run_idx, :steps, :]
        U = U_all[run_idx, :steps, :]
        ref_xyz = extract_ref_xyz(ref_traj_list[run_idx])[:steps]

        X_edmd = rollout_edmd_windowed(model, X_pid, U, PREDICTION_WINDOW_STEPS)
        X_lin = rollout_linear_windowed(
            A_lin, B_lin, c_lin, x_lin_scaler, u_lin_scaler,
            X_pid, U, PREDICTION_WINDOW_STEPS
        )

        pid_rmse = rmse(X_pid[:, 0:3], ref_xyz)
        edmd_rmse = rmse(X_edmd[:, 0:3], X_pid[:, 0:3])
        lin_rmse = rmse(X_lin[:, 0:3], X_pid[:, 0:3])
        yaw_edmd = float(np.sqrt(np.mean(wrap_angle_pi(X_edmd[:, 8] - X_pid[:, 8])**2)))
        yaw_lin = float(np.sqrt(np.mean(wrap_angle_pi(X_lin[:, 8] - X_pid[:, 8])**2)))
        r_edmd = rmse(X_edmd[:, 11], X_pid[:, 11])
        r_lin = rmse(X_lin[:, 11], X_pid[:, 11])
        rows.append((name, pid_rmse, edmd_rmse, lin_rmse, yaw_edmd, yaw_lin, r_edmd, r_lin))

        ax = fig.add_subplot(1, len(test_cases), plot_idx, projection="3d")
        ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], "k", lw=2.0, label="Reference")
        ax.plot(
            X_pid[:, 0], X_pid[:, 1], X_pid[:, 2],
            color="#777777", lw=1.6,
            label=f"PID/PX4 sim ({pid_rmse:.4f} m)"
        )
        ax.plot(
            X_edmd[:, 0], X_edmd[:, 1], X_edmd[:, 2],
            color="#2ca02c", lw=1.4, ls="--",
            label=f"EDMDc ({edmd_rmse:.4f} m)"
        )
        ax.plot(
            X_lin[:, 0], X_lin[:, 1], X_lin[:, 2],
            color="#1f77b4", lw=1.2, ls=":",
            label=f"Linear ({lin_rmse:.4f} m)"
        )
        ax.set_title(name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        set_readable_axes_3d(ax, np.vstack([ref_xyz, X_pid[:, 0:3]]))
        ax.legend(fontsize=8)

        err_edmd = np.linalg.norm(X_edmd[:, 0:3] - X_pid[:, 0:3], axis=1)
        err_lin = np.linalg.norm(X_lin[:, 0:3] - X_pid[:, 0:3], axis=1)
        err_ax = err_axes[plot_idx - 1]
        err_ax.plot(t, err_edmd, color="#2ca02c", lw=1.2, label="EDMDc")
        err_ax.plot(t, err_lin, color="#1f77b4", lw=1.0, ls=":", label="Linear")
        err_ax.set_title(
            f"{name}: {PREDICTION_WINDOW_STEPS * (t[1] - t[0]):.1f}s reset-window position error"
        )
        err_ax.set_ylabel("error [m]")
        err_ax.grid(True, alpha=0.3)
        err_ax.legend()

        fig_yaw, ax_yaw = plt.subplots(figsize=(10, 4))
        ax_yaw.plot(t, np.unwrap(X_pid[:, 8]), color="#777777", label="PID/PX4 sim")
        ax_yaw.plot(t, np.unwrap(X_edmd[:, 8]), color="#2ca02c", ls="--", label="EDMDc")
        ax_yaw.plot(t, np.unwrap(X_lin[:, 8]), color="#1f77b4", ls=":", label="Linear")
        ax_yaw.set_title(f"{name}: yaw rollout")
        ax_yaw.set_xlabel("time [s]")
        ax_yaw.set_ylabel("yaw [rad]")
        ax_yaw.grid(True, alpha=0.3)
        ax_yaw.legend()
        fig_yaw.tight_layout()

    err_axes[-1].set_xlabel("time [s]")
    fig.suptitle("PID/PX4 Simulation vs EDMDc vs Linear Windowed Rollout")
    fig.tight_layout()
    fig_err.tight_layout()

    print("\nComparison over logged wrench-input rollouts")
    print(
        "EDMDc/Linear RMSE are prediction errors relative to the PID/PX4 sim trace "
        f"with {PREDICTION_WINDOW_STEPS}-step reset windows."
    )
    print(
        f"{'case':<10s} {'PID-ref':>10s} {'EDMD-pos':>10s} {'LIN-pos':>10s} "
        f"{'EDMD-yaw':>10s} {'LIN-yaw':>10s} {'EDMD-r':>10s} {'LIN-r':>10s}"
    )
    for name, pid_e, edmd_e, lin_e, yaw_e, yaw_l, r_e, r_l in rows:
        print(
            f"{name:<10s} {pid_e:10.4f} {edmd_e:10.4f} {lin_e:10.4f} "
            f"{yaw_e:10.4f} {yaw_l:10.4f} {r_e:10.4f} {r_l:10.4f}"
        )

    plt.show()


if __name__ == "__main__":
    main()
