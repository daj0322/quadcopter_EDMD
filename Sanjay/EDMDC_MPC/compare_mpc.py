"""
Closed-loop MPC comparison for the current yaw-wrench simulator.

Compares:
  - PID/PX4-like controller from Simulation.py
  - EDMDc MPC using edmdc_model_yaw_wrench.pkl
  - Linear MPC fit from runs_mixed_n150.pkl

All controllers command the real simulated plant through wrench inputs:
    [thrust, tau_roll, tau_pitch, tau_yaw]
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from compare_three import fit_linear_baseline
from edmdc_mpc import (
    EDMDcMPC_QP,
    STATE_DIM,
    build_ref_horizon,
    extract_ref_xyz,
    lifted_state_from_x,
    load_edmdc_model,
    load_simulation_runs,
    precompute_ref_std,
    reference_yaw_arrays,
    rmse,
    wrap_angle_pi,
)
from PID_Mixer import pid_mixer
from Simulation import quad_sim


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.environ.get("EDMDC_DATA_FILE", SCRIPT_DIR / "runs_mixed_n150.pkl"))
MODEL_FILE = Path(os.environ.get("EDMDC_MODEL_FILE", SCRIPT_DIR / "edmdc_model_yaw_wrench.pkl"))
DEFAULT_TEST_INDICES = (39, 59, 129)

USE_PID_NOMINAL = True

N_MPC = 30
NC_MPC = 10
LEGACY_N_MPC = 18
LEGACY_NC_MPC = 6

Q_DIAG = np.array([
    45.0, 45.0, 55.0,
    4.0, 4.0, 5.0,
    0.2, 0.2, 4.0,
    0.05, 0.05, 1.0,
], dtype=float)
R_DIAG = np.array([0.12, 1.2, 1.2, 0.8], dtype=float)
RD_DIAG = np.array([0.04, 0.35, 0.35, 0.25], dtype=float)
Q_TERMINAL_SCALE = 3.0
# Both MPC controllers must apply the same optimized correction. Different
# blending factors would confound a comparison of prediction models.
EDMD_CORRECTION_BLEND = 1.0
LINEAR_CORRECTION_BLEND = 1.0
LEGACY_R_DIAG = np.array([0.25, 5.0, 5.0, 3.0], dtype=float)
LEGACY_RD_DIAG = np.array([0.12, 1.5, 1.5, 1.0], dtype=float)
LEGACY_Q_TERMINAL_SCALE = 1.0
LEGACY_EDMD_CORRECTION_BLEND = 1.0
LEGACY_LINEAR_CORRECTION_BLEND = 1.0

DU_RAW_MAX = np.array([2.0, 0.18, 0.18, 0.18], dtype=float)
LEGACY_DU_SCALED_MAX = np.array([0.7, 0.06, 0.06, 0.06], dtype=float)
DU_MIN = -LEGACY_DU_SCALED_MAX
DU_MAX = LEGACY_DU_SCALED_MAX


class InputScalerView:
    def __init__(self, scaler, n_inputs):
        self.mean_ = np.asarray(scaler.mean_[:n_inputs], dtype=float)
        self.scale_ = np.asarray(scaler.scale_[:n_inputs], dtype=float)
        self.n_features_in_ = int(n_inputs)

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=float)
        return values * self.scale_ + self.mean_


def scaled_delta_bounds(u_scaler, raw_delta_max):
    raw_delta_max = np.asarray(raw_delta_max, dtype=float)
    scales = np.asarray(u_scaler.scale_[:raw_delta_max.size], dtype=float)
    scaled = raw_delta_max / scales
    return -scaled, scaled


def timing_summary(solve_times):
    """Return mean, 95th-percentile, and worst-case controller latency in ms."""
    values_ms = 1e3 * np.asarray(solve_times, dtype=float)
    if values_ms.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (
        float(np.mean(values_ms)),
        float(np.percentile(values_ms, 95)),
        float(np.max(values_ms)),
    )


def yaw_tracking_metrics(states, ref_traj, dt):
    """Return wrapped yaw and yaw-rate RMSE against a reference trajectory."""
    yaw_ref, yaw_rate_ref = reference_yaw_arrays(ref_traj, dt=dt)
    yaw_error = wrap_angle_pi(states[:, 8] - yaw_ref)
    return (
        float(np.sqrt(np.mean(yaw_error**2))),
        float(rmse(states[:, 11], yaw_rate_ref)),
    )


def parse_test_cases(raw_indices, n_runs):
    """Create deterministic held-out comparison cases from comma-separated indices."""
    try:
        requested = [int(value.strip()) for value in raw_indices.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--test-indices must be a comma-separated list of integers") from exc
    if not requested:
        raise ValueError("--test-indices must contain at least one index")
    invalid = [idx for idx in requested if idx < 0 or idx >= n_runs]
    if invalid:
        raise ValueError(f"Requested test indices outside [0, {n_runs - 1}]: {invalid}")
    return [(idx, f"run-{idx}") for idx in requested]


def edmd_mpc_matrices(model):
    A = np.asarray(model["A"], dtype=float).copy()
    B = np.asarray(model["B"], dtype=float).copy()
    bias_idx = A.shape[0] - 1
    A[bias_idx, :] = 0.0
    B[bias_idx, :] = 0.0
    A[bias_idx, bias_idx] = 1.0
    return A, B


def controller_nominal_wrench(sim, state, ref, nominal_controller, use_pid_nominal):
    if use_pid_nominal:
        _, u_nom = nominal_controller.fct_step(state, ref, sim.dt)
        return clamp_wrench(sim, u_nom)

    acc_ref = np.asarray(ref.get("acc", np.zeros(3)), dtype=float)
    force_world = sim.q_mass * (acc_ref + np.array([0.0, 0.0, sim.g]))
    thrust_nom = np.linalg.norm(force_world)
    return clamp_wrench(sim, np.array([thrust_nom, 0.0, 0.0, 0.0], dtype=float))


def clamp_wrench(sim, u):
    """Project a wrench through the shared motor allocator."""
    _, u_applied, _, _ = pid_mixer.fct_allocate_wrench(
        u, sim.quad.kT, sim.quad.kD, sim.quad.l,
        min_omega=0.0, max_omega=sim.max_speed,
        prop_efficiency=sim.quad.prop_efficiency,
    )
    return u_applied


def allocation_parameters(sim):
    """Return the exact motor constraints used by the simulated plant."""
    return (
        pid_mixer.fct_allocation_matrix(sim.quad.kT, sim.quad.kD, sim.quad.l),
        np.zeros(4, dtype=float),
        pid_mixer.fct_max_motor_forces(
            sim.quad.kT, sim.max_speed,
            prop_efficiency=sim.quad.prop_efficiency,
        ),
    )


def plant_step(sim, state, u, dt):
    omega_cmd, u_applied, _, _ = pid_mixer.fct_allocate_wrench(
        u, sim.quad.kT, sim.quad.kD, sim.quad.l,
        min_omega=0.0, max_omega=sim.max_speed,
        prop_efficiency=sim.quad.prop_efficiency,
    )

    def ode(_, s_local):
        return sim.quad.fct_dynamics(0.0, s_local, omega_cmd)

    sol = solve_ivp(ode, [0.0, dt], state, method="RK45")
    return sol.y[:, -1], u_applied


def build_edmd_mpc(model, sim, legacy_pid_nominal=False):
    n_obs = model["n_obs"]
    Cz = np.zeros((STATE_DIM, n_obs))
    Cz[:STATE_DIM, :STATE_DIM] = np.eye(STATE_DIM)
    hover = np.array([sim.q_mass * sim.g, 0.0, 0.0, 0.0], dtype=float)
    A_mpc, B_mpc = edmd_mpc_matrices(model)
    if legacy_pid_nominal:
        du_min, du_max = -LEGACY_DU_SCALED_MAX, LEGACY_DU_SCALED_MAX
    else:
        du_min, du_max = scaled_delta_bounds(model["u_scaler"], DU_RAW_MAX)
    allocation_matrix, motor_force_min, motor_force_max = allocation_parameters(sim)
    return EDMDcMPC_QP(
        A=A_mpc, B=B_mpc, Cz=Cz,
        N=N_MPC, NC=NC_MPC,
        Q=np.diag(Q_DIAG), R=np.diag(R_DIAG), Rd=np.diag(RD_DIAG),
        u_scaler=model["u_scaler"],
        du_min=du_min, du_max=du_max,
        u_nominal_raw=hover,
        state_scaler=model["scaler"],
        input_lift_type=model.get("input_lift_type"),
        raw_input_dim=int(model.get("raw_input_dim", 4)),
        Q_terminal=np.diag(Q_TERMINAL_SCALE * Q_DIAG),
        allocation_matrix=allocation_matrix,
        motor_force_min=motor_force_min,
        motor_force_max=motor_force_max,
    )


def build_linear_mpc(A_lin, B_lin, c_lin, x_scaler, u_scaler, sim, legacy_pid_nominal=False):
    hover = np.array([sim.q_mass * sim.g, 0.0, 0.0, 0.0], dtype=float)
    A_aug = np.eye(STATE_DIM + 1)
    A_aug[:STATE_DIM, :STATE_DIM] = A_lin
    A_aug[:STATE_DIM, STATE_DIM] = c_lin
    A_aug[STATE_DIM, :STATE_DIM] = 0.0
    B_aug = np.zeros((STATE_DIM + 1, 4))
    B_aug[:STATE_DIM, :] = B_lin
    Cz = np.zeros((STATE_DIM, STATE_DIM + 1))
    Cz[:, :STATE_DIM] = np.eye(STATE_DIM)
    if legacy_pid_nominal:
        du_min, du_max = -LEGACY_DU_SCALED_MAX, LEGACY_DU_SCALED_MAX
    else:
        du_min, du_max = scaled_delta_bounds(u_scaler, DU_RAW_MAX)
    allocation_matrix, motor_force_min, motor_force_max = allocation_parameters(sim)
    return EDMDcMPC_QP(
        A=A_aug, B=B_aug, Cz=Cz,
        N=N_MPC, NC=NC_MPC,
        Q=np.diag(Q_DIAG), R=np.diag(R_DIAG), Rd=np.diag(RD_DIAG),
        u_scaler=u_scaler,
        du_min=du_min, du_max=du_max,
        u_nominal_raw=hover,
        Q_terminal=np.diag(Q_TERMINAL_SCALE * Q_DIAG),
        allocation_matrix=allocation_matrix,
        motor_force_min=motor_force_min,
        motor_force_max=motor_force_max,
    )


def run_pid_baseline(sim, ref_traj, steps):
    state = np.zeros(STATE_DIM)
    state[8] = float(ref_traj[0].get("yaw", 0.0))
    X = np.zeros((steps, STATE_DIM))
    U = np.zeros((steps, 4))
    sim.controller_PX4.fct_reset()
    X[0] = state
    for k in range(steps - 1):
        omega_cmd, u = sim.controller_PX4.fct_step(state, ref_traj[k], sim.dt)

        def ode(_, s_local):
            return sim.quad.fct_dynamics(0.0, s_local, omega_cmd)

        sol = solve_ivp(ode, [0.0, sim.dt], state, method="RK45")
        state = sol.y[:, -1]
        X[k + 1] = state
        U[k] = u
    U[-1] = U[-2]
    sim.controller_PX4.fct_reset()
    return X, U


def run_mpc_closed_loop(
    mpc, sim, ref_traj, state_scaler, nominal_controller, steps, label,
    lifted, use_pid_nominal=False, z_builder=None, correction_blend=1.0
):
    ref_std = precompute_ref_std(ref_traj[:steps], state_scaler, dt=sim.dt)
    state = np.zeros(STATE_DIM)
    state[8] = float(ref_traj[0].get("yaw", 0.0))
    X = np.zeros((steps, STATE_DIM))
    U = np.zeros((steps, 4))
    solve_times = []
    X[0] = state
    nominal_controller.fct_reset()

    for k in range(steps - 1):
        if z_builder is not None:
            z = z_builder(state, state_scaler)
        elif lifted:
            z = lifted_state_from_x(state, state_scaler)
        else:
            z = state_scaler.transform(state.reshape(1, -1)).flatten()
        ref_h = build_ref_horizon(ref_std, k, N_MPC)
        u_nom = controller_nominal_wrench(
            sim, state, ref_traj[k], nominal_controller, use_pid_nominal
        )

        t0 = time.perf_counter()
        u_cmd = mpc.compute(z, ref_h, u_nominal_raw=u_nom)
        solve_times.append(time.perf_counter() - t0)
        u_cmd = u_nom + float(correction_blend) * (u_cmd - u_nom)

        state, u_applied = plant_step(sim, state, u_cmd, sim.dt)
        X[k + 1] = state
        U[k] = u_applied

        if (k + 1) % 500 == 0:
            print(f"  {label}: step {k + 1}/{steps - 1}")

    U[-1] = U[-2]
    nominal_controller.fct_reset()
    return X, U, solve_times


def affine_linear_state(state, state_scaler):
    x_std = state_scaler.transform(state.reshape(1, -1)).flatten()
    return np.concatenate([x_std, [1.0]])


def set_axes_readable(ax, xyz):
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    pad = 0.08 * spans + 0.8
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])


def main():
    parser = argparse.ArgumentParser(description="Run closed-loop EDMD/linear MPC comparisons.")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of 0.01 s steps per trajectory. Defaults to 0 for full length.",
    )
    parser.add_argument(
        "--legacy-pid-nominal",
        action="store_true",
        help="Use the previous PID-centered, conservative MPC settings.",
    )
    parser.add_argument(
        "--feedforward-nominal",
        action="store_true",
        help="Use reference acceleration feedforward instead of the PID/PX4 nominal command.",
    )
    parser.add_argument(
        "--test-indices",
        default=os.environ.get("EDMDC_TEST_INDICES", ",".join(map(str, DEFAULT_TEST_INDICES))),
        help="Comma-separated held-out dataset indices to evaluate.",
    )
    args = parser.parse_args()

    global N_MPC, NC_MPC, R_DIAG, RD_DIAG, Q_TERMINAL_SCALE
    global EDMD_CORRECTION_BLEND, LINEAR_CORRECTION_BLEND, USE_PID_NOMINAL
    USE_PID_NOMINAL = not bool(args.feedforward_nominal)
    if args.legacy_pid_nominal:
        USE_PID_NOMINAL = True
        N_MPC = LEGACY_N_MPC
        NC_MPC = LEGACY_NC_MPC
        R_DIAG = LEGACY_R_DIAG.copy()
        RD_DIAG = LEGACY_RD_DIAG.copy()
        Q_TERMINAL_SCALE = LEGACY_Q_TERMINAL_SCALE
        EDMD_CORRECTION_BLEND = LEGACY_EDMD_CORRECTION_BLEND
        LINEAR_CORRECTION_BLEND = LEGACY_LINEAR_CORRECTION_BLEND

    if not MODEL_FILE.exists() or not DATA_FILE.exists():
        raise FileNotFoundError(
            "Run python parallel_sim.py, python mix_traj.py, and python EDMDc_training.py first."
        )

    model = load_edmdc_model(MODEL_FILE)
    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(DATA_FILE)
    n_runs = states_all.shape[0]
    test_cases = parse_test_cases(args.test_indices, n_runs)
    held_out = {idx for idx, _ in test_cases}
    model_train_indices = [
        int(idx) for idx in model.get("train_indices", [])
        if 0 <= int(idx) < n_runs
    ]
    if model_train_indices:
        if held_out & set(model_train_indices):
            raise ValueError("The EDMDc model was trained on a requested test run.")
        train_indices = model_train_indices
    else:
        train_indices = [i for i in range(n_runs) if i not in held_out]
    if not train_indices:
        raise ValueError("No training runs are available for the linear baseline.")
    validation_indices = [
        int(idx) for idx in model.get("validation_indices", [])
        if 0 <= int(idx) < n_runs and int(idx) not in held_out
    ]
    if not validation_indices:
        validation_indices = train_indices[:1]
    A_lin, B_lin, c_lin, x_lin_scaler, u_lin_scaler = fit_linear_baseline(
        t_all, states_all, U_all, train_indices, validation_indices
    )
    mode = "legacy PID nominal" if args.legacy_pid_nominal else (
        "PID nominal with physical delta limits" if USE_PID_NOMINAL
        else "reference feedforward nominal"
    )
    print(f"MPC mode: {mode}")
    print(f"MPC horizons: N={N_MPC}, NC={NC_MPC}")
    print(
        "MPC correction blends: "
        f"EDMD={EDMD_CORRECTION_BLEND:g}, Linear={LINEAR_CORRECTION_BLEND:g}"
    )

    fig = plt.figure(figsize=(6.5 * len(test_cases), 6))
    fig_err, err_axes = plt.subplots(len(test_cases), 1, figsize=(12, 3.2 * len(test_cases)))
    err_axes = np.atleast_1d(err_axes)
    rows = []

    for plot_idx, (run_idx, name) in enumerate(test_cases, start=1):
        ref_traj = ref_traj_list[run_idx]
        steps = len(ref_traj) if args.steps == 0 else min(args.steps, len(ref_traj))
        ref_xyz = extract_ref_xyz(ref_traj)[:steps]
        t = t_all[run_idx, :steps]

        print(f"\nRunning {name}: steps={steps}")
        sim_pid = quad_sim()
        X_pid, U_pid = run_pid_baseline(sim_pid, ref_traj, steps)

        sim_edmd = quad_sim()
        mpc_edmd = build_edmd_mpc(model, sim_edmd, legacy_pid_nominal=args.legacy_pid_nominal)
        X_edmd, U_edmd, st_edmd = run_mpc_closed_loop(
            mpc_edmd, sim_edmd, ref_traj, model["scaler"],
            quad_sim().controller_PX4, steps, "EDMD-MPC", lifted=True,
            use_pid_nominal=USE_PID_NOMINAL,
            correction_blend=EDMD_CORRECTION_BLEND
        )

        sim_lin = quad_sim()
        mpc_lin = build_linear_mpc(
            A_lin, B_lin, c_lin, x_lin_scaler, u_lin_scaler, sim_lin,
            legacy_pid_nominal=args.legacy_pid_nominal
        )
        X_lin, U_lin, st_lin = run_mpc_closed_loop(
            mpc_lin, sim_lin, ref_traj, x_lin_scaler,
            quad_sim().controller_PX4, steps, "Linear-MPC", lifted=False,
            use_pid_nominal=USE_PID_NOMINAL,
            z_builder=affine_linear_state,
            correction_blend=LINEAR_CORRECTION_BLEND
        )

        pid_e = rmse(X_pid[:, 0:3], ref_xyz)
        edmd_e = rmse(X_edmd[:, 0:3], ref_xyz)
        lin_e = rmse(X_lin[:, 0:3], ref_xyz)
        pid_yaw, pid_r = yaw_tracking_metrics(X_pid, ref_traj[:steps], sim_pid.dt)
        edmd_yaw, edmd_r = yaw_tracking_metrics(X_edmd, ref_traj[:steps], sim_edmd.dt)
        lin_yaw, lin_r = yaw_tracking_metrics(X_lin, ref_traj[:steps], sim_lin.dt)
        edmd_timing = timing_summary(st_edmd)
        lin_timing = timing_summary(st_lin)
        rows.append((
            name, pid_e, edmd_e, lin_e,
            pid_yaw, edmd_yaw, lin_yaw,
            pid_r, edmd_r, lin_r,
            edmd_timing, lin_timing,
        ))

        ax = fig.add_subplot(1, len(test_cases), plot_idx, projection="3d")
        ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], "k", lw=2.0, label="Reference")
        ax.plot(X_pid[:, 0], X_pid[:, 1], X_pid[:, 2], color="#777777", lw=1.4, label=f"PID/PX4 ({pid_e:.3f} m)")
        ax.plot(X_edmd[:, 0], X_edmd[:, 1], X_edmd[:, 2], color="#2ca02c", lw=1.3, ls="--", label=f"EDMD-MPC ({edmd_e:.3f} m)")
        ax.plot(X_lin[:, 0], X_lin[:, 1], X_lin[:, 2], color="#1f77b4", lw=1.1, ls=":", label=f"Linear-MPC ({lin_e:.3f} m)")
        ax.set_title(name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        set_axes_readable(ax, np.vstack([ref_xyz, X_pid[:, 0:3]]))
        ax.legend(fontsize=8)

        err_ax = err_axes[plot_idx - 1]
        err_ax.plot(t, np.linalg.norm(X_pid[:, 0:3] - ref_xyz, axis=1), color="#777777", label="PID/PX4")
        err_ax.plot(t, np.linalg.norm(X_edmd[:, 0:3] - ref_xyz, axis=1), color="#2ca02c", ls="--", label="EDMD-MPC")
        err_ax.plot(t, np.linalg.norm(X_lin[:, 0:3] - ref_xyz, axis=1), color="#1f77b4", ls=":", label="Linear-MPC")
        err_ax.set_title(f"{name}: position error")
        err_ax.set_ylabel("error [m]")
        err_ax.grid(True, alpha=0.3)
        err_ax.legend()

    err_axes[-1].set_xlabel("time [s]")
    fig.suptitle("Closed-Loop MPC Comparison")
    fig.tight_layout()
    fig_err.tight_layout()

    print("\nClosed-loop MPC comparison")
    print(
        f"{'case':<10s} {'PID-pos':>9s} {'EDMD-pos':>9s} {'LIN-pos':>9s} "
        f"{'PID-yaw':>9s} {'EDMD-yaw':>9s} {'LIN-yaw':>9s}"
    )
    for row in rows:
        print(
            f"{row[0]:<10s} {row[1]:9.4f} {row[2]:9.4f} {row[3]:9.4f} "
            f"{row[4]:9.4f} {row[5]:9.4f} {row[6]:9.4f}"
        )
        print(
            f"{'':<10s} yaw-rate RMSE PID/EDMD/linear = "
            f"{row[7]:.4f}/{row[8]:.4f}/{row[9]:.4f} rad/s; "
            f"solve mean EDMD/linear = {row[10][0]:.2f}/{row[11][0]:.2f} ms"
        )
        print(
            f"{'':<10s} EDMDc p95/max = {row[10][1]:.2f}/{row[10][2]:.2f} ms; "
            f"linear p95/max = {row[11][1]:.2f}/{row[11][2]:.2f} ms"
        )

    plt.show()


if __name__ == "__main__":
    main()
