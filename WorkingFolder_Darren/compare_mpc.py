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
    rmse,
)
from PID_Mixer import pid_mixer
from Simulation import quad_sim


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "runs_mixed_n150.pkl"
MODEL_FILE = SCRIPT_DIR / "edmdc_model_yaw_wrench.pkl"
TEST_CASES = [(39, "helix"), (59, "figure-8"), (129, "lissajous")]

N_MPC = 25
NC_MPC = 10

Q_DIAG = np.array([
    45.0, 45.0, 55.0,
    4.0, 4.0, 5.0,
    0.2, 0.2, 4.0,
    0.05, 0.05, 1.0,
], dtype=float)
R_DIAG = np.array([0.03, 1.2, 1.2, 0.7], dtype=float)
RD_DIAG = np.array([0.01, 0.35, 0.35, 0.25], dtype=float)

DU_MIN = np.array([-2.0, -0.18, -0.18, -0.18], dtype=float)
DU_MAX = np.array([2.0, 0.18, 0.18, 0.18], dtype=float)


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


def clamp_wrench(sim, u):
    u = np.asarray(u, dtype=float).copy()
    u[0] = np.clip(u[0], 0.1 * sim.q_mass * sim.g, sim.q_mass * (sim.g + 10.0))
    u[1] = np.clip(u[1], -sim.controller_PX4.torque_max[0], sim.controller_PX4.torque_max[0])
    u[2] = np.clip(u[2], -sim.controller_PX4.torque_max[1], sim.controller_PX4.torque_max[1])
    u[3] = np.clip(u[3], -sim.controller_PX4.torque_max[2], sim.controller_PX4.torque_max[2])
    return u


def plant_step(sim, state, u, dt):
    u = clamp_wrench(sim, u)
    omega_cmd = pid_mixer.fct_mixer(
        u, sim.quad.kT, sim.quad.kD, sim.quad.l,
        min_omega=0.0, max_omega=sim.max_speed,
    )

    def ode(_, s_local):
        return sim.quad.fct_dynamics(0.0, s_local, omega_cmd)

    sol = solve_ivp(ode, [0.0, dt], state, method="RK45")
    return sol.y[:, -1], u


def build_edmd_mpc(model, sim):
    n_obs = model["n_obs"]
    Cz = np.zeros((STATE_DIM, n_obs))
    Cz[:STATE_DIM, :STATE_DIM] = np.eye(STATE_DIM)
    hover = np.array([sim.q_mass * sim.g, 0.0, 0.0, 0.0], dtype=float)
    return EDMDcMPC_QP(
        A=model["A"], B=model["B"], Cz=Cz,
        N=N_MPC, NC=NC_MPC,
        Q=np.diag(Q_DIAG), R=np.diag(R_DIAG), Rd=np.diag(RD_DIAG),
        u_scaler=model["u_scaler"],
        du_min=DU_MIN, du_max=DU_MAX,
        u_nominal_raw=hover,
        state_scaler=model["scaler"],
        input_lift_type=model.get("input_lift_type"),
        raw_input_dim=int(model.get("raw_input_dim", 4)),
    )


def build_linear_mpc(A_lin, B_lin, x_scaler, u_scaler, sim):
    hover = np.array([sim.q_mass * sim.g, 0.0, 0.0, 0.0], dtype=float)
    return EDMDcMPC_QP(
        A=A_lin, B=B_lin, Cz=np.eye(STATE_DIM),
        N=N_MPC, NC=NC_MPC,
        Q=np.diag(Q_DIAG), R=np.diag(R_DIAG), Rd=np.diag(RD_DIAG),
        u_scaler=u_scaler,
        du_min=DU_MIN, du_max=DU_MAX,
        u_nominal_raw=hover,
    )


def run_pid_baseline(sim, ref_traj, steps):
    state = np.zeros(STATE_DIM)
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


def run_mpc_closed_loop(mpc, sim, ref_traj, state_scaler, nominal_controller, steps, label, lifted):
    ref_std = precompute_ref_std(ref_traj[:steps], state_scaler, dt=sim.dt)
    state = np.zeros(STATE_DIM)
    X = np.zeros((steps, STATE_DIM))
    U = np.zeros((steps, 4))
    solve_times = []
    X[0] = state
    nominal_controller.fct_reset()

    for k in range(steps - 1):
        if lifted:
            z = lifted_state_from_x(state, state_scaler)
        else:
            z = state_scaler.transform(state.reshape(1, -1)).flatten()
        ref_h = build_ref_horizon(ref_std, k, N_MPC)
        _, u_nom = nominal_controller.fct_step(state, ref_traj[k], sim.dt)
        u_nom = clamp_wrench(sim, u_nom)

        t0 = time.perf_counter()
        u_cmd = mpc.compute(z, ref_h, u_nominal_raw=u_nom)
        solve_times.append(time.perf_counter() - t0)

        state, u_applied = plant_step(sim, state, u_cmd, sim.dt)
        X[k + 1] = state
        U[k] = u_applied

        if (k + 1) % 500 == 0:
            print(f"  {label}: step {k + 1}/{steps - 1}")

    U[-1] = U[-2]
    nominal_controller.fct_reset()
    return X, U, solve_times


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
    args = parser.parse_args()

    if not MODEL_FILE.exists() or not DATA_FILE.exists():
        raise FileNotFoundError(
            "Run python parallel_sim.py, python mix_traj.py, and python EDMDc_training.py first."
        )

    model = load_edmdc_model(MODEL_FILE)
    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(DATA_FILE)
    n_runs = states_all.shape[0]
    held_out = {idx for idx, _ in TEST_CASES if idx < n_runs}
    train_indices = [i for i in range(n_runs) if i not in held_out]
    A_lin, B_lin, _, x_lin_scaler, u_lin_scaler = fit_linear_baseline(
        t_all, states_all, U_all, train_indices
    )

    fig = plt.figure(figsize=(6.5 * len(TEST_CASES), 6))
    fig_err, err_axes = plt.subplots(len(TEST_CASES), 1, figsize=(12, 3.2 * len(TEST_CASES)))
    err_axes = np.atleast_1d(err_axes)
    rows = []

    for plot_idx, (run_idx, name) in enumerate(TEST_CASES, start=1):
        ref_traj = ref_traj_list[run_idx]
        steps = len(ref_traj) if args.steps == 0 else min(args.steps, len(ref_traj))
        ref_xyz = extract_ref_xyz(ref_traj)[:steps]
        t = t_all[run_idx, :steps]

        print(f"\nRunning {name}: steps={steps}")
        sim_pid = quad_sim()
        X_pid, U_pid = run_pid_baseline(sim_pid, ref_traj, steps)

        sim_edmd = quad_sim()
        mpc_edmd = build_edmd_mpc(model, sim_edmd)
        X_edmd, U_edmd, st_edmd = run_mpc_closed_loop(
            mpc_edmd, sim_edmd, ref_traj, model["scaler"],
            quad_sim().controller_PX4, steps, "EDMD-MPC", lifted=True
        )

        sim_lin = quad_sim()
        mpc_lin = build_linear_mpc(A_lin, B_lin, x_lin_scaler, u_lin_scaler, sim_lin)
        X_lin, U_lin, st_lin = run_mpc_closed_loop(
            mpc_lin, sim_lin, ref_traj, x_lin_scaler,
            quad_sim().controller_PX4, steps, "Linear-MPC", lifted=False
        )

        pid_e = rmse(X_pid[:, 0:3], ref_xyz)
        edmd_e = rmse(X_edmd[:, 0:3], ref_xyz)
        lin_e = rmse(X_lin[:, 0:3], ref_xyz)
        edmd_ms = 1000.0 * float(np.mean(st_edmd))
        lin_ms = 1000.0 * float(np.mean(st_lin))
        rows.append((name, pid_e, edmd_e, lin_e, edmd_ms, lin_ms))

        ax = fig.add_subplot(1, len(TEST_CASES), plot_idx, projection="3d")
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
    print(f"{'case':<10s} {'PID':>9s} {'EDMD-MPC':>9s} {'LIN-MPC':>9s} {'EDMD ms':>9s} {'LIN ms':>9s}")
    for row in rows:
        print(f"{row[0]:<10s} {row[1]:9.4f} {row[2]:9.4f} {row[3]:9.4f} {row[4]:9.2f} {row[5]:9.2f}")

    plt.show()


if __name__ == "__main__":
    main()
