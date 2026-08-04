"""Interception comparison for the current yaw-wrench simulator."""

import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_mpc import (
    N_MPC, build_edmd_mpc, build_linear_mpc, plant_step,
    set_axes_readable,
)
from compare_three import fit_linear_baseline
from edmdc_mpc import (
    build_ref_horizon,
    extract_ref_xyz,
    lifted_state_from_x,
    load_edmdc_model,
    load_simulation_runs,
    precompute_ref_std,
)
from Simulation import quad_sim


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.environ.get("EDMDC_DATA_FILE", SCRIPT_DIR / "runs_mixed_n150.pkl"))
MODEL_FILE = Path(os.environ.get("EDMDC_MODEL_FILE", SCRIPT_DIR / "edmdc_model_yaw_wrench.pkl"))
CAPTURE_RADIUS = 0.75
T_MAX = 20.0
LEAD_TIME = 0.8


class StraightLineTarget:
    def __init__(self, p0, velocity):
        self.p0 = np.asarray(p0, dtype=float)
        self.v = np.asarray(velocity, dtype=float)

    def position(self, t):
        return self.p0 + self.v * t

    def velocity(self, t):
        return self.v.copy()


class HelixTarget:
    def __init__(self, center, radius, z0, climb_rate, speed):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.z0 = float(z0)
        self.climb_rate = float(climb_rate)
        self.omega = float(speed) / self.radius

    def position(self, t):
        return np.array([
            self.center[0] + self.radius * np.cos(self.omega * t),
            self.center[1] + self.radius * np.sin(self.omega * t),
            self.z0 + self.climb_rate * t,
        ])

    def velocity(self, t):
        return np.array([
            -self.radius * self.omega * np.sin(self.omega * t),
            self.radius * self.omega * np.cos(self.omega * t),
            self.climb_rate,
        ])


class WeavingTarget:
    def __init__(self, p0, vx, ay, fy, az, fz):
        self.p0 = np.asarray(p0, dtype=float)
        self.vx = float(vx)
        self.ay = float(ay)
        self.fy = float(fy)
        self.az = float(az)
        self.fz = float(fz)

    def position(self, t):
        return np.array([
            self.p0[0] + self.vx * t,
            self.p0[1] + self.ay * np.sin(2.0 * np.pi * self.fy * t),
            self.p0[2] + self.az * np.sin(2.0 * np.pi * self.fz * t),
        ])

    def velocity(self, t):
        return np.array([
            self.vx,
            self.ay * 2.0 * np.pi * self.fy * np.cos(2.0 * np.pi * self.fy * t),
            self.az * 2.0 * np.pi * self.fz * np.cos(2.0 * np.pi * self.fz * t),
        ])


def target_ref(target, t_now, dt, horizon, lead_time=0.0):
    ref = []
    for i in range(horizon):
        tf = t_now + lead_time + i * dt
        pos = target.position(tf)
        vel = target.velocity(tf)
        yaw = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel[:2]) > 1e-9 else 0.0
        ref.append({"pos": pos, "vel": vel, "acc": np.zeros(3), "yaw": float(yaw), "yaw_rate": 0.0})
    return ref


def scenarios():
    return [
        ("straight", StraightLineTarget([12.0, -6.0, 5.0], [-1.0, 0.7, 0.15])),
        ("helix", HelixTarget([8.0, 0.0], 5.0, 4.0, 0.15, 2.0)),
        ("weaving", WeavingTarget([10.0, 0.0, 6.0], -0.75, 4.0, 0.08, 2.0, 0.10)),
    ]


def run_pid(sim, target, steps):
    state = np.zeros(12)
    X = np.zeros((steps, 12))
    U = np.zeros((steps, 4))
    X[0] = state
    sim.controller_PX4.fct_reset()
    capture = None
    for k in range(steps - 1):
        ref = target_ref(target, k * sim.dt, sim.dt, 1, lead_time=LEAD_TIME)[0]
        omega, u = sim.controller_PX4.fct_step(state, ref, sim.dt)
        state, u_applied = plant_step(sim, state, u, sim.dt)
        X[k + 1] = state
        U[k] = u_applied
        if np.linalg.norm(state[:3] - target.position((k + 1) * sim.dt)) <= CAPTURE_RADIUS:
            capture = (k + 1) * sim.dt
            X = X[:k + 2]
            U = U[:k + 2]
            break
    sim.controller_PX4.fct_reset()
    return X, U, capture


def run_mpc(mpc, sim, target, scaler, steps, lifted, label):
    state = np.zeros(12)
    X = np.zeros((steps, 12))
    U = np.zeros((steps, 4))
    X[0] = state
    nominal = quad_sim().controller_PX4
    nominal.fct_reset()
    capture = None
    solve_times = []
    for k in range(steps - 1):
        ref = target_ref(target, k * sim.dt, sim.dt, N_MPC, lead_time=LEAD_TIME)
        ref_std = precompute_ref_std(ref, scaler, dt=sim.dt)
        if lifted:
            z = lifted_state_from_x(state, scaler)
        else:
            # The linear baseline augments its standardized 12-state model
            # with a constant state to represent the affine offset.
            z = np.concatenate([
                scaler.transform(state.reshape(1, -1)).flatten(),
                [1.0],
            ])
        _, u_nom = nominal.fct_step(state, ref[0], sim.dt)
        t0 = time.perf_counter()
        u_cmd = mpc.compute(z, build_ref_horizon(ref_std, 0, N_MPC), u_nominal_raw=u_nom)
        solve_times.append(time.perf_counter() - t0)
        state, u_applied = plant_step(sim, state, u_cmd, sim.dt)
        X[k + 1] = state
        U[k] = u_applied
        if (k + 1) % 500 == 0:
            print(f"  {label}: step {k + 1}/{steps - 1}")
        if np.linalg.norm(state[:3] - target.position((k + 1) * sim.dt)) <= CAPTURE_RADIUS:
            capture = (k + 1) * sim.dt
            X = X[:k + 2]
            U = U[:k + 2]
            break
    nominal.fct_reset()
    return X, U, capture, solve_times


def separation(X, target, dt):
    return np.array([np.linalg.norm(x[:3] - target.position(k * dt)) for k, x in enumerate(X)])


def main():
    parser = argparse.ArgumentParser(description="Run interception graphs with current yaw-wrench sim.")
    parser.add_argument("--tmax", type=float, default=T_MAX)
    parser.add_argument(
        "--cases", default=None,
        help="Optional comma-separated scenario names to run (for example: straight,helix).",
    )
    args = parser.parse_args()
    if args.tmax <= 0:
        parser.error("--tmax must be positive")

    model = load_edmdc_model(MODEL_FILE)
    t_all, states_all, U_all, refs = load_simulation_runs(DATA_FILE)
    train_indices = [
        int(idx) for idx in model.get("train_indices", [])
        if 0 <= int(idx) < states_all.shape[0]
    ]
    if not train_indices:
        train_indices = list(range(states_all.shape[0]))
    validation_indices = [
        int(idx) for idx in model.get("validation_indices", [])
        if 0 <= int(idx) < states_all.shape[0] and int(idx) not in train_indices
    ]
    if not validation_indices:
        validation_indices = train_indices[:1]
    A_lin, B_lin, c_lin, x_lin_scaler, u_lin_scaler = fit_linear_baseline(
        t_all, states_all, U_all, train_indices, validation_indices
    )

    rows = []
    cases = scenarios()
    if args.cases is not None:
        requested = {name.strip() for name in args.cases.split(",") if name.strip()}
        cases = [(name, target) for name, target in cases if name in requested]
        unknown = requested - {name for name, _ in scenarios()}
        if unknown:
            parser.error(f"Unknown scenario name(s): {', '.join(sorted(unknown))}")
    if not cases:
        parser.error("No interception scenarios selected")
    fig3d = plt.figure(figsize=(6.4 * len(cases), 6))
    figsep, sep_axes = plt.subplots(len(cases), 1, figsize=(12, 3.2 * len(cases)), sharex=False)
    sep_axes = np.atleast_1d(sep_axes)

    for plot_i, (name, target) in enumerate(cases, start=1):
        steps = int(args.tmax / quad_sim().dt)
        print(f"\nRunning intercept case: {name}")

        X_pid, U_pid, cap_pid = run_pid(quad_sim(), target, steps)
        sim_edmd = quad_sim()
        X_edmd, U_edmd, cap_edmd, st_edmd = run_mpc(
            build_edmd_mpc(model, sim_edmd), sim_edmd, target, model["scaler"], steps, True, "EDMD-MPC"
        )
        sim_lin = quad_sim()
        X_lin, U_lin, cap_lin, st_lin = run_mpc(
            build_linear_mpc(A_lin, B_lin, c_lin, x_lin_scaler, u_lin_scaler, sim_lin),
            sim_lin, target, x_lin_scaler, steps, False, "Linear-MPC"
        )

        sep_pid = separation(X_pid, target, quad_sim().dt)
        sep_edmd = separation(X_edmd, target, quad_sim().dt)
        sep_lin = separation(X_lin, target, quad_sim().dt)
        rows.append((name, cap_pid, cap_edmd, cap_lin, sep_pid.min(), sep_edmd.min(), sep_lin.min()))

        Tplot = max(len(X_pid), len(X_edmd), len(X_lin))
        target_xyz = np.array([target.position(k * quad_sim().dt) for k in range(Tplot)])
        ax = fig3d.add_subplot(1, len(cases), plot_i, projection="3d")
        ax.plot(target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2], "r", lw=2.0, label="Target")
        ax.plot(X_pid[:, 0], X_pid[:, 1], X_pid[:, 2], color="#777777", label=f"PID min {sep_pid.min():.2f} m")
        ax.plot(X_edmd[:, 0], X_edmd[:, 1], X_edmd[:, 2], "--", color="#2ca02c", label=f"EDMD-MPC min {sep_edmd.min():.2f} m")
        ax.plot(X_lin[:, 0], X_lin[:, 1], X_lin[:, 2], ":", color="#1f77b4", label=f"Linear-MPC min {sep_lin.min():.2f} m")
        ax.set_title(name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        set_axes_readable(ax, np.vstack([target_xyz, X_pid[:, :3], X_edmd[:, :3], X_lin[:, :3]]))
        ax.legend(fontsize=8)

        sep_ax = sep_axes[plot_i - 1]
        sep_ax.plot(np.arange(len(sep_pid)) * quad_sim().dt, sep_pid, color="#777777", label="PID")
        sep_ax.plot(np.arange(len(sep_edmd)) * quad_sim().dt, sep_edmd, "--", color="#2ca02c", label="EDMD-MPC")
        sep_ax.plot(np.arange(len(sep_lin)) * quad_sim().dt, sep_lin, ":", color="#1f77b4", label="Linear-MPC")
        sep_ax.axhline(CAPTURE_RADIUS, color="r", ls="--", lw=1.0, label="capture")
        sep_ax.set_title(f"{name}: separation")
        sep_ax.set_ylabel("distance [m]")
        sep_ax.grid(True, alpha=0.3)
        sep_ax.legend()

    sep_axes[-1].set_xlabel("time [s]")
    fig3d.suptitle("Interception Trajectories")
    figsep.suptitle("Interception Separation")
    fig3d.tight_layout()
    figsep.tight_layout()

    print("\nInterception summary")
    print(f"{'case':<10s} {'PID cap':>9s} {'EDMD cap':>9s} {'LIN cap':>9s} {'PID min':>9s} {'EDMD min':>9s} {'LIN min':>9s}")
    for name, cp, ce, cl, sp, se, sl in rows:
        fmt = lambda x: f"{x:.2f}s" if x is not None else "FAIL"
        print(f"{name:<10s} {fmt(cp):>9s} {fmt(ce):>9s} {fmt(cl):>9s} {sp:9.3f} {se:9.3f} {sl:9.3f}")

    plt.show()


if __name__ == "__main__":
    main()
