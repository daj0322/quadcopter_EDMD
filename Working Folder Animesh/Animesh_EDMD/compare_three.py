"""
Compare trajectory tracking performance of three outer-loop controllers:
PID, linear MPC, and EDMDc MPC.

All controllers command the same plant through the attitude-level input
[u1, phi_des, theta_des] and are evaluated against the same reference
trajectories.
"""

import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from Simulation import quad_sim
from edmdc_mpc import (
    EDMDcMPC_QP,
    load_edmdc_model,
    load_simulation_runs,
    lifted_state_from_x,
    drop_to_12state,
    precompute_ref_std,
    build_ref_horizon,
    extract_ref_xyz,
    reference_yaw_arrays,
    wrap_angle_pi,
    rmse,
)

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR       = Path(__file__).resolve().parent
EDMDC_MODEL_FILE = "edmdc_model_300.pkl"
DATA_FILE        = "runs_mixed_n300.pkl"
# Use the full downsampled 100 s trajectory. A 300-step cap only shows the
# first 30 s, so the figure-8 reference never completes its full loop.
MAX_STEPS        = None

# Test indices — one per trajectory family
TEST_CASES = [
    (39,  "helix (small)"),
    (59,  "figure-8"),
    (129, "lissajous"),
]

# MPC config for the 0.01 s EDMD model produced by EDMDc_training.py.
N_MPC   = 20
NC_MPC  = 15

Q_DIAG = np.array([
    300000.0, 480000.0, 300000.0,
        25.0,     40.0,     25.0,
         0.0,      0.0,      0.0,
         0.0,      0.0,      0.0,
], dtype=float)

R_DIAG  = np.array([0.0001, 0.75, 0.75], dtype=float)
RD_DIAG = np.array([1e-05, 0.075, 0.075], dtype=float)
R_YAW   = 0.25
RD_YAW  = 0.025

USE_PID_NOMINAL = True
# Fixed-yaw experiment: yaw remains controlled by the inner yaw PID, but the
# outer EDMDc QP does not optimize a separate yaw-command correction. This
# mirrors the paper's fixed-yaw 10-state objective while keeping the 12-state
# plant and logged yaw/r states.
EDMDC_YAW_CORRECTION = False
USE_CONSTANT_YAW_REFERENCE = True
CONSTANT_YAW = 0.0

# With USE_PID_NOMINAL, these are corrections around the cascaded PID command
# in standardized input units. Keeping them small holds EDMD-MPC near the
# behavior distribution it was trained on.
DU_MIN = np.array([-0.1, -0.01, -0.01], dtype=float)
DU_MAX = np.array([ 0.1,  0.01,  0.01], dtype=float)
DU_YAW = 0.0


def apply_yaw_reference_mode(ref_traj):
    if not USE_CONSTANT_YAW_REFERENCE:
        return ref_traj
    return [
        {**wp, "yaw": float(CONSTANT_YAW), "yaw_rate": 0.0}
        for wp in ref_traj
    ]


# Linear hover model
def build_linear_hover_model(sim, dt):
    """
    Build a discrete-time linear hover model for the attitude-commanded plant.

    The model uses the full 12-state representation
    [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r] with control input
    [thrust, phi_des, theta_des]. The inner attitude loop is approximated
    as PD control about hover, and the continuous model is discretized
    with a matrix exponential.
    """
    m = sim.q_mass
    g = sim.g
    Ixx = sim.Ixx
    Iyy = sim.Iyy
    Izz = sim.Izz
    k_drag_angular = sim.k_drag_angular

    # Inner-loop PD gains used for hover linearization.
    Kp_phi = sim.kp_ang[0]
    Kd_phi = sim.kd_ang[0]
    Kp_theta = sim.kp_ang[1]
    Kd_theta = sim.kd_ang[1]

    nx, nu = 12, 3

    # Continuous-time state matrix.
    Ac = np.zeros((nx, nx))
    Ac[0, 3] = 1.0
    Ac[1, 4] = 1.0
    Ac[2, 5] = 1.0
    Ac[3, 7] = g
    Ac[4, 6] = -g
    Ac[6, 9] = 1.0
    Ac[7, 10] = 1.0
    Ac[8, 11] = 1.0
    Ac[9, 6] = -Kp_phi / Ixx
    Ac[9, 9] = -Kd_phi / Ixx
    Ac[10, 7] = -Kp_theta / Iyy
    Ac[10, 10] = -Kd_theta / Iyy
    Ac[11, 11] = -k_drag_angular / Izz

    # Continuous-time input matrix.
    Bc = np.zeros((nx, nu))
    Bc[5, 0] = 1.0 / m
    Bc[9, 1] = Kp_phi / Ixx
    Bc[10, 2] = Kp_theta / Iyy

    # Discretize with a matrix exponential.
    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = Ac * dt
    M[:nx, nx:] = Bc * dt
    eM = la.expm(M)
    Ad = eM[:nx, :nx]
    Bd = eM[:nx, nx:]

    print("\n========== LINEAR MODEL DEBUG ==========")
    print(f"Continuous A eigenvalues (real): {np.sort(np.real(la.eigvals(Ac)))}")
    print(f"Discrete A eigenvalues (abs):   {np.sort(np.abs(la.eigvals(Ad)))}")
    print(f"Max abs eigenvalue of Ad: {np.max(np.abs(la.eigvals(Ad))):.6f}")
    print(f"Ad shape: {Ad.shape}, Bd shape: {Bd.shape}")
    print("=========================================")

    return Ad, Bd


# Closed-loop rollout
def run_mpc_closedloop(mpc, sim, X_init, ref_traj, scaler, dt, horizon, n_steps):
    """Run closed-loop tracking with the EDMDc-based MPC controller."""
    ref_std = precompute_ref_std(ref_traj[:n_steps], scaler, dt=dt)
    sim.controller_PID.fct_reset()
    nominal_sim = quad_sim() if USE_PID_NOMINAL else None

    X_mpc = np.zeros((n_steps, 12))
    U_mpc = np.zeros((n_steps, mpc.nu))

    x_current_12 = drop_to_12state(X_init[0])

    X_mpc[0] = drop_to_12state(x_current_12)
    solve_times = []

    for k in range(n_steps - 1):
        x12 = drop_to_12state(x_current_12)
        z_k = lifted_state_from_x(x12, scaler)
        x_ref_h = build_ref_horizon(ref_std, k, horizon)
        u_nom = np.zeros(mpc.nu, dtype=float)
        if USE_PID_NOMINAL:
            _, _, u_att_nom = nominal_sim.controller_PID.fct_step(
                x_current_12, ref_traj[k], dt
            )
            u_nom[:3] = u_att_nom[:3]
            if mpc.nu > 3:
                u_nom[3] = u_att_nom[3]
        else:
            u_nom[:3] = [sim.q_mass * sim.g, 0.0, 0.0]
            if mpc.nu > 3:
                u_nom[3] = sim.controller_PID.fct_desired_yaw(
                    ref_traj[k], x_current_12[8]
                )

        t0 = time.perf_counter()
        u_cmd = mpc.compute(z_k, x_ref_h, u_nominal_raw=u_nom)
        solve_times.append(time.perf_counter() - t0)

        u_cmd[0] = np.clip(u_cmd[0], 0.5 * sim.q_mass * sim.g, 2.0 * sim.q_mass * sim.g)
        u_cmd[1] = np.clip(u_cmd[1], -sim.controller_PID.tilt_max, sim.controller_PID.tilt_max)
        u_cmd[2] = np.clip(u_cmd[2], -sim.controller_PID.tilt_max, sim.controller_PID.tilt_max)

        U_mpc[k] = u_cmd
        if mpc.nu > 3:
            psi_des = u_cmd[3]
        else:
            psi_des = sim.controller_PID.fct_desired_yaw(
                ref_traj[k], x_current_12[8]
            )

        x_next_12 = sim.sim_PID.fct_step_attitude(
            x_current_12,
            u1=u_cmd[0],
            phi_des=u_cmd[1],
            theta_des=u_cmd[2],
            dt=dt,
            psi_des=psi_des,
        )
        x_current_12 = x_next_12
        X_mpc[k + 1] = drop_to_12state(x_next_12)

    U_mpc[-1] = U_mpc[-2]
    sim.controller_PID.fct_reset()
    if nominal_sim is not None:
        nominal_sim.controller_PID.fct_reset()
    return X_mpc, U_mpc, solve_times

def run_linear_mpc_closedloop(mpc, sim, X_init, ref_traj, scaler, dt, horizon, n_steps):
    """Run closed-loop tracking with the linear MPC controller."""
    ref_std = precompute_ref_std(ref_traj[:n_steps], scaler, dt=dt)
    sim.controller_PID.fct_reset()
    nominal_sim = quad_sim() if USE_PID_NOMINAL else None

    X_mpc = np.zeros((n_steps, 12))
    U_mpc = np.zeros((n_steps, 3))

    x_current_12 = drop_to_12state(X_init[0])

    X_mpc[0] = drop_to_12state(x_current_12)
    solve_times = []
    u_prev = np.array([sim.q_mass * sim.g, 0.0, 0.0], dtype=float)

    for k in range(n_steps - 1):
        x12 = drop_to_12state(x_current_12)

        # Linear MPC uses the standardized physical state directly.
        z_k = scaler.transform(x12.reshape(1, -1)).flatten()

        x_ref_h = build_ref_horizon(ref_std, k, horizon)

        t0 = time.perf_counter()
        if USE_PID_NOMINAL:
            _, _, u_att_nom = nominal_sim.controller_PID.fct_step(
                x_current_12, ref_traj[k], dt
            )
            u_nom = u_att_nom[:3].copy()
            psi_des = u_att_nom[3]
        else:
            u_nom = np.array([sim.q_mass * sim.g, 0.0, 0.0], dtype=float)
            psi_des = sim.controller_PID.fct_desired_yaw(
                ref_traj[k], x_current_12[8]
            )
        u_cmd = mpc.compute(z_k, x_ref_h, u_nominal_raw=u_nom)
        solve_times.append(time.perf_counter() - t0)

        u_cmd[0] = np.clip(u_cmd[0], 0.5 * sim.q_mass * sim.g,
                                      2.0 * sim.q_mass * sim.g)
        u_cmd[1] = np.clip(u_cmd[1], -sim.controller_PID.tilt_max,
                                       sim.controller_PID.tilt_max)
        u_cmd[2] = np.clip(u_cmd[2], -sim.controller_PID.tilt_max,
                                       sim.controller_PID.tilt_max)

        U_mpc[k] = u_cmd

        x_next_12 = sim.sim_PID.fct_step_attitude(
            x_current_12,
            u1=u_cmd[0], phi_des=u_cmd[1], theta_des=u_cmd[2],
            dt=dt,
            psi_des=psi_des,
        )
        x_current_12 = x_next_12
        X_mpc[k + 1] = drop_to_12state(x_next_12)

    U_mpc[-1] = U_mpc[-2]

    sim.controller_PID.fct_reset()
    if nominal_sim is not None:
        nominal_sim.controller_PID.fct_reset()
    return X_mpc, U_mpc, solve_times


# Model scaling
def scale_linear_model(Ad, Bd, state_scaler, u_scaler):
    """
    Map the linear model from physical coordinates to standardized coordinates.

    If x_s = (x - mu_x) / sigma_x and u_s = (u - mu_u) / sigma_u, then

        x_s(k+1) = A_s x_s(k) + B_s u_s(k) + c_s
    """
    sx = state_scaler.scale_
    mx = state_scaler.mean_
    su = u_scaler.scale_
    mu = u_scaler.mean_

    Sx_inv = np.diag(1.0 / sx)
    Sx = np.diag(sx)
    Su = np.diag(su)

    A_s = Sx_inv @ Ad @ Sx
    B_s = Sx_inv @ Bd @ Su
    c_s = Sx_inv @ (Ad @ mx + Bd @ mu - mx)

    return A_s, B_s, c_s


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


def set_axes_equal_3d(ax, xyz):
    xyz = np.asarray(xyz, dtype=float)
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.55 * max(np.max(maxs - mins), 1.0)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


# PID baseline at the controller update rate
def run_pid_at_dt(sim, ref_traj, X_true, dt_mpc, n_steps):
    """
    Re-simulate the PID controller at dt_mpc (e.g. 0.1s) instead of
    the original sim dt (0.01s). This is fair: PID gets the same
    control update rate as MPC.
    """
    X_pid = np.zeros((n_steps, 12))

    x_current_12 = drop_to_12state(X_true[0])

    X_pid[0] = drop_to_12state(x_current_12)

    sim.controller_PID.fct_reset()

    for k in range(n_steps - 1):
        ref_k = ref_traj[k]

        # PID computes control from current state
        omega_cmd, u, u_att = sim.controller_PID.fct_step(
            x_current_12, ref_k, dt_mpc
        )

        # Execute on plant at dt_mpc
        from scipy.integrate import solve_ivp
        def ode(t_local, s_local):
            return sim.quad.fct_dynamics(t_local, s_local, omega_cmd)

        sol = solve_ivp(ode, [0, dt_mpc], x_current_12, method="RK45")
        x_current_12 = sol.y[:, -1]

        X_pid[k + 1] = drop_to_12state(x_current_12)

    sim.controller_PID.fct_reset()

    return X_pid


def tracking_detail(X, ref_traj, ref_xyz, dt):
    ref_vel = np.array([wp["vel"][:3] for wp in ref_traj], dtype=float)
    ref_yaw, ref_r = reference_yaw_arrays(ref_traj, dt=dt)
    yaw_err = wrap_angle_pi(X[:, 8] - ref_yaw)
    return {
        "y": rmse(X[:, 1], ref_xyz[:, 1]),
        "vy": rmse(X[:, 4], ref_vel[:, 1]),
        "yaw": float(np.sqrt(np.mean(yaw_err**2))),
        "r": rmse(X[:, 11], ref_r),
    }




# Main experiment
def main():
    # --- Load EDMDc model ---
    model    = load_edmdc_model(SCRIPT_DIR / EDMDC_MODEL_FILE)
    A_edmd   = model["A"]
    B_edmd   = model["B"]
    scaler   = model["scaler"]
    u_scaler = model["u_scaler"]
    dt       = model["dt"]
    n_obs    = model["n_obs"]

    model_input_dim_edmd = B_edmd.shape[1]
    input_lift_type = model.get("input_lift_type")
    if input_lift_type == "thrust_direction":
        input_dim_edmd = int(model.get("raw_input_dim", 4))
    else:
        input_dim_edmd = model_input_dim_edmd

    print(
        f"EDMDc model: A={A_edmd.shape} B={B_edmd.shape} dt={dt} "
        f"raw_inputs={input_dim_edmd}"
    )

    # --- Load test data ---
    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(
        SCRIPT_DIR / DATA_FILE)

    if U_all.shape[2] == 4:
        U_all = U_all[:, :, :3]

    sim_dt = t_all[0, 1] - t_all[0, 0]
    step   = int(round(dt / sim_dt))
    idx_ds = np.arange(0, t_all.shape[1], step)
    t_all      = t_all[:, idx_ds]
    states_all = states_all[:, idx_ds, :]
    U_all      = U_all[:, idx_ds, :]
    ref_traj_list = [r[::step] for r in ref_traj_list]

    sim = quad_sim()
    if input_dim_edmd not in (3, 4):
        raise ValueError(f"Expected EDMD input dimension 3 or 4, got {input_dim_edmd}")

    u_nominal_edmd = np.zeros(input_dim_edmd, dtype=float)
    u_nominal_edmd[:3] = [sim.q_mass * sim.g, 0.0, 0.0]
    u_scaler_linear = (
        u_scaler
        if input_dim_edmd == 3 and model_input_dim_edmd == 3
        else InputScalerView(u_scaler, 3)
    )
    u_nominal_linear = np.array([sim.q_mass * sim.g, 0.0, 0.0], dtype=float)

    # --- Build linear model ---
    Ad_phys, Bd_phys = build_linear_hover_model(sim, dt)

    # Scale to standardized space
    A_lin_s, B_lin_s, c_lin_s = scale_linear_model(Ad_phys, Bd_phys, scaler, u_scaler_linear)

    print(f"\nLinear model (scaled): A={A_lin_s.shape} B={B_lin_s.shape}")
    print(f"Affine offset norm: {np.linalg.norm(c_lin_s):.6f}")

    # --- Build EDMDc MPC ---
    Q = np.diag(Q_DIAG)
    if input_dim_edmd == 4:
        R_edmd = np.diag(np.r_[R_DIAG, R_YAW])
        Rd_edmd = np.diag(np.r_[RD_DIAG, RD_YAW])
        du_min_edmd = np.r_[DU_MIN, -DU_YAW]
        du_max_edmd = np.r_[DU_MAX,  DU_YAW]
        if not EDMDC_YAW_CORRECTION:
            du_min_edmd[3] = 0.0
            du_max_edmd[3] = 0.0
    else:
        R_edmd = np.diag(R_DIAG)
        Rd_edmd = np.diag(RD_DIAG)
        du_min_edmd = DU_MIN
        du_max_edmd = DU_MAX

    Cz_edmd = np.zeros((12, n_obs))
    Cz_edmd[:12, :12] = np.eye(12)

    mpc_edmd = EDMDcMPC_QP(
        A=A_edmd, B=B_edmd, Cz=Cz_edmd,
        N=N_MPC, NC=NC_MPC,
        Q=Q, R=R_edmd, Rd=Rd_edmd,
        u_scaler=u_scaler,
        du_min=du_min_edmd, du_max=du_max_edmd,
        u_nominal_raw=u_nominal_edmd,
        state_scaler=scaler,
        input_lift_type=input_lift_type,
        raw_input_dim=input_dim_edmd,
    )

    # --- Build Linear MPC (same QP structure, linear A/B, Cz=I) ---
    Cz_lin = np.eye(12)

    mpc_linear = EDMDcMPC_QP(
        A=A_lin_s, B=B_lin_s, Cz=Cz_lin,
        N=N_MPC, NC=NC_MPC,
        Q=Q, R=np.diag(R_DIAG), Rd=np.diag(RD_DIAG),
        u_scaler=u_scaler_linear,
        du_min=DU_MIN, du_max=DU_MAX,
        u_nominal_raw=u_nominal_linear,
    )

    # ============================================================
    # RUN COMPARISON ON ALL TEST CASES
    # ============================================================
    print(f"\n{'='*80}")
    print(f"COMPARISON: EDMDc MPC vs Linear MPC vs PID")
    print(f"N={N_MPC}, NC={NC_MPC}, dt={dt}")
    print(f"{'='*80}")

    all_results = []

    for run_idx, traj_name in TEST_CASES:
        ri = run_idx % states_all.shape[0]
        t_ref   = t_all[ri]
        X_true  = states_all[ri]
        ref_traj = ref_traj_list[ri]
        ref_xyz = extract_ref_xyz(ref_traj)
        T = min(len(t_ref), X_true.shape[0], ref_xyz.shape[0])
        if MAX_STEPS is not None:
            T = min(T, MAX_STEPS)
        t_ref   = t_ref[:T]
        X_true  = X_true[:T]
        ref_xyz = ref_xyz[:T]
        ref_traj = apply_yaw_reference_mode(ref_traj[:T])

        pid_data_rmse = rmse(X_true[:, 0:3], ref_xyz)

        print(f"\n--- {traj_name} (idx={run_idx}, T={T}) ---")

        # Re-simulate PID at MPC rate (fair comparison)
        X_pid_slow = run_pid_at_dt(sim, ref_traj, X_true, dt, T)
        pid_slow_rmse = rmse(X_pid_slow[:, 0:3], ref_xyz)

        if USE_CONSTANT_YAW_REFERENCE:
            print(f"  Stored PID data: {pid_data_rmse:.4f} m  (original heading yaw)")
        print(f"  PID @{dt}s:  {pid_slow_rmse:.4f} m")

        # EDMDc MPC
        X_edmd, U_edmd, st_edmd = run_mpc_closedloop(
            mpc_edmd, sim, X_true, ref_traj, scaler, dt, N_MPC, T
        )
        edmd_rmse = rmse(X_edmd[:, 0:3], ref_xyz)
        edmd_time = 1e3 * np.mean(st_edmd)
        edmd_detail = tracking_detail(X_edmd, ref_traj, ref_xyz, dt)

        # Linear MPC
        X_lin, U_lin, st_lin = run_linear_mpc_closedloop(
            mpc_linear, sim, X_true, ref_traj, scaler, dt, N_MPC, T
        )
        lin_rmse = rmse(X_lin[:, 0:3], ref_xyz)
        lin_time = 1e3 * np.mean(st_lin)
        lin_detail = tracking_detail(X_lin, ref_traj, ref_xyz, dt)

        # Winner
        best = min(pid_slow_rmse, edmd_rmse, lin_rmse)
        winner = "PID" if best == pid_slow_rmse else ("EDMDc" if best == edmd_rmse else "Linear")

        print(f"  PID:    {pid_slow_rmse:.4f} m")
        print(f"  EDMDc:  {edmd_rmse:.4f} m  ({edmd_time:.2f} ms/step)")
        print(f"          y={edmd_detail['y']:.4f}m  vy={edmd_detail['vy']:.4f}m/s  "
              f"yaw={edmd_detail['yaw']:.4f}rad  r={edmd_detail['r']:.4f}rad/s")
        print(f"  Linear: {lin_rmse:.4f} m  ({lin_time:.2f} ms/step)")
        print(f"          y={lin_detail['y']:.4f}m  vy={lin_detail['vy']:.4f}m/s  "
              f"yaw={lin_detail['yaw']:.4f}rad  r={lin_detail['r']:.4f}rad/s")
        print(f"  Winner: {winner}")

        all_results.append({
            "name": traj_name,
            "idx": run_idx,
            "pid_fast": pid_data_rmse,
            "pid":pid_slow_rmse,
            "edmdc": edmd_rmse,
            "linear": lin_rmse,
            "edmdc_time": edmd_time,
            "linear_time": lin_time,
            "t_ref": t_ref,
            "ref_traj": ref_traj,
            "ref_xyz": ref_xyz,
            "X_true": X_true,
            "X_pid":X_pid_slow,
            "X_edmd": X_edmd,
            "X_lin": X_lin,
            "U_edmd": U_edmd,
            "U_lin": U_lin,
        })

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    header = f"{'Trajectory':<18s}  {'PID':>8s}  {'EDMDc':>8s}  {'Linear':>8s}  {'EDMDc/PID':>10s}  {'Lin/PID':>10s}  {'Winner':>8s}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        ratio_e = r["edmdc"] / r["pid"] if r["pid"] > 0 else float("inf")
        ratio_l = r["linear"] / r["pid"] if r["pid"] > 0 else float("inf")
        best = min(r["pid"], r["edmdc"], r["linear"])
        winner = "PID" if best == r["pid"] else ("EDMDc" if best == r["edmdc"] else "Linear")

        print(f"{r['name']:<18s}  {r['pid']:8.4f}  {r['edmdc']:8.4f}  {r['linear']:8.4f}  "
              f"{ratio_e:10.2f}x  {ratio_l:10.2f}x  {winner:>8s}")

    # Averages
    avg_pid   = np.mean([r["pid"]    for r in all_results])
    avg_edmdc = np.mean([r["edmdc"]  for r in all_results])
    avg_lin   = np.mean([r["linear"] for r in all_results])
    avg_et    = np.mean([r["edmdc_time"] for r in all_results])
    avg_lt    = np.mean([r["linear_time"] for r in all_results])

    print(f"\n{'Average':<18s}  {avg_pid:8.4f}  {avg_edmdc:8.4f}  {avg_lin:8.4f}")
    print(f"\nSolve time — EDMDc: {avg_et:.2f} ms  Linear: {avg_lt:.2f} ms")

    # ============================================================
    # PLOTS
    # ============================================================
    n_cases = len(all_results)

    # Downsample for plotting (10000 points is too dense)
    def thin(arr, factor=10):
        return arr[::factor]

    # Colors
    C_REF   = "black"
    C_PID   = "#888888"
    C_EDMDC = "#2ca02c"
    C_LIN   = "#1f77b4"

    # ---------------------------------------------------------------
    # PLOT 1: Bar chart comparison
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(n_cases)
    width = 0.25

    ax.bar(x_pos - width, [r["pid"]    for r in all_results], width,
           label="PID", color=C_PID, edgecolor="white")
    ax.bar(x_pos,         [r["edmdc"]  for r in all_results], width,
           label="EDMDc MPC", color=C_EDMDC, edgecolor="white")
    ax.bar(x_pos + width, [r["linear"] for r in all_results], width,
           label="Linear MPC", color=C_LIN, edgecolor="white")

    ax.set_xlabel("Trajectory Type", fontsize=12)
    ax.set_ylabel("Position RMSE [m]", fontsize=12)
    ax.set_title("Tracking Performance Comparison", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r["name"] for r in all_results], rotation=15, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    # ---------------------------------------------------------------
    # PLOT 2: 3D trajectory plots (reference clearly visible)
    # ---------------------------------------------------------------
    fig_3d, axes_3d = plt.subplots(
        1, n_cases, figsize=(6.5 * n_cases, 6),
        subplot_kw={"projection": "3d"}, squeeze=False
    )
    ds = 10  # downsample factor
    for i, (r, ax) in enumerate(zip(all_results, axes_3d.flat)):
        ref = r["ref_xyz"]
        # Reference FIRST — thick black
        ax.plot(ref[::ds, 0], ref[::ds, 1], ref[::ds, 2],
                C_REF, lw=3, label="Reference", zorder=1)
        # Responses on top
        ax.plot(r["X_pid"][::ds, 0], r["X_pid"][::ds, 1], r["X_pid"][::ds, 2],
                color=C_PID, lw=1.2, alpha=0.6, label=f"PID ({r['pid']:.3f}m)", zorder=2)
        ax.plot(r["X_edmd"][::ds, 0], r["X_edmd"][::ds, 1], r["X_edmd"][::ds, 2],
                color=C_EDMDC, lw=1.5, label=f"EDMDc ({r['edmdc']:.3f}m)", zorder=3)
        ax.plot(r["X_lin"][::ds, 0], r["X_lin"][::ds, 1], r["X_lin"][::ds, 2],
                color=C_LIN, lw=1.2, ls="--", label=f"Linear ({r['linear']:.3f}m)", zorder=3)
        # Start marker
        ax.scatter([ref[0, 0]], [ref[0, 1]], [ref[0, 2]],
                   c="red", s=60, marker="o", zorder=4, label="Start")
        ax.set_title(f"{r['name']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("x [m]", fontsize=9)
        ax.set_ylabel("y [m]", fontsize=9)
        ax.set_zlabel("z [m]", fontsize=9)
        xyz_plot = np.vstack([
            ref[:, 0:3],
            r["X_pid"][:, 0:3],
            r["X_edmd"][:, 0:3],
            r["X_lin"][:, 0:3],
        ])
        set_axes_equal_3d(ax, xyz_plot)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes_3d.flat[len(all_results):]:
        ax.set_axis_off()
    fig_3d.suptitle("3D Trajectory Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    print("Generated 3D Trajectory Comparison plot.")

    # ---------------------------------------------------------------
    # PLOT 3: Per-axis X, Y, Z over time for each trajectory
    # ---------------------------------------------------------------
    axis_labels = ["x [m]", "y [m]", "z [m]"]
    for r in all_results:
        ref = r["ref_xyz"]
        t   = r["t_ref"]

        fig_xyz, axes_xyz = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        for j in range(3):
            ax = axes_xyz[j]
            ax.plot(thin(t), thin(ref[:, j]),
                    C_REF, lw=2.5, label="Reference", zorder=1)
            ax.plot(thin(t), thin(r["X_pid"][:, j]),
                    color=C_PID, lw=1.2, alpha=0.6, label="PID", zorder=2)
            ax.plot(thin(t), thin(r["X_edmd"][:, j]),
                    color=C_EDMDC, lw=1.5, label="EDMDc MPC", zorder=3)
            ax.plot(thin(t), thin(r["X_lin"][:, j]),
                    color=C_LIN, lw=1.2, ls="--", label="Linear MPC", zorder=3)
            ax.set_ylabel(axis_labels[j], fontsize=11)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(fontsize=9, ncol=4, loc="upper right")
        axes_xyz[-1].set_xlabel("Time [s]", fontsize=11)
        fig_xyz.suptitle(
            f"{r['name']} — Position Tracking\n"
            f"PID={r['pid']:.4f}m   EDMDc={r['edmdc']:.4f}m   Linear={r['linear']:.4f}m",
            fontsize=13, fontweight="bold")
        plt.tight_layout()

    # ---------------------------------------------------------------
    # PLOT 4: Position error magnitude over time
    # ---------------------------------------------------------------
    fig_err, axes_err = plt.subplots(
        1, n_cases, figsize=(6 * n_cases, 4.8), squeeze=False
    )
    for i, (r, ax) in enumerate(zip(all_results, axes_err.flat)):
        ref = r["ref_xyz"]
        t   = r["t_ref"]
        err_pid   = np.linalg.norm(r["X_pid"][:, 0:3] - ref, axis=1)
        err_edmdc = np.linalg.norm(r["X_edmd"][:, 0:3] - ref, axis=1)
        err_lin   = np.linalg.norm(r["X_lin"][:, 0:3]  - ref, axis=1)

        ax.plot(thin(t), thin(err_pid),   color=C_PID, lw=1, alpha=0.6, label="PID")
        ax.plot(thin(t), thin(err_edmdc), color=C_EDMDC, lw=1.2, label="EDMDc")
        ax.plot(thin(t), thin(err_lin),   color=C_LIN, lw=1, ls="--", label="Linear")
        ax.set_title(f"{r['name']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("t [s]", fontsize=10)
        ax.set_ylabel("||pos error|| [m]", fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    for ax in axes_err.flat[len(all_results):]:
        ax.set_axis_off()
    fig_err.suptitle("Position Error Magnitude Over Time", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # ---------------------------------------------------------------
    # PLOT 5: Weak-state diagnostics for figure-8 case
    # ---------------------------------------------------------------
    fig8_idx = next(i for i, r in enumerate(all_results) if "figure" in r["name"])
    r = all_results[fig8_idx]
    ref_vel = np.array([wp["vel"][:3] for wp in r["ref_traj"]], dtype=float)
    ref_yaw, ref_rate = reference_yaw_arrays(r["ref_traj"], dt=dt)

    def yaw_on_ref_branch(X):
        return ref_yaw + wrap_angle_pi(X[:, 8] - ref_yaw)

    diag_specs = [
        ("y [m]", r["ref_xyz"][:, 1], r["X_pid"][:, 1], r["X_edmd"][:, 1], r["X_lin"][:, 1]),
        ("vy [m/s]", ref_vel[:, 1], r["X_pid"][:, 4], r["X_edmd"][:, 4], r["X_lin"][:, 4]),
        (r"$\psi$ [rad]", ref_yaw, yaw_on_ref_branch(r["X_pid"]),
         yaw_on_ref_branch(r["X_edmd"]), yaw_on_ref_branch(r["X_lin"])),
        ("r [rad/s]", ref_rate, r["X_pid"][:, 11], r["X_edmd"][:, 11], r["X_lin"][:, 11]),
    ]

    fig_diag, axes_diag = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, (label, ref_sig, pid_sig, edmd_sig, lin_sig) in zip(axes_diag, diag_specs):
        ax.plot(thin(r["t_ref"]), thin(ref_sig), C_REF, lw=2.3, label="Reference")
        ax.plot(thin(r["t_ref"]), thin(pid_sig), color=C_PID, lw=1.1, alpha=0.65, label="PID")
        ax.plot(thin(r["t_ref"]), thin(edmd_sig), color=C_EDMDC, lw=1.3, label="EDMDc MPC")
        ax.plot(thin(r["t_ref"]), thin(lin_sig), color=C_LIN, lw=1.1, ls="--", label="Linear MPC")
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
    axes_diag[0].legend(fontsize=9, ncol=4, loc="upper right")
    axes_diag[-1].set_xlabel("Time [s]", fontsize=11)
    fig_diag.suptitle(f"Weak-State Diagnostics - {r['name']}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # ---------------------------------------------------------------
    # PLOT 6: Control inputs for figure-8 case
    # ---------------------------------------------------------------
    u_labels = ["Thrust [N]", r"$\phi_{des}$ [rad]", r"$\theta_{des}$ [rad]"]
    if r["U_edmd"].shape[1] > 3:
        u_labels.append(r"$\psi_{des}$ [rad]")
    fig_u, axes_u = plt.subplots(len(u_labels), 1, figsize=(14, 9), sharex=True)
    axes_u = np.atleast_1d(axes_u)
    for j, label in enumerate(u_labels):
        ax = axes_u[j]
        ax.plot(thin(r["t_ref"]), thin(r["U_edmd"][:, j]),
                color=C_EDMDC, lw=1.2, label="EDMDc MPC")
        if j < r["U_lin"].shape[1]:
            ax.plot(thin(r["t_ref"]), thin(r["U_lin"][:, j]),
                    color=C_LIN, lw=1, ls="--", label="Linear MPC")
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=10)
    axes_u[-1].set_xlabel("Time [s]", fontsize=11)
    fig_u.suptitle(f"Control Inputs — {r['name']}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # ---------------------------------------------------------------
    # PLOT 7: Solve time comparison
    # ---------------------------------------------------------------
    fig_st, ax_st = plt.subplots(figsize=(8, 4))
    names  = [r["name"] for r in all_results]
    t_edmd = [r["edmdc_time"] for r in all_results]
    t_lin  = [r["linear_time"] for r in all_results]
    x_pos  = np.arange(n_cases)
    ax_st.bar(x_pos - 0.15, t_edmd, 0.3, label="EDMDc MPC", color=C_EDMDC)
    ax_st.bar(x_pos + 0.15, t_lin,  0.3, label="Linear MPC", color=C_LIN)
    ax_st.set_ylabel("Solve time [ms]", fontsize=11)
    ax_st.set_title("Computational Cost Comparison", fontsize=13, fontweight="bold")
    ax_st.set_xticks(x_pos)
    ax_st.set_xticklabels(names, rotation=15, fontsize=10)
    ax_st.legend(fontsize=10)
    ax_st.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
