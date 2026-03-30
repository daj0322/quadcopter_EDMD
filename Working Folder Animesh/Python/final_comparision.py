"""
compare_mpc.py
==============
Head-to-head comparison:
  1) PID baseline (from saved data)
  2) Linear MPC (Jacobian linearization around hover)
  3) EDMDc MPC (data-driven lifted model)

All three use the same plant for execution (inner PID + drone via fct_step_attitude).
Both MPCs use the same OSQP QP solver, same horizon, same Q/R weights.
The only difference is the prediction model.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt

from Simulation import quad_sim
from edmdc_mpc import (
    EDMDcMPC_QP,
    load_edmdc_model,
    load_simulation_runs,
    observables,
    lifted_state_from_x,
    drop_to_10state,
    precompute_ref_std,
    build_ref_horizon,
    extract_ref_xyz,
    rmse,
)

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR       = Path(__file__).resolve().parent
EDMDC_MODEL_FILE = "edmdc_model_0.1.pkl"
DATA_FILE        = "runs_mixed_n300.pkl"

# Test indices — one per trajectory family
TEST_CASES = [
    (39,  "helix (small)"),
    (59,  "figure-8"),
    (99,  "helix (large)"),
    (129, "lissajous"),
    (155, "waypoint"),
    (210, "hover excitation"),
]

# MPC config (use tuned values)
N_MPC   = 20
NC_MPC  = 15

Q_DIAG = np.array([
    100000.0, 100000.0, 100000.0,
    25.0, 25.0, 25.0,
    0.0, 0.0,
    0.0, 0.0,
], dtype=float)

R_DIAG = np.array([0.0002, 0.5, 0.5], dtype=float)
RD_DIAG = np.array([2e-05, 0.05, 0.05], dtype=float)

DU_MIN = np.array([-5.0, -3.5, -3.5], dtype=float)
DU_MAX = np.array([ 5.0,  3.5,  3.5], dtype=float)


# ============================================================
# BUILD LINEAR MODEL (Jacobian around hover)
# ============================================================
def build_linear_hover_model(sim, dt):
    """
    Build discretized linear model of the plant (inner PID + drone)
    linearized around hover.

    State: [x, y, z, vx, vy, vz, phi, theta, p, q]  (10)
    Input: [thrust, phi_des, theta_des]               (3)

    Continuous dynamics at hover:
        x_dot   = vx
        y_dot   = vy
        z_dot   = vz
        vx_dot  = g * theta          (small angle, thrust along z)
        vy_dot  = -g * phi           (small angle)
        vz_dot  = thrust/m - g       (perturbation: 1/m * delta_thrust)
        phi_dot = p
        theta_dot = q
        p_dot   = (-Kp_phi * phi - Kd_phi * p + Kp_phi * phi_des) / Ixx
        q_dot   = (-Kp_theta * theta - Kd_theta * q + Kp_theta * theta_des) / Iyy
    """
    m   = sim.q_mass
    g   = sim.g
    Ixx = sim.Ixx
    Iyy = sim.Iyy

    # Inner PID gains (PD approximation — ignore integral for linearization)
    Kp_phi   = sim.kp_ang[0]
    Kd_phi   = sim.kd_ang[0]
    Kp_theta = sim.kp_ang[1]
    Kd_theta = sim.kd_ang[1]

    nx, nu = 10, 3

    # Continuous A matrix
    Ac = np.zeros((nx, nx))
    # Position from velocity
    Ac[0, 3] = 1.0   # dx/dt = vx
    Ac[1, 4] = 1.0   # dy/dt = vy
    Ac[2, 5] = 1.0   # dz/dt = vz
    # Velocity from angles (small angle gravity projection)
    Ac[3, 7] = g     # dvx/dt = g * theta
    Ac[4, 6] = -g    # dvy/dt = -g * phi
    # vz: no state dependence at hover (thrust perturbation is in B)
    # Angle from angular rate
    Ac[6, 8] = 1.0   # dphi/dt = p
    Ac[7, 9] = 1.0   # dtheta/dt = q
    # Angular rate from inner PID
    Ac[8, 6] = -Kp_phi / Ixx     # dp/dt depends on phi
    Ac[8, 8] = -Kd_phi / Ixx     # dp/dt depends on p (damping)
    Ac[9, 7] = -Kp_theta / Iyy   # dq/dt depends on theta
    Ac[9, 9] = -Kd_theta / Iyy   # dq/dt depends on q (damping)

    # Continuous B matrix
    Bc = np.zeros((nx, nu))
    Bc[5, 0] = 1.0 / m           # dvz/dt = delta_thrust / m
    Bc[8, 1] = Kp_phi / Ixx      # dp/dt from phi_des
    Bc[9, 2] = Kp_theta / Iyy    # dq/dt from theta_des

    # Discretize via matrix exponential: [Ad, Bd] from [Ac, Bc; 0, 0]
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


# ============================================================
# RUN MPC CLOSED-LOOP
# ============================================================
def run_mpc_closedloop(mpc, sim, X_true, ref_traj, ref_xyz, scaler,
                       dt, N, n_steps, label="MPC"):
    """
    Generic closed-loop runner for any MPC that has .compute(z, ref_horizon).
    """
    ref_std = precompute_ref_std(ref_traj[:n_steps], scaler, n_states=10)

    X_mpc = np.zeros((n_steps, 10))
    U_mpc = np.zeros((n_steps, 3))

    x_current_12 = np.zeros(12)
    x10_init = X_true[0]
    x_current_12[0:6]  = x10_init[0:6]
    x_current_12[6:8]  = x10_init[6:8]
    x_current_12[9:11] = x10_init[8:10]

    X_mpc[0] = drop_to_10state(x_current_12)
    solve_times = []

    for k in range(n_steps - 1):
        x10 = drop_to_10state(x_current_12)
        z_k = lifted_state_from_x(x10, scaler)
        x_ref_h = build_ref_horizon(ref_std, k, N)

        t0 = time.perf_counter()
        u_cmd = mpc.compute(z_k, x_ref_h)
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
            dt=dt
        )
        x_current_12 = x_next_12
        X_mpc[k + 1] = drop_to_10state(x_next_12)

    U_mpc[-1] = U_mpc[-2]

    return X_mpc, U_mpc, solve_times


def run_linear_mpc_closedloop(mpc, sim, X_true, ref_traj, ref_xyz, scaler,
                              u_scaler_lin, dt, N, n_steps):
    """
    Closed-loop runner for linear MPC.
    The linear MPC operates in its own scaled space but uses the same plant.
    """
    ref_std = precompute_ref_std(ref_traj[:n_steps], scaler, n_states=10)

    X_mpc = np.zeros((n_steps, 10))
    U_mpc = np.zeros((n_steps, 3))

    x_current_12 = np.zeros(12)
    x10_init = X_true[0]
    x_current_12[0:6]  = x10_init[0:6]
    x_current_12[6:8]  = x10_init[6:8]
    x_current_12[9:11] = x10_init[8:10]

    X_mpc[0] = drop_to_10state(x_current_12)
    solve_times = []

    for k in range(n_steps - 1):
        x10 = drop_to_10state(x_current_12)

        # Linear MPC: lift = just scale (no nonlinear observables)
        z_k = scaler.transform(x10.reshape(1, -1)).flatten()

        x_ref_h = build_ref_horizon(ref_std, k, N)

        t0 = time.perf_counter()
        u_cmd = mpc.compute(z_k, x_ref_h)
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
            dt=dt
        )
        x_current_12 = x_next_12
        X_mpc[k + 1] = drop_to_10state(x_next_12)

    U_mpc[-1] = U_mpc[-2]

    return X_mpc, U_mpc, solve_times


# ============================================================
# SCALE LINEAR MODEL TO MATCH EDMDC SCALER SPACE
# ============================================================
def scale_linear_model(Ad, Bd, state_scaler, u_scaler):
    """
    Transform linear model from physical space to standardized space
    so it can be used with the same Q/R weights as EDMDc MPC.

    If x_s = (x - mu_x) / sigma_x, u_s = (u - mu_u) / sigma_u
    Then: x_s_{k+1} = A_s @ x_s_k + B_s @ u_s_k + c_s

    A_s = diag(1/sigma_x) @ Ad @ diag(sigma_x)
    B_s = diag(1/sigma_x) @ Bd @ diag(sigma_u)
    c_s = diag(1/sigma_x) @ (Ad @ mu_x + Bd @ mu_u - mu_x)
        (affine offset — absorbed into bias or ignored)
    """
    sx = state_scaler.scale_
    mx = state_scaler.mean_
    su = u_scaler.scale_
    mu = u_scaler.mean_

    Sx_inv = np.diag(1.0 / sx)
    Sx     = np.diag(sx)
    Su     = np.diag(su)

    A_s = Sx_inv @ Ad @ Sx
    B_s = Sx_inv @ Bd @ Su

    # Affine offset from linearization around hover (not zero in scaled space)
    c_s = Sx_inv @ (Ad @ mx + Bd @ mu - mx)

    return A_s, B_s, c_s

def run_pid_at_dt(sim, ref_traj, X_true, dt_mpc, n_steps):
    """
    Re-simulate the PID controller at dt_mpc (e.g. 0.1s) instead of
    the original sim dt (0.01s). This is fair: PID gets the same
    control update rate as MPC.
    """
    X_pid = np.zeros((n_steps, 10))

    x_current_12 = np.zeros(12)
    x10_init = X_true[0]
    x_current_12[0:6]  = x10_init[0:6]
    x_current_12[6:8]  = x10_init[6:8]
    x_current_12[9:11] = x10_init[8:10]

    X_pid[0] = drop_to_10state(x_current_12)

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

        X_pid[k + 1] = drop_to_10state(x_current_12)

    sim.controller_PID.fct_reset()

    return X_pid




# ============================================================
# MAIN
# ============================================================
def main():
    # --- Load EDMDc model ---
    model    = load_edmdc_model(SCRIPT_DIR / EDMDC_MODEL_FILE)
    A_edmd   = model["A"]
    B_edmd   = model["B"]
    scaler   = model["scaler"]
    u_scaler = model["u_scaler"]
    dt       = model["dt"]
    n_obs    = model["n_obs"]

    print(f"EDMDc model: A={A_edmd.shape} B={B_edmd.shape} dt={dt}")

    # --- Load test data ---
    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(
        SCRIPT_DIR / DATA_FILE)

    if states_all.shape[2] == 12:
        states_all = states_all[:, :, [0,1,2,3,4,5,6,7,9,10]]
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
    u_nominal = np.array([sim.q_mass * sim.g, 0.0, 0.0], dtype=float)

    # --- Build linear model ---
    Ad_phys, Bd_phys = build_linear_hover_model(sim, dt)

    # Scale to standardized space
    A_lin_s, B_lin_s, c_lin_s = scale_linear_model(Ad_phys, Bd_phys, scaler, u_scaler)

    print(f"\nLinear model (scaled): A={A_lin_s.shape} B={B_lin_s.shape}")
    print(f"Affine offset norm: {np.linalg.norm(c_lin_s):.6f}")

    # --- Build EDMDc MPC ---
    Q  = np.diag(Q_DIAG)
    R  = np.diag(R_DIAG)
    Rd = np.diag(RD_DIAG)

    Cz_edmd = np.zeros((10, n_obs))
    Cz_edmd[:10, :10] = np.eye(10)

    mpc_edmd = EDMDcMPC_QP(
        A=A_edmd, B=B_edmd, Cz=Cz_edmd,
        N=N_MPC, NC=NC_MPC,
        Q=Q, R=R, Rd=Rd,
        u_scaler=u_scaler,
        du_min=DU_MIN, du_max=DU_MAX,
        u_nominal_raw=u_nominal,
    )

    # --- Build Linear MPC (same QP structure, linear A/B, Cz=I) ---
    Cz_lin = np.eye(10)

    mpc_linear = EDMDcMPC_QP(
        A=A_lin_s, B=B_lin_s, Cz=Cz_lin,
        N=N_MPC, NC=NC_MPC,
        Q=Q, R=R, Rd=Rd,
        u_scaler=u_scaler,
        du_min=DU_MIN, du_max=DU_MAX,
        u_nominal_raw=u_nominal,
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
        t_ref   = t_ref[:T]
        X_true  = X_true[:T]
        ref_xyz = ref_xyz[:T]

        pid_rmse = rmse(X_true[:, 0:3], ref_xyz)

        print(f"\n--- {traj_name} (idx={run_idx}, T={T}) ---")

        # Re-simulate PID at MPC rate (fair comparison)
        X_pid_slow = run_pid_at_dt(sim, ref_traj, X_true, dt, T)
        pid_slow_rmse = rmse(X_pid_slow[:, 0:3], ref_xyz)

        print(f"  PID @{dt}s:  {pid_slow_rmse:.4f} m")

        # EDMDc MPC
        X_edmd, U_edmd, st_edmd = run_mpc_closedloop(
            mpc_edmd, sim, X_true, ref_traj, ref_xyz, scaler,
            dt, N_MPC, T, label="EDMDc"
        )
        edmd_rmse = rmse(X_edmd[:, 0:3], ref_xyz)
        edmd_time = 1e3 * np.mean(st_edmd)

        # Linear MPC
        X_lin, U_lin, st_lin = run_linear_mpc_closedloop(
            mpc_linear, sim, X_true, ref_traj, ref_xyz, scaler,
            u_scaler, dt, N_MPC, T
        )
        lin_rmse = rmse(X_lin[:, 0:3], ref_xyz)
        lin_time = 1e3 * np.mean(st_lin)

        # Winner
        best = min(pid_slow_rmse, edmd_rmse, lin_rmse)
        winner = "PID" if best == pid_rmse else ("EDMDc" if best == edmd_rmse else "Linear")

        print(f"  PID:    {pid_rmse:.4f} m")
        print(f"  EDMDc:  {edmd_rmse:.4f} m  ({edmd_time:.2f} ms/step)")
        print(f"  Linear: {lin_rmse:.4f} m  ({lin_time:.2f} ms/step)")
        print(f"  Winner: {winner}")

        all_results.append({
            "name": traj_name,
            "idx": run_idx,
            "pid_fast": pid_rmse,
            "pid":pid_slow_rmse,
            "edmdc": edmd_rmse,
            "linear": lin_rmse,
            "edmdc_time": edmd_time,
            "linear_time": lin_time,
            "t_ref": t_ref,
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
    fig_3d, axes_3d = plt.subplots(2, 3, figsize=(20, 12),
                                    subplot_kw={"projection": "3d"})
    ds = 10  # downsample factor
    for i, (r, ax) in enumerate(zip(all_results, axes_3d.flat)):
        ref = r["ref_xyz"]
        # Reference FIRST — thick black
        ax.plot(ref[::ds, 0], ref[::ds, 1], ref[::ds, 2],
                C_REF, lw=3, label="Reference", zorder=1)
        # Responses on top
        ax.plot(r["X_true"][::ds, 0], r["X_pid"][::ds, 1], r["X_true"][::ds, 2],
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
        ax.legend(fontsize=7, loc="upper left")
    fig_3d.suptitle("3D Trajectory Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()

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
    fig_err, axes_err = plt.subplots(2, 3, figsize=(18, 9))
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
    fig_err.suptitle("Position Error Magnitude Over Time", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # ---------------------------------------------------------------
    # PLOT 5: Control inputs for figure-8 case
    # ---------------------------------------------------------------
    fig8_idx = next(i for i, r in enumerate(all_results) if "figure" in r["name"])
    r = all_results[fig8_idx]
    u_labels = ["Thrust [N]", r"$\phi_{des}$ [rad]", r"$\theta_{des}$ [rad]"]
    fig_u, axes_u = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for j in range(3):
        ax = axes_u[j]
        ax.plot(thin(r["t_ref"]), thin(r["U_edmd"][:, j]),
                color=C_EDMDC, lw=1.2, label="EDMDc MPC")
        ax.plot(thin(r["t_ref"]), thin(r["U_lin"][:, j]),
                color=C_LIN, lw=1, ls="--", label="Linear MPC")
        ax.set_ylabel(u_labels[j], fontsize=11)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=10)
    axes_u[-1].set_xlabel("Time [s]", fontsize=11)
    fig_u.suptitle(f"Control Inputs — {r['name']}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # ---------------------------------------------------------------
    # PLOT 6: Solve time comparison
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