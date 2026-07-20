"""
tune_mpc.py
===========
Sweep MPC Q/R weights to find the best tracking performance.
Uses a shortened simulation (first 300 steps) for speed,
then validates top candidates on the full run.
"""

import pickle
import time
import itertools
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import osqp

from Simulation import quad_sim
from newmpc import (
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
EDMDC_MODEL_FILE = "edmdc_model.pkl"
DATA_FILE        = "runs_mixed_n300.pkl"
TEST_RUN_IDX     = 99

# Fixed MPC structure
N_EDMD  = 50
NC_EDMD = 25

# ============================================================
# PARAMETER GRID
# ============================================================
# Coarse sweep — each combo takes ~0.2s for 300 steps
GRID = {
    "Q_pos":    [5000, 20000, 50000, 100000, 200000],
    "Q_vel":    [100, 500, 1000, 5000, 10000],
    "R_thrust": [0.0001, 0.001, 0.01, 0.1],
    "R_angle":  [0.001, 0.01, 0.1, 0.5],
}

# DU bounds (fixed — wide enough to not constrain)
DU_MIN = np.array([-5.0, -3.5, -3.5], dtype=float)
DU_MAX = np.array([ 5.0,  3.5,  3.5], dtype=float)

# How many steps for fast eval
FAST_STEPS = 300

# How many top candidates to validate on full run
TOP_K = 5


# ============================================================
# SINGLE EVALUATION
# ============================================================
def evaluate_config(config, model, sim, t_ref, X_true, ref_traj,
                    ref_xyz, n_steps, label=""):
    """
    Run MPC closed-loop for n_steps and return position RMSE.
    """
    A_edmd   = model["A"]
    B_edmd   = model["B"]
    scaler   = model["scaler"]
    u_scaler = model["u_scaler"]
    dt       = model["dt"]
    n_obs    = model["n_obs"]

    Q_pos, Q_vel, R_thrust, R_angle = (
        config["Q_pos"], config["Q_vel"],
        config["R_thrust"], config["R_angle"]
    )

    Q_diag = np.array([
        Q_pos, Q_pos, Q_pos,
        Q_vel, Q_vel, Q_vel,
        0.0, 0.0,
        0.0, 0.0,
    ], dtype=float)

    R_diag  = np.array([R_thrust, R_angle, R_angle], dtype=float)
    Rd_diag = R_diag * 0.1  # rate penalty = 10% of R

    Cz = np.zeros((10, n_obs))
    Cz[:10, :10] = np.eye(10)

    u_nominal = np.array([sim.q_mass * sim.g, 0.0, 0.0], dtype=float)

    try:
        mpc = EDMDcMPC_QP(
            A=A_edmd, B=B_edmd, Cz=Cz,
            N=N_EDMD, NC=NC_EDMD,
            Q=np.diag(Q_diag), R=np.diag(R_diag), Rd=np.diag(Rd_diag),
            u_scaler=u_scaler,
            du_min=DU_MIN, du_max=DU_MAX,
            u_nominal_raw=u_nominal,
        )
    except Exception as e:
        print(f"  MPC setup failed: {e}")
        return float("inf"), None

    ref_std = precompute_ref_std(ref_traj[:n_steps], scaler, n_states=10)

    X_mpc = np.zeros((n_steps, 10))

    x_current_12 = np.zeros(12)
    x10_init = X_true[0]
    x_current_12[0:6]  = x10_init[0:6]
    x_current_12[6:8]  = x10_init[6:8]
    x_current_12[9:11] = x10_init[8:10]

    X_mpc[0] = drop_to_10state(x_current_12)

    for k in range(n_steps - 1):
        x10 = drop_to_10state(x_current_12)
        z_k = lifted_state_from_x(x10, scaler)
        x_ref_h = build_ref_horizon(ref_std, k, N_EDMD)
        u_cmd = mpc.compute(z_k, x_ref_h)

        u_cmd[0] = np.clip(u_cmd[0], 0.5 * sim.q_mass * sim.g,
                                      2.0 * sim.q_mass * sim.g)
        u_cmd[1] = np.clip(u_cmd[1], -sim.controller_PID.tilt_max,
                                       sim.controller_PID.tilt_max)
        u_cmd[2] = np.clip(u_cmd[2], -sim.controller_PID.tilt_max,
                                       sim.controller_PID.tilt_max)

        x_next_12 = sim.sim_PID.fct_step_attitude(
            x_current_12,
            u1=u_cmd[0], phi_des=u_cmd[1], theta_des=u_cmd[2],
            dt=dt
        )
        x_current_12 = x_next_12
        X_mpc[k + 1] = drop_to_10state(x_next_12)

    pos_rmse = rmse(X_mpc[:, 0:3], ref_xyz[:n_steps])
    vel_rmse = rmse(X_mpc[:, 3:6], X_true[:n_steps, 3:6])

    return pos_rmse, vel_rmse


# ============================================================
# MAIN
# ============================================================
def main():
    # --- load ---
    model = load_edmdc_model(SCRIPT_DIR / EDMDC_MODEL_FILE)
    dt    = model["dt"]

    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(
        SCRIPT_DIR / DATA_FILE)

    if states_all.shape[2] == 12:
        states_all = states_all[:, :, [0,1,2,3,4,5,6,7,9,10]]
    if U_all.shape[2] == 4:
        U_all = U_all[:, :, :3]

    sim_dt = t_all[0, 1] - t_all[0, 0]
    step   = int(round(dt / sim_dt))
    idx    = np.arange(0, t_all.shape[1], step)
    t_all      = t_all[:, idx]
    states_all = states_all[:, idx, :]
    U_all      = U_all[:, idx, :]
    ref_traj_list = [r[::step] for r in ref_traj_list]

    run_idx = TEST_RUN_IDX % states_all.shape[0]
    t_ref   = t_all[run_idx]
    X_true  = states_all[run_idx]
    ref_traj = ref_traj_list[run_idx]
    ref_xyz = extract_ref_xyz(ref_traj)
    T       = min(len(t_ref), X_true.shape[0], ref_xyz.shape[0])
    t_ref   = t_ref[:T]
    X_true  = X_true[:T]
    ref_xyz = ref_xyz[:T]

    sim = quad_sim()

    print(f"Test run: {run_idx}  T={T}  duration={t_ref[-1]:.1f}s")
    print(f"PID baseline RMSE: {rmse(X_true[:, 0:3], ref_xyz):.4f} m")

    # --- build grid ---
    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))
    n_combos = len(combos)

    print(f"\n{'='*60}")
    print(f"COARSE SWEEP: {n_combos} combos × {FAST_STEPS} steps each")
    print(f"{'='*60}")

    results = []
    t0_all = time.perf_counter()

    for idx_c, combo in enumerate(combos):
        config = dict(zip(keys, combo))

        t0 = time.perf_counter()
        pos_rmse, vel_rmse = evaluate_config(
            config, model, sim,
            t_ref, X_true, ref_traj, ref_xyz,
            n_steps=FAST_STEPS
        )
        elapsed = time.perf_counter() - t0

        results.append((pos_rmse, vel_rmse, config))

        if (idx_c + 1) % 20 == 0 or idx_c == 0:
            print(f"  [{idx_c+1:4d}/{n_combos}]  "
                  f"pos_RMSE={pos_rmse:8.4f}  "
                  f"Q_pos={config['Q_pos']:>8.0f}  "
                  f"Q_vel={config['Q_vel']:>6.0f}  "
                  f"R_thr={config['R_thrust']:.4f}  "
                  f"R_ang={config['R_angle']:.3f}  "
                  f"({elapsed:.2f}s)")

    total_time = time.perf_counter() - t0_all
    print(f"\nCoarse sweep done in {total_time:.1f}s")

    # --- sort by pos RMSE ---
    results.sort(key=lambda r: r[0])

    print(f"\n{'='*60}")
    print(f"TOP {min(20, len(results))} CONFIGS (by position RMSE, {FAST_STEPS} steps)")
    print(f"{'='*60}")
    print(f"{'Rank':>4s}  {'pos_RMSE':>10s}  {'vel_RMSE':>10s}  "
          f"{'Q_pos':>8s}  {'Q_vel':>6s}  {'R_thr':>8s}  {'R_ang':>6s}")
    print("-" * 70)
    for i, (pr, vr, cfg) in enumerate(results[:20]):
        print(f"{i+1:4d}  {pr:10.4f}  {vr:10.4f}  "
              f"{cfg['Q_pos']:8.0f}  {cfg['Q_vel']:6.0f}  "
              f"{cfg['R_thrust']:8.4f}  {cfg['R_angle']:6.3f}")

    # --- validate top K on full run ---
    print(f"\n{'='*60}")
    print(f"FULL VALIDATION: top {TOP_K} configs × {T} steps")
    print(f"{'='*60}")

    full_results = []
    for i, (_, _, config) in enumerate(results[:TOP_K]):
        print(f"\n  Validating config {i+1}/{TOP_K}: {config}")
        t0 = time.perf_counter()
        pos_rmse, vel_rmse = evaluate_config(
            config, model, sim,
            t_ref, X_true, ref_traj, ref_xyz,
            n_steps=T
        )
        elapsed = time.perf_counter() - t0
        full_results.append((pos_rmse, vel_rmse, config))
        print(f"    pos_RMSE={pos_rmse:.4f}  vel_RMSE={vel_rmse:.4f}  ({elapsed:.1f}s)")

    full_results.sort(key=lambda r: r[0])

    print(f"\n{'='*60}")
    print(f"FINAL RANKING (full {T}-step validation)")
    print(f"{'='*60}")
    for i, (pr, vr, cfg) in enumerate(full_results):
        print(f"  #{i+1}  pos_RMSE={pr:.4f}  vel_RMSE={vr:.4f}")
        print(f"       Q_pos={cfg['Q_pos']}  Q_vel={cfg['Q_vel']}  "
              f"R_thrust={cfg['R_thrust']}  R_angle={cfg['R_angle']}")

    best = full_results[0][2]
    print(f"\n{'='*60}")
    print(f"BEST CONFIG:")
    print(f"  Q_DIAG_EDMD = np.array([")
    print(f"      {best['Q_pos']}, {best['Q_pos']}, {best['Q_pos']},")
    print(f"      {best['Q_vel']}, {best['Q_vel']}, {best['Q_vel']},")
    print(f"      0.0, 0.0,")
    print(f"      0.0, 0.0,")
    print(f"  ], dtype=float)")
    print(f"  R_DIAG_EDMD  = np.array([{best['R_thrust']}, {best['R_angle']}, {best['R_angle']}], dtype=float)")
    print(f"  RD_DIAG_EDMD = np.array([{best['R_thrust']*0.1}, {best['R_angle']*0.1}, {best['R_angle']*0.1}], dtype=float)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()