"""
tune_mpc_full.py
================
Full MPC hyperparameter sweep: Q, R, N, NC, DU bounds.
Phase 1: coarse sweep on 300 steps
Phase 2: validate top K on full run
Phase 3: fine sweep around best config
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

# Test on multiple trajectory types for robustness
TEST_INDICES = [39, 59, 99, 129, 155, 210]
# Labels for readability
TEST_LABELS  = ["helix_sm", "fig8", "helix_lg", "lissajous", "waypoint", "hover"]

FAST_STEPS = 300
TOP_K      = 10

DU_MIN_FIXED = np.array([-5.0, -3.5, -3.5], dtype=float)
DU_MAX_FIXED = np.array([ 5.0,  3.5,  3.5], dtype=float)

# ============================================================
# PARAMETER GRID
# ============================================================
GRID_COARSE = {
    "Q_pos":    [50000, 100000, 200000, 500000],
    "Q_vel":    [100, 500, 1000, 5000],
    "R_thrust": [0.0001, 0.001, 0.01],
    "R_angle":  [0.01, 0.1, 0.5],
    "N":        [20, 30, 50],
    "NC":       [10, 15, 25],
}

# Fine grid around best (populated after coarse sweep)
def make_fine_grid(best):
    """Generate fine grid centered on best config."""
    def neighbors(val, factors=[0.5, 0.75, 1.0, 1.5, 2.0]):
        return sorted(set(val * f for f in factors))

    return {
        "Q_pos":    neighbors(best["Q_pos"], [0.5, 0.75, 1.0, 1.25, 1.5]),
        "Q_vel":    neighbors(best["Q_vel"], [0.25, 0.5, 1.0, 2.0, 4.0]),
        "R_thrust": neighbors(best["R_thrust"], [0.1, 0.5, 1.0, 2.0, 10.0]),
        "R_angle":  neighbors(best["R_angle"], [0.25, 0.5, 1.0, 2.0, 4.0]),
        "N":        [best["N"]],      # keep fixed in fine sweep
        "NC":       [best["NC"]],
    }


# ============================================================
# SINGLE EVALUATION
# ============================================================
def evaluate_config(config, model, sim, test_data_list, n_steps):
    """
    Run MPC closed-loop on multiple test trajectories.
    Returns average pos RMSE across all tests.
    """
    A_edmd   = model["A"]
    B_edmd   = model["B"]
    scaler   = model["scaler"]
    u_scaler = model["u_scaler"]
    dt       = model["dt"]
    n_obs    = model["n_obs"]

    Q_pos, Q_vel = config["Q_pos"], config["Q_vel"]
    R_thrust, R_angle = config["R_thrust"], config["R_angle"]
    N, NC = int(config["N"]), int(config["NC"])

    if NC > N:
        return float("inf"), {}

    Q_diag = np.array([
        Q_pos, Q_pos, Q_pos,
        Q_vel, Q_vel, Q_vel,
        0.0, 0.0,
        0.0, 0.0,
    ], dtype=float)

    R_diag  = np.array([R_thrust, R_angle, R_angle], dtype=float)
    Rd_diag = R_diag * 0.1

    Cz = np.zeros((10, n_obs))
    Cz[:10, :10] = np.eye(10)

    u_nominal = np.array([sim.q_mass * sim.g, 0.0, 0.0], dtype=float)

    try:
        mpc = EDMDcMPC_QP(
            A=A_edmd, B=B_edmd, Cz=Cz,
            N=N, NC=NC,
            Q=np.diag(Q_diag), R=np.diag(R_diag), Rd=np.diag(Rd_diag),
            u_scaler=u_scaler,
            du_min=DU_MIN_FIXED, du_max=DU_MAX_FIXED,
            u_nominal_raw=u_nominal,
        )
    except Exception as e:
        return float("inf"), {}

    per_traj_rmse = {}
    total_rmse_sum = 0.0

    for (t_ref, X_true, ref_traj, ref_xyz, label) in test_data_list:
        T_eval = min(n_steps, len(t_ref), X_true.shape[0])

        ref_std = precompute_ref_std(ref_traj[:T_eval], scaler, n_states=10)

        X_mpc = np.zeros((T_eval, 10))
        x_current_12 = np.zeros(12)
        x10_init = X_true[0]
        x_current_12[0:6]  = x10_init[0:6]
        x_current_12[6:8]  = x10_init[6:8]
        x_current_12[9:11] = x10_init[8:10]
        X_mpc[0] = drop_to_10state(x_current_12)

        for k in range(T_eval - 1):
            x10 = drop_to_10state(x_current_12)
            z_k = lifted_state_from_x(x10, scaler)
            x_ref_h = build_ref_horizon(ref_std, k, N)
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

        pos_rmse_traj = rmse(X_mpc[:, 0:3], ref_xyz[:T_eval])
        per_traj_rmse[label] = pos_rmse_traj
        total_rmse_sum += pos_rmse_traj

    avg_rmse = total_rmse_sum / len(test_data_list)
    return avg_rmse, per_traj_rmse


# ============================================================
# MAIN
# ============================================================
def main():
    # --- load model ---
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
    idx_ds = np.arange(0, t_all.shape[1], step)
    t_all      = t_all[:, idx_ds]
    states_all = states_all[:, idx_ds, :]
    U_all      = U_all[:, idx_ds, :]
    ref_traj_list = [r[::step] for r in ref_traj_list]

    sim = quad_sim()

    # --- prepare test data ---
    test_data_list = []
    print("Test trajectories:")
    for run_idx, label in zip(TEST_INDICES, TEST_LABELS):
        ri = run_idx % states_all.shape[0]
        t_ref   = t_all[ri]
        X_true  = states_all[ri]
        ref_traj = ref_traj_list[ri]
        ref_xyz = extract_ref_xyz(ref_traj)
        T = min(len(t_ref), X_true.shape[0], ref_xyz.shape[0])

        pid_rmse = rmse(X_true[:T, 0:3], ref_xyz[:T])
        print(f"  idx={run_idx:3d} ({label:>10s})  T={T}  PID_RMSE={pid_rmse:.4f} m")

        test_data_list.append((
            t_ref[:T], X_true[:T], ref_traj[:T], ref_xyz[:T], label
        ))

    # ====================================================
    # PHASE 1: COARSE SWEEP
    # ====================================================
    keys = list(GRID_COARSE.keys())
    values = list(GRID_COARSE.values())
    combos = list(itertools.product(*values))
    n_combos = len(combos)

    print(f"\n{'='*70}")
    print(f"PHASE 1: COARSE SWEEP — {n_combos} combos × {len(TEST_INDICES)} trajs × {FAST_STEPS} steps")
    print(f"{'='*70}")

    results = []
    t0_all = time.perf_counter()

    for idx_c, combo in enumerate(combos):
        config = dict(zip(keys, combo))

        # skip invalid NC > N
        if config["NC"] > config["N"]:
            continue

        t0 = time.perf_counter()
        avg_rmse, per_traj = evaluate_config(
            config, model, sim, test_data_list, n_steps=FAST_STEPS
        )
        elapsed = time.perf_counter() - t0

        results.append((avg_rmse, per_traj, config))

        if (idx_c + 1) % 50 == 0 or idx_c == 0:
            print(f"  [{idx_c+1:4d}/{n_combos}]  avg_RMSE={avg_rmse:8.4f}  "
                  f"N={config['N']:2d} NC={config['NC']:2d}  "
                  f"Q_pos={config['Q_pos']:>7.0f}  Q_vel={config['Q_vel']:>5.0f}  "
                  f"R_thr={config['R_thrust']:.4f}  R_ang={config['R_angle']:.3f}  "
                  f"({elapsed:.1f}s)")

    total_time = time.perf_counter() - t0_all
    results.sort(key=lambda r: r[0])

    print(f"\nCoarse sweep done in {total_time/60:.1f} min")

    # Print top 15
    print(f"\n{'='*70}")
    print(f"TOP 15 CONFIGS (coarse, {FAST_STEPS} steps, avg across {len(TEST_INDICES)} trajs)")
    print(f"{'='*70}")
    header = (f"{'#':>3s}  {'avg_RMSE':>9s}  {'N':>3s} {'NC':>3s}  "
              f"{'Q_pos':>8s}  {'Q_vel':>6s}  {'R_thr':>8s}  {'R_ang':>6s}")
    print(header)
    print("-" * len(header))
    for i, (ar, pt, cfg) in enumerate(results[:15]):
        detail = "  ".join(f"{lbl}={v:.3f}" for lbl, v in pt.items())
        print(f"{i+1:3d}  {ar:9.4f}  {cfg['N']:3d} {cfg['NC']:3d}  "
              f"{cfg['Q_pos']:8.0f}  {cfg['Q_vel']:6.0f}  "
              f"{cfg['R_thrust']:8.4f}  {cfg['R_angle']:6.3f}")
        print(f"     {detail}")

    best_coarse = results[0][2]

    # ====================================================
    # PHASE 2: FULL VALIDATION OF TOP K
    # ====================================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: FULL VALIDATION — top {TOP_K}")
    print(f"{'='*70}")

    full_results = []
    for i, (_, _, config) in enumerate(results[:TOP_K]):
        t0 = time.perf_counter()
        avg_rmse, per_traj = evaluate_config(
            config, model, sim, test_data_list, n_steps=1000
        )
        elapsed = time.perf_counter() - t0
        full_results.append((avg_rmse, per_traj, config))
        detail = "  ".join(f"{lbl}={v:.3f}" for lbl, v in per_traj.items())
        print(f"  [{i+1}/{TOP_K}] avg={avg_rmse:.4f}  ({elapsed:.1f}s)")
        print(f"    {detail}")

    full_results.sort(key=lambda r: r[0])

    best_full = full_results[0][2]

    # ====================================================
    # PHASE 3: FINE SWEEP AROUND BEST
    # ====================================================
    fine_grid = make_fine_grid(best_full)
    fine_keys = list(fine_grid.keys())
    fine_values = list(fine_grid.values())
    fine_combos = list(itertools.product(*fine_values))
    n_fine = len(fine_combos)

    print(f"\n{'='*70}")
    print(f"PHASE 3: FINE SWEEP — {n_fine} combos around best")
    print(f"{'='*70}")

    fine_results = []
    t0_all = time.perf_counter()

    for idx_c, combo in enumerate(fine_combos):
        config = dict(zip(fine_keys, combo))
        if config["NC"] > config["N"]:
            continue

        avg_rmse, per_traj = evaluate_config(
            config, model, sim, test_data_list, n_steps=FAST_STEPS
        )
        fine_results.append((avg_rmse, per_traj, config))

        if (idx_c + 1) % 100 == 0:
            print(f"  [{idx_c+1}/{n_fine}]  avg_RMSE={avg_rmse:.4f}")

    fine_results.sort(key=lambda r: r[0])
    fine_time = time.perf_counter() - t0_all
    print(f"Fine sweep done in {fine_time/60:.1f} min")

    # Validate top 5 fine on full run
    print(f"\nFine validation (full run):")
    final_results = []
    for i, (_, _, config) in enumerate(fine_results[:5]):
        avg_rmse, per_traj = evaluate_config(
            config, model, sim, test_data_list, n_steps=1000
        )
        final_results.append((avg_rmse, per_traj, config))
        detail = "  ".join(f"{lbl}={v:.3f}" for lbl, v in per_traj.items())
        print(f"  [{i+1}/5] avg={avg_rmse:.4f}  {detail}")

    final_results.sort(key=lambda r: r[0])

    # ====================================================
    # FINAL REPORT
    # ====================================================
    print(f"\n{'='*70}")
    print(f"FINAL RANKING")
    print(f"{'='*70}")
    for i, (ar, pt, cfg) in enumerate(final_results):
        print(f"\n  #{i+1}  avg_RMSE = {ar:.4f}")
        for lbl, v in pt.items():
            print(f"    {lbl:>12s}: {v:.4f} m")
        print(f"    Config: N={cfg['N']} NC={cfg['NC']} "
              f"Q_pos={cfg['Q_pos']} Q_vel={cfg['Q_vel']} "
              f"R_thrust={cfg['R_thrust']} R_angle={cfg['R_angle']}")

    best = final_results[0][2]
    Rd_t = best['R_thrust'] * 0.1
    Rd_a = best['R_angle'] * 0.1

    print(f"\n{'='*70}")
    print(f"PASTE INTO EDMDc_MPC.py:")
    print(f"{'='*70}")
    print(f"N_EDMD  = {best['N']}")
    print(f"NC_EDMD = {best['NC']}")
    print(f"")
    print(f"Q_DIAG_EDMD = np.array([")
    print(f"    {best['Q_pos']}, {best['Q_pos']}, {best['Q_pos']},")
    print(f"    {best['Q_vel']}, {best['Q_vel']}, {best['Q_vel']},")
    print(f"    0.0, 0.0,")
    print(f"    0.0, 0.0,")
    print(f"], dtype=float)")
    print(f"")
    print(f"R_DIAG_EDMD  = np.array([{best['R_thrust']}, {best['R_angle']}, {best['R_angle']}], dtype=float)")
    print(f"RD_DIAG_EDMD = np.array([{Rd_t}, {Rd_a}, {Rd_a}], dtype=float)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()