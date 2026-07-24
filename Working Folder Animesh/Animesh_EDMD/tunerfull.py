"""
tune_mpc_full.py
================
Full MPC hyperparameter sweep: Q, R, N, NC.
Parallelized across CPU cores.

Phase 1: coarse sweep on 300 steps
Phase 2: validate top K on full run
Phase 3: fine sweep around best config
"""

import pickle
import time
import itertools
import multiprocessing as mp
from pathlib import Path

import numpy as np

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR       = Path(__file__).resolve().parent
EDMDC_MODEL_FILE = "edmdc_model_300.pkl"
DATA_FILE        = "runs_mixed_n300.pkl"

TEST_INDICES = [39, 59, 129]
TEST_LABELS  = ["helix", "fig8", "lissajous"]

RUN_FOCUSED_FULL_ONLY = True
FAST_STEPS = 500
VALIDATION_STEPS = 10000
FINAL_VALIDATION_STEPS = 10000
TOP_K      = 5
FINAL_TOP_K = 5
MAX_WORKERS = 8

DU_MIN_FIXED = np.array([-0.1, -0.01, -0.01], dtype=float)
DU_MAX_FIXED = np.array([ 0.1,  0.01,  0.01], dtype=float)
DU_YAW_FIXED = 0.0
Q_Y_MULT_FIXED = 1.6
Q_VY_MULT_FIXED = 1.6
Q_YAW_FIXED = 0.0
Q_R_FIXED = 0.0
YAW_R_FIXED = 0.25
YAW_RD_FIXED = 0.025
SCORE_YAW_WEIGHT = 0.25
SCORE_R_WEIGHT = 0.03
SCORE_PID_RMSE_FLOOR = 0.5
SCORE_WORST_TRAJ_WEIGHT = 0.15
LISSAJOUS_RMSE_LIMIT = 8.0
LISSAJOUS_LIMIT_PENALTY = 2.0
USE_CONSTANT_YAW_REFERENCE = True
CONSTANT_YAW = 0.0

GRID_COARSE = {
    "Q_pos":    [250000, 300000, 350000, 400000],
    "Q_vel":    [25, 35, 50],
    "R_thrust": [2.5e-05, 5.0e-05, 0.0001],
    "R_angle":  [0.25, 0.40, 0.50, 0.75],
    "N":        [15, 20],
    "NC":       [15],
}

FOCUSED_FULL_CONFIGS = [
    # Current/previous manually useful blocks.
    {"Q_pos": 250000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.25, "N": 15, "NC": 15},
    {"Q_pos": 400000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.75, "N": 20, "NC": 15},
    {"Q_pos": 400000, "Q_vel": 50, "R_thrust": 2.5e-05, "R_angle": 0.50, "N": 20, "NC": 15},

    # N=15 local checks around the current block.
    {"Q_pos": 250000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.40, "N": 15, "NC": 15},
    {"Q_pos": 250000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.50, "N": 15, "NC": 15},

    # N=20 checks intended to recover lissajous without ruining fig8.
    {"Q_pos": 250000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.25, "N": 20, "NC": 15},
    {"Q_pos": 250000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.40, "N": 20, "NC": 15},
    {"Q_pos": 250000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.50, "N": 20, "NC": 15},
    {"Q_pos": 250000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.75, "N": 20, "NC": 15},
    {"Q_pos": 300000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.50, "N": 20, "NC": 15},
    {"Q_pos": 300000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.75, "N": 20, "NC": 15},
    {"Q_pos": 350000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.50, "N": 20, "NC": 15},
    {"Q_pos": 350000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.75, "N": 20, "NC": 15},
    {"Q_pos": 400000, "Q_vel": 25, "R_thrust": 0.0001,  "R_angle": 0.50, "N": 20, "NC": 15},
    {"Q_pos": 400000, "Q_vel": 35, "R_thrust": 0.0001,  "R_angle": 0.50, "N": 20, "NC": 15},
    {"Q_pos": 400000, "Q_vel": 35, "R_thrust": 0.0001,  "R_angle": 0.75, "N": 20, "NC": 15},
    {"Q_pos": 400000, "Q_vel": 25, "R_thrust": 5.0e-05, "R_angle": 0.50, "N": 20, "NC": 15},
    {"Q_pos": 400000, "Q_vel": 25, "R_thrust": 5.0e-05, "R_angle": 0.75, "N": 20, "NC": 15},
]


def make_fine_grid(best):
    def neighbors(val, factors):
        return sorted(set(val * f for f in factors))
    return {
        "Q_pos":    neighbors(best["Q_pos"], [0.8, 1.0, 1.2]),
        "Q_vel":    neighbors(best["Q_vel"], [0.75, 1.0, 1.25]),
        "R_thrust": neighbors(best["R_thrust"], [0.5, 1.0, 2.0]),
        "R_angle":  neighbors(best["R_angle"], [0.75, 1.0, 1.25]),
        "N":        [best["N"]],
        "NC":       [best["NC"]],
    }


def apply_yaw_reference_mode(ref_traj):
    if not USE_CONSTANT_YAW_REFERENCE:
        return ref_traj
    return [
        {**wp, "yaw": float(CONSTANT_YAW), "yaw_rate": 0.0}
        for wp in ref_traj
    ]


# ============================================================
# WORKER FUNCTION (runs in subprocess)
# ============================================================
def evaluate_single(args):
    """
    Evaluate one config. Model and sim recreated inside worker
    to avoid pickling issues.
    """
    config, test_data_list, model_file, n_steps = args

    import numpy as np
    from Simulation import quad_sim
    from edmdc_mpc import (
        EDMDcMPC_QP, load_edmdc_model, lifted_state_from_x,
        drop_to_12state, precompute_ref_std, build_ref_horizon,
        reference_yaw_arrays, wrap_angle_pi, rmse,
    )

    model    = load_edmdc_model(model_file)
    A_edmd   = model["A"]
    B_edmd   = model["B"]
    scaler   = model["scaler"]
    u_scaler = model["u_scaler"]
    dt       = model["dt"]
    n_obs    = model["n_obs"]
    model_input_dim = B_edmd.shape[1]
    input_lift_type = model.get("input_lift_type")
    if input_lift_type == "thrust_direction":
        input_dim = int(model.get("raw_input_dim", 4))
    else:
        input_dim = model_input_dim

    sim = quad_sim()
    nominal_sim = quad_sim()

    Q_pos, Q_vel = config["Q_pos"], config["Q_vel"]
    R_thrust, R_angle = config["R_thrust"], config["R_angle"]
    N, NC = int(config["N"]), int(config["NC"])

    if NC > N:
        return config, float("inf"), {}

    Q_diag = np.array([
        Q_pos, Q_Y_MULT_FIXED * Q_pos, Q_pos,
        Q_vel, Q_VY_MULT_FIXED * Q_vel, Q_vel,
        0.0, 0.0, Q_YAW_FIXED,
        0.0, 0.0, Q_R_FIXED,
    ], dtype=float)
    R_diag  = np.array([R_thrust, R_angle, R_angle], dtype=float)
    Rd_diag = R_diag * 0.1
    if input_dim == 4:
        R_diag = np.r_[R_diag, YAW_R_FIXED]
        Rd_diag = np.r_[Rd_diag, YAW_RD_FIXED]
        du_min = np.r_[DU_MIN_FIXED, -DU_YAW_FIXED]
        du_max = np.r_[DU_MAX_FIXED,  DU_YAW_FIXED]
    elif input_dim == 3:
        du_min = DU_MIN_FIXED
        du_max = DU_MAX_FIXED
    else:
        return config, float("inf"), {}

    Cz = np.zeros((12, n_obs))
    Cz[:12, :12] = np.eye(12)
    u_nominal = np.zeros(input_dim, dtype=float)
    u_nominal[:3] = [sim.q_mass * sim.g, 0.0, 0.0]

    try:
        mpc = EDMDcMPC_QP(
            A=A_edmd, B=B_edmd, Cz=Cz,
            N=N, NC=NC,
            Q=np.diag(Q_diag), R=np.diag(R_diag), Rd=np.diag(Rd_diag),
            u_scaler=u_scaler,
            du_min=du_min, du_max=du_max,
            u_nominal_raw=u_nominal,
            state_scaler=scaler,
            input_lift_type=input_lift_type,
            raw_input_dim=input_dim,
        )
    except Exception:
        return config, float("inf"), {}

    per_traj_rmse = {}
    traj_scores = []

    for (t_ref, X_true, ref_traj_dicts, ref_xyz, label, pid_rmse_ref) in test_data_list:
        T_eval = min(n_steps, len(t_ref), X_true.shape[0])
        ref_std = precompute_ref_std(ref_traj_dicts[:T_eval], scaler, dt=dt)
        sim.controller_PID.fct_reset()
        nominal_sim.controller_PID.fct_reset()

        X_mpc = np.zeros((T_eval, 12))
        x_current_12 = drop_to_12state(X_true[0])
        X_mpc[0] = drop_to_12state(x_current_12)

        for k in range(T_eval - 1):
            x12 = drop_to_12state(x_current_12)
            z_k = lifted_state_from_x(x12, scaler)
            x_ref_h = build_ref_horizon(ref_std, k, N)
            u_nom = np.zeros(input_dim, dtype=float)
            _, _, u_att_nom = nominal_sim.controller_PID.fct_step(
                x_current_12, ref_traj_dicts[k], dt
            )
            u_nom[:3] = u_att_nom[:3]
            if input_dim == 4:
                u_nom[3] = u_att_nom[3]
            u_cmd = mpc.compute(z_k, x_ref_h, u_nominal_raw=u_nom)

            u_cmd[0] = np.clip(u_cmd[0], 0.5 * sim.q_mass * sim.g,
                                          2.0 * sim.q_mass * sim.g)
            u_cmd[1] = np.clip(u_cmd[1], -sim.controller_PID.tilt_max,
                                           sim.controller_PID.tilt_max)
            u_cmd[2] = np.clip(u_cmd[2], -sim.controller_PID.tilt_max,
                                           sim.controller_PID.tilt_max)
            if input_dim == 4:
                psi_des = u_cmd[3]
            else:
                psi_des = u_att_nom[3]

            x_next_12 = sim.sim_PID.fct_step_attitude(
                x_current_12,
                u1=u_cmd[0], phi_des=u_cmd[1], theta_des=u_cmd[2],
                dt=dt,
                psi_des=psi_des,
            )
            x_current_12 = x_next_12
            X_mpc[k + 1] = drop_to_12state(x_next_12)

        pos_rmse_traj = rmse(X_mpc[:, 0:3], ref_xyz[:T_eval])
        pos_score = pos_rmse_traj / max(float(pid_rmse_ref), SCORE_PID_RMSE_FLOOR)
        ref_yaw, ref_r = reference_yaw_arrays(ref_traj_dicts[:T_eval], dt=dt)
        yaw_err = wrap_angle_pi(X_mpc[:, 8] - ref_yaw)
        yaw_rmse_traj = float(np.sqrt(np.mean(yaw_err**2)))
        r_rmse_traj = rmse(X_mpc[:, 11], ref_r)
        traj_score = (
            pos_score
            + SCORE_YAW_WEIGHT * yaw_rmse_traj
            + SCORE_R_WEIGHT * r_rmse_traj
        )
        if label.lower().startswith("liss"):
            liss_excess = max(0.0, pos_rmse_traj - LISSAJOUS_RMSE_LIMIT)
            traj_score += LISSAJOUS_LIMIT_PENALTY * (
                liss_excess / LISSAJOUS_RMSE_LIMIT
            )
        per_traj_rmse[label] = pos_rmse_traj
        traj_scores.append(traj_score)

    avg_score = float(np.mean(traj_scores))
    worst_score = float(np.max(traj_scores))
    avg_rmse = avg_score + SCORE_WORST_TRAJ_WEIGHT * worst_score
    return config, avg_rmse, per_traj_rmse


# ============================================================
# PARALLEL SWEEP HELPER
# ============================================================
def parallel_sweep(configs, test_data, model_file, n_steps,
                   n_workers, phase_name="Sweep"):
    n_valid = len(configs)
    n_workers = max(1, min(int(n_workers), n_valid))
    print(f"\n{phase_name}: {n_valid} configs on {n_workers} cores, {n_steps} steps each")

    args_list = [
        (cfg, test_data, str(model_file), n_steps)
        for cfg in configs
    ]

    t0 = time.perf_counter()
    with mp.Pool(n_workers) as pool:
        results_raw = pool.map(evaluate_single, args_list)
    elapsed = time.perf_counter() - t0

    results = [(avg, per, cfg) for cfg, avg, per in results_raw]
    results.sort(key=lambda r: r[0])

    print(f"  Done in {elapsed/60:.1f} min ({elapsed/max(n_valid,1):.2f}s per combo)")
    return results


def grid_to_configs(grid):
    """Convert grid dict to list of config dicts, filtering NC > N."""
    keys = list(grid.keys())
    values = list(grid.values())
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        if cfg.get("NC", 0) <= cfg.get("N", float("inf")):
            configs.append(cfg)
    return configs


# ============================================================
# MAIN
# ============================================================
def main():
    from edmdc_mpc import load_edmdc_model, load_simulation_runs, extract_ref_xyz, rmse

    n_workers = min(MAX_WORKERS, mp.cpu_count())
    print(f"CPU cores: {mp.cpu_count()}  using workers: {n_workers}")

    model_file = SCRIPT_DIR / EDMDC_MODEL_FILE
    data_file  = SCRIPT_DIR / DATA_FILE

    model = load_edmdc_model(model_file)
    dt = model["dt"]
    print(f"Model dt: {dt}")

    # --- Load and downsample ---
    t_all, states_all, U_all, ref_traj_list = load_simulation_runs(data_file)

    if U_all.shape[2] == 4:
        U_all = U_all[:, :, :3]

    sim_dt = t_all[0, 1] - t_all[0, 0]
    step = int(round(dt / sim_dt))
    idx_ds = np.arange(0, t_all.shape[1], step)
    t_all      = t_all[:, idx_ds]
    states_all = states_all[:, idx_ds, :]
    U_all      = U_all[:, idx_ds, :]
    ref_traj_list = [r[::step] for r in ref_traj_list]

    # --- Prepare test data ---
    test_data = []
    print("\nTest trajectories:")
    for run_idx, label in zip(TEST_INDICES, TEST_LABELS):
        ri = run_idx % states_all.shape[0]
        t_ref    = t_all[ri]
        X_true   = states_all[ri]
        ref_traj = apply_yaw_reference_mode(ref_traj_list[ri])
        ref_xyz  = extract_ref_xyz(ref_traj)
        T = min(len(t_ref), X_true.shape[0], ref_xyz.shape[0])

        pid_rmse = rmse(X_true[:T, 0:3], ref_xyz[:T])
        print(f"  idx={run_idx:3d} ({label:>10s})  T={T}  PID_RMSE={pid_rmse:.4f} m")

        test_data.append((
            t_ref[:T], X_true[:T], ref_traj[:T], ref_xyz[:T], label, pid_rmse
        ))

    if RUN_FOCUSED_FULL_ONLY:
        seen = set()
        focused_configs = []
        for cfg in FOCUSED_FULL_CONFIGS:
            key = tuple((k, cfg[k]) for k in ("Q_pos", "Q_vel", "R_thrust", "R_angle", "N", "NC"))
            if key not in seen and cfg["NC"] <= cfg["N"]:
                seen.add(key)
                focused_configs.append(cfg)

        final_results = parallel_sweep(
            focused_configs, test_data, model_file,
            n_steps=FINAL_VALIDATION_STEPS,
            n_workers=min(n_workers, len(focused_configs)),
            phase_name="FOCUSED FULL VALIDATION"
        )

        print(f"\n{'='*70}")
        print(f"FINAL RANKING")
        print(
            "avg_score uses EDMDc position RMSE normalized by the PID baseline "
            f"(floor={SCORE_PID_RMSE_FLOOR:.2f} m), plus yaw/r penalties, "
            f"{SCORE_WORST_TRAJ_WEIGHT:.2f}x worst-trajectory score, and a "
            f"lissajous penalty above {LISSAJOUS_RMSE_LIMIT:.1f} m."
        )
        print(f"{'='*70}")
        for i, (ar, pt, cfg) in enumerate(final_results[:10]):
            print(f"\n  #{i+1}  avg_score = {ar:.4f}")
            for lbl, v in pt.items():
                print(f"    {lbl:>12s}: {v:.4f} m")
            print(f"    Config: N={cfg['N']} NC={cfg['NC']} "
                  f"Q_pos={cfg['Q_pos']} Q_vel={cfg['Q_vel']} "
                  f"R_thrust={cfg['R_thrust']} R_angle={cfg['R_angle']}")

        best = final_results[0][2]
        Rd_t = best['R_thrust'] * 0.1
        Rd_a = best['R_angle'] * 0.1

        print(f"\n{'='*70}")
        print(f"PASTE INTO compare_three.py / final_comparison.py:")
        print(f"{'='*70}")
        print(f"N_MPC  = {best['N']}")
        print(f"NC_MPC = {best['NC']}")
        print(f"")
        print(f"Q_DIAG = np.array([")
        print(f"    {best['Q_pos']}, {Q_Y_MULT_FIXED * best['Q_pos']}, {best['Q_pos']},")
        print(f"    {best['Q_vel']}, {Q_VY_MULT_FIXED * best['Q_vel']}, {best['Q_vel']},")
        print(f"    0.0, 0.0, {Q_YAW_FIXED},")
        print(f"    0.0, 0.0, {Q_R_FIXED},")
        print(f"], dtype=float)")
        print(f"")
        print(f"R_DIAG  = np.array([{best['R_thrust']}, {best['R_angle']}, {best['R_angle']}], dtype=float)")
        print(f"RD_DIAG = np.array([{Rd_t}, {Rd_a}, {Rd_a}], dtype=float)")
        print(f"R_YAW   = {YAW_R_FIXED}")
        print(f"RD_YAW  = {YAW_RD_FIXED}")
        print(f"DU_YAW  = {DU_YAW_FIXED}")
        print(f"{'='*70}")
        return

    # ====================================================
    # PHASE 1: COARSE SWEEP
    # ====================================================
    coarse_configs = grid_to_configs(GRID_COARSE)

    results = parallel_sweep(
        coarse_configs, test_data, model_file,
        n_steps=FAST_STEPS, n_workers=n_workers,
        phase_name="PHASE 1 (COARSE)"
    )

    print(f"\n{'='*70}")
    print(f"TOP 15 CONFIGS (coarse, {FAST_STEPS} steps)")
    print(f"{'='*70}")
    header = f"{'#':>3s}  {'avg_score':>9s}  {'N':>3s} {'NC':>3s}  {'Q_pos':>8s}  {'Q_vel':>6s}  {'R_thr':>8s}  {'R_ang':>6s}"
    print(header)
    print("-" * len(header))
    for i, (ar, pt, cfg) in enumerate(results[:15]):
        detail = "  ".join(f"{lbl}={v:.3f}" for lbl, v in pt.items())
        print(f"{i+1:3d}  {ar:9.4f}  {cfg['N']:3d} {cfg['NC']:3d}  "
              f"{cfg['Q_pos']:8.0f}  {cfg['Q_vel']:6.0f}  "
              f"{cfg['R_thrust']:8.4f}  {cfg['R_angle']:6.3f}")
        print(f"     {detail}")

    # ====================================================
    # PHASE 2: FULL VALIDATION OF TOP K
    # ====================================================
    top_configs = [cfg for _, _, cfg in results[:TOP_K]]

    full_results = parallel_sweep(
        top_configs, test_data, model_file,
        n_steps=VALIDATION_STEPS, n_workers=min(n_workers, TOP_K),
        phase_name="PHASE 2 (FULL VALIDATION)"
    )

    print(f"\n{'='*70}")
    print(f"FULL VALIDATION TOP {TOP_K}")
    print(f"{'='*70}")
    for i, (ar, pt, cfg) in enumerate(full_results[:TOP_K]):
        detail = "  ".join(f"{lbl}={v:.3f}" for lbl, v in pt.items())
        print(f"  [{i+1}] avg={ar:.4f}  {detail}")

    best_full = full_results[0][2]

    # ====================================================
    # PHASE 3: FINE SWEEP AROUND BEST
    # ====================================================
    fine_grid = make_fine_grid(best_full)
    fine_configs = grid_to_configs(fine_grid)

    fine_results = parallel_sweep(
        fine_configs, test_data, model_file,
        n_steps=FAST_STEPS, n_workers=n_workers,
        phase_name="PHASE 3 (FINE SWEEP)"
    )

    # Validate top 5
    top_fine = [cfg for _, _, cfg in fine_results[:FINAL_TOP_K]]

    final_results = parallel_sweep(
        top_fine, test_data, model_file,
        n_steps=FINAL_VALIDATION_STEPS, n_workers=min(n_workers, FINAL_TOP_K),
        phase_name="PHASE 3 VALIDATION"
    )

    # ====================================================
    # FINAL REPORT
    # ====================================================
    print(f"\n{'='*70}")
    print(f"FINAL RANKING")
    print(
        "avg_score uses EDMDc position RMSE normalized by the PID baseline "
        f"(floor={SCORE_PID_RMSE_FLOOR:.2f} m), plus yaw/r penalties, "
        f"{SCORE_WORST_TRAJ_WEIGHT:.2f}x worst-trajectory score, and a "
        f"lissajous penalty above {LISSAJOUS_RMSE_LIMIT:.1f} m."
    )
    print(f"{'='*70}")
    for i, (ar, pt, cfg) in enumerate(final_results[:5]):
        print(f"\n  #{i+1}  avg_score = {ar:.4f}")
        for lbl, v in pt.items():
            print(f"    {lbl:>12s}: {v:.4f} m")
        print(f"    Config: N={cfg['N']} NC={cfg['NC']} "
              f"Q_pos={cfg['Q_pos']} Q_vel={cfg['Q_vel']} "
              f"R_thrust={cfg['R_thrust']} R_angle={cfg['R_angle']}")

    best = final_results[0][2]
    Rd_t = best['R_thrust'] * 0.1
    Rd_a = best['R_angle'] * 0.1

    print(f"\n{'='*70}")
    print(f"PASTE INTO compare_three.py / final_comparison.py:")
    print(f"{'='*70}")
    print(f"N_MPC  = {best['N']}")
    print(f"NC_MPC = {best['NC']}")
    print(f"")
    print(f"Q_DIAG = np.array([")
    print(f"    {best['Q_pos']}, {Q_Y_MULT_FIXED * best['Q_pos']}, {best['Q_pos']},")
    print(f"    {best['Q_vel']}, {Q_VY_MULT_FIXED * best['Q_vel']}, {best['Q_vel']},")
    print(f"    0.0, 0.0, {Q_YAW_FIXED},")
    print(f"    0.0, 0.0, {Q_R_FIXED},")
    print(f"], dtype=float)")
    print(f"")
    print(f"R_DIAG  = np.array([{best['R_thrust']}, {best['R_angle']}, {best['R_angle']}], dtype=float)")
    print(f"RD_DIAG = np.array([{Rd_t}, {Rd_a}, {Rd_a}], dtype=float)")
    print(f"R_YAW   = {YAW_R_FIXED}")
    print(f"RD_YAW  = {YAW_RD_FIXED}")
    print(f"DU_YAW  = {DU_YAW_FIXED}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
