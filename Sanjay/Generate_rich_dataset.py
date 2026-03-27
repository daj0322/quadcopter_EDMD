import pickle
import random
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from Simulation import quad_sim


# ============================================================
# CONFIG
# ============================================================
SOURCE_FIG8_FILE = "runs_traj2_n200.pkl"
OUTPUT_FILE = "runs_rich_mixed_n600.pkl"

# how many runs to generate in each family
N_NOMINAL_FIG8 = 120
N_PERTURBED_FIG8 = 160
N_HOVER_LOCAL = 140
N_LOCAL_PROBING = 180

# local probing settings
LOCAL_PROBE_DURATION = 2.0   # seconds of meaningful probing
PROBE_SEGMENT_SEC = 0.12     # piecewise-constant control segment length


# ============================================================
# LOADERS
# ============================================================
def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


# ============================================================
# REFERENCE HELPERS
# ============================================================
def smooth_random_signal(time, rng, amp, n_sines_range=(2, 5), freq_range=(0.03, 0.25)):
    time = np.asarray(time, dtype=float)
    y = np.zeros_like(time)
    n_sines = rng.randint(n_sines_range[0], n_sines_range[1])

    for _ in range(n_sines):
        A = rng.uniform(0.3 * amp, amp) / n_sines
        f = rng.uniform(freq_range[0], freq_range[1])
        w = 2.0 * np.pi * f
        ph = rng.uniform(0.0, 2.0 * np.pi)
        y += A * np.sin(w * time + ph)

    return y


def perturb_reference_traj(ref_traj, time, rng,
                           pos_amp=(0.12, 0.12, 0.08),
                           yaw_amp_deg=4.0):
    """
    Add smooth perturbations to an existing reference trajectory.
    Keeps the same structure: list of dicts with pos / vel / yaw.
    """
    time = np.asarray(time, dtype=float)

    dx = smooth_random_signal(time, rng, pos_amp[0])
    dy = smooth_random_signal(time, rng, pos_amp[1])
    dz = smooth_random_signal(time, rng, pos_amp[2])

    yaw_amp = np.deg2rad(yaw_amp_deg)
    dyaw = smooth_random_signal(time, rng, yaw_amp, n_sines_range=(1, 3), freq_range=(0.02, 0.10))

    out = []
    for k, r in enumerate(ref_traj):
        pos = np.asarray(r["pos"], dtype=float).copy()
        vel = np.asarray(r["vel"], dtype=float).copy()
        yaw = float(r["yaw"])

        pos_pert = pos + np.array([dx[k], dy[k], dz[k]], dtype=float)
        yaw_pert = yaw + dyaw[k]

        out.append({
            "pos": pos_pert,
            "vel": vel,
            "yaw": yaw_pert,
        })

    return out


def make_hover_excitation_reference(time, rng,
                                    xyz_amp=(0.15, 0.15, 0.10),
                                    yaw_amp_deg=5.0):
    """
    Smooth local reference near hover / origin.
    """
    time = np.asarray(time, dtype=float)
    T = len(time)

    dx = smooth_random_signal(time, rng, xyz_amp[0], n_sines_range=(2, 4), freq_range=(0.03, 0.16))
    dy = smooth_random_signal(time, rng, xyz_amp[1], n_sines_range=(2, 4), freq_range=(0.03, 0.16))
    dz = smooth_random_signal(time, rng, xyz_amp[2], n_sines_range=(2, 4), freq_range=(0.04, 0.20))

    dz = dz - dz[0] + 0.10

    yaw_amp = np.deg2rad(yaw_amp_deg)
    dyaw = smooth_random_signal(time, rng, yaw_amp, n_sines_range=(1, 3), freq_range=(0.02, 0.08))

    vx = np.gradient(dx, time[1] - time[0])
    vy = np.gradient(dy, time[1] - time[0])
    vz = np.gradient(dz, time[1] - time[0])

    ref = []
    for k in range(T):
        ref.append({
            "pos": np.array([dx[k], dy[k], dz[k]], dtype=float),
            "vel": np.array([vx[k], vy[k], vz[k]], dtype=float),
            "yaw": float(dyaw[k]),
        })

    p0 = ref[0]["pos"].copy()
    for k in range(T):
        ref[k]["pos"] = ref[k]["pos"] - p0

    return ref


def make_constant_anchor_reference(time, anchor_state):
    """
    Constant position/yaw reference around an anchor state.
    Used for local probing runs.
    """
    pos0 = np.asarray(anchor_state[:3], dtype=float)
    yaw0 = float(anchor_state[8])

    ref = []
    for _ in range(len(time)):
        ref.append({
            "pos": pos0.copy(),
            "vel": np.zeros(3),
            "yaw": yaw0,
        })
    return ref


def sample_initial_state(rng,
                         pos_std=(0.08, 0.08, 0.05),
                         vel_std=(0.05, 0.05, 0.04),
                         ang_std=(0.04, 0.04, 0.03),
                         rate_std=(0.08, 0.08, 0.04)):
    x0 = np.zeros(12)

    x0[0] = rng.gauss(0.0, pos_std[0])
    x0[1] = rng.gauss(0.0, pos_std[1])
    x0[2] = max(-0.05, rng.gauss(0.0, pos_std[2]))

    x0[3] = rng.gauss(0.0, vel_std[0])
    x0[4] = rng.gauss(0.0, vel_std[1])
    x0[5] = rng.gauss(0.0, vel_std[2])

    x0[6] = rng.gauss(0.0, ang_std[0])   # phi
    x0[7] = rng.gauss(0.0, ang_std[1])   # theta
    x0[8] = rng.gauss(0.0, ang_std[2])   # psi

    x0[9] = rng.gauss(0.0, rate_std[0])  # p
    x0[10] = rng.gauss(0.0, rate_std[1]) # q
    x0[11] = rng.gauss(0.0, rate_std[2]) # r

    return x0


# ============================================================
# LOCAL PROBING VIA TRUE NONLINEAR MODEL
# ============================================================
def generalized_input_to_rotor_speeds(quad, u):
    """
    Convert generalized input u = [u1,u2,u3,u4]
    to rotor speeds omega.
    """
    u1, u2, u3, u4 = [float(v) for v in u]

    arm = quad.l / np.sqrt(2.0)
    gamma = quad.kD / quad.kT

    A = np.array([
        [1.0,   1.0,   1.0,   1.0],
        [-arm, -arm,   arm,   arm],
        [ arm, -arm,  -arm,   arm],
        [gamma, -gamma, gamma, -gamma],
    ], dtype=float)

    b = np.array([u1, u2, u3, u4], dtype=float)
    T = np.linalg.solve(A, b)
    T = np.clip(T, 0.0, None)

    eff = getattr(quad, "prop_efficiency", np.ones(4))
    eff = np.asarray(eff, dtype=float)
    eff = np.clip(eff, 1e-8, None)

    omega_sq = T / (quad.kT * eff)
    omega_sq = np.clip(omega_sq, 0.0, None)
    omega = np.sqrt(omega_sq)

    return omega


def simulate_open_loop_generalized_inputs(sim, x0, U_seq):
    dt = sim.dt
    time_local = np.arange(U_seq.shape[0]) * dt

    state = np.array(x0, dtype=float)
    states = np.zeros((U_seq.shape[0], 12), dtype=float)

    for k in range(U_seq.shape[0]):
        u_k = U_seq[k]
        omega_k = generalized_input_to_rotor_speeds(sim.quad, u_k)

        def ode(t_local, s_local):
            return sim.quad.fct_dynamics(t_local, s_local, omega_k)

        sol = solve_ivp(
            ode,
            [0.0, dt],
            state,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        state = sol.y[:, -1]
        states[k] = state

    return time_local, states


def build_local_probe_input_sequence(u_nom, n_steps, dt, rng):
    """
    Piecewise-constant generalized inputs around an anchor nominal input.
    Designed to excite local state-input sensitivity.
    """
    U_seq = np.zeros((n_steps, 4), dtype=float)

    seg_len = max(1, int(round(PROBE_SEGMENT_SEC / dt)))

    k = 0
    while k < n_steps:
        du = np.array([
            rng.uniform(-0.14, 0.14),    # thrust
            rng.uniform(-0.012, 0.012),  # roll torque
            rng.uniform(-0.012, 0.012),  # pitch torque
            rng.uniform(-0.0035, 0.0035) # yaw torque
        ], dtype=float)

        u_cmd = np.array(u_nom, dtype=float) + du
        u_cmd[0] = max(0.10, u_cmd[0])

        k_end = min(n_steps, k + seg_len)
        U_seq[k:k_end] = u_cmd
        k = k_end

    return U_seq


# ============================================================
# MAIN GENERATOR
# ============================================================
if __name__ == "__main__":
    source = load_simulation_runs(SOURCE_FIG8_FILE)

    sim = quad_sim()
    time = np.asarray(source["time"], dtype=float)
    dt = float(source["sim_dt"])
    T = len(time)

    if not np.isclose(dt, sim.dt):
        print(f"Warning: source dt={dt} but quad_sim dt={sim.dt}; using quad_sim dt")

    t_runs = []
    states_runs = []
    U_runs = []
    ref_traj_list = []
    family_labels = []

    # ----------------------------
    # Family 1: nominal figure-8 subset
    # ----------------------------
    n_nom = min(N_NOMINAL_FIG8, int(source["n"]))
    for i in range(n_nom):
        t_runs.append(np.asarray(source["t"][i], dtype=float))
        states_runs.append(np.asarray(source["states"][i], dtype=float))
        U_runs.append(np.asarray(source["U"][i], dtype=float))
        ref_traj_list.append(source["ref_traj_list"][i])
        family_labels.append("fig8_nominal")

    # ----------------------------
    # Family 2: perturbed figure-8
    # ----------------------------
    for i in range(N_PERTURBED_FIG8):
        src_idx = i % int(source["n"])
        rng = random.Random(20000 + i)

        ref_base = source["ref_traj_list"][src_idx]
        ref_pert = perturb_reference_traj(
            ref_base,
            time,
            rng=rng,
            pos_amp=(
                rng.uniform(0.05, 0.16),
                rng.uniform(0.05, 0.16),
                rng.uniform(0.03, 0.10),
            ),
            yaw_amp_deg=rng.uniform(1.0, 6.0),
        )

        x0 = sample_initial_state(
            rng,
            pos_std=(0.06, 0.06, 0.04),
            vel_std=(0.04, 0.04, 0.03),
            ang_std=(0.03, 0.03, 0.02),
            rate_std=(0.06, 0.06, 0.03),
        )

        t_i, states_i, omegas_i, U_i = sim.sim_PID.fct_simulate(
            sim.time, sim.dt, ref_pert, x0
        )

        t_runs.append(np.asarray(t_i, dtype=float))
        states_runs.append(np.asarray(states_i, dtype=float))
        U_runs.append(np.asarray(U_i, dtype=float))
        ref_traj_list.append(ref_pert)
        family_labels.append("fig8_perturbed")

    # ----------------------------
    # Family 3: hover/local excitation
    # ----------------------------
    for i in range(N_HOVER_LOCAL):
        rng = random.Random(30000 + i)

        ref_hover = make_hover_excitation_reference(
            time,
            rng=rng,
            xyz_amp=(
                rng.uniform(0.05, 0.18),
                rng.uniform(0.05, 0.18),
                rng.uniform(0.04, 0.12),
            ),
            yaw_amp_deg=rng.uniform(1.5, 7.0),
        )

        x0 = sample_initial_state(
            rng,
            pos_std=(0.05, 0.05, 0.03),
            vel_std=(0.03, 0.03, 0.02),
            ang_std=(0.025, 0.025, 0.02),
            rate_std=(0.05, 0.05, 0.03),
        )

        t_i, states_i, omegas_i, U_i = sim.sim_PID.fct_simulate(
            sim.time, sim.dt, ref_hover, x0
        )

        t_runs.append(np.asarray(t_i, dtype=float))
        states_runs.append(np.asarray(states_i, dtype=float))
        U_runs.append(np.asarray(U_i, dtype=float))
        ref_traj_list.append(ref_hover)
        family_labels.append("hover_excitation")

    # ----------------------------
    # Family 4: local probing around figure-8 anchor states
    # ----------------------------
    n_steps_probe = int(round(LOCAL_PROBE_DURATION / sim.dt))
    candidate_run_count = min(60, int(source["n"]))

    for i in range(N_LOCAL_PROBING):
        rng = random.Random(40000 + i)

        run_idx = rng.randrange(candidate_run_count)
        max_anchor = max(20, T - n_steps_probe - 2)
        anchor_idx = rng.randrange(10, max_anchor)

        x_anchor = np.asarray(source["states"][run_idx, anchor_idx], dtype=float).copy()
        u_anchor = np.asarray(source["U"][run_idx, anchor_idx], dtype=float).copy()

        # add a tiny state perturbation around the anchor
        x0 = x_anchor.copy()
        x0[:3] += np.array([
            rng.uniform(-0.04, 0.04),
            rng.uniform(-0.04, 0.04),
            rng.uniform(-0.03, 0.03)
        ])
        x0[3:6] += np.array([
            rng.uniform(-0.03, 0.03),
            rng.uniform(-0.03, 0.03),
            rng.uniform(-0.02, 0.02)
        ])
        x0[6:9] += np.array([
            rng.uniform(-0.02, 0.02),
            rng.uniform(-0.02, 0.02),
            rng.uniform(-0.01, 0.01)
        ])
        x0[9:12] += np.array([
            rng.uniform(-0.03, 0.03),
            rng.uniform(-0.03, 0.03),
            rng.uniform(-0.015, 0.015)
        ])

        U_probe_short = build_local_probe_input_sequence(
            u_anchor, n_steps_probe, sim.dt, rng
        )

        t_probe, X_probe = simulate_open_loop_generalized_inputs(sim, x0, U_probe_short)

        # pad to full length so saved file shape matches the rest
        X_full = np.zeros((T, 12), dtype=float)
        U_full = np.zeros((T, 4), dtype=float)

        X_full[:n_steps_probe] = X_probe
        U_full[:n_steps_probe] = U_probe_short

        # hold last state and nominal input after probing window
        X_full[n_steps_probe:] = X_probe[-1]
        U_full[n_steps_probe:] = u_anchor

        ref_probe = make_constant_anchor_reference(time, x0)

        t_runs.append(time.copy())
        states_runs.append(X_full)
        U_runs.append(U_full)
        ref_traj_list.append(ref_probe)
        family_labels.append("local_probing")

    # ----------------------------
    # Save output
    # ----------------------------
    t_runs = np.stack(t_runs, axis=0)
    states_runs = np.stack(states_runs, axis=0)
    U_runs = np.stack(U_runs, axis=0)

    out = {
        "traj": "rich_mixed",
        "n": t_runs.shape[0],
        "sim_dt": sim.dt,
        "time": sim.time,
        "t": t_runs,
        "states": states_runs,
        "U": U_runs,
        "ref_traj_list": ref_traj_list,
        "family_labels": family_labels,
        "source_files": [str(Path(SOURCE_FIG8_FILE).resolve())],
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(out, f)

    print(f"Saved rich dataset to: {Path(OUTPUT_FILE).resolve()}")
    print(f"Total runs: {out['n']}")
    print(f"t shape: {out['t'].shape}")
    print(f"states shape: {out['states'].shape}")
    print(f"U shape: {out['U'].shape}")

    unique_labels, counts = np.unique(np.array(family_labels), return_counts=True)
    print("family counts:")
    for lab, cnt in zip(unique_labels, counts):
        print(f"  {lab}: {cnt}")