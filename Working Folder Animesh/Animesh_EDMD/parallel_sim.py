import pickle
import numpy as np
import multiprocessing as mp
from Simulation import quad_sim

def run_single(args):
    traj, i = args
    import random
    sim = quad_sim()
    rng = random.Random(1000 * traj + i)   # ← add this line

    ref_traj = sim.fct_sample_trajectory(traj, rng)

    p0 = ref_traj[0]["pos"].copy()
    for k in range(len(ref_traj)):
        ref_traj[k]["pos"] = ref_traj[k]["pos"] - p0

    init_state = np.zeros(12)
    t_i, states_i, _, _, u_att_i = sim.sim_PID.fct_simulate(
        sim.time, sim.dt, ref_traj, init_state
    )
    return t_i, states_i, u_att_i, ref_traj

def run_prbs_single(i):
    import random
    from scipy.integrate import solve_ivp
    from PID_Mixer import pid_mixer

    sim = quad_sim()
    rng = random.Random(7000 + i)

    T          = len(sim.time)
    thrust_nom = sim.q_mass * sim.g
    delta_thrust = 1.0
    delta_phi    = 0.7
    delta_theta  = 0.7

    u_att_seq   = np.zeros((T, 3))
    switch_steps = rng.randint(2, 20)
    thrust_cmd = thrust_nom
    phi_cmd    = 0.0
    theta_cmd  = 0.0

    for k in range(T):
        if k % switch_steps == 0:
            switch_steps = rng.randint(2, 20)
            thrust_cmd = float(np.clip(
                thrust_nom + rng.uniform(-delta_thrust, delta_thrust),
                thrust_nom * 0.5, thrust_nom * 2.0))
            phi_cmd   = float(np.clip(rng.uniform(-delta_phi,   delta_phi),
                              -sim.controller_PID.tilt_max, sim.controller_PID.tilt_max))
            theta_cmd = float(np.clip(rng.uniform(-delta_theta, delta_theta),
                              -sim.controller_PID.tilt_max, sim.controller_PID.tilt_max))
        u_att_seq[k] = [thrust_cmd, phi_cmd, theta_cmd]

    state      = np.zeros(12)
    state[2]   = 1.0   # start at 1m altitude
    states_i   = np.zeros((T, 12))
    u_att_log  = np.zeros((T, 4))

    for k in range(T):
        u1        = u_att_seq[k, 0]
        phi_des   = u_att_seq[k, 1]
        theta_des = u_att_seq[k, 2]

        phi, theta = state[6], state[7]

        u2 = sim.controller_PID.pid_phi.fct_control(phi, phi_des, sim.dt)
        u3 = sim.controller_PID.pid_theta.fct_control(theta, theta_des, sim.dt)
        u2 = float(np.clip(u2, -sim.controller_PID.torque_max, sim.controller_PID.torque_max))
        u3 = float(np.clip(u3, -sim.controller_PID.torque_max, sim.controller_PID.torque_max))

        u = [u1, u2, u3, 0.0]
        omega_cmd = pid_mixer.fct_mixer(
            u, sim.quad.kT, sim.quad.kD, sim.quad.l,
            min_omega=0.0, max_omega=sim.max_speed)

        def ode(t_local, s_local):
            return sim.quad.fct_dynamics(t_local, s_local, omega_cmd)

        sol = solve_ivp(ode, [sim.time[k], sim.time[k] + sim.dt], state, method="RK45")
        state = sol.y[:, -1]

        states_i[k]  = state
        u_att_log[k] = [u1, phi_des, theta_des,0]

    sim.controller_PID.pid_phi.fct_reset()
    sim.controller_PID.pid_theta.fct_reset()

    T_ref = [{"pos": np.zeros(3), "vel": np.zeros(3), "yaw": 0.0}] * T
    return sim.time, states_i, u_att_log, T_ref




def save_parallel(traj, n, filename, n_workers=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    print(f"Running {n} simulations on {n_workers} cores...")

    args = [(traj, i) for i in range(n)]
    with mp.Pool(n_workers) as pool:
        results = pool.map(run_single, args)

    t      = np.stack([r[0] for r in results])
    states = np.stack([r[1] for r in results])
    U      = np.stack([r[2] for r in results])
    refs   = [r[3] for r in results]

    sim = quad_sim()
    with open(filename, "wb") as f:
        pickle.dump({"traj": traj, "n": n, "sim_dt": sim.dt,
                     "time": sim.time, "t": t, "states": states,
                     "U": U, "ref_traj_list": refs}, f)
    print(f"Saved {filename}")

def save_prbs_parallel(n, filename, n_workers=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    print(f"Generating PRBS, n={n} on {n_workers} cores...")
    t0 = time.perf_counter()

    with mp.Pool(n_workers) as pool:
        results = pool.map(run_prbs_single, range(n))

    t_arr      = np.stack([r[0] for r in results])
    states_arr = np.stack([r[1] for r in results])
    U_arr      = np.stack([r[2] for r in results])
    refs       = [r[3] for r in results]

    sim = quad_sim()
    with open(filename, "wb") as f:
        pickle.dump({"traj": "prbs", "n": n, "sim_dt": sim.dt,
                     "time": sim.time, "t": t_arr, "states": states_arr,
                     "U": U_arr, "ref_traj_list": refs}, f)

    print(f"Saved {filename}  ({time.perf_counter()-t0:.1f} s)")



if __name__ == "__main__":
    import time
    t0 = time.perf_counter()

    save_parallel(traj=1, n=50, filename="runs_traj1_n50.pkl")  # helix
    save_parallel(traj=2, n=50, filename="runs_traj2_n50.pkl")  # fig8
    save_parallel(traj=3, n=50, filename="runs_traj3_n50.pkl")  # lissajous
    save_parallel(traj=4, n=50, filename="runs_traj4_n50.pkl")  # waypoint
    save_parallel(traj=5, n=30, filename="runs_traj5_n30.pkl")  # hover
    save_prbs_parallel(n=70, filename="runs_prbs_n70.pkl")  # PRBS


    print(f"Total time: {(time.perf_counter()-t0)/60:.1f} min")