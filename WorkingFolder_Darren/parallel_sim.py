import pickle
import time

from Simulation import quad_sim


def save_runs(traj, n, filename):
    """
    Generate simulation runs with the current Darren simulator.

    Logged inputs are PX4-style plant wrench commands:
    [thrust, tau_roll, tau_pitch, tau_yaw].
    """
    print(f"Running trajectory {traj}, n={n}...")
    start = time.perf_counter()

    sim = quad_sim()
    t, states, U, refs = sim.fct_run_simulation(traj, n)

    with open(filename, "wb") as f:
        pickle.dump(
            {
                "traj": traj,
                "n": n,
                "sim_dt": sim.dt,
                "time": sim.time,
                "t": t,
                "states": states,
                "U": U,
                "ref_traj_list": refs,
                "input_type": "wrench",
                "input_labels": ["thrust", "tau_roll", "tau_pitch", "tau_yaw"],
            },
            f,
        )

    elapsed = time.perf_counter() - start
    print(f"Saved {filename} in {elapsed:.1f} s")


if __name__ == "__main__":
    t0 = time.perf_counter()

    save_runs(traj=1, n=50, filename="runs_traj1_n50.pkl")
    save_runs(traj=2, n=50, filename="runs_traj2_n50.pkl")
    save_runs(traj=3, n=50, filename="runs_traj3_n50.pkl")

    print(f"Total time: {(time.perf_counter() - t0) / 60:.1f} min")
