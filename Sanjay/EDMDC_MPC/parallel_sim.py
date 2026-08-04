"""Generate reproducible yaw-aware EDMDc simulation data.

The default ``paper`` profile recreates the original paper's data regime:
100-second runs at 100 Hz, with 50/50/50/50/30 trajectory-family runs and
70 PRBS runs.  It intentionally retains the corrected applied-wrench logs and
the yaw-aware closed-loop excitation introduced for the ACC revision.
"""

import argparse
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import numpy as np

from Simulation import quad_sim


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_RUN_COUNTS = ((1, 50), (2, 50), (3, 50), (4, 50), (5, 30))
PAPER_PRBS_RUNS = 70
PAPER_DURATION_SECONDS = 100.0
INPUT_LABELS = ["thrust", "tau_roll", "tau_pitch", "tau_yaw"]


def configure_duration(sim, duration):
    if duration is None:
        return
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    sim.time = np.arange(0.0, duration, sim.dt)


def default_worker_count(n_tasks):
    return max(1, min(n_tasks, os.cpu_count() or 1, 4))


def map_runs(worker, tasks, workers):
    """Map deterministic, independent simulations without oversubscribing RAM."""
    workers = default_worker_count(len(tasks)) if workers is None else workers
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        return [worker(task) for task in tasks]
    with mp.Pool(processes=min(workers, len(tasks))) as pool:
        return pool.map(worker, tasks)


def run_trajectory_single(task):
    """Top-level worker so that the task is pickleable on every platform."""
    traj, run_index, duration = task
    sim = quad_sim()
    configure_duration(sim, duration)
    return sim.fct_run_single_simulation(traj, run_index)


def save_runs(traj, n, filename, duration=None, workers=None, profile="custom"):
    """Save one trajectory family with motor-feasible applied wrench logs."""
    if n < 1:
        raise ValueError("n must be at least 1")
    print(f"Running trajectory {traj}, n={n}...")
    start = time.perf_counter()
    results = map_runs(
        run_trajectory_single,
        [(traj, run_index, duration) for run_index in range(n)],
        workers,
    )

    t = np.stack([entry[0] for entry in results])
    states = np.stack([entry[1] for entry in results])
    U = np.stack([entry[2] for entry in results])
    U_requested = np.stack([entry[3] for entry in results])
    refs = [entry[4] for entry in results]
    sim_dt = quad_sim.dt
    time_vector = np.arange(0.0, duration, sim_dt) if duration is not None else quad_sim().time

    with open(filename, "wb") as f:
        pickle.dump(
            {
                "traj": traj,
                "n": n,
                "sim_dt": sim_dt,
                "time": time_vector,
                "t": t,
                "states": states,
                "U": U,
                "U_requested": U_requested,
                "ref_traj_list": refs,
                "input_type": "applied_wrench",
                "input_labels": INPUT_LABELS,
                "dataset_profile": profile,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved {filename} in {time.perf_counter() - start:.1f} s")


def make_yaw_prbs_reference(time_vector, seed):
    """Create bounded closed-loop yaw PRBS, replacing old raw-angle PRBS.

    The run count and deterministic seed sequence match the original setup.
    The old input log contained desired roll/pitch angles rather than plant
    inputs, so it cannot be reused for applied-wrench EDMDc identification.
    """
    rng = np.random.default_rng(seed)
    time_vector = np.asarray(time_vector, dtype=float)
    n_samples = len(time_vector)
    dt = float(time_vector[1] - time_vector[0]) if n_samples > 1 else 0.01

    yaw_rate = np.zeros(n_samples)
    k = 0
    while k < n_samples:
        hold_steps = int(rng.integers(40, 121))  # 0.4--1.2 s at 100 Hz
        yaw_rate[k:k + hold_steps] = rng.uniform(-0.8, 0.8)
        k += hold_steps
    yaw = np.cumsum(yaw_rate) * dt

    # Keep a small translational component so the PRBS family covers a local
    # flight regime rather than only a perfectly stationary hover.
    ramp_duration = min(4.0, max(time_vector[-1] if n_samples else 0.0, dt))
    ramp = np.clip(time_vector / max(ramp_duration, dt), 0.0, 1.0)
    rise = ramp * ramp * (3.0 - 2.0 * ramp)
    rise_dot = np.where(
        ramp < 1.0,
        6.0 * ramp * (1.0 - ramp) / max(ramp_duration, dt),
        0.0,
    )
    x = 0.4 * np.sin(0.35 * time_vector)
    y = 0.3 * np.sin(0.27 * time_vector + 0.5)
    z = 1.5 * rise + 0.08 * np.sin(0.45 * time_vector)
    vx = 0.14 * np.cos(0.35 * time_vector)
    vy = 0.081 * np.cos(0.27 * time_vector + 0.5)
    vz = 1.5 * rise_dot + 0.036 * np.cos(0.45 * time_vector)
    ax = -0.049 * np.sin(0.35 * time_vector)
    ay = -0.02187 * np.sin(0.27 * time_vector + 0.5)
    az = -0.0162 * np.sin(0.45 * time_vector)

    return [
        {
            "pos": np.array([x[i], y[i], z[i]], dtype=float),
            "vel": np.array([vx[i], vy[i], vz[i]], dtype=float),
            "acc": np.array([ax[i], ay[i], az[i]], dtype=float),
            "yaw": float(yaw[i]),
            "yaw_rate": float(yaw_rate[i]),
        }
        for i in range(n_samples)
    ]


def run_yaw_prbs_single(task):
    run_index, duration = task
    sim = quad_sim()
    configure_duration(sim, duration)
    ref = make_yaw_prbs_reference(sim.time, seed=7000 + run_index)
    init_state = np.zeros(12)
    init_state[8] = float(ref[0]["yaw"])
    t, states, _, U, U_requested = sim.sim_PID.fct_simulate(
        sim.time, sim.dt, ref, init_state, return_requested=True
    )
    return t, states, U, U_requested, ref


def save_prbs_runs(n, filename, duration=None, workers=None, profile="custom"):
    """Save bounded yaw-PRBS flights with the old 7000+i seed convention."""
    if n < 1:
        raise ValueError("n must be at least 1")
    print(f"Running yaw PRBS excitation, n={n}...")
    start = time.perf_counter()
    results = map_runs(
        run_yaw_prbs_single,
        [(run_index, duration) for run_index in range(n)],
        workers,
    )
    t = np.stack([entry[0] for entry in results])
    states = np.stack([entry[1] for entry in results])
    U = np.stack([entry[2] for entry in results])
    U_requested = np.stack([entry[3] for entry in results])
    refs = [entry[4] for entry in results]
    sim_dt = quad_sim.dt
    time_vector = np.arange(0.0, duration, sim_dt) if duration is not None else quad_sim().time

    with open(filename, "wb") as f:
        pickle.dump(
            {
                "traj": "yaw_prbs",
                "n": n,
                "sim_dt": sim_dt,
                "time": time_vector,
                "t": t,
                "states": states,
                "U": U,
                "U_requested": U_requested,
                "ref_traj_list": refs,
                "input_type": "applied_wrench",
                "input_labels": INPUT_LABELS,
                "dataset_profile": profile,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved {filename} in {time.perf_counter() - start:.1f} s")


def resolve_generation_spec(parser, args):
    if args.profile == "paper":
        if any(value is not None for value in (
            args.runs_per_family, args.prbs_runs, args.duration,
        )):
            parser.error(
                "The paper profile has fixed 100 s and 50/50/50/50/30/70 "
                "counts. Use --profile compact for a custom run."
            )
        return PAPER_RUN_COUNTS, PAPER_PRBS_RUNS, PAPER_DURATION_SECONDS

    runs_per_family = 50 if args.runs_per_family is None else args.runs_per_family
    prbs_runs = 0 if args.prbs_runs is None else args.prbs_runs
    if runs_per_family < 1:
        parser.error("--runs-per-family must be at least 1")
    if prbs_runs < 0:
        parser.error("--prbs-runs cannot be negative")
    return tuple((traj, runs_per_family) for traj in (1, 2, 3)), prbs_runs, args.duration


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic yaw-aware EDMDc training trajectories."
    )
    parser.add_argument(
        "--profile", choices=("paper", "compact"), default="paper",
        help="paper recreates the old 300-run data regime (default); compact is custom.",
    )
    parser.add_argument(
        "--runs-per-family", type=int, default=None,
        help="Compact profile only: runs for each of trajectories 1--3 (default: 50).",
    )
    parser.add_argument(
        "--prbs-runs", type=int, default=None,
        help="Compact profile only: bounded yaw-PRBS runs (default: 0).",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Compact profile only: duration in seconds (default: simulator 45 s).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Independent simulation workers (default: up to 4).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR,
        help="Directory for generated pickle files (default: this script's directory).",
    )
    args = parser.parse_args()
    family_counts, prbs_runs, duration = resolve_generation_spec(parser, args)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for traj, n in family_counts:
        save_runs(
            traj=traj,
            n=n,
            filename=output_dir / f"runs_traj{traj}_n{n}.pkl",
            duration=duration,
            workers=args.workers,
            profile=args.profile,
        )
    if prbs_runs:
        save_prbs_runs(
            n=prbs_runs,
            filename=output_dir / f"runs_prbs_n{prbs_runs}.pkl",
            duration=duration,
            workers=args.workers,
            profile=args.profile,
        )
    print(f"Total time: {(time.perf_counter() - start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
