import argparse
import pickle
import numpy as np
from pathlib import Path


def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def infer_family_label(data, filename):
    traj = data.get("traj", None)
    if traj == 1:
        return "helix"
    elif traj == 2:
        return "fig8"
    elif traj == 3:
        return "lissajous"
    elif traj == 4:
        return "waypoint"
    elif traj == 5:
        return "hover_excitation"
    elif traj in ("prbs", "yaw_prbs"):
        return "yaw_prbs"
    elif isinstance(traj, str):
        return traj

    name = Path(filename).stem.lower()
    if "traj1" in name or "helix" in name:
        return "helix"
    if "traj2" in name or "fig8" in name:
        return "fig8"
    if "traj3" in name or "lissa" in name:
        return "lissajous"
    if "traj4" in name or "wayp" in name:
        return "waypoint"
    if "traj5" in name or "hover" in name:
        return "hover_excitation"
    if "prbs" in name:
        return "yaw_prbs"
    return "unknown"


def combine_run_files(file_list, output_file):
    datasets = [load_simulation_runs(f) for f in file_list]

    if not all(d.get("input_type") == "applied_wrench" for d in datasets):
        raise ValueError(
            "Every source file must log applied_wrench inputs. Regenerate legacy data "
            "with parallel_sim.py before mixing."
        )
    if not all("U_requested" in d for d in datasets):
        raise ValueError(
            "Every source file must retain U_requested actuator diagnostics. "
            "Regenerate legacy data with parallel_sim.py before mixing."
        )

    # Keep full yaw-aware logs:
    # states = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    # U      = realized [thrust, tau_roll, tau_pitch, tau_yaw]
    # U_requested is retained when each source file provides it.
    for f, d in zip(file_list, datasets):
        print(f"{Path(f).name}: states shape = {d['states'].shape}, U shape = {d['U'].shape}")

    # Compatibility checks against first file
    ref = datasets[0]
    for i, data in enumerate(datasets[1:], 1):
        if not np.isclose(ref["sim_dt"], data["sim_dt"]):
            raise ValueError(f"sim_dt mismatch at file {i}")
        if not np.array_equal(ref["time"], data["time"]):
            raise ValueError(f"time vector mismatch at file {i}")
        if ref["states"].shape[1:] != data["states"].shape[1:]:
            raise ValueError(f"states shape mismatch at file {i}")
        if ref["U"].shape[1:] != data["U"].shape[1:]:
            raise ValueError(f"U shape mismatch at file {i}")
        if ref["t"].shape[1:] != data["t"].shape[1:]:
            raise ValueError(f"t shape mismatch at file {i}")

    t_combined = np.concatenate([d["t"] for d in datasets], axis=0)
    states_combined = np.concatenate([d["states"] for d in datasets], axis=0)
    U_combined = np.concatenate([d["U"] for d in datasets], axis=0)
    has_requested_inputs = True
    U_requested_combined = np.concatenate(
        [d["U_requested"] for d in datasets], axis=0
    )
    ref_combined = sum([list(d["ref_traj_list"]) for d in datasets], [])

    family_labels = sum([
        [infer_family_label(d, f)] * d["n"]
        for d, f in zip(datasets, file_list)
    ], [])

    combined_data = {
        "traj": "mixed",
        "n": sum(d["n"] for d in datasets),
        "sim_dt": ref["sim_dt"],
        "time": ref["time"],
        "t": t_combined,
        "states": states_combined,
        "U": U_combined,
        "ref_traj_list": ref_combined,
        "family_labels": family_labels,
        "source_files": [str(f) for f in file_list],
        "dataset_profile": ref.get("dataset_profile", "custom"),
        "input_type": "applied_wrench",
        "input_labels": ["thrust", "tau_roll", "tau_pitch", "tau_yaw"],
    }
    combined_data["U_requested"] = U_requested_combined

    with open(output_file, "wb") as f:
        pickle.dump(combined_data, f)

    print(f"Saved: {Path(output_file).resolve()}")
    print(f"Total runs:   {combined_data['n']}")
    print(f"t shape:      {combined_data['t'].shape}")
    print(f"states shape: {combined_data['states'].shape}")
    print(f"U shape:      {combined_data['U'].shape}")
    print(f"Requested U:  {'present' if has_requested_inputs else 'not available'}")
    print(f"Families:     {set(family_labels)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine yaw-aware trajectory families into one EDMDc dataset."
    )
    parser.add_argument(
        "--profile", choices=("paper", "compact"), default="paper",
        help="paper combines the old 300-run family composition (default).",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path(__file__).resolve().parent,
        help="Directory containing generated trajectory pickle files.",
    )
    parser.add_argument(
        "--runs-per-family", type=int, default=None,
        help="Compact profile only: count used to name each trajectory file (default: 50).",
    )
    parser.add_argument(
        "--prbs-runs", type=int, default=None,
        help="Compact profile only: bounded yaw-PRBS run count (default: 0).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output dataset path (default: runs_mixed_n<3*runs>.pkl in --input-dir).",
    )
    args = parser.parse_args()
    if args.profile == "paper":
        if args.runs_per_family is not None or args.prbs_runs is not None:
            parser.error(
                "The paper profile has fixed 50/50/50/50/30/70 counts. "
                "Use --profile compact for a custom mix."
            )
        file_specs = [(1, 50), (2, 50), (3, 50), (4, 50), (5, 30)]
        prbs_runs = 70
    else:
        runs_per_family = 50 if args.runs_per_family is None else args.runs_per_family
        prbs_runs = 0 if args.prbs_runs is None else args.prbs_runs
        if runs_per_family < 1:
            parser.error("--runs-per-family must be at least 1")
        if prbs_runs < 0:
            parser.error("--prbs-runs cannot be negative")
        file_specs = [(traj, runs_per_family) for traj in (1, 2, 3)]

    input_dir = args.input_dir.resolve()
    output = args.output
    if output is None:
        output = input_dir / f"runs_mixed_n{sum(n for _, n in file_specs) + prbs_runs}.pkl"
    elif not output.is_absolute():
        output = Path.cwd() / output

    file_list = [
        input_dir / f"runs_traj{traj}_n{n}.pkl"
        for traj, n in file_specs
    ]
    if prbs_runs:
        file_list.append(input_dir / f"runs_prbs_n{prbs_runs}.pkl")
    combine_run_files(file_list=file_list, output_file=output)
