import pickle
import numpy as np
from pathlib import Path


def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def infer_family_label(data, filename):
    # Prefer explicit traj code if available
    traj = data.get("traj", None)

    if traj == 1:
        return "helix"
    elif traj == 2:
        return "fig8"
    elif traj == 3:
        return "hover_excitation"
    elif isinstance(traj, str):
        return traj

    # Fallback: infer from filename
    name = Path(filename).stem.lower()
    if "traj1" in name or "helix" in name:
        return "helix"
    if "traj2" in name or "fig8" in name:
        return "fig8"
    if "hover" in name or "excitation" in name:
        return "hover_excitation"

    return "unknown"


def combine_run_files(file1, file2, output_file):
    data1 = load_simulation_runs(file1)
    data2 = load_simulation_runs(file2)

    required_keys = ["traj", "n", "sim_dt", "time", "t", "states", "U", "ref_traj_list"]
    for k in required_keys:
        if k not in data1 or k not in data2:
            raise KeyError(f"Missing required key '{k}' in one of the files")

    # Basic compatibility checks
    if not np.isclose(data1["sim_dt"], data2["sim_dt"]):
        raise ValueError(f"sim_dt mismatch: {data1['sim_dt']} vs {data2['sim_dt']}")

    if not np.array_equal(data1["time"], data2["time"]):
        raise ValueError("time vectors do not match")

    if data1["states"].shape[1:] != data2["states"].shape[1:]:
        raise ValueError(
            f"states shape mismatch: {data1['states'].shape} vs {data2['states'].shape}"
        )

    if data1["U"].shape[1:] != data2["U"].shape[1:]:
        raise ValueError(
            f"input shape mismatch: {data1['U'].shape} vs {data2['U'].shape}"
        )

    if data1["t"].shape[1:] != data2["t"].shape[1:]:
        raise ValueError(
            f"time-array shape mismatch: {data1['t'].shape} vs {data2['t'].shape}"
        )

    # Combine along run axis
    t_combined = np.concatenate([data1["t"], data2["t"]], axis=0)
    states_combined = np.concatenate([data1["states"], data2["states"]], axis=0)
    U_combined = np.concatenate([data1["U"], data2["U"]], axis=0)
    ref_combined = list(data1["ref_traj_list"]) + list(data2["ref_traj_list"])

    label1 = infer_family_label(data1, file1)
    label2 = infer_family_label(data2, file2)

    family_labels = (
        [label1] * data1["n"] +
        [label2] * data2["n"]
    )

    combined_data = {
        "traj": "mixed",
        "n": data1["n"] + data2["n"],
        "sim_dt": data1["sim_dt"],
        "time": data1["time"],
        "t": t_combined,
        "states": states_combined,
        "U": U_combined,
        "ref_traj_list": ref_combined,
        "family_labels": family_labels,
        "source_files": [str(file1), str(file2)],
    }

    with open(output_file, "wb") as f:
        pickle.dump(combined_data, f)

    print(f"Saved combined file to: {Path(output_file).resolve()}")
    print(f"Total runs: {combined_data['n']}")
    print(f"t shape: {combined_data['t'].shape}")
    print(f"states shape: {combined_data['states'].shape}")
    print(f"U shape: {combined_data['U'].shape}")
    print(f"ref_traj_list length: {len(combined_data['ref_traj_list'])}")
    print(f"family_labels length: {len(combined_data['family_labels'])}")
    print(f"labels used: {label1}, {label2}")


if __name__ == "__main__":
    combine_run_files(
        "runs_traj2_n200.pkl",
        "runs_hover_excitation_n100.pkl",
        "runs_traj2_plus_hover_n300.pkl"
    )