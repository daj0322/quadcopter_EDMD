import pickle
import numpy as np
from pathlib import Path


def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def infer_family_label(data, filename):
    traj = data.get("traj", None)
    if traj == 1:                    return "helix"
    elif traj == 2:                  return "fig8"
    elif traj == 3:                  return "lissajous"
    elif traj == 4:                  return "waypoint"
    elif traj == "hover_excitation": return "hover_excitation"
    elif traj == "prbs":             return "prbs"
    elif isinstance(traj, str):      return traj

    name = Path(filename).stem.lower()
    if "traj1" in name or "helix"  in name: return "helix"
    if "traj2" in name or "fig8"   in name: return "fig8"
    if "traj3" in name or "lissa"  in name: return "lissajous"
    if "traj4" in name or "wayp"   in name: return "waypoint"
    if "hover" in name:                      return "hover_excitation"
    if "prbs"  in name:                      return "prbs"
    return "unknown"


def combine_run_files(file_list, output_file):
    datasets = [load_simulation_runs(f) for f in file_list]

    # Drop degenerate psi_des column if present (col 3, always zero)
    for d in datasets:
        if d["U"].shape[2] == 4:
            d["U"] = d["U"][:, :, :3]

    # Drop psi (col 8), p (col 9), q (col 10), r (col 11)
    # Keep only [x, y, z, vx, vy, vz, phi, theta] — 8 states
    # Drop psi, p, q, r — keep only [x, y, z, vx, vy, vz, phi, theta]
    for d in datasets:
        if d["states"].shape[2] == 12:
            # drop psi(8) and r(11) only — keep p(9) and q(10)
            d["states"] = d["states"][:, :, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]

        # Debug — check actual shapes before dropping
    for f, d in zip(file_list, datasets):
        print(f"{Path(f).name}: states shape = {d['states'].shape}")

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

    # Combine
    t_combined      = np.concatenate([d["t"]      for d in datasets], axis=0)
    states_combined = np.concatenate([d["states"] for d in datasets], axis=0)

    U_combined      = np.concatenate([d["U"]      for d in datasets], axis=0)
    ref_combined    = sum([list(d["ref_traj_list"]) for d in datasets], [])

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
    }

    with open(output_file, "wb") as f:
        pickle.dump(combined_data, f)

    print(f"Saved: {Path(output_file).resolve()}")
    print(f"Total runs:   {combined_data['n']}")
    print(f"t shape:      {combined_data['t'].shape}")
    print(f"states shape: {combined_data['states'].shape}")
    print(f"U shape:      {combined_data['U'].shape}")
    print(f"Families:     {set(family_labels)}")


if __name__ == "__main__":
    combine_run_files(
        file_list=[
            "runs_traj1_n50.pkl",
            "runs_traj2_n50.pkl",
            "runs_traj3_n50.pkl",
            "runs_traj4_n50.pkl",
            "runs_traj5_n30.pkl",
            "runs_prbs_n70.pkl",
        ],
        output_file="runs_mixed_n300.pkl"
    )