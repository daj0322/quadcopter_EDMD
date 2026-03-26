import pickle

for fname in ["runs_traj1_n50.pkl", "runs_traj2_n50.pkl"]:
    with open(fname, "rb") as f:
        data = pickle.load(f)

    print("\nFILE:", fname)
    print("top-level keys:", data.keys())
    print("traj:", data["traj"])
    print("n field:", data["n"])
    print("t shape:", data["t"].shape)
    print("states shape:", data["states"].shape)
    print("U shape:", data["U"].shape)
    print("num ref trajectories:", len(data["ref_traj_list"]))