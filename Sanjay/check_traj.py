import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_one_run(fname, run_idx=0):
    with open(fname, "rb") as f:
        data = pickle.load(f)

    X = data["states"][run_idx]   # shape (3500, 12)
    t = data["t"][run_idx]

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    fig = plt.figure(figsize=(12,5))

    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(x, y)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(f"{fname} run {run_idx} : XY")
    ax1.axis("equal")
    ax1.grid(True)

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(t, z)
    ax2.set_xlabel("time")
    ax2.set_ylabel("z")
    ax2.set_title(f"{fname} run {run_idx} : z vs time")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

plot_one_run("runs_traj1_n50.pkl", 0)
plot_one_run("runs_traj2_n50.pkl", 0)