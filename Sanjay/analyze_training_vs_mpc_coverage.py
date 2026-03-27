import pickle
import numpy as np
import matplotlib.pyplot as plt


# =========================
# SETTINGS
# =========================
TRAIN_FILE = "runs_traj2_n200.pkl"

# Put your saved MPC rollout outputs here if you saved them.
# If you have not saved them yet, see notes below.
LINEAR_MPC_FILE = "linear_mpc_results.pkl"
EDMD_MPC_FILE = "edmd_mpc_results.pkl"

# Which rollout time indices to inspect for nearest-neighbor coverage
QUERY_INDICES = [0, 10, 25, 50, 100, 200]

# Number of nearest neighbors to report
K_NEIGHBORS = 5


# =========================
# HELPERS
# =========================
def load_pkl(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def flatten_training_pairs(train_data):
    """
    Flatten the training data into per-time-step state-input samples.

    Returns
    -------
    X_flat : (M, 12)
    U_flat : (M, 4)
    """
    X = np.asarray(train_data["states"], dtype=float)   # shape: (n_runs, T, 12)
    U = np.asarray(train_data["U"], dtype=float)        # shape: (n_runs, T, 4)

    X_flat = X.reshape(-1, X.shape[-1])
    U_flat = U.reshape(-1, U.shape[-1])

    return X_flat, U_flat


def print_control_stats(name, U):
    print(f"\n{name}")
    print("-" * len(name))
    for j in range(U.shape[1]):
        print(
            f"u{j+1}: min={U[:, j].min(): .6f}, "
            f"max={U[:, j].max(): .6f}, "
            f"mean={U[:, j].mean(): .6f}, "
            f"std={U[:, j].std(): .6f}"
        )


def plot_control_histograms(U_train, U_lin=None, U_edmd=None):
    labels = ["u1", "u2", "u3", "u4"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    for j, ax in enumerate(axs):
        ax.hist(U_train[:, j], bins=60, alpha=0.5, label="training", density=True)
        if U_lin is not None:
            ax.hist(U_lin[:, j], bins=60, alpha=0.5, label="linear mpc", density=True)
        if U_edmd is not None:
            ax.hist(U_edmd[:, j], bins=60, alpha=0.5, label="edmd mpc", density=True)

        ax.set_title(labels[j])
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


def normalize_features(X, eps=1e-8):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return mu, sigma


def nearest_neighbors_state_input(X_train, U_train, xq, uq, k=5, state_weight=1.0, input_weight=1.0):
    """
    Find nearest training state-input pairs to a query (xq, uq).

    Distance is computed in normalized coordinates, combining state and input.
    """
    XU_train = np.hstack([state_weight * X_train, input_weight * U_train])
    xuq = np.hstack([state_weight * xq, input_weight * uq])[None, :]

    mu, sigma = normalize_features(XU_train)
    XU_train_n = (XU_train - mu) / sigma
    xuq_n = (xuq - mu) / sigma

    dists = np.linalg.norm(XU_train_n - xuq_n, axis=1)
    idx = np.argsort(dists)[:k]

    return idx, dists[idx]


def coverage_report(name, X_train, U_train, X_query, U_query, query_indices, k=5):
    print(f"\n{name} state-input coverage check")
    print("=" * (len(name) + 28))

    for qi in query_indices:
        if qi >= len(X_query):
            continue

        xq = X_query[qi]
        uq = U_query[qi]

        idx, dists = nearest_neighbors_state_input(
            X_train, U_train, xq, uq, k=k, state_weight=1.0, input_weight=1.0
        )

        print(f"\nQuery step {qi}")
        print(f"xq[:6] = {xq[:6]}")
        print(f"uq     = {uq}")
        print("nearest distances:", np.round(dists, 6))

        nn_states = X_train[idx]
        nn_inputs = U_train[idx]

        print("nearest input samples:")
        for row in nn_inputs:
            print(" ", np.round(row, 6))

        print("nearest state samples first 6 entries:")
        for row in nn_states[:, :6]:
            print(" ", np.round(row, 6))


def hover_region_report(X_train, U_train):
    """
    Look at training controls near hover-like states.
    """
    mask = (
        (np.abs(X_train[:, 3]) < 0.10) &   # vx
        (np.abs(X_train[:, 4]) < 0.10) &   # vy
        (np.abs(X_train[:, 5]) < 0.10) &   # vz
        (np.abs(X_train[:, 6]) < 0.05) &   # phi
        (np.abs(X_train[:, 7]) < 0.05) &   # theta
        (np.abs(X_train[:, 9]) < 0.10) &   # p
        (np.abs(X_train[:,10]) < 0.10)     # q
    )

    U_hover = U_train[mask]
    print("\nHover-like training samples")
    print("==========================")
    print("count:", len(U_hover))

    if len(U_hover) == 0:
        print("No samples matched the hover-like mask.")
        return

    print_control_stats("training controls in hover-like region", U_hover)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train_data = load_pkl(TRAIN_FILE)
    X_train, U_train = flatten_training_pairs(train_data)

    print_control_stats("training controls", U_train)
    hover_region_report(X_train, U_train)

    # Load MPC outputs if available
    U_lin = None
    X_lin = None
    U_edmd = None
    X_edmd = None

    try:
        lin_data = load_pkl(LINEAR_MPC_FILE)
        X_lin = np.asarray(lin_data["X"], dtype=float)
        U_lin = np.asarray(lin_data["U"], dtype=float)
        print_control_stats("linear MPC controls", U_lin)
    except FileNotFoundError:
        print(f"\nDid not find {LINEAR_MPC_FILE}, skipping linear MPC analysis.")

    try:
        edmd_data = load_pkl(EDMD_MPC_FILE)
        X_edmd = np.asarray(edmd_data["X"], dtype=float)
        U_edmd = np.asarray(edmd_data["U"], dtype=float)
        print_control_stats("EDMD-MPC controls", U_edmd)
    except FileNotFoundError:
        print(f"\nDid not find {EDMD_MPC_FILE}, skipping EDMD-MPC analysis.")

    plot_control_histograms(U_train, U_lin, U_edmd)

    if X_lin is not None and U_lin is not None:
        coverage_report(
            "Linear MPC",
            X_train, U_train,
            X_lin, U_lin,
            QUERY_INDICES,
            k=K_NEIGHBORS
        )

    if X_edmd is not None and U_edmd is not None:
        coverage_report(
            "EDMD-MPC",
            X_train, U_train,
            X_edmd, U_edmd,
            QUERY_INDICES,
            k=K_NEIGHBORS
        )