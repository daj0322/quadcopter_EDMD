import pickle
import numpy as np
import scipy.sparse as sp
import osqp

# File I/O
def load_edmdc_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["t"], data["states"], data["U"], data["ref_traj_list"]

# State lifting
# The lifted model uses a 10-state reduced vector:
# [x, y, z, vx, vy, vz, phi, theta, p, q]
#
# Observable ordering must remain consistent with EDMDc_training.py:
#  - states
#  - sin/cos of roll and pitch
#  - selected cross terms
#  - quadratic energy-like terms
#  - constant bias
def observables(x_std, scaler):
    """
    Return the lifted observable vector for a standardized 10-state input.

    The observable definition must match the lifting used during training.
    """
    x = np.asarray(x_std).flatten()
    assert len(x) == 10, f"Expected 10-state vector, got {len(x)}"

    obs = list(x)  # 10 linear terms

    phi_rad   = x[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_rad = x[7] * scaler.scale_[7] + scaler.mean_[7]

    obs.append(np.sin(phi_rad))
    obs.append(np.cos(phi_rad))
    obs.append(np.sin(theta_rad))
    obs.append(np.cos(theta_rad))

    obs.append(x[6] * x[8])   # phi * p
    obs.append(x[7] * x[9])   # theta * q
    obs.append(x[3] * x[6])   # vx * phi
    obs.append(x[4] * x[7])   # vy * theta

    obs.append(x[3]**2 + x[4]**2 + x[5]**2)   # v_sq
    obs.append(x[8]**2 + x[9]**2)             # omega_sq

    obs.append(x[3] * x[7])  # vx * theta
    obs.append(x[4] * x[6])  # vy * phi
    obs.append(x[5] * x[5])  # vz²
    obs.append(x[6] * x[6])  # phi²
    obs.append(x[7] * x[7])  # theta²
    obs.append(x[8] * x[9])  # p * q

    obs.append(1.0)

    return np.asarray(obs, dtype=float)


def lifted_state_from_x(x10, scaler):
    """Map a physical 10-state vector to the lifted observable space."""
    x_std = scaler.transform(x10.reshape(1, -1)).flatten()
    return observables(x_std, scaler)


def drop_to_10state(x12):
    """Convert a 12-state vector to the reduced 10-state representation."""
    return x12[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]


# MPC solver
class EDMDcMPC_QP:
    """
    Quadratic-program MPC controller built on a lifted linear EDMDc model.

    The optimizer penalizes tracking error in the physical-state coordinates
    selected by Cz while optimizing control increments over the control horizon.
    """
    def __init__(self, A, B, Cz, N, NC, Q, R, Rd,
                 u_scaler, du_min, du_max, u_nominal_raw):
        self.A  = np.asarray(A, dtype=float)
        self.B  = np.asarray(B, dtype=float)
        self.Cz = np.asarray(Cz, dtype=float)
        self.N  = int(N)
        self.NC = int(NC)
        self.Q  = np.asarray(Q,  dtype=float)
        self.R  = np.asarray(R,  dtype=float)
        self.Rd = np.asarray(Rd, dtype=float)

        self.u_scaler      = u_scaler
        self.u_nom_raw     = np.asarray(u_nominal_raw, dtype=float)
        self.u_nom_scaled  = u_scaler.transform(
            self.u_nom_raw.reshape(1, -1)).flatten()

        self.du_min = np.asarray(du_min, dtype=float)
        self.du_max = np.asarray(du_max, dtype=float)

        self.nz   = self.A.shape[0]   # observable dimension
        self.nu   = self.B.shape[1]   # input dimension
        self.nx   = self.Cz.shape[0]  # tracked physical-state dimension
        self.nvar = self.NC * self.nu

        self._du_prev = np.zeros(self.nvar)

        self.Sz, self.Su = self._build_prediction_matrices()

        # Map lifted-state predictions to the tracked physical-state coordinates.
        Su_dense = self.Su.toarray()
        Su_phys  = np.zeros((self.N * self.nx, self.nvar))
        for i in range(self.N):
            for j in range(self.NC):
                Su_phys[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = \
                    self.Cz @ Su_dense[i*self.nz:(i+1)*self.nz, j*self.nu:(j+1)*self.nu]
        self.Su_phys = sp.csc_matrix(Su_phys)

        self.Qbar = sp.block_diag(
            [sp.csc_matrix(self.Q) for _ in range(self.N)], format="csc")
        self.Rbar = sp.block_diag(
            [sp.csc_matrix(self.R) for _ in range(self.NC)], format="csc")
        self.D    = self._build_difference_matrix()
        self.Rdbar = (
            sp.block_diag([sp.csc_matrix(self.Rd)
                           for _ in range(self.NC - 1)], format="csc")
            if self.NC > 1 else None
        )

        P     = self._build_hessian()
        Aineq = sp.eye(self.nvar, format="csc")
        l     = np.tile(self.du_min, self.NC)
        u     = np.tile(self.du_max, self.NC)

        self.prob = osqp.OSQP()
        self.prob.setup(P=P, q=np.zeros(self.nvar),
                        A=Aineq, l=l, u=u,
                        warm_start=True, verbose=False, polish=False)

    def _build_prediction_matrices(self):
        Sz = np.zeros((self.N * self.nz, self.nz))
        Su = np.zeros((self.N * self.nz, self.NC * self.nu))
        A_pow = [np.eye(self.nz)]
        for _ in range(self.N):
            A_pow.append(A_pow[-1] @ self.A)
        for i in range(self.N):
            Sz[i*self.nz:(i+1)*self.nz, :] = A_pow[i+1]
            for j in range(min(i+1, self.NC)):
                Su[i*self.nz:(i+1)*self.nz, j*self.nu:(j+1)*self.nu] = \
                    A_pow[i-j] @ self.B
        return sp.csc_matrix(Sz), sp.csc_matrix(Su)

    def _build_difference_matrix(self):
        if self.NC <= 1:
            return None
        rows, cols, vals = [], [], []
        for k in range(self.NC - 1):
            for j in range(self.nu):
                r = k * self.nu + j
                rows.extend([r, r])
                cols.extend([k*self.nu+j, (k+1)*self.nu+j])
                vals.extend([-1.0, 1.0])
        return sp.coo_matrix(
            (vals, (rows, cols)),
            shape=((self.NC-1)*self.nu, self.NC*self.nu)).tocsc()

    def _build_hessian(self):
        P = self.Su_phys.T @ self.Qbar @ self.Su_phys + self.Rbar
        if self.D is not None and self.Rdbar is not None:
            P = P + self.D.T @ self.Rdbar @ self.D
        return (0.5 * (P + P.T)).tocsc()

    def _build_q(self, z0, x_ref_std_horizon):
        z_free = self.Sz @ z0
        x_free = np.array([
            self.Cz @ z_free[i*self.nz:(i+1)*self.nz]
            for i in range(self.N)
        ]).reshape(-1)
        x_ref = x_ref_std_horizon.reshape(-1)
        return np.asarray(
            self.Su_phys.T @ (self.Qbar @ (x_free - x_ref))
        ).reshape(-1)

    def compute(self, z0, x_ref_std_horizon):
        q = self._build_q(z0, x_ref_std_horizon)
        self.prob.update(q=q)
        self.prob.warm_start(x=self._du_prev)
        res = self.prob.solve()

        if res.info.status not in ("solved", "solved inaccurate"):
            print(f"Warning OSQP: {res.info.status}")
            du0 = self._du_prev[:self.nu]
        else:
            du_opt = np.asarray(res.x).reshape(-1)
            self._du_prev = du_opt.copy()
            du0 = du_opt[:self.nu]

        u0_scaled = self.u_nom_scaled + du0
        u0_raw    = self.u_scaler.inverse_transform(
            u0_scaled.reshape(1, -1)).flatten()
        return u0_raw  # [thrust, phi_des, theta_des]


# Reference processing
def extract_ref_xyz(ref_traj):
    return np.array([wp["pos"][:3] for wp in ref_traj], dtype=float)


def precompute_ref_std(ref_traj, scaler, n_states=10):
    """Build a standardized reference trajectory using position and velocity."""
    T = len(ref_traj)
    X_ref = np.zeros((T, n_states))
    for k in range(T):
        X_ref[k, 0:3] = ref_traj[k]["pos"][:3]
        X_ref[k, 3:6] = ref_traj[k]["vel"][:3]
    return scaler.transform(X_ref)


def build_ref_horizon(ref_std, k, N):
    T = ref_std.shape[0]
    h = np.zeros((N, ref_std.shape[1]))
    for i in range(N):
        h[i] = ref_std[min(k + i, T - 1)]
    return h

# Metrics
def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))
