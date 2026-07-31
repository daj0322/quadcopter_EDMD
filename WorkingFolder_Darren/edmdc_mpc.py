import pickle
import numpy as np
import scipy.sparse as sp

try:
    import osqp
except ModuleNotFoundError:
    osqp = None

STATE_DIM = 12
STATE_LABELS = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"]
REDUCED_10_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
RAW_INPUT_DIM = 4
RAW_INPUT_LABELS = ["thrust", "tau_roll", "tau_pitch", "tau_yaw"]
INPUT_LIFT_TYPE = "thrust_direction_rate_coupling"
LEGACY_INPUT_LIFT_TYPE = "thrust_direction"
INPUT_LIFT_LABELS = RAW_INPUT_LABELS + [
    "thrust_x", "thrust_y", "thrust_z",
    "tau_roll_p", "tau_pitch_q", "tau_yaw_r",
]
LEGACY_INPUT_LIFT_LABELS = RAW_INPUT_LABELS + ["thrust_x", "thrust_y", "thrust_z"]


def thrust_direction_from_state_phys(states_phys):
    """Return the world-frame thrust direction from physical 12-state samples."""
    states = np.asarray(states_phys, dtype=float)
    scalar = states.ndim == 1
    states_2d = np.atleast_2d(states)

    if states_2d.shape[1] < STATE_DIM:
        raise ValueError(
            f"Expected at least {STATE_DIM} state entries, got {states_2d.shape[1]}"
        )

    phi = states_2d[:, 6]
    theta = states_2d[:, 7]
    psi = states_2d[:, 8]

    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_theta, c_theta = np.sin(theta), np.cos(theta)
    s_psi, c_psi = np.sin(psi), np.cos(psi)

    dirs = np.column_stack([
        c_psi*s_theta*c_phi + s_psi*s_phi,
        s_psi*s_theta*c_phi - c_psi*s_phi,
        c_theta*c_phi,
    ])
    return dirs[0] if scalar else dirs


def lift_inputs_from_phys(states_phys, raw_inputs):
    """Map raw commands to the learned EDMDc input vector."""
    states = np.asarray(states_phys, dtype=float)
    raw = np.asarray(raw_inputs, dtype=float)
    scalar = states.ndim == 1 and raw.ndim == 1

    states_2d = np.atleast_2d(states)
    raw_2d = np.atleast_2d(raw)

    if raw_2d.shape[1] < RAW_INPUT_DIM:
        raise ValueError(
            f"Expected {RAW_INPUT_DIM} raw input channels, got {raw_2d.shape[1]}"
        )

    if states_2d.shape[0] == 1 and raw_2d.shape[0] > 1:
        states_2d = np.repeat(states_2d, raw_2d.shape[0], axis=0)
    elif raw_2d.shape[0] == 1 and states_2d.shape[0] > 1:
        raw_2d = np.repeat(raw_2d, states_2d.shape[0], axis=0)

    if states_2d.shape[0] != raw_2d.shape[0]:
        raise ValueError(
            f"State/input sample mismatch: {states_2d.shape[0]} vs {raw_2d.shape[0]}"
        )

    raw_4 = raw_2d[:, :RAW_INPUT_DIM]
    thrust = raw_4[:, :1]
    thrust_dir = thrust_direction_from_state_phys(states_2d)
    lifted_parts = [raw_4, thrust * thrust_dir]
    if raw_2d.shape[1] >= RAW_INPUT_DIM:
        rates = states_2d[:, 9:12]
        lifted_parts.append(raw_4[:, 1:4] * rates)
    lifted = np.hstack(lifted_parts)
    return lifted[0] if scalar else lifted


def scaled_lifted_input_from_phys(state_phys, raw_input, u_scaler):
    lifted = lift_inputs_from_phys(state_phys, raw_input)
    expected = int(getattr(u_scaler, "n_features_in_", np.asarray(lifted).shape[-1]))
    lifted = np.asarray(lifted, dtype=float)[..., :expected]
    return u_scaler.transform(np.atleast_2d(lifted)).flatten()


# File I/O
def load_edmdc_model(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_simulation_runs(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["t"], data["states"], data["U"], data["ref_traj_list"]

# State lifting
# The current lifted model uses the full 12-state vector:
# [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
#
# Observable ordering must remain consistent with EDMDc_training.py:
#  - states
#  - sin/cos of roll, pitch, and yaw
#  - selected cross terms
#  - quadratic energy-like terms
#  - constant bias

def _scaler_state_dim(scaler):
    return int(getattr(scaler, "n_features_in_", len(getattr(scaler, "mean_", []))))


def _observables_10state(x_std, scaler):
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


def _observables_12state(x_std, scaler):
    x = np.asarray(x_std).flatten()
    assert len(x) == STATE_DIM, f"Expected 12-state vector, got {len(x)}"

    obs = list(x)  # 12 linear terms

    phi_rad   = x[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_rad = x[7] * scaler.scale_[7] + scaler.mean_[7]
    psi_rad   = x[8] * scaler.scale_[8] + scaler.mean_[8]

    s_phi, c_phi = np.sin(phi_rad), np.cos(phi_rad)
    s_theta, c_theta = np.sin(theta_rad), np.cos(theta_rad)
    s_psi, c_psi = np.sin(psi_rad), np.cos(psi_rad)

    obs.extend([
        s_phi, c_phi,
        s_theta, c_theta,
        s_psi, c_psi,
    ])

    obs.extend([
        x[6] * x[9],     # phi * p
        x[7] * x[10],    # theta * q
        x[8] * x[11],    # psi * r
        x[3] * x[6],     # vx * phi
        x[4] * x[7],     # vy * theta
        x[3] * x[8],     # vx * psi
        x[4] * x[8],     # vy * psi
        x[5] * x[7],     # vz * theta
    ])

    obs.extend([
        x[3]**2 + x[4]**2 + x[5]**2,       # v_sq
        x[9]**2 + x[10]**2 + x[11]**2,     # omega_sq
    ])

    obs.extend([
        x[5] * x[5],     # vz^2
        x[6] * x[6],     # phi^2
        x[7] * x[7],     # theta^2
        x[8] * x[8],     # psi^2
        x[9] * x[10],    # p * q
        x[10] * x[11],   # q * r
        x[9] * x[11],    # p * r
    ])

    obs.extend([
        x[0] * x[1],     # x * y
        x[0] * x[3],     # x * vx
        x[1] * x[4],     # y * vy
        x[0] * x[4],     # x * vy
        x[1] * x[3],     # y * vx
        x[3] * x[4],     # vx * vy
        x[0] * x[0],     # x^2
        x[1] * x[1],     # y^2
        x[3] * x[3],     # vx^2
        x[4] * x[4],     # vy^2
        x[0] * x[7],     # x * theta
        x[1] * x[6],     # y * phi
        x[3] * x[7],     # vx * theta
        x[4] * x[6],     # vy * phi
    ])

    vx, vy, vz = x[3], x[4], x[5]
    body_vx = c_theta*c_psi*vx + c_theta*s_psi*vy - s_theta*vz
    body_vy = (s_phi*s_theta*c_psi - c_phi*s_psi)*vx + \
              (s_phi*s_theta*s_psi + c_phi*c_psi)*vy + \
              s_phi*c_theta*vz
    body_vz = (c_phi*s_theta*c_psi + s_phi*s_psi)*vx + \
              (c_phi*s_theta*s_psi - s_phi*c_psi)*vy + \
              c_phi*c_theta*vz

    thrust_dir_x = c_psi*s_theta*c_phi + s_psi*s_phi
    thrust_dir_y = s_psi*s_theta*c_phi - c_psi*s_phi
    thrust_dir_z = c_theta*c_phi

    obs.extend([
        body_vx, body_vy, body_vz,
        thrust_dir_x, thrust_dir_y, thrust_dir_z,
    ])

    obs.append(1.0)

    return np.asarray(obs, dtype=float)


def observables(x_std, scaler):
    """
    Return the lifted observable vector for a standardized state.

    The observable definition must match the lifting used during training.
    A 10-state branch is kept only so old 10-state model files can still load.
    """
    expected_dim = _scaler_state_dim(scaler)
    if expected_dim == 10:
        return _observables_10state(x_std, scaler)
    if expected_dim == STATE_DIM:
        return _observables_12state(x_std, scaler)
    raise ValueError(f"Unsupported scaler state dimension: {expected_dim}")


def drop_to_10state(x12):
    """Convert a 12-state vector to the legacy reduced 10-state representation."""
    x = np.asarray(x12, dtype=float).flatten()
    if len(x) == 10:
        return x.copy()
    if len(x) < STATE_DIM:
        raise ValueError(f"Expected at least 12 entries, got {len(x)}")
    return x[REDUCED_10_INDICES].copy()


def drop_to_12state(x_state):
    """Convert a plant/logged state to the full EDMD 12-state representation."""
    x = np.asarray(x_state, dtype=float).flatten()
    if len(x) >= STATE_DIM:
        return x[:STATE_DIM].copy()
    if len(x) == 10:
        x12 = np.zeros(STATE_DIM)
        x12[0:6] = x[0:6]
        x12[6:8] = x[6:8]
        x12[9:11] = x[8:10]
        return x12
    raise ValueError(f"Expected 10-state or 12-state vector, got {len(x)}")


def lifted_state_from_x(x_state, scaler):
    """Map a physical state vector to the lifted observable space."""
    expected_dim = _scaler_state_dim(scaler)
    if expected_dim == 10:
        x_phys = drop_to_10state(x_state)
    elif expected_dim == STATE_DIM:
        x_phys = drop_to_12state(x_state)
    else:
        raise ValueError(f"Unsupported scaler state dimension: {expected_dim}")

    x_std = scaler.transform(x_phys.reshape(1, -1)).flatten()
    return observables(x_std, scaler)


# MPC solver
class EDMDcMPC_QP:
    """
    Quadratic-program MPC controller built on a lifted linear EDMDc model.

    The optimizer penalizes tracking error in the physical-state coordinates
    selected by Cz while optimizing control increments over the control horizon.
    """
    def __init__(self, A, B, Cz, N, NC, Q, R, Rd,
                 u_scaler, du_min, du_max, u_nominal_raw,
                 state_scaler=None, input_lift_type=None, raw_input_dim=None,
                 Q_terminal=None):
        self.A  = np.asarray(A, dtype=float)
        self.B_model = np.asarray(B, dtype=float)
        self.Cz = np.asarray(Cz, dtype=float)
        self.N  = int(N)
        self.NC = int(NC)
        self.Q  = np.asarray(Q,  dtype=float)
        self.Q_terminal = (
            np.asarray(Q_terminal, dtype=float)
            if Q_terminal is not None else self.Q
        )
        self.R  = np.asarray(R,  dtype=float)
        self.Rd = np.asarray(Rd, dtype=float)

        self.u_scaler = u_scaler
        self.state_scaler = state_scaler
        self.input_lift_type = input_lift_type
        self.raw_input_dim = raw_input_dim

        self.du_min = np.asarray(du_min, dtype=float)
        self.du_max = np.asarray(du_max, dtype=float)

        self.nz   = self.A.shape[0]   # observable dimension
        self.model_nu = self.B_model.shape[1]
        nominal_input_size = np.asarray(u_nominal_raw, dtype=float).size
        self.uses_lifted_input = (
            self.input_lift_type in (INPUT_LIFT_TYPE, LEGACY_INPUT_LIFT_TYPE)
            or (self.model_nu in (len(INPUT_LIFT_LABELS), len(LEGACY_INPUT_LIFT_LABELS))
                and getattr(self.u_scaler, "n_features_in_", self.model_nu) == self.model_nu
                and (raw_input_dim == RAW_INPUT_DIM or nominal_input_size == RAW_INPUT_DIM))
        )
        if self.uses_lifted_input:
            if self.state_scaler is None:
                raise ValueError("state_scaler is required for thrust-direction input lifting")
            self.raw_input_dim = RAW_INPUT_DIM if self.raw_input_dim is None else int(self.raw_input_dim)
            self.nu = self.raw_input_dim
            self._lift_state_phys = np.zeros(STATE_DIM)
        else:
            self.nu = self.model_nu
        self.B = self.B_model[:, :self.nu] if self.uses_lifted_input else self.B_model

        self.nx   = self.Cz.shape[0]  # tracked physical-state dimension
        self.nvar = self.NC * self.nu

        if self.du_min.size != self.nu or self.du_max.size != self.nu:
            raise ValueError(
                f"du_min/du_max must have length {self.nu}, got "
                f"{self.du_min.size}/{self.du_max.size}"
            )

        self._set_nominal_input(u_nominal_raw)
        self._du_prev = np.zeros(self.nvar)

        q_blocks = [sp.csc_matrix(self.Q) for _ in range(max(self.N - 1, 0))]
        q_blocks.append(sp.csc_matrix(self.Q_terminal))
        self.Qbar = sp.block_diag(q_blocks, format="csc")
        self.Rbar = sp.block_diag(
            [sp.csc_matrix(self.R) for _ in range(self.NC)], format="csc")
        self.D    = self._build_difference_matrix()
        self.Rdbar = (
            sp.block_diag([sp.csc_matrix(self.Rd)
                           for _ in range(self.NC - 1)], format="csc")
            if self.NC > 1 else None
        )

        self.Aineq = sp.eye(self.nvar, format="csc")
        self.l = np.tile(self.du_min, self.NC)
        self.u_bound = np.tile(self.du_max, self.NC)

        self._refresh_prediction_model()
        if osqp is None:
            raise ModuleNotFoundError(
                "osqp is required for EDMDcMPC_QP closed-loop optimization. "
                "Install osqp or use compare_three.py for rollout comparison."
            )
        self.prob = osqp.OSQP()
        self.prob.setup(P=self.P, q=np.zeros(self.nvar),
                        A=self.Aineq, l=self.l, u=self.u_bound,
                        warm_start=True, verbose=False, polish=False)

    def _set_nominal_input(self, u_nominal_raw):
        self.u_nom_raw = np.asarray(u_nominal_raw, dtype=float).flatten()
        if self.u_nom_raw.size < self.nu:
            raise ValueError(f"Expected at least {self.nu} nominal inputs, got {self.u_nom_raw.size}")
        self.u_nom_raw = self.u_nom_raw[:self.nu]

        if self.uses_lifted_input:
            self.u_nom_model_scaled = scaled_lifted_input_from_phys(
                self._lift_state_phys, self.u_nom_raw, self.u_scaler
            )
            self.u_nom_scaled = (
                (self.u_nom_raw - self.u_scaler.mean_[:self.nu])
                / self.u_scaler.scale_[:self.nu]
            )
        else:
            self.u_nom_scaled = self.u_scaler.transform(
                self.u_nom_raw.reshape(1, -1)).flatten()
            self.u_nom_model_scaled = self.u_nom_scaled

        if hasattr(self, "NC"):
            self.u_nom_horizon_scaled = np.tile(self.u_nom_scaled, self.NC)
            self.u_nom_model_horizon_scaled = np.tile(self.u_nom_model_scaled, self.NC)

    def _input_lift_jacobian(self):
        """
        Map raw standardized command deltas to lifted standardized input deltas.

        The learned model input starts with raw commands, then may include
        thrust-direction and torque-rate coupling channels. The QP still
        optimizes the 4 real commands, so lifted columns need local
        state-dependent sensitivities.
        """
        if not self.uses_lifted_input:
            return np.eye(self.model_nu)

        J = np.zeros((self.model_nu, self.nu))
        J[:self.nu, :self.nu] = np.eye(self.nu)

        thrust_dir = thrust_direction_from_state_phys(self._lift_state_phys)
        thrust_scale = self.u_scaler.scale_[0]
        for axis in range(3):
            lifted_idx = RAW_INPUT_DIM + axis
            if lifted_idx >= self.model_nu:
                continue
            J[lifted_idx, 0] = (
                thrust_scale * thrust_dir[axis] / self.u_scaler.scale_[lifted_idx]
            )
        rate_start = RAW_INPUT_DIM + 3
        rates = self._lift_state_phys[9:12]
        for axis in range(3):
            lifted_idx = rate_start + axis
            raw_idx = 1 + axis
            if lifted_idx >= self.model_nu or raw_idx >= self.nu:
                continue
            J[lifted_idx, raw_idx] = (
                self.u_scaler.scale_[raw_idx] * rates[axis]
                / self.u_scaler.scale_[lifted_idx]
            )
        return J

    def _refresh_prediction_model(self):
        if self.uses_lifted_input:
            self.B = self.B_model @ self._input_lift_jacobian()
        else:
            self.B = self.B_model

        self.Sz, self.Su, self.Su_model = self._build_prediction_matrices()

        Su_dense = self.Su.toarray()
        Su_phys = np.zeros((self.N * self.nx, self.nvar))
        for i in range(self.N):
            for j in range(self.NC):
                Su_phys[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = (
                    self.Cz @ Su_dense[i*self.nz:(i+1)*self.nz,
                                       j*self.nu:(j+1)*self.nu]
                )
        self.Su_phys = sp.csc_matrix(Su_phys)
        self.P = self._build_hessian()

    def _setup_solver(self):
        self.prob = osqp.OSQP()
        self.prob.setup(P=self.P, q=np.zeros(self.nvar),
                        A=self.Aineq, l=self.l, u=self.u_bound,
                        warm_start=True, verbose=False, polish=False)

    def _set_lift_state_from_z(self, z0):
        if not self.uses_lifted_input:
            return
        z0 = np.asarray(z0, dtype=float).flatten()
        x_std = z0[:STATE_DIM]
        self._lift_state_phys = self.state_scaler.inverse_transform(
            x_std.reshape(1, -1)
        ).flatten()
        self._set_nominal_input(self.u_nom_raw)

    def _build_prediction_matrices(self):
        Sz = np.zeros((self.N * self.nz, self.nz))
        Su = np.zeros((self.N * self.nz, self.NC * self.nu))
        Su_model = np.zeros((self.N * self.nz, self.NC * self.model_nu))
        A_pow = [np.eye(self.nz)]
        for _ in range(self.N):
            A_pow.append(A_pow[-1] @ self.A)
        for i in range(self.N):
            Sz[i*self.nz:(i+1)*self.nz, :] = A_pow[i+1]
            for input_step in range(i + 1):
                # The optimized command is held constant after the control
                # horizon, so the last decision block affects all later steps.
                j = min(input_step, self.NC - 1)
                Su[i*self.nz:(i+1)*self.nz, j*self.nu:(j+1)*self.nu] += \
                    A_pow[i-input_step] @ self.B
                Su_model[i*self.nz:(i+1)*self.nz, j*self.model_nu:(j+1)*self.model_nu] += \
                    A_pow[i-input_step] @ self.B_model
        return sp.csc_matrix(Sz), sp.csc_matrix(Su), sp.csc_matrix(Su_model)

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
        # The learned EDMD model uses absolute standardized inputs. The QP
        # variables are deltas around the nominal input, so the nominal input
        # response must be part of the free trajectory.
        z_free = self.Sz @ z0 + self.Su_model @ self.u_nom_model_horizon_scaled
        x_free = np.array([
            self.Cz @ z_free[i*self.nz:(i+1)*self.nz]
            for i in range(self.N)
        ]).reshape(-1)
        x_ref = x_ref_std_horizon.reshape(-1)
        return np.asarray(
            self.Su_phys.T @ (self.Qbar @ (x_free - x_ref))
        ).reshape(-1)

    def compute(self, z0, x_ref_std_horizon, u_nominal_raw=None):
        self._set_lift_state_from_z(z0)
        if u_nominal_raw is not None:
            self._set_nominal_input(u_nominal_raw)

        if self.uses_lifted_input:
            self._refresh_prediction_model()
            self._setup_solver()

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
        if self.uses_lifted_input:
            u0_raw = u0_scaled * self.u_scaler.scale_[:self.nu] + self.u_scaler.mean_[:self.nu]
        else:
            u0_raw = self.u_scaler.inverse_transform(
                u0_scaled.reshape(1, -1)).flatten()
        return u0_raw  # [thrust, tau_roll, tau_pitch, optional tau_yaw]


# Reference processing
def wrap_angle_pi(angle):
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def reference_yaw_arrays(ref_traj, dt=None):
    """Return unwrapped yaw and yaw-rate references for a trajectory list."""
    yaw = np.unwrap([
        float(wp.get("yaw", 0.0)) for wp in ref_traj
    ])

    yaw_rate = np.zeros_like(yaw)
    explicit = [
        float(wp["yaw_rate"]) if "yaw_rate" in wp else np.nan
        for wp in ref_traj
    ]
    explicit = np.asarray(explicit, dtype=float)
    has_explicit = np.isfinite(explicit)
    if np.any(has_explicit):
        yaw_rate[has_explicit] = explicit[has_explicit]

    if np.any(~has_explicit) and dt is not None and len(yaw) > 1:
        computed = np.gradient(yaw, float(dt))
        yaw_rate[~has_explicit] = computed[~has_explicit]

    return yaw, yaw_rate


def extract_ref_xyz(ref_traj):
    return np.array([wp["pos"][:3] for wp in ref_traj], dtype=float)


def precompute_ref_std(ref_traj, scaler, n_states=None, dt=None):
    """Build a standardized reference trajectory using position, velocity, and yaw."""
    T = len(ref_traj)
    expected_dim = _scaler_state_dim(scaler)
    if n_states is None or n_states != expected_dim:
        n_states = expected_dim

    X_ref = np.zeros((T, n_states))
    yaw_values = None
    yaw_rate_values = None
    if n_states >= STATE_DIM:
        yaw_values, yaw_rate_values = reference_yaw_arrays(ref_traj, dt=dt)

    for k in range(T):
        X_ref[k, 0:3] = ref_traj[k]["pos"][:3]
        X_ref[k, 3:6] = ref_traj[k]["vel"][:3]
        if yaw_values is not None:
            X_ref[k, 8] = yaw_values[k]
        if yaw_rate_values is not None:
            X_ref[k, 11] = yaw_rate_values[k]
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
