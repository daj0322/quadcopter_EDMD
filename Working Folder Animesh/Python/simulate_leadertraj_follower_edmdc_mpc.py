"""
simulate_leadertraj_follower_edmdc_mpc.py
=========================================
Python translation + EDMD-MPC integration of simulate_leadertraj_follower_mpc.m

Pipeline
--------
  1. parallel_sim.py   – generate trajectory families + PRBS data
  2. mix_traj.py       – combine all .pkl files into runs_mixed_n300.pkl
  3. EDMDc_training.py – train A, B; saves edmdc_model_0.1.pkl
  4. edmdc_mpc.py      – validates EDMD-MPC on held-out test run
  5. THIS FILE         – leader-follower simulation using EDMD-MPC for the follower

Reference (leader) trajectory
-------------------------------
  A purely geometric multi-segment helical path — no drone dynamics.
  Segment 1  [0   → T1=1  s]  Vertical lift-off to helix base
  Segment 2  [T1  → T2=6  s]  Rising helix, 1 full turn, +4 m
  Segment 3  [T2  → T3=9  s]  Flat circle at peak altitude, half turn
  Segment 4  [T3  → T4=12 s]  Descending helix, half turn, −2 m
  Hold                         Final position held for remainder

Follower
--------
  EDMD-MPC (from edmdc_mpc.py) driving the Simulation inner-PID.
  The inner PID converts [thrust, phi_des, theta_des] → rotor speeds.
  State is re-lifted from the true 12-state every EDMD step (dt=0.1 s).

Capture / dwell logic  (identical to MATLAB)
---------------------------------------------
  Capture radius  R_capture = 0.05 m
  Required dwell  T_dwell   = 2.0 s  continuously inside capture radius
  Simulation ends immediately on confirmation.

Plots (clipped to interception + 0.5 s buffer)
------------------------------------------------
  1. 3D — full reference path + follower
  2. 3D — reference clipped at confirmed interception
  3. x / y / z position over time
  4. Separation distance vs time
  5. Follower EDMD-MPC control inputs  [thrust, phi_des, theta_des]
  6. All 10 follower states — EDMD-MPC vs ground-truth
"""

# ============================================================
# IMPORTS
# ============================================================
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import osqp
import scipy.sparse as sp

# -- project imports (same folder or on PYTHONPATH) ----------
from Simulation import quad_sim          # quad dynamics + PID

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR       = Path(__file__).resolve().parent
EDMDC_MODEL_FILE = SCRIPT_DIR / "edmdc_model_0.1.pkl"

# EDMD-MPC horizon  (must match training dt = 0.1 s)
N_EDMD  = 50     # prediction steps → 5 s look-ahead
NC_EDMD = 10     # control horizon

# Cost weights — position heavily penalised, angles lightly
Q_DIAG_EDMD = np.array([
    750_000.0, 750_000.0, 750_000.0,   # x, y, z
        100.0,     100.0,     100.0,   # vx, vy, vz
          0.0,       0.0,              # phi, theta
          0.0,       0.0,              # p, q
], dtype=float)

R_DIAG_EDMD  = np.array([0.001, 0.1, 0.1], dtype=float)
RD_DIAG_EDMD = np.array([0.0001, 0.01, 0.01], dtype=float)

# Delta-u bounds in scaled space  [thrust, phi_des, theta_des]
DU_MIN_EDMD = np.array([-5.0, -3.5, -3.5], dtype=float)
DU_MAX_EDMD = np.array([ 5.0,  3.5,  3.5], dtype=float)

# Reference (leader) trajectory geometry
R_HELIX = 2.0   # [m]  helix radius
H_HELIX = 4.0   # [m]  total rise in segment 2
H_DOWN  = 2.0   # [m]  descent in segment 4
Z_BASE  = 0.5   # [m]  lift-off altitude (end of segment 1)

T1 = 1.0        # [s] end of lift-off
T2 = T1 + 5.0  # [s] end of rising helix
T3 = T2 + 3.0  # [s] end of flat circle
T4 = T3 + 3.0  # [s] end of descending helix
T_END = 20.0    # [s] total simulation window

# Simulation time step for geometry evaluation (= inner PID step)
Ts_inner = 0.01  # [s]  high-frequency reference evaluation
Ts_edmd  = 0.1   # [s]  EDMD-MPC step  (must match model dt)

# Capture / dwell parameters  (match MATLAB)
R_CAPTURE = 0.5   # [m]
T_DWELL   = 2.0    # [s]  continuous dwell required
PLOT_BUF  = 0.5    # [s]  post-interception plot buffer

# Initial follower offset (easy to modify)
#FOLLOWER_INIT_POS = np.array([0.0, -1.0, 0.0])  # 1 m to the side
FOLLOWER_INIT_POS = np.array([0.0, -30.0, 0.0])

# ============================================================
# OBSERVABLES  (must match EDMDc_training.py EXACTLY)
# ============================================================
def observables(x_std: np.ndarray, scaler) -> np.ndarray:
    """
    Lift a standardised 10-state vector to the EDMD observable space (21-dim).

    State layout after trimming 12→10:
      [x, y, z, vx, vy, vz, phi, theta, p, q]
    Observables:
      [0-9]   10 linear states
      [10-13] sin(phi), cos(phi), sin(theta), cos(theta)
      [14-17] phi*p, theta*q, vx*phi, vy*theta
      [18]    v_sq  = vx²+vy²+vz²
      [19]    omega_sq = p²+q²
      [20]    bias  = 1
    """
    x = np.asarray(x_std).flatten()
    assert len(x) == 10, f"Expected 10-state vector, got {len(x)}"

    obs = list(x)   # 10 linear terms

    # Trig terms — unscale to radians first
    phi_rad   = x[6] * scaler.scale_[6] + scaler.mean_[6]
    theta_rad = x[7] * scaler.scale_[7] + scaler.mean_[7]
    obs += [np.sin(phi_rad), np.cos(phi_rad),
            np.sin(theta_rad), np.cos(theta_rad)]

    # Cross terms
    obs.append(x[6] * x[8])   # phi * p
    obs.append(x[7] * x[9])   # theta * q
    obs.append(x[3] * x[6])   # vx * phi
    obs.append(x[4] * x[7])   # vy * theta

    # Energy-like scalars
    obs.append(x[3]**2 + x[4]**2 + x[5]**2)   # v_sq
    obs.append(x[8]**2 + x[9]**2)              # omega_sq

    # Bias
    obs.append(1.0)

    return np.asarray(obs, dtype=float)


def lifted_state_from_x12(x12: np.ndarray, scaler) -> np.ndarray:
    """12-state → 10-state → standardise → lift to observable space."""
    x10  = drop_to_10state(x12)
    x_std = scaler.transform(x10.reshape(1, -1)).flatten()
    return observables(x_std, scaler)


def drop_to_10state(x12: np.ndarray) -> np.ndarray:
    """Drop psi (index 8) and r (index 11) from the 12-state vector."""
    return x12[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]


# ============================================================
# EDMD-MPC QP CLASS  (identical to edmdc_mpc.py)
# ============================================================
class EDMDcMPC_QP:
    """
    Receding-horizon MPC built on the linear EDMD surrogate model.

    Decision variables : delta-u in scaled input space over NC steps.
    Objective          : track reference physical states via Cz projection.
    Constraints        : box bounds on delta-u.
    Solver             : OSQP (warm-started every step).
    """

    def __init__(self, A, B, Cz, N, NC, Q, R, Rd,
                 u_scaler, du_min, du_max, u_nominal_raw):
        self.A  = np.asarray(A,  dtype=float)
        self.B  = np.asarray(B,  dtype=float)
        self.Cz = np.asarray(Cz, dtype=float)
        self.N  = int(N)
        self.NC = int(NC)
        self.Q  = np.asarray(Q,  dtype=float)
        self.R  = np.asarray(R,  dtype=float)
        self.Rd = np.asarray(Rd, dtype=float)

        self.u_scaler     = u_scaler
        self.u_nom_raw    = np.asarray(u_nominal_raw, dtype=float)
        self.u_nom_scaled = u_scaler.transform(
            self.u_nom_raw.reshape(1, -1)).flatten()

        self.du_min = np.asarray(du_min, dtype=float)
        self.du_max = np.asarray(du_max, dtype=float)

        self.nz   = self.A.shape[0]   # observable dim (21)
        self.nu   = self.B.shape[1]   # input dim (3)
        self.nx   = self.Cz.shape[0]  # physical state dim (10)
        self.nvar = self.NC * self.nu

        self._du_prev = np.zeros(self.nvar)

        self.Sz, self.Su = self._build_prediction_matrices()

        # Project Su to physical state space via Cz
        Su_dense = self.Su.toarray()
        Su_phys  = np.zeros((self.N * self.nx, self.nvar))
        for i in range(self.N):
            for j in range(self.NC):
                Su_phys[i*self.nx:(i+1)*self.nx,
                         j*self.nu:(j+1)*self.nu] = \
                    self.Cz @ Su_dense[i*self.nz:(i+1)*self.nz,
                                        j*self.nu:(j+1)*self.nu]
        self.Su_phys = sp.csc_matrix(Su_phys)

        self.Qbar  = sp.block_diag(
            [sp.csc_matrix(self.Q) for _ in range(self.N)], format="csc")
        self.Rbar  = sp.block_diag(
            [sp.csc_matrix(self.R) for _ in range(self.NC)], format="csc")
        self.D     = self._build_difference_matrix()
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

    # ---- internal builders ----------------------------------------
    def _build_prediction_matrices(self):
        Sz = np.zeros((self.N * self.nz, self.nz))
        Su = np.zeros((self.N * self.nz, self.NC * self.nu))
        A_pow = [np.eye(self.nz)]
        for _ in range(self.N):
            A_pow.append(A_pow[-1] @ self.A)
        for i in range(self.N):
            Sz[i*self.nz:(i+1)*self.nz, :] = A_pow[i + 1]
            for j in range(min(i + 1, self.NC)):
                Su[i*self.nz:(i+1)*self.nz,
                    j*self.nu:(j+1)*self.nu] = A_pow[i - j] @ self.B
        return sp.csc_matrix(Sz), sp.csc_matrix(Su)

    def _build_difference_matrix(self):
        if self.NC <= 1:
            return None
        rows, cols, vals = [], [], []
        for k in range(self.NC - 1):
            for j in range(self.nu):
                r = k * self.nu + j
                rows.extend([r, r])
                cols.extend([k * self.nu + j, (k + 1) * self.nu + j])
                vals.extend([-1.0, 1.0])
        return sp.coo_matrix(
            (vals, (rows, cols)),
            shape=((self.NC - 1) * self.nu, self.NC * self.nu)).tocsc()

    def _build_hessian(self):
        P = self.Su_phys.T @ self.Qbar @ self.Su_phys + self.Rbar
        if self.D is not None and self.Rdbar is not None:
            P = P + self.D.T @ self.Rdbar @ self.D
        return (0.5 * (P + P.T)).tocsc()

    def _build_q(self, z0, x_ref_std_horizon):
        z_free = self.Sz @ z0
        x_free = np.array([
            self.Cz @ z_free[i * self.nz:(i + 1) * self.nz]
            for i in range(self.N)
        ]).reshape(-1)
        x_ref = x_ref_std_horizon.reshape(-1)
        return np.asarray(
            self.Su_phys.T @ (self.Qbar @ (x_free - x_ref))
        ).reshape(-1)

    # ---- public API -----------------------------------------------
    def compute(self, z0: np.ndarray, x_ref_std_horizon: np.ndarray) -> np.ndarray:
        """Return optimal [thrust, phi_des, theta_des] in raw (physical) units."""
        q = self._build_q(z0, x_ref_std_horizon)
        self.prob.update(q=q)
        self.prob.warm_start(x=self._du_prev)
        res = self.prob.solve()

        if res.info.status not in ("solved", "solved inaccurate"):
            print(f"  [OSQP] Warning: {res.info.status}")
            du0 = self._du_prev[:self.nu]
        else:
            du_opt = np.asarray(res.x).reshape(-1)
            self._du_prev = du_opt.copy()
            du0 = du_opt[:self.nu]

        u0_scaled = self.u_nom_scaled + du0
        u0_raw    = self.u_scaler.inverse_transform(
            u0_scaled.reshape(1, -1)).flatten()
        return u0_raw   # [thrust [N], phi_des [rad], theta_des [rad]]


# ============================================================
# REFERENCE TRAJECTORY  (pure geometry — no dynamics)
# ============================================================
def build_leader_reference(t_vec: np.ndarray) -> np.ndarray:
    """
    Evaluate the multi-segment helical leader path at each time in t_vec.

    Returns
    -------
    ref_pos : (3, N) array of [x, y, z] positions.
    """
    nstep   = len(t_vec)
    ref_pos = np.zeros((3, nstep))

    k_T4 = int(min(np.searchsorted(t_vec, T4), nstep - 1))

    for k, tk in enumerate(t_vec):
        if tk <= T1:
            # Segment 1: vertical lift-off at (R_HELIX, 0)
            alpha        = tk / T1
            ref_pos[:, k] = [R_HELIX, 0.0, alpha * Z_BASE]

        elif tk <= T2:
            # Segment 2: rising helix, 1 full turn
            alpha        = (tk - T1) / (T2 - T1)
            theta_h      = 2.0 * np.pi * alpha
            ref_pos[:, k] = [R_HELIX * np.cos(theta_h),
                              R_HELIX * np.sin(theta_h),
                              Z_BASE + H_HELIX * alpha]

        elif tk <= T3:
            # Segment 3: flat circle at peak altitude, half turn
            z_peak       = Z_BASE + H_HELIX
            alpha        = (tk - T2) / (T3 - T2)
            theta_h      = 2.0 * np.pi + np.pi * alpha   # 2π → 3π
            ref_pos[:, k] = [R_HELIX * np.cos(theta_h),
                              R_HELIX * np.sin(theta_h),
                              z_peak]

        elif tk <= T4:
            # Segment 4: descending helix, half turn
            z_peak       = Z_BASE + H_HELIX
            alpha        = (tk - T3) / (T4 - T3)
            theta_h      = 3.0 * np.pi + np.pi * alpha   # 3π → 4π
            ref_pos[:, k] = [R_HELIX * np.cos(theta_h),
                              R_HELIX * np.sin(theta_h),
                              z_peak - H_DOWN * alpha]

        else:
            # Hold final position
            ref_pos[:, k] = ref_pos[:, k_T4]

    return ref_pos


# ============================================================
# REFERENCE HORIZON HELPER  (for EDMD-MPC)
# ============================================================
def build_ref_horizon_from_pos(ref_pos_col: np.ndarray,
                                k_edmd: int,
                                N: int,
                                scaler,
                                n_states: int = 10) -> np.ndarray:
    """
    Build a standardised (N, 10) reference array for the MPC.

    Positions come from ref_pos_col (3, M_edmd).
    Velocities are finite-difference estimated and zero-padded where needed.
    Only x, y, z (indices 0-2) are tracked — rest are zero-reference.
    """
    M = ref_pos_col.shape[1]
    horizon = np.zeros((N, n_states))

    for i in range(N):
        ki = min(k_edmd + i, M - 1)
        horizon[i, 0:3] = ref_pos_col[:, ki]
        # Approximate velocity via finite difference
        if ki < M - 1:
            horizon[i, 3:6] = (ref_pos_col[:, ki + 1]
                                - ref_pos_col[:, ki]) / Ts_edmd
        else:
            horizon[i, 3:6] = 0.0
        # Angles, rates → 0 (hover trim)

    # Standardise
    horizon_std = scaler.transform(horizon)
    return horizon_std


# ============================================================
# LOAD EDMD MODEL
# ============================================================
def load_edmdc_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# MAIN
# ============================================================
def main():
    # ----------------------------------------------------------
    # 1. Load EDMD model
    # ----------------------------------------------------------
    print(f"Loading EDMD model from: {EDMDC_MODEL_FILE}")
    model    = load_edmdc_model(EDMDC_MODEL_FILE)
    A_edmd   = model["A"]
    B_edmd   = model["B"]
    scaler   = model["scaler"]
    u_scaler = model["u_scaler"]
    dt_model = model["dt"]
    n_obs    = model["n_obs"]

    assert np.isclose(dt_model, Ts_edmd), \
        f"Model dt={dt_model} ≠ Ts_edmd={Ts_edmd} — update CONFIG."

    print(f"  A: {A_edmd.shape}   B: {B_edmd.shape}   n_obs: {n_obs}")
    print(f"  Model dt: {dt_model} s")

    # ----------------------------------------------------------
    # 2. Initialise simulation
    # ----------------------------------------------------------
    sim = quad_sim()

    # ----------------------------------------------------------
    # 3. Build reference (leader) trajectory at EDMD resolution
    # ----------------------------------------------------------
    # High-res time grid (for geometry evaluation and dwell check)
    t_hi  = np.arange(0.0, T_END + Ts_inner, Ts_inner)
    nstep = len(t_hi)

    # EDMD-step time grid
    t_edmd = np.arange(0.0, T_END + Ts_edmd, Ts_edmd)
    M_edmd = len(t_edmd)

    ref_pos_hi   = build_leader_reference(t_hi)    # (3, nstep)  high-res
    ref_pos_edmd = build_leader_reference(t_edmd)  # (3, M_edmd) EDMD-res

    print(f"\nReference trajectory: 4 segments, active path = {T4:.1f} s")
    print(f"  High-res steps : {nstep}  (dt = {Ts_inner} s)")
    print(f"  EDMD steps     : {M_edmd} (dt = {Ts_edmd} s)")

    # ----------------------------------------------------------
    # 4. Build EDMD-MPC
    # ----------------------------------------------------------
    u_nominal = np.array([sim.q_mass * sim.g, 0.0, 0.0], dtype=float)

    # Cz extracts the 10 physical states from the 21-dim observable
    Cz = np.zeros((10, n_obs))
    Cz[:10, :10] = np.eye(10)

    mpc = EDMDcMPC_QP(
        A=A_edmd, B=B_edmd, Cz=Cz,
        N=N_EDMD, NC=NC_EDMD,
        Q=np.diag(Q_DIAG_EDMD),
        R=np.diag(R_DIAG_EDMD),
        Rd=np.diag(RD_DIAG_EDMD),
        u_scaler=u_scaler,
        du_min=DU_MIN_EDMD,
        du_max=DU_MAX_EDMD,
        u_nominal_raw=u_nominal,
    )

    # ----------------------------------------------------------
    # 5. Initialise follower state
    # ----------------------------------------------------------
    x_follower = np.zeros(12)
    x_follower[0:3] = FOLLOWER_INIT_POS  # offset from ref start

    # ----------------------------------------------------------
    # 6. Storage
    # ----------------------------------------------------------
    X_ref_log      = np.zeros((3,  nstep))
    X_follower_log = np.zeros((12, nstep))   # actual 12-state trajectory
    U_follower_log = np.zeros((3,  nstep))   # [thrust, phi_des, theta_des]

    # ----------------------------------------------------------
    # 7. Capture / dwell bookkeeping
    # ----------------------------------------------------------
    in_radius     = False
    dwell_count   = 0
    N_dwell       = round(T_DWELL / Ts_inner)
    capture_k     = nstep - 1    # sentinel
    confirmed_k   = nstep - 1   # sentinel
    sim_confirmed = False

    # Current EDMD command (held until next EDMD step)
    u_cmd = u_nominal.copy()

    # EDMD step counter
    edmd_k   = 0            # index into t_edmd / ref_pos_edmd
    next_edmd_t = 0.0       # time of next EDMD solve

    solve_times = []

    print("\nRunning leader-follower simulation...")
    t_wall_start = time.perf_counter()

    # ----------------------------------------------------------
    # 8. Main loop  — runs at Ts_inner, EDMD solves at Ts_edmd
    # ----------------------------------------------------------
    for k in range(nstep):
        tk = t_hi[k]

        # Record reference and follower state
        X_ref_log[:, k]      = ref_pos_hi[:, k]
        X_follower_log[:, k] = x_follower

        # ---- Capture / dwell check --------------------------
        sep = np.linalg.norm(x_follower[0:3] - ref_pos_hi[:, k])

        if sep <= R_CAPTURE:
            if not in_radius:
                in_radius   = True
                capture_k   = k
                dwell_count = 1
            else:
                dwell_count += 1

            if dwell_count >= N_dwell:
                confirmed_k   = k
                sim_confirmed = True
                print(f"\nINTERCEPTION CONFIRMED at t = {tk:.2f} s  "
                      f"({T_DWELL:.1f} s inside capture radius)")
                # Fill remaining log with last state
                if k + 1 < nstep:
                    X_ref_log[:, k+1:]      = ref_pos_hi[:, k+1:]
                    X_follower_log[:, k+1:] = x_follower[:, None]
                    U_follower_log[:, k+1:] = u_cmd[:, None]
                break
        else:
            in_radius   = False
            dwell_count = 0

        # ---- EDMD-MPC solve (every Ts_edmd seconds) ---------
        if tk >= next_edmd_t - 1e-9:
            # Re-lift current true state
            z_k = lifted_state_from_x12(x_follower, scaler)

            # Build reference horizon in standardised space
            x_ref_h = build_ref_horizon_from_pos(
                ref_pos_edmd, edmd_k, N_EDMD, scaler)

            # Solve QP
            t0    = time.perf_counter()
            u_cmd = mpc.compute(z_k, x_ref_h)
            solve_times.append(time.perf_counter() - t0)

            # Clip to physical limits
            u_cmd[0] = np.clip(u_cmd[0],
                               0.5 * sim.q_mass * sim.g,
                               2.0 * sim.q_mass * sim.g)
            u_cmd[1] = np.clip(u_cmd[1],
                               -sim.controller_PID.tilt_max,
                                sim.controller_PID.tilt_max)
            u_cmd[2] = np.clip(u_cmd[2],
                               -sim.controller_PID.tilt_max,
                                sim.controller_PID.tilt_max)

            edmd_k      += 1
            next_edmd_t += Ts_edmd

        U_follower_log[:, k] = u_cmd

        # ---- Step inner PID + dynamics (high-res Ts_inner) --
        x_follower = sim.sim_PID.fct_step_attitude(
            x_follower,
            u1        = u_cmd[0],
            phi_des   = u_cmd[1],
            theta_des = u_cmd[2],
            dt        = Ts_inner,
        )

        # Progress log every 200 steps
        if k % 200 == 0:
            print(f"  t={tk:.2f} s  pos=({x_follower[0]:.2f}, "
                  f"{x_follower[1]:.2f}, {x_follower[2]:.2f})  "
                  f"sep={sep:.3f} m")

    t_wall_elapsed = time.perf_counter() - t_wall_start
    print(f"\nSimulation wall time: {t_wall_elapsed:.1f} s")
    if solve_times:
        print(f"EDMD-MPC solve time — "
              f"avg: {1e3*np.mean(solve_times):.2f} ms  "
              f"max: {1e3*np.max(solve_times):.2f} ms")

    # ----------------------------------------------------------
    # 9. Determine plot window
    # ----------------------------------------------------------
    if sim_confirmed:
        sim_end_k = confirmed_k
    else:
        sim_end_k = nstep - 1
        print("WARNING: Interception NOT confirmed within simulation window.")

    # Safety clamp
    if not sim_confirmed or capture_k >= sim_end_k:
        capture_k = sim_end_k

    t_confirmed     = t_hi[sim_end_k]
    t_capture_start = t_hi[capture_k]

    buf_steps  = min(round(PLOT_BUF / Ts_inner), nstep - 1 - sim_end_k)
    plot_end_k = sim_end_k + buf_steps
    t_plot     = t_hi[:plot_end_k + 1]

    print(f"\nConfirmed interception time : {t_confirmed:.2f} s")
    print(f"Dwell began at              : {t_capture_start:.2f} s")
    print(f"Plot window ends at         : {t_hi[plot_end_k]:.2f} s")

    # Convenience slices
    pe  = plot_end_k + 1   # exclusive end for slices
    se  = sim_end_k  + 1

    X_ref_plot = X_ref_log[:,      :pe]
    X_fol_plot = X_follower_log[:, :pe]
    U_fol_plot = U_follower_log[:, :pe]

    dist_vec = np.linalg.norm(
        X_ref_log[:3, :pe] - X_follower_log[:3, :pe], axis=0)

    # ----------------------------------------------------------
    # 10. Plot 1 — 3D full view
    # ----------------------------------------------------------
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(ref_pos_hi[0], ref_pos_hi[1], ref_pos_hi[2],
            "b:", lw=1.0, label="Reference path (full)")
    ax.plot(X_ref_log[0, :pe], X_ref_log[1, :pe], X_ref_log[2, :pe],
            "b", lw=2.0, label="Reference (active)")
    ax.plot(X_follower_log[0, :pe], X_follower_log[1, :pe],
            X_follower_log[2, :pe],
            "g-", lw=1.8, label="Follower EDMD-MPC")

    # Markers
    ax.scatter(*ref_pos_hi[:, 0],   s=80, c="b", marker="s", label="Ref start")
    ax.scatter(*X_follower_log[:3, 0], s=80, c="g", marker="o",
               label="Follower start")
    ax.scatter(*X_follower_log[:3, capture_k], s=100, c="lime", marker="s",
               label=f"Entered capture r (t={t_capture_start:.2f} s)")
    ax.scatter(*X_follower_log[:3, sim_end_k], s=140, c="lime", marker="^",
               label=f"Confirmed (t={t_confirmed:.2f} s)")

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"3D — Interception confirmed at t = {t_confirmed:.2f} s")
    ax.legend(loc="best", fontsize=8); ax.grid(True)
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()

    # ----------------------------------------------------------
    # 11. Plot 2 — 3D clipped at confirmed interception
    # ----------------------------------------------------------
    fig2 = plt.figure(figsize=(10, 7))
    ax2  = fig2.add_subplot(111, projection="3d")
    ax2.plot(X_ref_log[0, :se], X_ref_log[1, :se], X_ref_log[2, :se],
             "b", lw=2.0, label="Reference (clipped)")
    ax2.plot(X_follower_log[0, :pe], X_follower_log[1, :pe],
             X_follower_log[2, :pe],
             "g-", lw=1.8, label="Follower EDMD-MPC")
    ax2.scatter(*ref_pos_hi[:, 0],       s=80,  c="b", marker="s")
    ax2.scatter(*X_follower_log[:3, 0],  s=80,  c="g", marker="o")
    ax2.scatter(*X_follower_log[:3, sim_end_k], s=140, c="lime", marker="^",
                label=f"Confirmed (t={t_confirmed:.2f} s)")
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]"); ax2.set_zlabel("z [m]")
    ax2.set_title("3D — Reference clipped at confirmed interception")
    ax2.legend(loc="best", fontsize=8); ax2.grid(True)
    ax2.view_init(elev=30, azim=45)
    plt.tight_layout()

    # ----------------------------------------------------------
    # 12. Plot 3 — x / y / z position vs time
    # ----------------------------------------------------------
    pos_labels = ["x [m]", "y [m]", "z [m]"]
    fig3, axs3 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for i in range(3):
        axs3[i].plot(t_hi[:se], X_ref_log[i, :se],
                     "b", lw=1.8, label="Reference")
        axs3[i].plot(t_plot, X_follower_log[i, :pe],
                     "g-", lw=1.5, label="Follower EDMD-MPC")
        axs3[i].axvspan(t_capture_start, t_confirmed,
                        color="green", alpha=0.12)
        axs3[i].axvline(t_capture_start, color="g", ls="--", lw=1.0,
                        label=f"Entered r<{R_CAPTURE:.2f} m")
        axs3[i].axvline(t_confirmed,     color="g", ls="-",  lw=1.2,
                        label="Confirmed")
        axs3[i].set_ylabel(pos_labels[i])
        axs3[i].grid(True)
        if i == 0:
            axs3[i].legend(loc="best", fontsize=8)
    axs3[-1].set_xlabel("t [s]")
    fig3.suptitle(f"Position — Interception confirmed at t = {t_confirmed:.2f} s")
    plt.tight_layout()

    # ----------------------------------------------------------
    # 13. Plot 4 — Separation distance
    # ----------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(t_plot, dist_vec, "g", lw=1.5, label="Separation")
    ax4.axhline(R_CAPTURE, color="r", ls="--", lw=1.0,
                label=f"Capture radius {R_CAPTURE:.2f} m")
    ax4.axvline(t_capture_start, color="g", ls="--", lw=1.0,
                label=f"Entered radius (t={t_capture_start:.2f} s)")
    ax4.axvline(t_confirmed,     color="g", ls="-",  lw=1.2,
                label=f"Confirmed (t={t_confirmed:.2f} s)")
    ylims = ax4.get_ylim()
    ax4.fill_betweenx(ylims,
                      t_capture_start, t_confirmed,
                      color="green", alpha=0.12)
    ax4.set_xlabel("t [s]")
    ax4.set_ylabel("Distance [m]")
    ax4.set_title(
        f"Separation: Reference vs Follower — Confirmed at t = {t_confirmed:.2f} s")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True)
    plt.tight_layout()

    # ----------------------------------------------------------
    # 14. Plot 5 — Follower control inputs
    # ----------------------------------------------------------
    u_labels = ["thrust [N]", "phi_des [rad]", "theta_des [rad]"]
    fig5, axs5 = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        axs5[i].plot(t_plot, U_follower_log[i, :pe], "g-", lw=1.2)
        axs5[i].axvline(t_confirmed, color="g", ls="-", lw=1.0,
                        label="Confirmed")
        axs5[i].set_title(u_labels[i])
        axs5[i].set_xlabel("t [s]")
        axs5[i].grid(True)
        axs5[i].legend(loc="best", fontsize=8)
    fig5.suptitle(
        f"Follower Control Inputs — Confirmed at t = {t_confirmed:.2f} s")
    plt.tight_layout()

    # ----------------------------------------------------------
    # 15. Plot 6 — All 10 follower states (EDMD-MPC actual)
    # ----------------------------------------------------------
    state_names = ["x","y","z","vx","vy","vz","phi","theta","p","q"]
    state_units = ["m","m","m","m/s","m/s","m/s","rad","rad","rad/s","rad/s"]
    # Map from 12-state index to 10-state index
    idx_12to10  = [0,1,2,3,4,5,6,7,9,10]

    fig6, axs6 = plt.subplots(2, 5, figsize=(20, 7))
    for i in range(10):
        r, c  = divmod(i, 5)
        ax    = axs6[r, c]
        idx12 = idx_12to10[i]
        ax.plot(t_plot, X_follower_log[idx12, :pe], "g-", lw=1.2,
                label="Follower actual")
        ax.axvline(t_confirmed, color="g", ls="-", lw=1.0)
        ax.set_title(f"{state_names[i]} [{state_units[i]}]")
        ax.set_xlabel("t [s]")
        ax.grid(True)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    fig6.suptitle(
        f"Follower States — Confirmed at t = {t_confirmed:.2f} s")
    plt.tight_layout()

    plt.show()


# ============================================================
if __name__ == "__main__":
    main()
