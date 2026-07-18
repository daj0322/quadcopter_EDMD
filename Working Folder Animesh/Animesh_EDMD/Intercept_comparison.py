"""
intercept_comparison.py
=======================
Compare PID, Linear MPC, and EDMDc MPC for intercepting a moving target.

The target moves along a predefined path. Each controller tries to
reach the target from a different starting position.

Architecture:
  - MPC controllers: predict target position over horizon, track it
  - PID: reactively tracks the target's current position
  - All use the same plant (inner PID + drone via fct_step_attitude)
"""

import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from Simulation import quad_sim
from edmdc_mpc import (
    EDMDcMPC_QP,
    load_edmdc_model,
    load_simulation_runs,
    lifted_state_from_x,
    drop_to_10state,
    precompute_ref_std,
    build_ref_horizon,
    rmse,
)

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR       = Path(__file__).resolve().parent
EDMDC_MODEL_FILE = "edmdc_model_300_0.01.pkl"
DATA_FILE        = "runs_mixed_n300.pkl"

# Control rate
DT = 0.01

# MPC tuned config (dt=0.01)
N_MPC   = 100
NC_MPC  = 20

Q_DIAG = np.array([
    300000.0, 300000.0, 300000.0,
        300.0,     300.0,     300.0,
         0.0,      0.0,
         0.0,      0.0,
], dtype=float)

R_DIAG  = np.array([0.02, 0.08, 0.08], dtype=float)
RD_DIAG = np.array([0.002, 0.01, 0.01], dtype=float)

DU_MIN = np.array([-5.0, -3.5, -3.5], dtype=float)
DU_MAX = np.array([ 5.0,  3.5,  3.5], dtype=float)

# Interception scenario
CAPTURE_RADIUS = 0.5    # meters
T_MAX          = 30.0   # max simulation time (s)

# ============================================================
# TARGET MODELS
# ============================================================
class StraightLineTarget:
    """Target moves at constant velocity in 3D."""
    def __init__(self, p0, velocity, head_start=0.0):
        self.p0 = np.array(p0, dtype=float)
        self.v  = np.array(velocity, dtype=float)
        self.head_start = head_start

    def position(self, t):
        return self.p0 + self.v * (t + self.head_start)

    def velocity(self, t):
        return self.v.copy()

    def name(self):
        return "straight-line"


class HelicalTarget:
    """Target flies a helix — circular in XY, climbing in Z."""
    def __init__(self, center, radius, z_start, climb_rate, speed,
                 head_start=0.0):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.z0     = float(z_start)
        self.vz     = float(climb_rate)
        self.omega  = float(speed) / float(radius)
        self.head_start = head_start

    def position(self, t):
        tt = t + self.head_start
        return np.array([
            self.center[0] + self.radius * np.cos(self.omega * tt),
            self.center[1] + self.radius * np.sin(self.omega * tt),
            self.z0 + self.vz * tt
        ])

    def velocity(self, t):
        tt = t + self.head_start
        return np.array([
            -self.radius * self.omega * np.sin(self.omega * tt),
             self.radius * self.omega * np.cos(self.omega * tt),
             self.vz
        ])

    def name(self):
        return "helical"


class Figure8Target:
    """Target flies a figure-8 with altitude variation."""
    def __init__(self, center, a, b, altitude, tilt_deg, speed,
                 head_start=0.0):
        self.cx, self.cy = float(center[0]), float(center[1])
        self.a     = float(a)
        self.b     = float(b)
        self.alt   = float(altitude)
        self.tilt  = np.deg2rad(float(tilt_deg))
        self.omega = float(speed) / float(max(a, b))
        self.head_start = head_start

    def position(self, t):
        tt = t + self.head_start
        s = self.omega * tt
        x_local = self.a * np.sin(s)
        y_local = self.b * np.sin(s) * np.cos(s)
        return np.array([
            self.cx + x_local,
            self.cy + y_local * np.cos(self.tilt),
            self.alt + y_local * np.sin(self.tilt)
        ])

    def velocity(self, t):
        tt = t + self.head_start
        s = self.omega * tt
        dx = self.a * self.omega * np.cos(s)
        dy = self.b * self.omega * (np.cos(s)**2 - np.sin(s)**2)
        return np.array([
            dx,
            dy * np.cos(self.tilt),
            dy * np.sin(self.tilt)
        ])

    def name(self):
        return "figure-8"


class SinusoidalTarget:
    """Target weaves laterally and vertically while moving forward."""
    def __init__(self, p0, forward_speed, lateral_amp, lateral_freq,
                 vert_amp, vert_freq, head_start=0.0):
        self.p0 = np.array(p0, dtype=float)
        self.vx = float(forward_speed)
        self.amp_y  = float(lateral_amp)
        self.freq_y = float(lateral_freq)
        self.amp_z  = float(vert_amp)
        self.freq_z = float(vert_freq)
        self.head_start = head_start

    def position(self, t):
        tt = t + self.head_start
        return np.array([
            self.p0[0] + self.vx * tt,
            self.p0[1] + self.amp_y * np.sin(2*np.pi*self.freq_y * tt),
            self.p0[2] + self.amp_z * np.sin(2*np.pi*self.freq_z * tt)
        ])

    def velocity(self, t):
        tt = t + self.head_start
        return np.array([
            self.vx,
            self.amp_y * 2*np.pi*self.freq_y * np.cos(2*np.pi*self.freq_y * tt),
            self.amp_z * 2*np.pi*self.freq_z * np.cos(2*np.pi*self.freq_z * tt)
        ])

    def name(self):
        return "sinusoidal-3D"


class DivingTarget:
    """Target approaches at altitude then dives — forces vertical maneuver."""
    def __init__(self, p0, approach_speed, dive_time, dive_speed,
                 head_start=0.0):
        self.p0 = np.array(p0, dtype=float)
        self.vx = float(approach_speed)
        self.t_dive = float(dive_time)
        self.vz_dive = float(dive_speed)
        self.head_start = head_start

    def position(self, t):
        tt = t + self.head_start
        x = self.p0[0] + self.vx * tt
        y = self.p0[1]
        if tt < self.t_dive:
            z = self.p0[2]
        else:
            z = self.p0[2] + self.vz_dive * (tt - self.t_dive)
        return np.array([x, y, z])

    def velocity(self, t):
        tt = t + self.head_start
        if tt < self.t_dive:
            return np.array([self.vx, 0.0, 0.0])
        else:
            return np.array([self.vx, 0.0, self.vz_dive])

    def name(self):
        return "diving"


# ============================================================
# BUILD TARGET REFERENCE FOR MPC
# ============================================================
def build_target_ref_traj(target, t_now, dt, N):
    """
    Build a reference trajectory (list of dicts) for MPC by predicting
    where the target will be over the next N steps.
    """
    ref = []
    for i in range(N):
        t_future = t_now + i * dt
        pos = target.position(t_future)
        vel = target.velocity(t_future)
        ref.append({
            "pos": pos,
            "vel": vel,
            "yaw": 0.0,
        })
    return ref


def build_target_ref_pid(target, t_now):
    """Build a single-step reference dict for PID."""
    pos = target.position(t_now)
    vel = target.velocity(t_now)
    return {
        "pos": pos,
        "vel": vel,
        "yaw": 0.0,
    }


# ============================================================
# LINEAR MODEL (same as final_comparison.py)
# ============================================================
def build_linear_hover_model(sim, dt):
    m, g = sim.q_mass, sim.g
    Ixx, Iyy = sim.Ixx, sim.Iyy
    Kp_phi, Kd_phi = sim.kp_ang[0], sim.kd_ang[0]
    Kp_theta, Kd_theta = sim.kp_ang[1], sim.kd_ang[1]

    nx, nu = 10, 3
    Ac = np.zeros((nx, nx))
    Ac[0,3] = 1; Ac[1,4] = 1; Ac[2,5] = 1
    Ac[3,7] = g; Ac[4,6] = -g
    Ac[6,8] = 1; Ac[7,9] = 1
    Ac[8,6] = -Kp_phi/Ixx;   Ac[8,8] = -Kd_phi/Ixx
    Ac[9,7] = -Kp_theta/Iyy; Ac[9,9] = -Kd_theta/Iyy

    Bc = np.zeros((nx, nu))
    Bc[5,0] = 1/m; Bc[8,1] = Kp_phi/Ixx; Bc[9,2] = Kp_theta/Iyy

    M = np.zeros((nx+nu, nx+nu))
    M[:nx,:nx] = Ac*dt; M[:nx,nx:] = Bc*dt
    eM = la.expm(M)
    return eM[:nx,:nx], eM[:nx,nx:]


def scale_linear_model(Ad, Bd, scaler, u_scaler):
    sx, mx = scaler.scale_, scaler.mean_
    su, mu = u_scaler.scale_, u_scaler.mean_
    Sx_inv = np.diag(1/sx); Sx = np.diag(sx); Su = np.diag(su)
    A_s = Sx_inv @ Ad @ Sx
    B_s = Sx_inv @ Bd @ Su
    return A_s, B_s


# ============================================================
# INTERCEPTION RUNNERS
# ============================================================
def run_pid_intercept(sim, target, x0_12, dt, t_max, capture_radius):
    """PID reactively chases target's current position."""
    n_max = int(t_max / dt)
    X = np.zeros((n_max, 10))
    U = np.zeros((n_max, 3))
    x12 = x0_12.copy()
    X[0] = drop_to_10state(x12)

    sim.controller_PID.fct_reset()
    capture_step = n_max - 1
    solve_times = []

    for k in range(n_max - 1):
        t_now = k * dt
        ref_k = build_target_ref_pid(target, t_now)

        t0 = time.perf_counter()
        omega_cmd, u, u_att = sim.controller_PID.fct_step(x12, ref_k, dt)
        solve_times.append(time.perf_counter() - t0)

        U[k] = u_att[:3]

        from scipy.integrate import solve_ivp
        def ode(t_local, s_local):
            return sim.quad.fct_dynamics(t_local, s_local, omega_cmd)
        sol = solve_ivp(ode, [0, dt], x12, method="RK45")
        x12 = sol.y[:, -1]
        X[k+1] = drop_to_10state(x12)

        # Check capture
        dist = np.linalg.norm(X[k+1, 0:3] - target.position(t_now + dt))
        if dist <= capture_radius:
            capture_step = k + 1
            break

    sim.controller_PID.fct_reset()
    T = capture_step + 1
    return X[:T], U[:T], solve_times[:T-1], capture_step * dt


def run_edmdc_mpc_intercept(mpc, sim, scaler, target, x0_12, dt, N,
                            t_max, capture_radius):
    """EDMDc MPC tracks predicted target trajectory."""
    n_max = int(t_max / dt)
    X = np.zeros((n_max, 10))
    U = np.zeros((n_max, 3))
    x12 = x0_12.copy()
    X[0] = drop_to_10state(x12)

    capture_step = n_max - 1
    solve_times = []

    for k in range(n_max - 1):
        t_now = k * dt
        x10 = drop_to_10state(x12)
        z_k = lifted_state_from_x(x10, scaler)

        # Build predicted target reference over MPC horizon
        ref_traj_k = build_target_ref_traj(target, t_now, dt, N)
        ref_std = precompute_ref_std(ref_traj_k, scaler, n_states=10)
        x_ref_h = ref_std[:N]

        t0 = time.perf_counter()
        u_cmd = mpc.compute(z_k, x_ref_h)
        solve_times.append(time.perf_counter() - t0)

        u_cmd[0] = np.clip(u_cmd[0], 0.5*sim.q_mass*sim.g, 2.0*sim.q_mass*sim.g)
        u_cmd[1] = np.clip(u_cmd[1], -sim.controller_PID.tilt_max, sim.controller_PID.tilt_max)
        u_cmd[2] = np.clip(u_cmd[2], -sim.controller_PID.tilt_max, sim.controller_PID.tilt_max)
        U[k] = u_cmd

        x12 = sim.sim_PID.fct_step_attitude(
            x12, u1=u_cmd[0], phi_des=u_cmd[1], theta_des=u_cmd[2], dt=dt)
        X[k+1] = drop_to_10state(x12)

        dist = np.linalg.norm(X[k+1, 0:3] - target.position(t_now + dt))
        if dist <= capture_radius:
            capture_step = k + 1
            break

    T = capture_step + 1
    return X[:T], U[:T], solve_times[:T-1], capture_step * dt


def run_linear_mpc_intercept(mpc, sim, scaler, target, x0_12, dt, N,
                             t_max, capture_radius):
    """Linear MPC tracks predicted target trajectory."""
    n_max = int(t_max / dt)
    X = np.zeros((n_max, 10))
    U = np.zeros((n_max, 3))
    x12 = x0_12.copy()
    X[0] = drop_to_10state(x12)

    capture_step = n_max - 1
    solve_times = []

    for k in range(n_max - 1):
        t_now = k * dt
        x10 = drop_to_10state(x12)
        z_k = scaler.transform(x10.reshape(1, -1)).flatten()

        ref_traj_k = build_target_ref_traj(target, t_now, dt, N)
        ref_std = precompute_ref_std(ref_traj_k, scaler, n_states=10)
        x_ref_h = ref_std[:N]

        t0 = time.perf_counter()
        u_cmd = mpc.compute(z_k, x_ref_h)
        solve_times.append(time.perf_counter() - t0)

        u_cmd[0] = np.clip(u_cmd[0], 0.5*sim.q_mass*sim.g, 2.0*sim.q_mass*sim.g)
        u_cmd[1] = np.clip(u_cmd[1], -sim.controller_PID.tilt_max, sim.controller_PID.tilt_max)
        u_cmd[2] = np.clip(u_cmd[2], -sim.controller_PID.tilt_max, sim.controller_PID.tilt_max)
        U[k] = u_cmd

        x12 = sim.sim_PID.fct_step_attitude(
            x12, u1=u_cmd[0], phi_des=u_cmd[1], theta_des=u_cmd[2], dt=dt)
        X[k+1] = drop_to_10state(x12)

        dist = np.linalg.norm(X[k+1, 0:3] - target.position(t_now + dt))
        if dist <= capture_radius:
            capture_step = k + 1
            break

    T = capture_step + 1
    return X[:T], U[:T], solve_times[:T-1], capture_step * dt


# ============================================================
# COMPUTE METRICS
# ============================================================
def compute_metrics(X, target, dt):
    """Compute separation distance over time."""
    T = X.shape[0]
    sep = np.zeros(T)
    for k in range(T):
        sep[k] = np.linalg.norm(X[k, 0:3] - target.position(k * dt))
    return sep


# ============================================================
# SCENARIOS
# ============================================================
def get_scenarios():
    """
    Define interception scenarios.

    All interceptors start from the origin [0, 0, 0].
    Targets have a head start (already moving when chase begins).
    More nonlinear paths force aggressive 3D maneuvering.
    """
    return [
        {
            "name": "Straight-line (3D diagonal)",
            "target": StraightLineTarget(
                p0=[3.0, 2.0, 0.0],
                velocity=[1.0, 0.5, 0.3],
                head_start=2.0),
            "interceptor_start": [0.0, 0.0, 0.0],
        },
        {
            "name": "Helix (climbing spiral)",
            "target": HelicalTarget(
                center=[5.0, 0.0], radius=4.0, z_start=1.0,
                climb_rate=0.3, speed=1.5, head_start=3.0),
            "interceptor_start": [0.0, 0.0, 0.0],
        },
        {
            "name": "Figure-8 (tilted)",
            "target": Figure8Target(
                center=[8.0, 0.0], a=5.0, b=3.0, altitude=2.0,
                tilt_deg=30.0, speed=1.5, head_start=3.0),
            "interceptor_start": [0.0, 0.0, 0.0],
        },
        {
            "name": "3D sinusoidal evasion",
            "target": SinusoidalTarget(
                p0=[3.0, 0.0, 1.0],
                forward_speed=0.8,
                lateral_amp=3.0, lateral_freq=0.15,
                vert_amp=1.5, vert_freq=0.2,
                head_start=2.0),
            "interceptor_start": [0.0, 0.0, 0.0],
        },
        {
            "name": "Diving target",
            "target": DivingTarget(
                p0=[8.0, 0.0, 5.0],
                approach_speed=0.8, dive_time=4.0,
                dive_speed=-1.0, head_start=2.0),
            "interceptor_start": [0.0, 0.0, 0.0],
        },
        {
            "name": "Fast helix (aggressive)",
            "target": HelicalTarget(
                center=[0.0, 5.0], radius=3.0, z_start=0.5,
                climb_rate=0.5, speed=2.5, head_start=2.0),
            "interceptor_start": [0.0, 0.0, 0.0],
        },
    ]


# ============================================================
# MAIN
# ============================================================
def main():
    # --- Load model ---
    model    = load_edmdc_model(SCRIPT_DIR / EDMDC_MODEL_FILE)
    A_edmd   = model["A"]
    B_edmd   = model["B"]
    scaler   = model["scaler"]
    u_scaler = model["u_scaler"]
    dt       = model["dt"]
    n_obs    = model["n_obs"]

    # Override dt if needed
    dt = DT

    print(f"EDMDc model: A={A_edmd.shape} B={B_edmd.shape}")
    print(f"Control dt: {dt}")

    sim = quad_sim()
    u_nominal = np.array([sim.q_mass * sim.g, 0.0, 0.0])

    # --- Build controllers ---
    # EDMDc MPC
    Cz = np.zeros((10, n_obs))
    Cz[:10, :10] = np.eye(10)

    mpc_edmd = EDMDcMPC_QP(
        A=A_edmd, B=B_edmd, Cz=Cz,
        N=N_MPC, NC=NC_MPC,
        Q=np.diag(Q_DIAG), R=np.diag(R_DIAG), Rd=np.diag(RD_DIAG),
        u_scaler=u_scaler,
        du_min=DU_MIN, du_max=DU_MAX,
        u_nominal_raw=u_nominal,
    )

    # Linear MPC
    Ad, Bd = build_linear_hover_model(sim, dt)
    A_lin, B_lin = scale_linear_model(Ad, Bd, scaler, u_scaler)

    mpc_linear = EDMDcMPC_QP(
        A=A_lin, B=B_lin, Cz=np.eye(10),
        N=N_MPC, NC=NC_MPC,
        Q=np.diag(Q_DIAG), R=np.diag(R_DIAG), Rd=np.diag(RD_DIAG),
        u_scaler=u_scaler,
        du_min=DU_MIN, du_max=DU_MAX,
        u_nominal_raw=u_nominal,
    )

    scenarios = get_scenarios()

    # ============================================================
    # RUN ALL SCENARIOS
    # ============================================================
    print(f"\n{'='*80}")
    print(f"INTERCEPTION COMPARISON")
    print(f"dt={dt}, N={N_MPC}, NC={NC_MPC}, capture_radius={CAPTURE_RADIUS}m")
    print(f"{'='*80}")

    all_results = []

    for scenario in scenarios:
        name   = scenario["name"]
        target = scenario["target"]
        p0     = scenario["interceptor_start"]

        print(f"\n--- {name} ---")
        print(f"  Target type: {target.name()}")
        print(f"  Target initial pos: {target.position(0)}")
        print(f"  Interceptor start: {p0}")

        # Initial 12-state
        x0_12 = np.zeros(12)
        x0_12[0:3] = p0

        # PID
        print("  Running PID...")
        X_pid, U_pid, st_pid, t_cap_pid = run_pid_intercept(
            sim, target, x0_12.copy(), dt, T_MAX, CAPTURE_RADIUS)
        pid_time = 1e3 * np.mean(st_pid) if len(st_pid) > 0 else 0

        # EDMDc MPC
        print("  Running EDMDc MPC...")
        X_edmd, U_edmd, st_edmd, t_cap_edmd = run_edmdc_mpc_intercept(
            mpc_edmd, sim, scaler, target, x0_12.copy(), dt, N_MPC,
            T_MAX, CAPTURE_RADIUS)
        edmd_time = 1e3 * np.mean(st_edmd) if len(st_edmd) > 0 else 0

        # Linear MPC
        print("  Running Linear MPC...")
        X_lin, U_lin, st_lin, t_cap_lin = run_linear_mpc_intercept(
            mpc_linear, sim, scaler, target, x0_12.copy(), dt, N_MPC,
            T_MAX, CAPTURE_RADIUS)
        lin_time = 1e3 * np.mean(st_lin) if len(st_lin) > 0 else 0

        # Compute separation histories
        sep_pid  = compute_metrics(X_pid, target, dt)
        sep_edmd = compute_metrics(X_edmd, target, dt)
        sep_lin  = compute_metrics(X_lin, target, dt)

        captured_pid  = t_cap_pid < T_MAX
        captured_edmd = t_cap_edmd < T_MAX
        captured_lin  = t_cap_lin < T_MAX

        print(f"  PID:    capture={'%.2fs' % t_cap_pid if captured_pid else 'FAILED'}  "
              f"min_sep={np.min(sep_pid):.3f}m  ({pid_time:.2f} ms/step)")
        print(f"  EDMDc:  capture={'%.2fs' % t_cap_edmd if captured_edmd else 'FAILED'}  "
              f"min_sep={np.min(sep_edmd):.3f}m  ({edmd_time:.2f} ms/step)")
        print(f"  Linear: capture={'%.2fs' % t_cap_lin if captured_lin else 'FAILED'}  "
              f"min_sep={np.min(sep_lin):.3f}m  ({lin_time:.2f} ms/step)")

        # Target trajectory for plotting
        T_plot = max(len(X_pid), len(X_edmd), len(X_lin))
        target_traj = np.array([target.position(k*dt) for k in range(T_plot)])

        # Detailed diagnostics
        print(f"\n  === DETAILED DIAGNOSTICS: {name} ===")
        print(f"  Target at t=0:     {target.position(0)}")
        print(f"  Target at t=1:     {target.position(1.0)}")
        print(f"  Target at t=5:     {target.position(5.0)}")
        print(f"  Target speed:      {np.linalg.norm(target.velocity(0)):.2f} m/s")

        # PID diagnostics
        print(f"\n  PID:")
        print(f"    Steps taken:     {len(X_pid)}")
        print(f"    Final pos:       {X_pid[-1, 0:3]}")
        print(f"    Final target:    {target.position(len(X_pid) * dt)}")
        print(f"    Final sep:       {np.min(sep_pid):.4f} m")
        print(f"    Max speed:       {np.max(np.linalg.norm(X_pid[:, 3:6], axis=1)):.2f} m/s")
        print(
            f"    Max tilt:        {np.max(np.abs(X_pid[:, 6:8])):.3f} rad ({np.rad2deg(np.max(np.abs(X_pid[:, 6:8]))):.1f} deg)")

        # EDMDc diagnostics
        print(f"\n  EDMDc MPC:")
        print(f"    Steps taken:     {len(X_edmd)}")
        print(f"    Final pos:       {X_edmd[-1, 0:3]}")
        print(f"    Final target:    {target.position(len(X_edmd) * dt)}")
        print(f"    Final sep:       {np.min(sep_edmd):.4f} m")
        print(f"    Max speed:       {np.max(np.linalg.norm(X_edmd[:, 3:6], axis=1)):.2f} m/s")
        print(
            f"    Max tilt:        {np.max(np.abs(X_edmd[:, 6:8])):.3f} rad ({np.rad2deg(np.max(np.abs(X_edmd[:, 6:8]))):.1f} deg)")
        print(f"    Max thrust:      {np.max(U_edmd[:, 0]):.3f} N")
        print(
            f"    Thrust range:    [{np.min(U_edmd[:len(X_edmd) - 1, 0]):.3f}, {np.max(U_edmd[:len(X_edmd) - 1, 0]):.3f}]")
        print(
            f"    phi_des range:   [{np.min(U_edmd[:len(X_edmd) - 1, 1]):.3f}, {np.max(U_edmd[:len(X_edmd) - 1, 1]):.3f}]")
        print(
            f"    theta_des range: [{np.min(U_edmd[:len(X_edmd) - 1, 2]):.3f}, {np.max(U_edmd[:len(X_edmd) - 1, 2]):.3f}]")

        # Linear diagnostics
        print(f"\n  Linear MPC:")
        print(f"    Steps taken:     {len(X_lin)}")
        print(f"    Final pos:       {X_lin[-1, 0:3]}")
        print(f"    Final target:    {target.position(len(X_lin) * dt)}")
        print(f"    Final sep:       {np.min(sep_lin):.4f} m")
        print(f"    Max speed:       {np.max(np.linalg.norm(X_lin[:, 3:6], axis=1)):.2f} m/s")
        print(
            f"    Max tilt:        {np.max(np.abs(X_lin[:, 6:8])):.3f} rad ({np.rad2deg(np.max(np.abs(X_lin[:, 6:8]))):.1f} deg)")
        print(f"    Max thrust:      {np.max(U_lin[:, 0]):.3f} N")
        print(
            f"    Thrust range:    [{np.min(U_lin[:len(X_lin) - 1, 0]):.3f}, {np.max(U_lin[:len(X_lin) - 1, 0]):.3f}]")
        print(
            f"    phi_des range:   [{np.min(U_lin[:len(X_lin) - 1, 1]):.3f}, {np.max(U_lin[:len(X_lin) - 1, 1]):.3f}]")
        print(
            f"    theta_des range: [{np.min(U_lin[:len(X_lin) - 1, 2]):.3f}, {np.max(U_lin[:len(X_lin) - 1, 2]):.3f}]")

        # Separation at key times
        print(f"\n  Separation at key times:")
        for t_check in [0.5, 1.0, 2.0, 3.0, 5.0]:
            k_check = int(t_check / dt)
            s_pid = sep_pid[min(k_check, len(sep_pid) - 1)]
            s_edmd = sep_edmd[min(k_check, len(sep_edmd) - 1)]
            s_lin = sep_lin[min(k_check, len(sep_lin) - 1)]
            print(f"    t={t_check:.1f}s:  PID={s_pid:.3f}m  EDMDc={s_edmd:.3f}m  Linear={s_lin:.3f}m")
        print(f"  ===================================")

        all_results.append({
            "name": name,
            "target": target,
            "target_traj": target_traj,
            "X_pid": X_pid, "U_pid": U_pid,
            "X_edmd": X_edmd, "U_edmd": U_edmd,
            "X_lin": X_lin, "U_lin": U_lin,
            "sep_pid": sep_pid, "sep_edmd": sep_edmd, "sep_lin": sep_lin,
            "t_cap_pid": t_cap_pid, "t_cap_edmd": t_cap_edmd, "t_cap_lin": t_cap_lin,
            "captured_pid": captured_pid, "captured_edmd": captured_edmd,
            "captured_lin": captured_lin,
            "pid_time": pid_time, "edmd_time": edmd_time, "lin_time": lin_time,
        })

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print(f"\n{'='*80}")
    print(f"INTERCEPTION SUMMARY")
    print(f"{'='*80}")
    header = f"{'Scenario':<25s}  {'PID':>10s}  {'EDMDc':>10s}  {'Linear':>10s}  {'Winner':>8s}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        def fmt(captured, t):
            return f"{t:.2f}s" if captured else "FAIL"

        t_pid  = r["t_cap_pid"]  if r["captured_pid"]  else float("inf")
        t_edmd = r["t_cap_edmd"] if r["captured_edmd"] else float("inf")
        t_lin  = r["t_cap_lin"]  if r["captured_lin"]  else float("inf")

        best = min(t_pid, t_edmd, t_lin)
        if best == float("inf"):
            winner = "NONE"
        else:
            names_map = [(t_pid, "PID"), (t_edmd, "EDMDc"), (t_lin, "Linear")]
            winner = min(names_map, key=lambda x: x[0])[1]

        print(f"{r['name']:<25s}  "
              f"{fmt(r['captured_pid'], r['t_cap_pid']):>10s}  "
              f"{fmt(r['captured_edmd'], r['t_cap_edmd']):>10s}  "
              f"{fmt(r['captured_lin'], r['t_cap_lin']):>10s}  "
              f"{winner:>8s}")

    print(f"\nSolve times (avg per step):")
    for r in all_results:
        print(f"  {r['name']:<25s}  PID: ~0ms  EDMDc: {r['edmd_time']:.2f}ms  Linear: {r['lin_time']:.2f}ms")

    # ============================================================
    # PLOTS
    # ============================================================
    C_TGT   = "red"
    C_PID   = "#888888"
    C_EDMDC = "#2ca02c"
    C_LIN   = "#1f77b4"

    n_scenarios = len(all_results)

    # --- 3D trajectories ---
    fig_3d, axes_3d = plt.subplots(2, 3, figsize=(20, 12),
                                    subplot_kw={"projection": "3d"})
    for i, (r, ax) in enumerate(zip(all_results, axes_3d.flat)):
        tgt = r["target_traj"]
        ds = max(1, len(tgt) // 500)

        ax.plot(tgt[::ds, 0], tgt[::ds, 1], tgt[::ds, 2],
                color=C_TGT, lw=2.5, label="Target", zorder=1)
        ax.plot(r["X_pid"][::ds, 0], r["X_pid"][::ds, 1], r["X_pid"][::ds, 2],
                color=C_PID, lw=1.2, label=f"PID", zorder=2)
        ax.plot(r["X_edmd"][::ds, 0], r["X_edmd"][::ds, 1], r["X_edmd"][::ds, 2],
                color=C_EDMDC, lw=1.5, label=f"EDMDc", zorder=3)
        ax.plot(r["X_lin"][::ds, 0], r["X_lin"][::ds, 1], r["X_lin"][::ds, 2],
                color=C_LIN, lw=1.2, ls="--", label=f"Linear", zorder=3)

        # Start/end markers
        ax.scatter(*r["X_pid"][0, :3], c="black", s=80, marker="o", zorder=4)
        ax.scatter(*tgt[0], c=C_TGT, s=80, marker="^", zorder=4)

        # Capture markers
        if r["captured_edmd"]:
            ax.scatter(*r["X_edmd"][-1, :3], c=C_EDMDC, s=120, marker="*", zorder=5)
        if r["captured_lin"]:
            ax.scatter(*r["X_lin"][-1, :3], c=C_LIN, s=120, marker="*", zorder=5)
        if r["captured_pid"]:
            ax.scatter(*r["X_pid"][-1, :3], c=C_PID, s=120, marker="*", zorder=5)

        ax.set_title(r["name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.legend(fontsize=8)

    fig_3d.suptitle("Interception Trajectories", fontsize=15, fontweight="bold")
    plt.tight_layout()

    # --- Separation distance ---
    fig_sep, axes_sep = plt.subplots(2, 3, figsize=(18, 10))
    for i, (r, ax) in enumerate(zip(all_results, axes_sep.flat)):
        t_pid  = np.arange(len(r["sep_pid"])) * dt
        t_edmd = np.arange(len(r["sep_edmd"])) * dt
        t_lin  = np.arange(len(r["sep_lin"])) * dt

        ds = max(1, len(t_pid) // 500)

        ax.plot(t_pid[::ds], r["sep_pid"][::ds], color=C_PID, lw=1.2, label="PID")
        ax.plot(t_edmd[::ds], r["sep_edmd"][::ds], color=C_EDMDC, lw=1.5, label="EDMDc")
        ax.plot(t_lin[::ds], r["sep_lin"][::ds], color=C_LIN, lw=1.2, ls="--", label="Linear")
        ax.axhline(CAPTURE_RADIUS, color="red", ls=":", lw=1, label="Capture radius")

        ax.set_title(r["name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("t [s]"); ax.set_ylabel("Separation [m]")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)

    fig_sep.suptitle("Interceptor-Target Separation", fontsize=15, fontweight="bold")
    plt.tight_layout()

    # --- Control inputs for first scenario ---
    r = all_results[0]
    u_labels = ["Thrust [N]", r"$\phi_{des}$ [rad]", r"$\theta_{des}$ [rad]"]
    fig_u, axes_u = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    ds = max(1, len(r["U_edmd"]) // 500)
    t_u = np.arange(len(r["U_edmd"])) * dt

    for j in range(3):
        ax = axes_u[j]
        ax.plot(t_u[::ds], r["U_edmd"][::ds, j], color=C_EDMDC, lw=1.2, label="EDMDc")
        ax.plot(t_u[::ds], r["U_lin"][:len(t_u)][::ds, j], color=C_LIN, lw=1, ls="--", label="Linear")
        ax.set_ylabel(u_labels[j])
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=10)
    axes_u[-1].set_xlabel("Time [s]")
    fig_u.suptitle(f"Control Inputs — {r['name']}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()