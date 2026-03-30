"""
nonlinear_mpc.py
================
Nonlinear MPC using CasADi + IPOPT for quadcopter trajectory tracking.

The prediction model encodes the full plant dynamics:
  inner PID (attitude tracking) + quadcopter rigid body dynamics.

This serves as the "best possible" MPC baseline — it has perfect
knowledge of the dynamics. The purpose is to show that EDMDc MPC
achieves comparable tracking at a fraction of the computation cost.

Interface matches EDMDcMPC_QP: call .compute(x10, x_ref_horizon)
and get back [thrust, phi_des, theta_des].
"""

import time
import numpy as np
import casadi as ca


class NonlinearMPC:
    """
    Nonlinear MPC for the attitude-commanded quadcopter plant.

    State (10):  [x, y, z, vx, vy, vz, phi, theta, p, q]
    Input (3):   [thrust, phi_des, theta_des]

    The dynamics include:
      - Full rotation matrix (no small-angle approximation)
      - Inner PID attitude loop (PD control on phi, theta)
      - Linear drag on velocity and angular rates
      - Gravity

    Discretized with RK4, solved with IPOPT.
    """

    def __init__(self, sim, N, NC, Q_diag, R_diag, Rd_diag, dt,
                 u_min, u_max):
        """
        Parameters
        ----------
        sim : quad_sim instance (provides physical parameters)
        N : int — prediction horizon
        NC : int — control horizon (NC <= N)
        Q_diag : (10,) — state cost weights in physical units
        R_diag : (3,) — input cost weights
        Rd_diag : (3,) — input rate cost weights
        dt : float — time step
        u_min, u_max : (3,) — input bounds [thrust, phi_des, theta_des]
        """
        self.N  = N
        self.NC = NC
        self.dt = dt
        self.nx = 10
        self.nu = 3

        # Physical parameters from sim
        self.m   = sim.q_mass
        self.g   = sim.g
        self.Ixx = sim.Ixx
        self.Iyy = sim.Iyy
        self.Izz = sim.Izz
        self.k_drag_linear  = sim.k_drag_linear
        self.k_drag_angular = sim.k_drag_angular

        # Inner PID gains
        self.Kp_phi   = sim.kp_ang[0]
        self.Kd_phi   = sim.kd_ang[0]
        self.Kp_theta = sim.kp_ang[1]
        self.Kd_theta = sim.kd_ang[1]
        self.torque_max = sim.controller_PID.torque_max

        # Build the NLP
        self._build_nlp(Q_diag, R_diag, Rd_diag, u_min, u_max)

        # Warm start storage
        self._u_prev = np.tile(
            np.array([self.m * self.g, 0.0, 0.0]), NC
        )

    def _continuous_dynamics(self, x, u):
        """
        CasADi symbolic continuous dynamics.

        x: [x, y, z, vx, vy, vz, phi, theta, p, q]  (10)
        u: [thrust, phi_des, theta_des]               (3)
        """
        # Unpack state
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        phi, theta  = x[6], x[7]
        p, q        = x[8], x[9]

        # Unpack input
        thrust   = u[0]
        phi_des  = u[1]
        theta_des = u[2]

        # Rotation matrix (psi = 0)
        cphi = ca.cos(phi)
        sphi = ca.sin(phi)
        cth  = ca.cos(theta)
        sth  = ca.sin(theta)

        # Thrust in world frame: R @ [0, 0, thrust]
        # R = Rz(0) @ Ry(theta) @ Rx(phi)
        Fx = thrust * cphi * sth
        Fy = -thrust * sphi
        Fz = thrust * cphi * cth

        # Translational dynamics
        ax = Fx / self.m - self.k_drag_linear * vx
        ay = Fy / self.m - self.k_drag_linear * vy
        az = Fz / self.m - self.g - self.k_drag_linear * vz

        # Euler angle rates (simplified, psi=0, r=0)
        dphi   = p
        dtheta = q

        # Inner PID torques (PD, no integral for smooth CasADi)
        tau_phi   = self.Kp_phi * (phi_des - phi) - self.Kd_phi * p
        tau_theta = self.Kp_theta * (theta_des - theta) - self.Kd_theta * q

        # Clip torques smoothly (soft clamp via tanh)
        tmax = self.torque_max
        tau_phi   = tmax * ca.tanh(tau_phi / tmax)
        tau_theta = tmax * ca.tanh(tau_theta / tmax)

        # Angular acceleration
        dp = (tau_phi   - self.k_drag_angular * p) / self.Ixx
        dq = (tau_theta - self.k_drag_angular * q) / self.Iyy

        return ca.vertcat(vx, vy, vz, ax, ay, az, dphi, dtheta, dp, dq)

    def _rk4_step(self, x, u, dt):
        """Single RK4 integration step."""
        k1 = self._continuous_dynamics(x, u)
        k2 = self._continuous_dynamics(x + dt/2 * k1, u)
        k3 = self._continuous_dynamics(x + dt/2 * k2, u)
        k4 = self._continuous_dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _build_nlp(self, Q_diag, R_diag, Rd_diag, u_min, u_max):
        """Build the CasADi NLP once; reuse at each MPC step."""
        nx, nu, N, NC = self.nx, self.nu, self.N, self.NC

        Q = ca.diag(ca.DM(Q_diag))
        R = ca.diag(ca.DM(R_diag))
        Rd = ca.diag(ca.DM(Rd_diag))

        # Decision variables: NC control inputs
        U = ca.MX.sym("U", nu, NC)

        # Parameters: initial state + N reference states
        P = ca.MX.sym("P", nx * (1 + N))

        x0 = P[:nx]

        # Build cost and dynamics rollout
        J = 0.0
        x_k = x0

        u_prev = ca.DM([self.m * self.g, 0.0, 0.0])

        for k in range(N):
            # Select control (hold last after NC)
            u_idx = min(k, NC - 1)
            u_k = U[:, u_idx]

            # Reference at step k
            x_ref_k = P[nx*(1+k) : nx*(2+k)]

            # State cost
            e = x_k - x_ref_k
            J += ca.mtimes([e.T, Q, e])

            # Input rate cost
            if k < NC:
                if k == 0:
                    du = u_k - u_prev
                else:
                    du = u_k - U[:, u_idx - 1]
                J += ca.mtimes([du.T, Rd, du])

            # Input cost (deviation from hover)
            if k < NC:
                u_hover = ca.DM([self.m * self.g, 0.0, 0.0])
                du_nom = u_k - u_hover
                J += ca.mtimes([du_nom.T, R, du_nom])

            # Dynamics
            x_k = self._rk4_step(x_k, u_k, self.dt)

        # Terminal cost on last predicted state
        x_ref_N = P[nx*N : nx*(N+1)]
        e_N = x_k - x_ref_N
        J += ca.mtimes([e_N.T, Q, e_N])

        # Flatten decision variables
        U_flat = ca.reshape(U, nu * NC, 1)

        # Bounds
        lbx = np.tile(u_min, NC)
        ubx = np.tile(u_max, NC)

        # NLP dict
        nlp = {
            "x": U_flat,
            "f": J,
            "p": P,
        }

        # Solver options
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
        }

        self.solver = ca.nlpsol("nmpc", "ipopt", nlp, opts)
        self.lbx = lbx
        self.ubx = ubx
        self.n_params = nx * (1 + N)

    def compute(self, x10, x_ref_horizon):
        """
        Solve one NMPC step.

        Parameters
        ----------
        x10 : (10,) — current physical state
        x_ref_horizon : (N, 10) — reference trajectory over horizon

        Returns
        -------
        u_cmd : (3,) — [thrust, phi_des, theta_des]
        """
        x10 = np.asarray(x10, dtype=float).flatten()
        x_ref = np.asarray(x_ref_horizon, dtype=float)

        # Pad reference if shorter than N
        if x_ref.shape[0] < self.N:
            pad = np.tile(x_ref[-1:], (self.N - x_ref.shape[0], 1))
            x_ref = np.vstack([x_ref, pad])

        # Build parameter vector: [x0, x_ref_0, x_ref_1, ..., x_ref_{N-1}]
        p = np.concatenate([x10, x_ref[:self.N].flatten()])

        print("    calling IPOPT...")

        # Solve
        sol = self.solver(
            x0=self._u_prev,
            lbx=self.lbx,
            ubx=self.ubx,
            p=p,
        )
        print("    IPOPT returned.")

        u_opt = np.array(sol["x"]).flatten()
        self._u_prev = u_opt.copy()

        # Return first control
        return u_opt[:self.nu].copy()


def build_nmpc(sim, dt, N=20, NC=10):
    """
    Convenience function to build an NMPC with reasonable defaults.

    Q/R are in physical units — tuned to produce similar tracking
    pressure as the scaled-space MPC weights.
    """
    # Physical-unit weights
    # Position: heavy penalty
    # Velocity: moderate
    # Angles, rates: free (let MPC use them)
    Q_diag = np.array([
        50.0, 50.0, 50.0,      # x, y, z [m]
         5.0,  5.0,  5.0,      # vx, vy, vz [m/s]
          0.0,   0.0,             # phi, theta [rad]
          0.0,   0.0,             # p, q [rad/s]
    ])

    R_diag = np.array([0.1, 5.0, 5.0])    # thrust, phi_des, theta_des
    Rd_diag = np.array([0.01, 0.5, 0.5])  # rate penalty

    u_min = np.array([0.5 * sim.q_mass * sim.g, -sim.controller_PID.tilt_max, -sim.controller_PID.tilt_max])
    u_max = np.array([2.0 * sim.q_mass * sim.g,  sim.controller_PID.tilt_max,  sim.controller_PID.tilt_max])

    return NonlinearMPC(sim, N, NC, Q_diag, R_diag, Rd_diag, dt, u_min, u_max)