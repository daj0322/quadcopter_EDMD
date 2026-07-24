import numpy as np
import random
from quadcopter import quadcopter
from Cascaded_Controllers import QuadPIDController6Fixed, QuadPX4LikeController
from Closed_loop import ClosedLoopQuad

class quad_sim:
    
    # Simulation Parameters
    q_mass = 3.33819 # kg
    g = 9.80665 # m/s^2
    q_l = 0.28881 # m
    kD = 4.8e-7 # aerodynamic drag/yaw torque factor
    kT = 3.44e-5 # N/(rad/s)^2

    # Drag estimates for a 5.5 inch cube center body.
    # Assumptions: rho=1.225 kg/m^3, Cd~=1.05 for a cube, linearized at
    # v_ref=15 m/s and omega_ref=220 deg/s to match this simulator's
    # F=-k*v and tau=-k*omega damping model.
    cube_side = 5.5 * 0.0254 # m
    k_drag_linear = 0.18826928071875 # kg/s
    k_drag_angular = 6.569729303413257e-5 # N*m*s/rad

    Ixx, Iyy, Izz = 0.04164, 0.03963, 0.04758
    I = np.diag([Ixx, Iyy, Izz])

    kp_pos = [2.0, 2.0, 15.] #[x,y,z]
    ki_pos = [0.2, 0.2, 5.] #[x,y,z]
    kd_pos = [3.2, 3.2, 15.] #[x,y,z]
    kp_ang = [6.5, 6.5, 2.8] #[phi,theta,psi]
    ki_ang = [0.1, 0.1, 0.1] #[phi,theta,psi]
    kd_ang = [3.7, 3.7, 5.] #[phi,theta,psi]

    max_speed = 16380.0 * 2.0 * np.pi / 60.0 # rad/s

    quad = quadcopter(q_mass, g, q_l, I, kD, kT, k_drag_linear, k_drag_angular, prop_efficiency=[1.0, 1.0, 1.0, 1.0])

    controller_PID = QuadPIDController6Fixed(
        quad,
        kp_pos, ki_pos, kd_pos,
        kp_ang, ki_ang, kd_ang,
        max_speed=max_speed,
        a_xy_max=10.0,
        a_z_max=10.0,
        tilt_max_deg=45.0,
        torque_roll_pitch_max=1.10,
        yaw_tau_max=1.19)

    controller_PX4 = QuadPX4LikeController(
        quad,
        max_speed=max_speed,
        pos_p=(1.0, 1.0, 2.0),
        vel_p=(3.0, 3.0, 5.0),
        vel_i=(0.08, 0.08, 1.0),
        att_p=(3.5, 3.5, 2.8),
        # PX4 rate gains are normalized actuator gains; this simulator's
        # rate_p outputs torque directly, so use a 2.5x torque-unit conversion.
        rate_p=(0.35, 0.35, 0.26879),
        rate_sp_max=(np.deg2rad(220.0), np.deg2rad(220.0), np.deg2rad(200.0)),
        vel_sp_max_xy=15.0,
        vel_sp_max_z=15.0,
        acc_max_xy=10.0,
        acc_max_z=10.0,
        tilt_max_deg=45.0,
        thrust_max=q_mass * (g + 10.0),
        torque_max=(1.14, 1.09, 1.19))

    sim_PID = ClosedLoopQuad(quad, controller_PX4)

    # Time setup
    dt = 0.01
    time = np.arange(0.0, 35.0, dt)

    def fct_unwrap_trajectory_yaw(self, traj):
        yaws = np.unwrap([r["yaw"] for r in traj])
        for r, yaw in zip(traj, yaws):
            r["yaw"] = float(yaw)
        return traj

    def fct_smooth_time_scaling(self, tau, T):
        ramp_fraction = 0.3
        r = ramp_fraction
        cruise_scale = 1.0 / (1.0 - r)

        def ramp_distance(xi):
            return xi**3 - 0.5 * xi**4

        def ramp_speed(xi):
            return 3.0 * xi**2 - 2.0 * xi**3

        def ramp_accel(xi):
            return 6.0 * xi - 6.0 * xi**2

        if tau < r:
            xi = tau / r
            sigma = cruise_scale * r * ramp_distance(xi)
            sigma_dot = cruise_scale * ramp_speed(xi) / T
            sigma_ddot = cruise_scale * ramp_accel(xi) / (r * T**2)
        elif tau > 1.0 - r:
            tau_remaining = 1.0 - tau
            xi = tau_remaining / r
            sigma = 1.0 - cruise_scale * r * ramp_distance(xi)
            sigma_dot = cruise_scale * ramp_speed(xi) / T
            sigma_ddot = -cruise_scale * ramp_accel(xi) / (r * T**2)
        else:
            sigma = cruise_scale * (0.5 * r + tau - r)
            sigma_dot = cruise_scale / T
            sigma_ddot = 0.0
        return sigma, sigma_dot, sigma_ddot

    def fct_make_helical_trajectory(self, time,
                                    center=(0.0, 0.0),
                                    radius=1.0,
                                    z_start=0.5,
                                    z_end=3.0,
                                    n_turns=3.0,
                                    yaw_follows_path=True):
        """
        Make a helical trajectory:
        - circle of given radius around (cx, cy)
        - altitude increases linearly from z_start to z_end
        - n_turns full revolutions over the full duration of 'time'
        - total trajectory duration is time[-1] - time[0]
        - constant speed along the path

        Returns a list of dicts with keys:
            "pos": np.array([x,y,z])
            "vel": np.array([vx,vy,vz])
            "yaw": float
        """

        time = np.asarray(time, dtype=float)
        t0 = float(time[0])
        T  = float(time[-1] - time[0])  # total duration

        if T <= 0.0:
            raise ValueError("time array must span a positive duration")

        cx, cy = float(center[0]), float(center[1])

        # Angle and altitude as functions of time
        # tau in [0,1]
        traj = []
        for t in time:
            tau = (t - t0) / T  # normalized time in [0,1]
            sigma, sigma_dot, sigma_ddot = self.fct_smooth_time_scaling(tau, T)

            # Angle (n_turns full revolutions)
            theta = 2.0 * np.pi * n_turns * sigma
            theta_dot = 2.0 * np.pi * n_turns * sigma_dot
            theta_ddot = 2.0 * np.pi * n_turns * sigma_ddot

            # Position
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = z_start + (z_end - z_start) * sigma

            # Velocity (derivatives)
            vx = -radius * np.sin(theta) * theta_dot
            vy =  radius * np.cos(theta) * theta_dot
            vz = (z_end - z_start) * sigma_dot

            # Acceleration (derivatives)
            ax = -radius * (np.cos(theta) * theta_dot**2 + np.sin(theta) * theta_ddot)
            ay = radius * (-np.sin(theta) * theta_dot**2 + np.cos(theta) * theta_ddot)
            az = (z_end - z_start) * sigma_ddot

            # Yaw: either follow the tangent direction or stay fixed
            if yaw_follows_path:
                tangent_x = -radius * np.sin(theta)
                tangent_y = radius * np.cos(theta)
                yaw = np.arctan2(tangent_y, tangent_x)  # heading along the path
                yaw_rate = theta_dot
            else:
                yaw = 0.0  # or any constant you like
                yaw_rate = 0.0

            traj.append({
                "pos": np.array([x, y, z], dtype=float),
                "vel": np.array([vx, vy, vz], dtype=float),
                "acc": np.array([ax, ay, az], dtype=float),
                "yaw": float(yaw),
                "yaw_rate": float(yaw_rate)
            })

        return self.fct_unwrap_trajectory_yaw(traj)

    def fct_make_figure8_trajectory(self, time,
                                    center=(0.0, 0.0, 1.0),
                                    a=1.0,
                                    b=0.5,
                                    n_loops=1.0,
                                    tilt_deg=30.0,
                                    yaw_follows_path=True,
                                    yaw_constant=0.0):
        """
        Make a 3D figure-8 trajectory.

        Base curve (before tilt) is a lemniscate of Gerono in the XY-plane:
            x' = a * sin(ω t)
            y' = b * sin(ω t) * cos(ω t) = 0.5*b*sin(2ω t)
            z' = 0

        Then we rotate that plane around the X-axis by `tilt_deg`, so z varies.

        Parameters
        ----------
        time : array-like
            Time stamps for the trajectory.
        center : (cx, cy, cz)
            Center of the figure-8 in world coordinates.
        a, b : float
            Horizontal/vertical scales of the figure-8.
        n_loops : float
            Number of figure-8 loops over the full time interval.
        tilt_deg : float
            Tilt angle (degrees) around the X-axis. 0° -> flat in XY, z = const.
        yaw_follows_path : bool
            If True, yaw is aligned with the XY projection of the velocity.
            If False, yaw is constant (yaw_constant).
        yaw_constant : float
            Constant yaw (rad) if yaw_follows_path is False.

        Returns
        -------
        traj : list of dict
            Each element has keys "pos", "vel", "yaw".
        """
        time = np.asarray(time, dtype=float)
        t0 = float(time[0])
        T  = float(time[-1] - time[0])
        if T <= 0.0:
            raise ValueError("time array must span a positive duration")

        cx, cy, cz = map(float, center)

        # Angular frequency to get n_loops over duration T
        omega = 2.0 * np.pi * n_loops / T

        # Rotation about x-axis
        tilt = np.deg2rad(tilt_deg)
        cth = np.cos(tilt)
        sth = np.sin(tilt)

        traj = []
        for t in time:
            tau = (t - t0) / T
            sigma, sigma_dot, sigma_ddot = self.fct_smooth_time_scaling(tau, T)

            # ---- base planar figure-8 (XY plane) ----
            s    = 2.0 * np.pi * n_loops * sigma
            s_dot = 2.0 * np.pi * n_loops * sigma_dot
            s_ddot = 2.0 * np.pi * n_loops * sigma_ddot
            sin_s = np.sin(s)
            cos_s = np.cos(s)

            # Position in local (unrotated) frame
            x_local = a * sin_s
            y_local = b * sin_s * cos_s   # 0.5*b*sin(2s)
            z_local = 0.0

            # Velocity in local frame (time derivatives)
            dx_local = a * cos_s * s_dot
            # derivative of b*sin(s)*cos(s) = b*omega*(cos^2 - sin^2) = b*omega*cos(2s)
            dy_local = b * (cos_s**2 - sin_s**2) * s_dot
            dz_local = 0.0

            ddx_local = a * (cos_s * s_ddot - sin_s * s_dot**2)
            ddy_local = b * ((cos_s**2 - sin_s**2) * s_ddot - 4.0 * sin_s * cos_s * s_dot**2)
            ddz_local = 0.0

            # ---- rotate around X-axis to introduce z-variation ----
            # x' = x
            # y' =  y*cos(tilt) - z*sin(tilt) = y*cos(tilt)
            # z' =  y*sin(tilt) + z*cos(tilt) = y*sin(tilt)
            x_world = x_local
            y_world = y_local * cth
            z_world = y_local * sth

            dx_world = dx_local
            dy_world = dy_local * cth
            dz_world = dy_local * sth

            ddx_world = ddx_local
            ddy_world = ddy_local * cth
            ddz_world = ddy_local * sth

            # ---- shift to center ----
            x = cx + x_world
            y = cy + y_world
            z = cz + z_world

            vx = dx_world
            vy = dy_world
            vz = dz_world

            ax = ddx_world
            ay = ddy_world
            az = ddz_world

            # ---- yaw ----
            if yaw_follows_path:
                # Heading in the XY plane
                tangent_x = a * cos_s
                tangent_y = b * (cos_s**2 - sin_s**2) * cth
                yaw = np.arctan2(tangent_y, tangent_x)
                yaw_rate = (vx * ay - vy * ax) / (vx**2 + vy**2 + 1e-12)
            else:
                yaw = float(yaw_constant)
                yaw_rate = 0.0

            traj.append({
                "pos": np.array([x, y, z], dtype=float),
                "vel": np.array([vx, vy, vz], dtype=float),
                "acc": np.array([ax, ay, az], dtype=float),
                "yaw": float(yaw),
                "yaw_rate": float(yaw_rate)
            })

        return self.fct_unwrap_trajectory_yaw(traj)

    def fct_run_simulation(self, traj, n):
        """
        Run n simulations using a single trajectory type.

        All runs start at:
            position = (0,0,0)
            velocity = (0,0,0)
            angles = 0

        traj = 1  -> helical trajectory
        traj = 2  -> figure-8 trajectory

        Each run gets different randomized trajectory parameters,
        but they are deterministic by run index. That means:

        - calling this function multiple times with the same traj and n
        will generate the exact same trajectories
        - run 0 always uses the same trajectory parameters
        - run 1 always uses the same trajectory parameters
        - etc.

        Returns
        -------
        t : (n, T)
        states : (n, T, 12)
        U : (n, T, n_inputs)
        ref_traj_list : list of reference trajectories used for each run
        """

        import random

        t_runs = []
        states_runs = []
        U_runs = []
        ref_traj_list = []

        for i in range(n):

            # =====================================================
            # Deterministic random generator for this run
            # =====================================================
            # This makes trajectory parameters repeatable across
            # separate calls to fct_run_simulation(...)
            seed = 1000 * traj + i
            rng = random.Random(seed)

            # =====================================================
            # Generate trajectory
            # =====================================================
            if traj == 1:

                ref_traj = self.fct_make_helical_trajectory(
                    self.time,
                    center=(0.0, 0.0),
                    radius=rng.uniform(1, 5),
                    z_start=0.0,
                    z_end=rng.uniform(5, 10),
                    n_turns=1,
                    yaw_follows_path=True
                )

            elif traj == 2:

                ref_traj = self.fct_make_figure8_trajectory(
                    self.time,
                    center=(0.0, 0.0, 0.0),
                    a=rng.uniform(1, 5),
                    b=rng.uniform(1, 5),
                    n_loops=1,
                    tilt_deg=rng.uniform(10, 80),
                    yaw_follows_path=True
                )

            else:
                raise ValueError("traj must be 1 or 2")

            # =====================================================
            # Shift trajectory so it starts at (0,0,0)
            # =====================================================
            p0 = ref_traj[0]["pos"].copy()

            for k in range(len(ref_traj)):
                ref_traj[k]["pos"] = ref_traj[k]["pos"] - p0

            ref_traj_list.append(ref_traj)

            # =====================================================
            # Initial state = ZERO
            # =====================================================
            init_state = np.zeros(12)

            # =====================================================
            # Run simulation
            # =====================================================
            t_i, states_i, omegas_i, U_i = self.sim_PID.fct_simulate(
                self.time, self.dt, ref_traj, init_state
            )

            t_runs.append(t_i)
            states_runs.append(states_i)
            U_runs.append(U_i)

        # =====================================================
        # Stack results
        # =====================================================
        t = np.stack(t_runs, axis=0)
        states = np.stack(states_runs, axis=0)
        U = np.stack(U_runs, axis=0)

        return t, states, U, ref_traj_list

    # def fct_run_simulation(self, traj, n):
    #     """
    #     Run n simulations cycling through available trajectory types.

    #     Deterministic randomization per run:
    #     - Same (traj, n) → same trajectories every time
    #     - Different runs → different parameters

    #     All runs start from:
    #         state = 0
    #         trajectory start = (0,0,0)
    #     """

    #     import random

    #     # ---------------------------------------------------------
    #     # Available trajectory types
    #     # ---------------------------------------------------------
    #     traj_ids = [1, 2]
    #     num_traj_types = len(traj_ids)

    #     start_index = (traj - 1) % num_traj_types

    #     t_runs = []
    #     states_runs = []
    #     U_runs = []
    #     ref_traj_list = []

    #     for i in range(n):

    #         # -----------------------------------------------------
    #         # Select trajectory type (cycling)
    #         # -----------------------------------------------------
    #         traj_id = traj_ids[(start_index + i) % num_traj_types]

    #         # -----------------------------------------------------
    #         # Deterministic RNG (KEY PART)
    #         # -----------------------------------------------------
    #         seed = 1000 * traj_id + i
    #         rng = random.Random(seed)

    #         # =====================================================
    #         # 1) Build trajectory (deterministic per run)
    #         # =====================================================
    #         if traj_id == 1:

    #             ref_traj = self.fct_make_helical_trajectory(
    #                 self.time,
    #                 center=(0.0, 0.0),
    #                 radius=rng.uniform(1, 5),
    #                 z_start=0.0,
    #                 z_end=rng.uniform(5, 10),
    #                 n_turns=1,
    #                 yaw_follows_path=True
    #             )

    #         elif traj_id == 2:

    #             ref_traj = self.fct_make_figure8_trajectory(
    #                 self.time,
    #                 center=(0.0, 0.0, 0.0),
    #                 a=rng.uniform(1, 5),
    #                 b=rng.uniform(1, 5),
    #                 n_loops=1,
    #                 tilt_deg=rng.uniform(10, 80),
    #                 yaw_follows_path=True
    #             )

    #         else:
    #             raise ValueError(f"Unknown trajectory id: {traj_id}")

    #         # =====================================================
    #         # 2) Shift trajectory to start at origin
    #         # =====================================================
    #         p0 = ref_traj[0]["pos"].copy()
    #         for k in range(len(ref_traj)):
    #             ref_traj[k]["pos"] -= p0

    #         ref_traj_list.append(ref_traj)

    #         # =====================================================
    #         # 3) Initial state = ZERO
    #         # =====================================================
    #         init_state = np.zeros(12)

    #         # =====================================================
    #         # 4) Run simulation
    #         # =====================================================
    #         t_i, states_i, omegas_i, U_i = self.sim_PID.fct_simulate(
    #             self.time, self.dt, ref_traj, init_state
    #         )

    #         t_runs.append(t_i)
    #         states_runs.append(states_i)
    #         U_runs.append(U_i)

    #     # =====================================================
    #     # 5) Stack results
    #     # =====================================================
    #     t = np.stack(t_runs, axis=0)
    #     states = np.stack(states_runs, axis=0)
    #     U = np.stack(U_runs, axis=0)

    #     return t, states, U, ref_traj_list

    def fct_save_simulation_runs(self, traj, n, filename="saved_runs.pkl"):
        """
        Run simulation using the current deterministic randomized
        fct_run_simulation(...) and save the results to a pickle file.

        This preserves the exact same trajectory generation behavior
        as the current fct_run_simulation method.
        """
        import pickle

        # Use the existing simulation function exactly as-is
        t, states, U, ref_traj_list = self.fct_run_simulation(traj, n)

        data = {
            "traj": traj,
            "n": n,
            "sim_dt": self.dt,
            "time": self.time,
            "t": t,
            "states": states,
            "U": U,
            "ref_traj_list": ref_traj_list,
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved simulation runs to {filename}")
