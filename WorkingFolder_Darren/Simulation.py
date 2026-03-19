import numpy as np
import random
from quadcopter import quadcopter
from Cascaded_Controllers import QuadPIDController6Fixed
from Closed_loop import ClosedLoopQuad

class quad_sim:
    
    # Simulation Parameters
    q_mass = 0.5 # kg
    g = 9.81 # m/s
    q_l = 0.2 # m
    kD = 1e-9
    kT = 3e-5
    k_drag_linear = 0.5
    k_drag_angular = 0.02
    Ixx, Iyy, Izz = 5e-3, 5e-3, 9e-3
    I = np.diag([Ixx, Iyy, Izz])

    kp_pos = [0.95, 0.95, 15.] #[x,y,z]
    ki_pos = [0.2, 0.2, 5.] #[x,y,z]
    kd_pos = [1.8, 1.8, 15.] #[x,y,z]
    kp_ang = [6.9, 6.9, 25.] #[phi,theta,psi]
    ki_ang = [0.1, 0.1, 0.1] #[phi,theta,psi]
    kd_ang = [3.7, 3.7, 9.] #[phi,theta,psi]

    max_speed = 400.0

    quad = quadcopter(q_mass, g, q_l, I, kD, kT, k_drag_linear, k_drag_angular, prop_efficiency=[1.0, 1.0, 1.0, 1.0])

    controller_PID = QuadPIDController6Fixed(
        quad,
        kp_pos, ki_pos, kd_pos,
        kp_ang, ki_ang, kd_ang,
        max_speed=max_speed,
        a_xy_max=2.0,
        a_z_max=4.0,
        tilt_max_deg=45.0,
        torque_roll_pitch_max=0.10)

    sim_PID = ClosedLoopQuad(quad, controller_PID)

    # Time setup
    dt = 0.01
    time = np.arange(0.0, 35.0, dt)

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

            # Angle (n_turns full revolutions)
            theta = 2.0 * np.pi * n_turns * tau
            theta_dot = 2.0 * np.pi * n_turns / T

            # Position
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = z_start + (z_end - z_start) * tau

            # Velocity (derivatives)
            vx = -radius * np.sin(theta) * theta_dot
            vy =  radius * np.cos(theta) * theta_dot
            vz = (z_end - z_start) / T

            # Yaw: either follow the tangent direction or stay fixed
            if yaw_follows_path:
                yaw = np.arctan2(vy, vx)  # heading along the path
            else:
                yaw = 0.0  # or any constant you like

            traj.append({
                "pos": np.array([x, y, z], dtype=float),
                "vel": np.array([vx, vy, vz], dtype=float),
                "yaw": float(yaw)
            })

        return traj

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
            tr = t - t0

            # ---- base planar figure-8 (XY plane) ----
            s    = omega * tr
            sin_s = np.sin(s)
            cos_s = np.cos(s)

            # Position in local (unrotated) frame
            x_local = a * sin_s
            y_local = b * sin_s * cos_s   # 0.5*b*sin(2s)
            z_local = 0.0

            # Velocity in local frame (time derivatives)
            dx_local = a * omega * cos_s
            # derivative of b*sin(s)*cos(s) = b*omega*(cos^2 - sin^2) = b*omega*cos(2s)
            dy_local = b * omega * (cos_s**2 - sin_s**2)
            dz_local = 0.0

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

            # ---- shift to center ----
            x = cx + x_world
            y = cy + y_world
            z = cz + z_world

            vx = dx_world
            vy = dy_world
            vz = dz_world

            # ---- yaw ----
            if yaw_follows_path:
                # Heading in the XY plane
                yaw = np.arctan2(vy, vx)
            else:
                yaw = float(yaw_constant)

            traj.append({
                "pos": np.array([x, y, z], dtype=float),
                "vel": np.array([vx, vy, vz], dtype=float),
                "yaw": float(yaw)
            })

        return traj

    # def fct_run_simulation(self, traj, n):
    #     """
    #     Run n simulations using a single trajectory type.

    #     All runs start at:
    #         position = (0,0,0)
    #         velocity = (0,0,0)
    #         angles = 0

    #     traj = 1  -> helical trajectory
    #     traj = 2  -> figure-8 trajectory

    #     Returns
    #     -------
    #     t : (n, T)
    #     states : (n, T, 12)
    #     U : (n, T, n_inputs)
    #     ref_traj_list : list of reference trajectories used for each run
    #     """

    #     import random

    #     t_runs = []
    #     states_runs = []
    #     U_runs = []
    #     ref_traj_list = []

    #     for i in range(n):

    #         # =====================================================
    #         # Generate trajectory
    #         # =====================================================

    #         if traj == 1:

    #             ref_traj = self.fct_make_helical_trajectory(
    #                 self.time,
    #                 center=(0.0, 0.0),
    #                 radius=random.uniform(1,5),
    #                 z_start=0.0,
    #                 z_end=random.uniform(5,10),
    #                 n_turns=1,
    #                 yaw_follows_path=True
    #             )

    #         elif traj == 2:

    #             ref_traj = self.fct_make_figure8_trajectory(
    #                 self.time,
    #                 center=(0.0, 0.0, 0.0),
    #                 a=random.uniform(1,5),
    #                 b=random.uniform(1,5),
    #                 n_loops=1,
    #                 tilt_deg=random.uniform(10,80),
    #                 yaw_follows_path=True
    #             )

    #         else:
    #             raise ValueError("traj must be 1 or 2")

    #         # =====================================================
    #         # Shift trajectory so it starts at (0,0,0)
    #         # =====================================================

    #         p0 = ref_traj[0]["pos"].copy()

    #         for k in range(len(ref_traj)):
    #             ref_traj[k]["pos"] = ref_traj[k]["pos"] - p0

    #         ref_traj_list.append(ref_traj)

    #         # =====================================================
    #         # Initial state = ZERO
    #         # =====================================================

    #         init_state = np.zeros(12)

    #         # =====================================================
    #         # Run simulation
    #         # =====================================================

    #         t_i, states_i, omegas_i, U_i = self.sim_PID.fct_simulate(
    #             self.time, self.dt, ref_traj, init_state
    #         )

    #         t_runs.append(t_i)
    #         states_runs.append(states_i)
    #         U_runs.append(U_i)

    #     # =====================================================
    #     # Stack results
    #     # =====================================================

    #     t = np.stack(t_runs, axis=0)
    #     states = np.stack(states_runs, axis=0)
    #     U = np.stack(U_runs, axis=0)

    #     return t, states, U, ref_traj_list

    def fct_run_simulation(self, traj, n):
        """
        Run n simulations cycling through available trajectory types.

        - Suppose you have traj_ids = [1, 2, 3, ..., T] (T traj types).
        - For a given call with n runs:
            n = T   -> one of each trajectory
            n = 2T  -> two of each (full cycle twice)
            n = T+k -> one of each, then first k again
            n < T   -> only the first n trajectories in the list

        The argument 'traj' is treated as a 1-based starting index
        into the traj_ids list (usually 1 if you just want to start
        from the first trajectory type).

        All runs start from:
            state = 0
            trajectory start = (0,0,0)

        Returns
        -------
        t : (n, T)
        states : (n, T, 12)
        U : (n, T, n_inputs)
        ref_traj_list : list of reference trajectories used for each run
        """

        import random

        # ---------------------------------------------------------
        # List of available trajectory types
        # Add more numbers here as you create more trajectory families
        # ---------------------------------------------------------
        traj_ids = [1, 2]     # 1: helical, 2: figure-8
        num_traj_types = len(traj_ids)

        # Convert 'traj' (1-based) to 0-based starting offset
        start_index = (traj - 1) % num_traj_types

        t_runs = []
        states_runs = []
        U_runs = []
        ref_traj_list = []

        for i in range(n):

            # Choose which trajectory type this run should use
            traj_id = traj_ids[(start_index + i) % num_traj_types]

            # =====================================================
            # 1) Build a NEW random reference trajectory
            # =====================================================
            if traj_id == 1:
                ref_traj = self.fct_make_helical_trajectory(
                    self.time,
                    center=(0.0, 0.0),
                    radius=random.uniform(1,5),
                    z_start=0.0,
                    z_end=random.uniform(5,10),
                    n_turns=1,
                    yaw_follows_path=True
                )

            elif traj_id == 2:
                ref_traj = self.fct_make_figure8_trajectory(
                    self.time,
                    center=(0.0, 0.0, 0.0),
                    a=random.uniform(1,5),
                    b=random.uniform(1,5),
                    n_loops=1,
                    tilt_deg=random.uniform(10,80),
                    yaw_follows_path=True
                )

            else:
                raise ValueError(f"Unknown trajectory id: {traj_id}")

            # =====================================================
            # 2) Shift trajectory so it starts at (0,0,0)
            # =====================================================
            p0 = ref_traj[0]["pos"].copy()
            for k in range(len(ref_traj)):
                ref_traj[k]["pos"] = ref_traj[k]["pos"] - p0

            ref_traj_list.append(ref_traj)

            # =====================================================
            # 3) Initial state = all zeros
            # =====================================================
            init_state = np.zeros(12)

            # =====================================================
            # 4) Simulate this run
            # =====================================================
            t_i, states_i, omegas_i, U_i = self.sim_PID.fct_simulate(
                self.time, self.dt, ref_traj, init_state
            )

            t_runs.append(t_i)
            states_runs.append(states_i)
            U_runs.append(U_i)

        # =====================================================
        # 5) Stack into arrays (n, T, ·) like before
        # =====================================================
        t = np.stack(t_runs, axis=0)        # (n, T)
        states = np.stack(states_runs, 0)   # (n, T, 12)
        U = np.stack(U_runs, 0)             # (n, T, n_inputs)

        return t, states, U, ref_traj_list