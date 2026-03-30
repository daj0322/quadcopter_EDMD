import numpy as np
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
    time = np.arange(0.0, 100.0, dt)

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

    def fct_make_hover_excitation_trajectory(
        self,
        time,
        rng,
        xyz_amp=(0.15, 0.15, 0.12),
        xyz_freq=(0.08, 0.10, 0.12),
        yaw_amp_deg=5.0,
        n_sines_range=(2, 4),
    ):

        time = np.asarray(time, dtype=float)
        T = len(time)

        ax, ay, az = xyz_amp
        fx_base, fy_base, fz_base = xyz_freq

        n_sines = rng.randint(n_sines_range[0], n_sines_range[1])

        x = np.zeros(T)
        y = np.zeros(T)
        z = np.zeros(T)

        for _ in range(n_sines):
            wx = 2.0 * np.pi * rng.uniform(0.5 * fx_base, 1.8 * fx_base)
            wy = 2.0 * np.pi * rng.uniform(0.5 * fy_base, 1.8 * fy_base)
            wz = 2.0 * np.pi * rng.uniform(0.5 * fz_base, 1.8 * fz_base)

            phx = rng.uniform(0.0, 2.0 * np.pi)
            phy = rng.uniform(0.0, 2.0 * np.pi)
            phz = rng.uniform(0.0, 2.0 * np.pi)

            x += (ax / n_sines) * np.sin(wx * time + phx)
            y += (ay / n_sines) * np.sin(wy * time + phy)
            z += (az / n_sines) * np.sin(wz * time + phz)

        z = z - z[0]
        z += 0.1

        yaw_amp = np.deg2rad(yaw_amp_deg)
        yaw_w = 2.0 * np.pi * rng.uniform(0.03, 0.10)
        yaw_ph = rng.uniform(0.0, 2.0 * np.pi)
        yaw = yaw_amp * np.sin(yaw_w * time + yaw_ph)

        vx = np.gradient(x, self.dt)
        vy = np.gradient(y, self.dt)
        vz = np.gradient(z, self.dt)

        traj = []
        for k in range(T):
            traj.append({
                "pos": np.array([x[k], y[k], z[k]], dtype=float),
                "vel": np.array([vx[k], vy[k], vz[k]], dtype=float),
                "yaw": float(yaw[k]),
            })

        p0 = traj[0]["pos"].copy()
        for k in range(T):
            traj[k]["pos"] = traj[k]["pos"] - p0

        return traj

    def fct_run_hover_excitation_simulation(self, n):

        t_runs = []
        states_runs = []
        U_runs = []
        ref_traj_list = []

        for i in range(n):
            seed = 3000 + i
            rng = random.Random(seed)

            ref_traj = self.fct_make_hover_excitation_trajectory(
                self.time,
                rng=rng,
                xyz_amp=(
                    rng.uniform(0.05, 0.20),
                    rng.uniform(0.05, 0.20),
                    rng.uniform(0.05, 0.15),
                ),
                xyz_freq=(
                    rng.uniform(0.05, 0.12),
                    rng.uniform(0.05, 0.12),
                    rng.uniform(0.06, 0.15),
                ),
                yaw_amp_deg=rng.uniform(2.0, 8.0),
                n_sines_range=(2, 4),
            )

            ref_traj_list.append(ref_traj)

            init_state = np.zeros(12)

            t_i, states_i, omegas_i, _, U_i = self.sim_PID.fct_simulate(
                self.time, self.dt, ref_traj, init_state
            )

            t_runs.append(t_i)
            states_runs.append(states_i)
            U_runs.append(U_i)

        t = np.stack(t_runs, axis=0)
        states = np.stack(states_runs, axis=0)
        U = np.stack(U_runs, axis=0)

        return t, states, U, ref_traj_list

    def fct_save_hover_excitation_runs(self, n, filename="runs_hover_excitation_n100.pkl"):
        import pickle

        t, states, U, ref_traj_list = self.fct_run_hover_excitation_simulation(n)

        data = {
            "traj": "hover_excitation",
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

        print(f"Saved hover excitation runs to {filename}")
        print("t shape:", t.shape)
        print("states shape:", states.shape)
        print("U shape:", U.shape)

    def fct_make_lissajous_trajectory(self, time,
                                      center=(0.0, 0.0, 2.0),
                                      ax=2.0, ay=1.5, az=1.0,
                                      fx=1.0, fy=2.0, fz=3.0,
                                      phase_y=np.pi / 2, phase_z=np.pi / 4,
                                      n_loops=1.0,
                                      yaw_follows_path=True):
        time = np.asarray(time, dtype=float)
        t0 = float(time[0])
        T = float(time[-1] - time[0])
        cx, cy, cz = map(float, center)
        omega = 2.0 * np.pi * n_loops / T

        traj = []
        for t in time:
            tr = t - t0
            x = cx + ax * np.sin(fx * omega * tr)
            y = cy + ay * np.sin(fy * omega * tr + phase_y)
            z = cz + az * np.sin(fz * omega * tr + phase_z)

            vx = ax * fx * omega * np.cos(fx * omega * tr)
            vy = ay * fy * omega * np.cos(fy * omega * tr + phase_y)
            vz = az * fz * omega * np.cos(fz * omega * tr + phase_z)

            yaw = np.arctan2(vy, vx) if yaw_follows_path else 0.0

            traj.append({
                "pos": np.array([x, y, z], dtype=float),
                "vel": np.array([vx, vy, vz], dtype=float),
                "yaw": float(yaw)
            })
        return traj

    def fct_make_random_waypoint_trajectory(self, time, rng,
                                            n_waypoints=8,
                                            xy_range=3.0,
                                            z_range=(0.5, 4.0),
                                            smooth_sigma=50):
        from scipy.ndimage import gaussian_filter1d

        time = np.asarray(time, dtype=float)
        T = len(time)

        wp_x = [rng.uniform(-xy_range, xy_range) for _ in range(n_waypoints)]
        wp_y = [rng.uniform(-xy_range, xy_range) for _ in range(n_waypoints)]
        wp_z = [rng.uniform(*z_range) for _ in range(n_waypoints)]

        wp_idx = np.linspace(0, T - 1, n_waypoints).astype(int)
        x_raw = np.interp(np.arange(T), wp_idx, wp_x)
        y_raw = np.interp(np.arange(T), wp_idx, wp_y)
        z_raw = np.interp(np.arange(T), wp_idx, wp_z)

        x = gaussian_filter1d(x_raw, smooth_sigma)
        y = gaussian_filter1d(y_raw, smooth_sigma)
        z = gaussian_filter1d(z_raw, smooth_sigma)

        vx = np.gradient(x, self.dt)
        vy = np.gradient(y, self.dt)
        vz = np.gradient(z, self.dt)

        traj = []
        for k in range(T):
            traj.append({
                "pos": np.array([x[k], y[k], z[k]], dtype=float),
                "vel": np.array([vx[k], vy[k], vz[k]], dtype=float),
                "yaw": float(np.arctan2(vy[k], vx[k]))
            })

        p0 = traj[0]["pos"].copy()
        for k in range(T):
            traj[k]["pos"] = traj[k]["pos"] - p0

        return traj



    def fct_run_simulation(self, traj, n):
        """
        Run n simulations for a selected trajectory family.

        Each run uses deterministic randomized trajectory parameters based on
        the run index so the generated dataset is reproducible.
        """

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
            ref_traj = self.fct_sample_trajectory(traj, rng)

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
            t_i, states_i, omegas_i, _, U_i = self.sim_PID.fct_simulate(
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


    def fct_sample_trajectory(self, traj, rng):
        if traj == 1:
            return self.fct_make_helical_trajectory(
                self.time, center=(0.0, 0.0),
                radius=rng.uniform(3, 10),
                z_start=0.0,
                z_end=rng.uniform(3, 10),
                n_turns=1,
                yaw_follows_path=True
            )
        elif traj == 2:
            return self.fct_make_figure8_trajectory(
                self.time, center=(0.0, 0.0, 0.0),
                a=rng.uniform(25, 35),
                b=rng.uniform(25, 35),
                n_loops=1,
                tilt_deg=rng.uniform(10, 80),
                yaw_follows_path=True
            )
        elif traj == 3:
            return self.fct_make_lissajous_trajectory(
                self.time,
                center=(0.0, 0.0, rng.uniform(1.0, 4.0)),
                ax=rng.uniform(15, 25),
                ay=rng.uniform(15, 25),
                az=rng.uniform(3, 7),
                fx=rng.choice([1, 2, 3]),
                fy=rng.choice([1, 2, 3]),
                fz=rng.choice([1, 2, 3]),
                phase_y=rng.uniform(0, np.pi),
                phase_z=rng.uniform(0, np.pi),
                n_loops=1.0,
                yaw_follows_path=True
            )
        elif traj == 4:
            return self.fct_make_random_waypoint_trajectory(
                self.time, rng=rng,
                n_waypoints=rng.randint(5, 15),
                xy_range=rng.uniform(15, 25),
                z_range=(0.5, rng.uniform(3.0, 8.0)),
                smooth_sigma=rng.randint(25, 50)
            )
        elif traj == 5:
            return self.fct_make_hover_excitation_trajectory(
                self.time, rng=rng,
                xyz_amp=(
                    rng.uniform(2.0, 4.0),
                    rng.uniform(2.0, 4.0),
                    rng.uniform(1.0, 2.0),
                ),
                xyz_freq=(
                    rng.uniform(0.05, 0.12),
                    rng.uniform(0.05, 0.12),
                    rng.uniform(0.06, 0.15),
                ),
                yaw_amp_deg=rng.uniform(2.0, 8.0),
                n_sines_range=(2, 4),
            )
        else:
            raise ValueError(f"Unknown traj: {traj}")

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