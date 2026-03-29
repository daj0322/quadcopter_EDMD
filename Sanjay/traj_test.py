import numpy as np
import multiprocessing as mp
from Simulation import quad_sim
import random


def check_traj(args):
    label, traj_type, params = args
    sim = quad_sim()

    if traj_type == "fig8":
        ref = sim.fct_make_figure8_trajectory(
            sim.time, center=(0.0, 0.0, 0.0),
            a=params["a"], b=params["b"], n_loops=1,
            tilt_deg=45.0, yaw_follows_path=True
        )
    elif traj_type == "helix":
        ref = sim.fct_make_helical_trajectory(
            sim.time, center=(0.0, 0.0),
            radius=params["r"], z_start=0.0,
            z_end=params.get("z_end", 30.0), n_turns=1,
            yaw_follows_path=True
        )
    elif traj_type == "lissajous":
        ref = sim.fct_make_lissajous_trajectory(
            sim.time,
            center=(0.0, 0.0, 5.0),
            ax=params["a"], ay=params["a"], az=params["a"]//5,
            fx=1, fy=2, fz=3,
            phase_y=np.pi/2, phase_z=np.pi/4,
            n_loops=1.0, yaw_follows_path=True
        )
    elif traj_type == "waypoint":
        rng = random.Random(42)
        ref = sim.fct_make_random_waypoint_trajectory(
            sim.time, rng=rng,
            n_waypoints=10,
            xy_range=params["r"],
            z_range=(1.0, 10.0),
            smooth_sigma=40
        )
    elif traj_type == "hover":
        rng = random.Random(42)
        ref = sim.fct_make_hover_excitation_trajectory(
            sim.time, rng=rng,
            xyz_amp=(params["amp"], params["amp"], params["amp"]/2),
            xyz_freq=(0.05, 0.05, 0.08),
            yaw_amp_deg=5.0,
            n_sines_range=(2, 4),
        )

    p0 = ref[0]["pos"].copy()
    for k in range(len(ref)):
        ref[k]["pos"] = ref[k]["pos"] - p0

    init_state = np.zeros(12)
    t, states, _, _, u_att = sim.sim_PID.fct_simulate(
        sim.time, sim.dt, ref, init_state
    )

    phi_des   = u_att[:, 1]
    theta_des = u_att[:, 2]
    speed     = np.linalg.norm(states[:, 3:6], axis=1)
    ref_pos   = np.array([wp["pos"] for wp in ref])
    pos_error = np.linalg.norm(ref_pos - states[:, :3], axis=1)
    tilt      = np.sqrt(phi_des**2 + theta_des**2)
    first_5s  = int(5.0 / sim.dt)

    return {
        "label":       label,
        "speed_mean":  speed.mean(),
        "speed_max":   speed.max(),
        "phi_std":     np.rad2deg(phi_des.std()),
        "phi_min":     np.rad2deg(phi_des.min()),
        "phi_max":     np.rad2deg(phi_des.max()),
        "theta_std":   np.rad2deg(theta_des.std()),
        "theta_min":   np.rad2deg(theta_des.min()),
        "theta_max":   np.rad2deg(theta_des.max()),
        "tilt_5":      100*np.mean(np.rad2deg(tilt) > 5),
        "tilt_10":     100*np.mean(np.rad2deg(tilt) > 10),
        "tilt_20":     100*np.mean(np.rad2deg(tilt) > 20),
        "err_mean":    pos_error.mean(),
        "err_max":     pos_error.max(),
        "err_first5":  pos_error[:first_5s].mean(),
        "err_after5":  pos_error[first_5s:].mean(),
    }


if __name__ == "__main__":
    configs = []

    # figure-8
    for a in [25, 35, 40, 50, 60]:
        configs.append((f"fig8 a=b={a}m", "fig8", {"a": a, "b": a}))

    # helix
    for r in [25, 35, 40, 50, 60]:
        configs.append((f"helix r={r}m", "helix", {"r": r, "z_end": 30.0}))

    # lissajous
    for a in [20, 30, 40, 50]:
        configs.append((f"lissajous a={a}m", "lissajous", {"a": a}))

    # waypoint
    for r in [20, 30, 40, 50]:
        configs.append((f"waypoint range={r}m", "waypoint", {"r": r}))

    # hover
    for amp in [1.0, 2.0, 4.0]:
        configs.append((f"hover amp={amp}m", "hover", {"amp": amp}))

    print(f"Running {len(configs)} configs on {mp.cpu_count()} cores...")

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(check_traj, configs)

    # Print results grouped by trajectory type
    current_type = ""
    for r in results:
        ttype = r["label"].split()[0]
        if ttype != current_type:
            current_type = ttype
            print(f"\n{'='*60}")
            print(f"  {current_type.upper()}")
            print(f"{'='*60}")

        print(f"\n  {r['label']}")
        print(f"  Speed mean/max:          {r['speed_mean']:.2f} / {r['speed_max']:.2f} m/s")
        print(f"  phi_des   min/std/max:   {r['phi_min']:.1f} / {r['phi_std']:.1f} / {r['phi_max']:.1f} deg")
        print(f"  theta_des min/std/max:   {r['theta_min']:.1f} / {r['theta_std']:.1f} / {r['theta_max']:.1f} deg")
        print(f"  % time tilt > 5/10/20:   {r['tilt_5']:.1f}% / {r['tilt_10']:.1f}% / {r['tilt_20']:.1f}%")
        print(f"  Tracking mean/max:       {r['err_mean']:.2f} / {r['err_max']:.2f} m")
        print(f"  First 5s / After 5s:     {r['err_first5']:.2f} / {r['err_after5']:.2f} m")