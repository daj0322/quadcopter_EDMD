import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from Helperfcts import helperfcts
from Simulation import quad_sim


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def body_points(position, phi, theta, psi, arm_length):
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    rz = np.array([[cpsi, -spsi, 0.0], [spsi, cpsi, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[ctheta, 0.0, stheta], [0.0, 1.0, 0.0], [-stheta, 0.0, ctheta]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cphi, -sphi], [0.0, sphi, cphi]])
    rotation = rz @ ry @ rx

    arm = arm_length / np.sqrt(2.0)
    local_points = np.array([
        [arm, -arm, 0.0],
        [-arm, -arm, 0.0],
        [-arm, arm, 0.0],
        [arm, arm, 0.0],
    ])
    return position + local_points @ rotation.T


def reference_attitude_from_trajectory(sim, ref_traj):
    ref_phi = np.zeros(len(ref_traj))
    ref_theta = np.zeros(len(ref_traj))
    ref_psi = np.unwrap([r["yaw"] for r in ref_traj])

    for k, ref in enumerate(ref_traj):
        acc = np.asarray(ref.get("acc", np.zeros(3)), dtype=float)
        force_world = sim.q_mass * (acc + np.array([0.0, 0.0, sim.g]))
        R_des = helperfcts.fct_desired_rotation_from_force_and_yaw(force_world, ref_psi[k])
        ref_phi[k], ref_theta[k], _ = helperfcts.fct_euler_from_R(R_des)

    return ref_phi, ref_theta, ref_psi


def plot_attitude_tracking(t, states, ref_traj, sim, traj_label, save_path):
    ref_phi, ref_theta, ref_psi = reference_attitude_from_trajectory(sim, ref_traj)
    actual_phi = states[:, 6]
    actual_theta = states[:, 7]
    actual_psi = np.unwrap(states[:, 8])

    actual = [actual_phi, actual_theta, actual_psi]
    reference = [ref_phi, ref_theta, ref_psi]
    labels = ["Roll phi [rad]", "Pitch theta [rad]", "Yaw psi [rad]"]

    fig_att, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, y_actual, y_ref, label in zip(axes, actual, reference, labels):
        rms = float(np.sqrt(np.mean((y_actual - y_ref) ** 2)))
        ax.plot(t, y_ref, color="#ff7f0e", linewidth=1.8, label="reference")
        ax.plot(t, y_actual, color="#1f77b4", linewidth=1.3, label=f"actual (RMS {rms:.4f})")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig_att.suptitle(f"{traj_label}: roll, pitch, yaw tracking")
    fig_att.tight_layout()

    if save_path:
        root = save_path.rsplit(".", 1)[0] if "." in save_path else save_path
        attitude_path = f"{root}_attitude.png"
        fig_att.savefig(attitude_path, dpi=160)
        print(f"Saved attitude plot to {attitude_path}")


def run_visualization(traj_id, duration, playback_speed, stride, save_path):
    traj_names = {
        1: "helix",
        2: "figure-8",
        3: "lissajous",
    }
    sim = quad_sim()
    if duration is not None:
        sim.time = np.arange(0.0, duration, sim.dt)

    t_all, states_all, u_all, ref_traj_list = sim.fct_run_simulation(traj_id, 1)
    t = t_all[0]
    states = states_all[0]
    controls = u_all[0]
    ref_traj = ref_traj_list[0]
    ref_pos = np.array([r["pos"] for r in ref_traj], dtype=float)
    ref_yaw = np.array([r["yaw"] for r in ref_traj], dtype=float)
    ref_vel = np.array([r["vel"] for r in ref_traj], dtype=float)
    traj_label = f"Trajectory {traj_id} ({traj_names.get(traj_id, 'unknown')})"

    frame_idx = np.arange(0, len(t), stride)
    interval_ms = max(1, int(1000.0 * sim.dt * stride / playback_speed))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], "--", color="0.55", linewidth=1.8, label="reference")
    actual_line, = ax.plot([], [], [], color="#1f77b4", linewidth=2.2, label="simulated drone path")
    arm_a, = ax.plot([], [], [], color="#222222", linewidth=3.0)
    arm_b, = ax.plot([], [], [], color="#222222", linewidth=3.0)
    motors = ax.scatter([], [], [], s=55, color="#d62728", depthshade=True)
    heading_line, = ax.plot([], [], [], color="#2ca02c", linewidth=3.0, label="actual yaw")
    ref_heading_line, = ax.plot([], [], [], color="#ff7f0e", linewidth=2.0, label="reference yaw")
    title = ax.set_title("")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc="upper left")
    ax.grid(True)

    all_pos = np.vstack([states[:, 0:3], ref_pos])
    pad = 0.6
    ax.set_xlim(all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ax.set_ylim(all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)
    ax.set_zlim(max(0.0, all_pos[:, 2].min() - pad), all_pos[:, 2].max() + pad)
    set_axes_equal(ax)

    yaw_arrow_len = max(0.35, 1.5 * sim.q_l)

    def set_line_3d(line, points):
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])

    def update(frame_number):
        k = frame_idx[frame_number]
        state = states[k]
        position = state[0:3]
        phi, theta, psi = state[6:9]

        points = body_points(position, phi, theta, psi, sim.q_l)
        set_line_3d(arm_a, points[[0, 2]])
        set_line_3d(arm_b, points[[1, 3]])
        motors._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

        set_line_3d(actual_line, states[: k + 1, 0:3])

        actual_heading = np.array([
            position,
            position + yaw_arrow_len * np.array([np.cos(psi), np.sin(psi), 0.0]),
        ])
        ref_heading = np.array([
            ref_pos[k],
            ref_pos[k] + yaw_arrow_len * np.array([np.cos(ref_yaw[k]), np.sin(ref_yaw[k]), 0.0]),
        ])
        set_line_3d(heading_line, actual_heading)
        set_line_3d(ref_heading_line, ref_heading)

        yaw_error = helperfcts.wrap_angle(psi - ref_yaw[k])
        speed_xy = np.linalg.norm(ref_vel[k, 0:2])
        title.set_text(
            f"{traj_label} | t={t[k]:.2f}s | "
            f"yaw={psi:.2f} rad | ref={ref_yaw[k]:.2f} rad | error={yaw_error:.2f} rad | u4={controls[k, 3]:.4f}"
            f" | ref speed xy={speed_xy:.2f} m/s"
        )
        return actual_line, arm_a, arm_b, motors, heading_line, ref_heading_line, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_idx),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    plot_attitude_tracking(t, states, ref_traj, sim, traj_label, save_path)

    if save_path:
        animation.save(save_path, dpi=140)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the quadcopter trajectory and yaw tracking in 3D.")
    parser.add_argument("--traj", type=int, default=2, choices=[1, 2, 3], help="1 for helix, 2 for figure-8, 3 for lissajous.")
    parser.add_argument("--duration", type=float, default=None, help="Optional simulation duration in seconds. Defaults to quad_sim.time.")
    parser.add_argument("--speed", type=float, default=2.0, help="Playback speed multiplier.")
    parser.add_argument("--stride", type=int, default=5, help="Use every Nth simulation sample for animation.")
    parser.add_argument("--save", default="", help="Optional output path, such as drone_yaw.gif or drone_yaw.mp4.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_visualization(args.traj, args.duration, args.speed, args.stride, args.save)
