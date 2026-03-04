import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Trajectory Definitions
# ============================================================

def fct_lissajous_trajectory(time, a=1.0, b=1.7, c=0.8, phase=np.pi/4):
    traj = []
    for t in time:
        fx = np.sin(a*t)
        fy = np.sin(b*t + phase)
        fz = np.sin(c*t)

        x = 7 + 3*fx
        y = 7 + 3*fy
        z = 7 + 3*fz

        traj.append({"pos": np.array([x,y,z])})
    return traj


def fct_trefoil_trajectory(time):
    traj = []
    for t in time:
        fx = (np.sin(t) + 2*np.sin(2*t)) / 3
        fy = (np.cos(t) - 2*np.cos(2*t)) / 3
        fz = np.sin(3*t)

        x = 7 + 3*fx
        y = 7 + 3*fy
        z = 7 + 3*fz

        traj.append({"pos": np.array([x,y,z])})
    return traj


def fct_spiral_trajectory(time, omega=1.0, z_freq=0.7):
    traj = []
    for t in time:
        fx = np.cos(omega*t)
        fy = np.sin(omega*t)
        fz = np.sin(z_freq*t)

        x = 7 + 3*fx
        y = 7 + 3*fy
        z = 7 + 3*fz

        traj.append({"pos": np.array([x,y,z])})
    return traj


def fct_multisine_trajectory(time):
    traj = []
    for t in time:
        fx = (np.sin(t) + 0.5*np.sin(2.3*t)) / 1.5
        fy = (np.sin(1.7*t) + 0.4*np.sin(3.1*t)) / 1.4
        fz = (0.8*np.sin(0.9*t) + 0.3*np.sin(2.5*t)) / 1.1

        x = 7 + 3*fx
        y = 7 + 3*fy
        z = 7 + 3*fz

        traj.append({"pos": np.array([x,y,z])})
    return traj


def fct_line_trajectory(time, slope_x=1.0, slope_y=0.5, slope_z=-0.7):
    T = time[-1] - time[0]
    traj = []
    for t in time:
        tau = (t - time[0]) / T
        base = 2*tau - 1

        fx = np.clip(slope_x * base, -1, 1)
        fy = np.clip(slope_y * base, -1, 1)
        fz = np.clip(slope_z * base, -1, 1)

        x = 7 + 3*fx
        y = 7 + 3*fy
        z = 7 + 3*fz

        traj.append({"pos": np.array([x,y,z])})
    return traj


def fct_drift_trajectory(time, slope=1.0):
    T = time[-1] - time[0]
    traj = []
    for t in time:
        tau = (t - time[0]) / T
        base = 2*tau - 1

        fx = np.clip(slope * base, -1, 1)
        fy = np.sin(t)
        fz = np.sin(1.5*t)

        x = 7 + 3*fx
        y = 7 + 3*fy
        z = 7 + 3*fz

        traj.append({"pos": np.array([x,y,z])})
    return traj


def fct_chirp_trajectory(time, w0=0.5, beta=0.05):
    traj = []
    for t in time:
        w = w0 + beta*t

        fx = np.sin(w*t)
        fy = np.cos(w*t)
        fz = np.sin(0.7*w*t)

        x = 7 + 3*fx
        y = 7 + 3*fy
        z = 7 + 3*fz

        traj.append({"pos": np.array([x,y,z])})
    return traj


# ============================================================
# Time Vector
# ============================================================

t = np.linspace(0, 20, 2000)

# ============================================================
# Generate Trajectories
# ============================================================

traj_list = [
    (fct_lissajous_trajectory(t), "Lissajous"),
    (fct_trefoil_trajectory(t), "Trefoil"),
    (fct_spiral_trajectory(t), "Spiral"),
    (fct_multisine_trajectory(t), "Multisine"),
    (fct_line_trajectory(t), "Line"),
    (fct_drift_trajectory(t), "Drift"),
    (fct_chirp_trajectory(t), "Chirp"),
]

# ============================================================
# Plot Subplots
# ============================================================

fig = plt.figure(figsize=(16, 12))

for i, (traj, name) in enumerate(traj_list):
    ax = fig.add_subplot(3, 3, i+1, projection='3d')

    xyz = np.array([p["pos"] for p in traj])
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], linewidth=1.5)

    ax.set_xlim(4, 10)
    ax.set_ylim(4, 10)
    ax.set_zlim(4, 10)

    ax.set_box_aspect([1,1,1])  # Equal scaling
    ax.set_title(name)

plt.tight_layout()
plt.show()