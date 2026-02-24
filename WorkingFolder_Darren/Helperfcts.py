import numpy as np

class helperfcts:
    def wrap_angle(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def fct_euler_from_R(R):
        theta = np.arcsin(-np.clip(R[2,0], -1.0, 1.0))
        phi   = np.arctan2(R[2,1], R[2,2])
        psi   = np.arctan2(R[1,0], R[0,0])
        return phi, theta, psi

    def fct_desired_rotation_from_force_and_yaw(F_world, psi_des):
        b3 = F_world / (np.linalg.norm(F_world) + 1e-12)

        cpsi, spsi = np.cos(psi_des), np.sin(psi_des)
        b1_des = np.array([cpsi, spsi, 0.0])

        b2 = np.cross(b3, b1_des)
        n = np.linalg.norm(b2)
        if n < 1e-9:
            b1_des = np.array([1.0, 0.0, 0.0])
            b2 = np.cross(b3, b1_des)
            n = np.linalg.norm(b2) + 1e-12
        b2 = b2 / n
        b1 = np.cross(b2, b3)
        return np.column_stack((b1, b2, b3))
    
    def fct_rms_error(states, ref_traj, dt):
        xr = np.array([r["pos"][0] for r in ref_traj], dtype=float)
        yr = np.array([r["pos"][1] for r in ref_traj], dtype=float)
        zr = np.array([r["pos"][2] for r in ref_traj], dtype=float)

        ex = states[:, 0] - xr
        ey = states[:, 1] - yr
        ez = states[:, 2] - zr

        exy = ex**2 + ey**2
        ez = ez**2

        Txy = (len(exy) - 1) * dt
        integral = np.trapezoid(exy, dx=dt)
        rms_xy = float(np.sqrt(integral / Txy))

        Tz = (len(ez) - 1) * dt
        integral = np.trapezoid(ez, dx=dt)
        rms_z = float(np.sqrt(integral / Tz))

        return rms_xy, rms_z