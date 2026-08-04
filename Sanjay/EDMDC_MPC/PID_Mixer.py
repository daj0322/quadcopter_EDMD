import numpy as np


class pid_mixer:
    @staticmethod
    def fct_allocation_matrix(kT, kD, l):
        """Return the wrench-to-rotor-thrust allocation matrix for the X frame.

        The input wrench is ``[total_thrust, tau_roll, tau_pitch, tau_yaw]``
        and the output is the four requested rotor thrusts. The same matrix is
        used by the simulator, data logger, and MPC feasibility constraints.
        """
        if kT <= 0.0 or kD <= 0.0 or l <= 0.0:
            raise ValueError("kT, kD, and l must be positive")
        arm = l / np.sqrt(2.0)
        yaw_force = kT / (4.0 * kD)
        return np.array([
            [0.25, -1.0 / (4.0 * arm),  1.0 / (4.0 * arm),  yaw_force],
            [0.25, -1.0 / (4.0 * arm), -1.0 / (4.0 * arm), -yaw_force],
            [0.25,  1.0 / (4.0 * arm), -1.0 / (4.0 * arm),  yaw_force],
            [0.25,  1.0 / (4.0 * arm),  1.0 / (4.0 * arm), -yaw_force],
        ], dtype=float)

    @staticmethod
    def fct_max_motor_forces(kT, max_omega, prop_efficiency=None):
        """Return per-rotor thrust limits for a motor-speed limit."""
        if kT <= 0.0 or max_omega < 0.0:
            raise ValueError("kT must be positive and max_omega nonnegative")
        efficiency = (
            np.ones(4, dtype=float) if prop_efficiency is None
            else np.asarray(prop_efficiency, dtype=float)
        )
        if efficiency.shape != (4,) or np.any(efficiency <= 0.0):
            raise ValueError("prop_efficiency must contain four positive entries")
        return efficiency * kT * float(max_omega)**2

    @staticmethod
    def fct_wrench_from_motor_forces(forces, kT, kD, l):
        """Recover the realized body wrench from four rotor thrusts."""
        forces = np.asarray(forces, dtype=float).reshape(-1)
        if forces.shape != (4,):
            raise ValueError("Expected four rotor thrusts")
        if kT <= 0.0 or kD <= 0.0 or l <= 0.0:
            raise ValueError("kT, kD, and l must be positive")
        arm = l / np.sqrt(2.0)
        yaw_ratio = kD / kT
        return np.array([
            np.sum(forces),
            arm * (-forces[0] - forces[1] + forces[2] + forces[3]),
            arm * (forces[0] - forces[1] - forces[2] + forces[3]),
            yaw_ratio * (forces[0] - forces[1] + forces[2] - forces[3]),
        ], dtype=float)

    @staticmethod
    def fct_allocate_wrench(u, kT, kD, l, min_omega=0.0, max_omega=2000.0,
                            prop_efficiency=None):
        """Allocate a requested wrench and return the wrench the plant receives.

        Per-motor thrust clipping is the actuator nonlinearity in this
        simulator. Returning the realized wrench makes that nonlinearity
        explicit instead of silently associating a requested wrench with a
        different state transition.
        """
        u_requested = np.asarray(u, dtype=float).reshape(-1)
        if u_requested.shape != (4,):
            raise ValueError("Expected wrench [thrust, tau_roll, tau_pitch, tau_yaw]")
        if min_omega < 0.0 or max_omega < min_omega:
            raise ValueError("Require 0 <= min_omega <= max_omega")

        efficiency = (
            np.ones(4, dtype=float) if prop_efficiency is None
            else np.asarray(prop_efficiency, dtype=float)
        )
        if efficiency.shape != (4,) or np.any(efficiency <= 0.0):
            raise ValueError("prop_efficiency must contain four positive entries")

        allocation = pid_mixer.fct_allocation_matrix(kT, kD, l)
        requested_forces = allocation @ u_requested
        min_forces = efficiency * kT * float(min_omega)**2
        max_forces = pid_mixer.fct_max_motor_forces(
            kT, max_omega, prop_efficiency=efficiency
        )
        applied_forces = np.clip(requested_forces, min_forces, max_forces)
        omega = np.sqrt(applied_forces / (efficiency * kT))
        u_applied = pid_mixer.fct_wrench_from_motor_forces(
            applied_forces, kT, kD, l
        )
        saturated = bool(np.any(~np.isclose(
            applied_forces, requested_forces, rtol=1e-12, atol=1e-12
        )))
        return omega, u_applied, applied_forces, saturated

    @staticmethod
    def fct_mixer(u, kT, kD, l, min_omega=0, max_omega=2000):
        """Compatibility wrapper returning allocated motor speeds only."""
        omega, _, _, _ = pid_mixer.fct_allocate_wrench(
            u, kT, kD, l, min_omega=min_omega, max_omega=max_omega
        )
        return omega
