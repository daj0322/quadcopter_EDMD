"""
Current-simulator MPC adapter.

The original Animesh NMPC file used attitude commands:
    [thrust, phi_des, theta_des]

Darren's current simulator uses wrench commands:
    [thrust, tau_roll, tau_pitch, tau_yaw]

This module keeps the old `build_nmpc(...)` import path alive by returning a
linear wrench-space QP MPC built with the same solver interface used by
compare_mpc.py.
"""

import numpy as np

from compare_mpc import (
    DU_MAX,
    DU_MIN,
    NC_MPC,
    N_MPC,
    Q_DIAG,
    R_DIAG,
    RD_DIAG,
)
from edmdc_mpc import EDMDcMPC_QP, STATE_DIM


def build_nmpc(sim, dt, A=None, B=None, x_scaler=None, u_scaler=None, N=N_MPC, NC=NC_MPC):
    """
    Build a current yaw-wrench MPC object.

    Parameters
    ----------
    sim : quad_sim
        Current Darren simulator instance.
    dt : float
        Kept for compatibility with the old builder. The supplied discrete
        model should already match this time step.
    A, B : ndarray
        Discrete standardized linear model matrices.
    x_scaler, u_scaler : sklearn scalers
        State and input scalers for the linear model.
    N, NC : int
        Prediction and control horizons.
    """
    if A is None or B is None or x_scaler is None or u_scaler is None:
        raise ValueError(
            "build_nmpc now requires A, B, x_scaler, and u_scaler for the "
            "current yaw-wrench simulator. See compare_mpc.py for the full "
            "closed-loop comparison workflow."
        )

    hover = np.array([sim.q_mass * sim.g, 0.0, 0.0, 0.0], dtype=float)
    return EDMDcMPC_QP(
        A=A,
        B=B,
        Cz=np.eye(STATE_DIM),
        N=N,
        NC=NC,
        Q=np.diag(Q_DIAG),
        R=np.diag(R_DIAG),
        Rd=np.diag(RD_DIAG),
        u_scaler=u_scaler,
        du_min=DU_MIN,
        du_max=DU_MAX,
        u_nominal_raw=hover,
    )
