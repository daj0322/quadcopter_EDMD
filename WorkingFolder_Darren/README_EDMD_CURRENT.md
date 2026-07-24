# Current EDMD Workflow

This folder uses Darren's current PX4-like simulation/controller stack.

Simulation state:

```text
[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
```

Logged control input:

```text
[thrust, tau_roll, tau_pitch, tau_yaw]
```

The yaw channel is now the commanded yaw torque from the controller, not a desired
yaw angle. Reference yaw and yaw rate remain in each trajectory dictionary so the
simulator can track heading along the path.

## Regenerate EDMD Data

```bash
python parallel_sim.py
python mix_traj.py
python EDMDc_training.py
python compare_three.py
python compare_mpc.py
python Intercept_comparison.py
```

Outputs:

```text
runs_traj1_n50.pkl
runs_traj2_n50.pkl
runs_traj3_n50.pkl
runs_mixed_n150.pkl
edmdc_model_yaw_wrench.pkl
```

`edmdc_mpc.py` contains the shared state/input lifting utilities used by the
trained model.

`compare_three.py` visualizes the current PID/PX4 simulation trace against EDMDc
and a linear least-squares baseline using the same logged wrench inputs. The
model traces use 1-second reset windows so the plot shows prediction quality
over the full trajectory without one long open-loop drift hiding the path.
`final_comparison.py` calls the same comparison entrypoint.

`compare_mpc.py` runs closed-loop PID/PX4, EDMD-MPC, and linear-MPC through the
current yaw-wrench plant. By default it evaluates the full trajectory; use a
command like `python compare_mpc.py --steps 1500` for a shorter development run.
The restored MPC entrypoints `final_comparison.py`, `tunerfull.py`,
`Intercept_comparison.py`, `intercept_comparison_w_pred.py`, and
`Real data edmdc mpc.py` now call this same current-simulator MPC comparison.

`Intercept_comparison.py` runs moving-target interception graphs with the
current yaw-wrench simulator. `intercept_comparison_w_pred.py` calls the same
updated interception workflow.
