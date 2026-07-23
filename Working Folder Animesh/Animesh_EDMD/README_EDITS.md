# EDMDc Quadcopter Code Edits

This note summarizes the active code changes made during the EDMDc/yaw/MPC work. It only describes the retained changes that are still part of the current folder state.

## Current Goal

The codebase now supports a 12-state quadcopter EDMDc workflow:

```text
[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
```

The learned raw command vector is:

```text
[thrust, phi_des, theta_des, psi_des]
```

The EDMDc model still controls the plant through attitude-level commands, but yaw is now part of both the state and the command path.

The active target-tracking setup is now a fixed-yaw-with-yaw-control experiment: yaw is not removed, but helix, figure-8, and lissajous use `psi_des = 0.0` so the yaw loop holds heading while EDMDc tracks position.

## Main Pipeline

Use this sequence when regenerating data and retraining:

```powershell
python parallel_sim.py
python mix_traj.py
python EDMDc_training.py
python tunerfull.py
python compare_three.py
```

Use `final_comparison.py` after `compare_three.py` when you want the fuller comparison script.

After changing the yaw convention, regenerate the run files before judging the EDMDc model. The old `runs_*.pkl` files and `edmdc_model_300.pkl` were produced under the previous yaw setup.

## Yaw Control Path

### `Cascaded_Controllers.py`

- Added yaw PID support through `pid_psi`.
- Added desired-yaw handling with `fct_desired_yaw`.
- Added yaw torque generation with `fct_yaw_torque`.
- The controller now returns attitude commands as:

```text
[thrust, phi_des, theta_des, psi_des]
```

- Desired yaw follows the explicit trajectory yaw when present.
- If no yaw is present in a reference, the fallback is heading from XY velocity.
- Roll and pitch commands are computed using the desired yaw frame.

### `Closed_loop.py`

- `fct_simulate` logs 4-channel attitude commands.
- `fct_step_attitude` now accepts `psi_des`.
- MPC comparisons can command:

```text
thrust, phi_des, theta_des, psi_des
```

- The inner attitude loop applies yaw torque while holding the MPC attitude command over the MPC step.

## Trajectory And Plotting

### `Simulation.py`

- Trajectory references include yaw.
- Helix, figure-8, and lissajous references use fixed yaw by default:

```python
TARGET_TRAJECTORY_YAW_FOLLOWS_PATH = False
TARGET_TRAJECTORY_YAW_CONSTANT = 0.0
```

- Random waypoint yaw is generated from XY velocity heading.
- Added a Matplotlib simulation plotting path:
  - 3D reference vs actual trajectory.
  - Position reference vs actual subplots.
  - Attitude reference vs actual subplots.
  - Velocity, body rates, commanded inputs, and tracking-error plots.
- Added an optional animated drone visualization in the Matplotlib figure.
- Plots are shown interactively with Matplotlib, not saved as PNGs.

## Data Mixing

### `mix_traj.py`

- Mixed datasets keep full yaw-aware logs.
- States are expected as 12 states:

```text
[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
```

- Inputs are expected as 4 commands:

```text
[thrust, phi_des, theta_des, psi_des]
```

- The mixed dataset records family labels and source files.

Current mixed data file:

```text
runs_mixed_n300.pkl
```

## EDMDc Training

### `EDMDc_training.py`

- Training is now at 100 Hz:

```text
dt = 0.01
```

- The active model uses 12 physical states.
- The active raw input dimension is 4.
- Training focuses on the trajectory families we care about:
  - helix
  - figure-8
  - lissajous
- Held-out test indices are:

```text
39, 59, 129
```

- Added early-transient weighting for the first 2 seconds of each run.
- Added regularization sweep over:

```text
[1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
```

- Lambda selection uses both:
  - rolling 2-second horizon error
  - first 2-second free-rollout diagnostic score

- Added exact kinematic rows for position and attitude integrators.

### Observables

The 12-state observable lift includes:

- the 12 standardized states
- sin/cos of roll, pitch, and yaw
- angle-rate terms such as `phi*p`, `theta*q`, `psi*r`
- velocity-angle terms
- velocity and angular-rate energy terms
- yaw-related terms
- extra x/y/vx/vy trajectory-shape terms
- body-frame velocity terms
- thrust-direction terms
- bias

The current observable dimension is:

```text
n_obs = 56
```

### Lifted Inputs

The raw 4-command input is lifted to 7 learned input channels:

```text
[thrust, phi_des, theta_des, psi_des, thrust_x, thrust_y, thrust_z]
```

The extra channels are thrust projected through the current attitude direction.

Current saved model file:

```text
edmdc_model_300.pkl
```

Current saved model shape:

```text
A = (56, 56)
B = (56, 7)
dt = 0.01
raw_input_dim = 4
input_lift_type = thrust_direction
```

### Training Diagnostics

- Added short-horizon diagnostic plots.
- Added one-step, single-rollout, rolling-position, rolling-velocity, and full-state RMSE reporting.
- Added a short-horizon gate message that prints whether the model is good enough to move forward.
- Fixed the subplot title/layout overlap in the short-horizon state-prediction plots.

## EDMDc MPC Runtime

### `edmdc_mpc.py`

- Added 12-state observable support while keeping legacy 10-state loading support.
- Added helpers for:
  - `thrust_direction_from_state_phys`
  - `lift_inputs_from_phys`
  - `scaled_lifted_input_from_phys`
  - `drop_to_12state`
  - `reference_yaw_arrays`
- `EDMDcMPC_QP` now detects when a model was trained with lifted thrust-direction inputs.
- The QP still optimizes the 4 real commands:

```text
[thrust, phi_des, theta_des, psi_des]
```

- Internally, the learned 7-input EDMDc model is locally linearized around the current physical state.
- The effective MPC input matrix is built as:

```text
B_eff = B_model @ d(lifted_input) / d(raw_input)
```

- The local lifted-input linearization is refreshed every MPC step.
- The EDMDc plant command path sends `psi_des` to the simulator when yaw correction is enabled.

## Comparison Scripts

### `compare_three.py`

This is the main comparison script for:

- PID
- EDMDc MPC
- Linear MPC

The active test set is now only:

```text
helix (small)
figure-8
lissajous
```

The non-target waypoint and hover-excitation cases are not part of the active comparison set.

Current active MPC tuning block:

```python
N_MPC  = 20
NC_MPC = 15

Q_DIAG = np.array([
    400000.0, 640000.0, 400000.0,
        50.0,     80.0,     50.0,
         0.0,      0.0,  20000.0,
         0.0,      0.0,   3000.0,
], dtype=float)

R_DIAG  = np.array([2.5e-05, 0.5, 0.5], dtype=float)
RD_DIAG = np.array([2.5e-06, 0.05, 0.05], dtype=float)
R_YAW   = 0.25
RD_YAW  = 0.025
DU_YAW  = 0.005
```

Other active settings:

```python
USE_PID_NOMINAL = True
EDMDC_YAW_CORRECTION = True
USE_CONSTANT_YAW_REFERENCE = True
```

Important behavior:

- EDMDc computes corrections around the nominal cascaded PID attitude command.
- EDMDc is allowed to adjust yaw through `psi_des`.
- Constant-yaw reference override is on for the target comparisons.
- Full 100-second trajectories are used.

Plots include:

- summary bar chart
- 3D trajectory comparison
- per-axis position tracking plots
- position-error magnitude over time
- weak-state diagnostics for figure-8
- control inputs for figure-8
- solve-time comparison

The 3D and error plot layouts were adjusted to match the three active trajectories.

### `final_comparison.py`

This script follows the same active EDMDc/linear/PID setup as `compare_three.py`, but also includes NMPC comparison code.

Active test set:

```text
helix (small)
figure-8
lissajous
```

The active EDMDc tuning block matches `compare_three.py`.

## MPC Tuning

### `tunerfull.py`

- Tunes MPC settings using the saved EDMDc model.
- Tuning is separate from EDMDc training.
- Uses the same target trajectories:

```text
helix, figure-8, lissajous
```

- `DU_YAW_FIXED` is:

```python
DU_YAW_FIXED = 0.005
```

- Full validation now runs the full 10000-step trajectory:

```python
VALIDATION_STEPS = 10000
FINAL_VALIDATION_STEPS = 10000
```

- The score was changed to normalize position RMSE by each trajectory's PID baseline. This prevents lissajous from dominating the tuner only because it naturally has larger raw-meter errors.
- The normalized scoring uses:

```python
SCORE_PID_RMSE_FLOOR = 0.5
```

- The final score is normalized position error plus yaw/r penalties.
- The script prints a block that can be pasted into `compare_three.py` and `final_comparison.py`.

## Most Recent MPC Tuning Result

The latest focused full-validation run in `tunerfull.py` tested 18 MPC blocks on the full 10000-step trajectories. The selected block above gave:

```text
helix (small): EDMDc=1.0009 m
figure-8:      EDMDc=1.8388 m
lissajous:     EDMDc=10.1287 m
```

This improves the previous active block on all three target trajectories, but lissajous is still the weak case.

## Notes

- EDMDc training and MPC tuning are separate steps.
- `EDMDc_training.py` learns and saves the EDMDc model.
- `tunerfull.py` loads the saved EDMDc model and tunes MPC weights/horizons.
- `compare_three.py` evaluates the tuned EDMDc MPC against PID and linear MPC.
- `final_comparison.py` is the broader comparison script that also includes NMPC.
