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

`U` is the wrench actually applied to the plant after motor allocation and
per-motor thrust limits. `U_requested` is saved alongside it for actuator
diagnostics. EDMDc trains only on `U`, so every state transition uses the input
the plant actually received.

The yaw channel is applied yaw torque, not a desired yaw angle. Reference yaw
and yaw rate remain in each trajectory dictionary. Path simulations start at
their reference heading and cap the heading rate at 0.8 rad/s, avoiding an
artificial, infeasible yaw step at startup.

## Regenerate EDMD Data

```bash
python parallel_sim.py --workers 4
python mix_traj.py
python EDMDc_training.py
python compare_three.py
python compare_mpc.py
python Intercept_comparison.py
```

The default `paper` profile deliberately matches the old paper's data regime:
100-second runs at 100 Hz, with 50 helix, 50 figure-eight, 50 Lissajous,
50 waypoint, 30 hover-excitation, and 70 PRBS runs (300 total). It preserves
the original deterministic seed convention and the original five held-out test
runs. EDMDc then uses the original 0.1 s identification sample time and 2 s
MPC horizon by default, while retaining the yaw-aware state and applied-wrench
data contract.

Regenerate all data and models after changes to the allocator or simulator.
`mix_traj.py` and `EDMDc_training.py` intentionally reject legacy datasets
that do not declare `input_type="applied_wrench"` and include `U_requested`.

The PRBS count and seed range are kept from the old code, but its implementation
is intentionally yaw-aware: it excites a bounded yaw-rate reference through the
same controller, so `U` remains a physical applied wrench rather than a log of
desired roll/pitch angles.

For a first reproducible validation set with disjoint train/validation/test
trajectory runs, use the simulator's intended 45-second duration:

```bash
mkdir -p artifacts/yaw_validation
python parallel_sim.py --profile compact --runs-per-family 4 --prbs-runs 4 --duration 45 --output-dir artifacts/yaw_validation
python mix_traj.py --profile compact --runs-per-family 4 --prbs-runs 4 --input-dir artifacts/yaw_validation
EDMDC_DATA_FILE=artifacts/yaw_validation/runs_mixed_n16.pkl \\
EDMDC_MODEL_FILE=artifacts/yaw_validation/edmdc_model_yaw_wrench.pkl \\
EDMDC_VALIDATION_INDICES=2,6,10 EDMDC_TEST_INDICES=3,7,11 \\
MPLBACKEND=Agg python EDMDc_training.py
```

For a quick deterministic smoke test that does not depend on Git LFS assets:

```bash
mkdir -p artifacts/smoke
python parallel_sim.py --profile compact --runs-per-family 3 --prbs-runs 3 --duration 8 --output-dir artifacts/smoke
python mix_traj.py --profile compact --runs-per-family 3 --prbs-runs 3 --input-dir artifacts/smoke
EDMDC_DATA_FILE=artifacts/smoke/runs_mixed_n12.pkl \\
EDMDC_MODEL_FILE=artifacts/smoke/edmdc_model_yaw_wrench.pkl \\
EDMDC_VALIDATION_INDICES=1,4,7 EDMDC_TEST_INDICES=2,5,8 \\
MPLBACKEND=Agg python EDMDc_training.py
EDMDC_DATA_FILE=artifacts/smoke/runs_mixed_n12.pkl \\
EDMDC_MODEL_FILE=artifacts/smoke/edmdc_model_yaw_wrench.pkl \\
EDMDC_TEST_INDICES=2,5,8 MPLBACKEND=Agg python compare_three.py
EDMDC_DATA_FILE=artifacts/smoke/runs_mixed_n12.pkl \\
EDMDC_MODEL_FILE=artifacts/smoke/edmdc_model_yaw_wrench.pkl \\
MPLBACKEND=Agg python compare_mpc.py --test-indices 2,5,8 --steps 100
EDMDC_DATA_FILE=artifacts/smoke/runs_mixed_n12.pkl \\
EDMDC_MODEL_FILE=artifacts/smoke/edmdc_model_yaw_wrench.pkl \\
MPLBACKEND=Agg python Intercept_comparison.py --cases straight --tmax 1
```

The shortened smoke trajectory is a software integration check only: reducing
the duration while retaining the full-scale path makes that reference much more
aggressive than the intended experiment. Do not use its model metrics in the
paper or to tune the controller.

Install dependencies with `python -m pip install -r requirements.txt` from the
repository root. The training script uses separate deterministic validation and
test runs: lambda selection uses validation only, balances position, yaw, and
yaw-rate error, and the final diagnostics use the untouched test split.

`compare_mpc.py` applies the full optimized correction for both EDMDc-MPC and
linear MPC. This keeps the comparison attributable to their prediction models,
rather than controller-specific post-processing.

The MPC also uses the same wrench-to-motor allocation matrix as the plant. Its
per-motor force inequalities prevent the optimizer from requesting a wrench
that would be altered by motor clipping.

The shared MPC module limits BLAS to one thread by default (`EDMDC_BLAS_THREADS=1`)
because its small, repeatedly assembled matrices otherwise create timing jitter.
Set the variable to `0` to leave the process's BLAS configuration unchanged.

Outputs:

```text
runs_traj1_n50.pkl ... runs_traj4_n50.pkl
runs_traj5_n30.pkl
runs_prbs_n70.pkl
runs_mixed_n300.pkl
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
