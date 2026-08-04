"""Run the current yaw-wrench closed-loop MPC comparison.

The old tuner optimized an attitude-commanded EDMDc MPC path. The maintained
MPC path now uses Darren's current yaw-wrench simulator and is implemented in
compare_mpc.py.
"""

from compare_mpc import main


if __name__ == "__main__":
    main()
