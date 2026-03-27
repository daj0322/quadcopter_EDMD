from Simulation import quad_sim

quad = quad_sim()
quad.fct_save_simulation_runs(traj=1, n=200, filename="runs_traj1_n200.pkl")

quad.fct_save_simulation_runs(traj=2, n=100, filename="runs_traj2_n100.pkl")

quad.fct_save_simulation_runs(traj=1, n=100, filename="runs_traj1_n100.pkl")