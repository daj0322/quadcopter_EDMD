from Simulation import quad_sim

sim = quad_sim()
sim.fct_save_hover_excitation_runs(
    n=100,
    filename="runs_hover_excitation_n100.pkl"
)