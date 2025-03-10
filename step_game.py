from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation

#config = Config("configs/operational_test_config.yml")
config = Config("configs/manual_config.yml")
#config = Config("configs/functional_config.yml")
sim = FireSimulation(config)

sim.rendering = True
# Run game minute by minute
while(input("Type 'exit' to finish simulation:\t") != "exit"):
    sim.run("1h")

# Now save a GIF and fire spread graph from the last 2 hours of simulation
sim.save_gif()
#sim.save_spread_graph()
# Saved to the location specified in the config: simulation.sf_home
