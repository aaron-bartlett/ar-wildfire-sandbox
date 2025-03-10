from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation

config = Config("configs/operational_config.yml")
sim = FireSimulation(config)

sim.rendering = True
# Run a 1 hour simulation
sim.run("1h")

# Run the same simulation for 30 more minutes
sim.run("30m")

sim.run("2h")

# Now save a GIF and fire spread graph from the last 2 hours of simulation
sim.save_gif()
#sim.save_spread_graph()
