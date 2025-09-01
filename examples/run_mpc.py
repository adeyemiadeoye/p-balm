from utils.bop_system import BOPSystem
from utils.MPCBase import solve_mpc

import random
import jax
import numpy as np

SEED = 1234
np.random.seed(SEED)
random.seed(SEED)
jax_key = jax.random.PRNGKey(SEED)

xi = 10
alpha = 12
init_state = "s_SW"
system = BOPSystem(init_state=init_state)
algo = "pbalm"
X, U, YS = solve_mpc(system, algo=algo, xi=xi, alpha=alpha, sim_steps=48, verbose=True)
if hasattr(system, 'plot_results') and callable(system.plot_results):
    system.plot_results(X, U, YS, algo, xi=xi, alpha=alpha)