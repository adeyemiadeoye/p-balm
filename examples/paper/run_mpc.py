from bop_system import BOPSystem
from MPCBase import solve_mpc

system = BOPSystem()
xi = 10
alpha = 12
# for mode in ['pbalm']:
for mode in ['pbalm-star', 'pbalm', 'balm', 'alm']:
    print(f"Running mode: {mode}")
    X, U, YS = solve_mpc(system, mode=mode, xi=xi, alpha=alpha, sim_steps=48, verbose=True, save_gevals=True)
    if hasattr(system, 'plot_results') and callable(system.plot_results):
        system.plot_results(X, U, YS, mode, xi=xi, alpha=alpha)