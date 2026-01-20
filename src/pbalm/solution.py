from .problem import GradEvalCounter
from .result import Result
from .pbalm import Solution
from jax import jacfwd
import jax
import jax.numpy as jnp
import numpy as np
import time

def solve(problem, x0, inner_solve_runner=None, lbda0=None, mu0=None, rho0=1e-3, nu0=1e-3, use_proximal=True,
            gamma0=1e-1, x_shapes=None, x_cumsizes=None, beta=0.5, alpha=10, delta=1.0, xi1=1.0, xi2=1.0, tol=1e-6, fp_tol=None, max_iter=1000, phase_I_tol=1e-7,
            start_feas=True, inner_solver=None, pa_direction=None, pa_solver_opts=None, verbosity=1, max_runtime=24.0,
            phi_strategy="pow", feas_reset_interval=None, no_reset=False, adaptive_fp_tol=False, max_iter_inner=1000):
    """
    Calls the PBALM solver on the given problem instance.

    Parameters:
        problem: An instance of the Problem class defining the optimization problem.
        x0: Initial guess for the decision variables.
        inner_solve_runner: (Optional) Class that holds the inner solver routine `train_fun`; train_fun(palm_obj_fun, x, max_iter_inner, fp_tol).
        lbda0: Initial Lagrange multipliers for equality constraints.
        mu0: Initial Lagrange multipliers for inequality constraints.
        rho0: Initial penalty parameters for equality constraints.
        nu0: Initial penalty parameters for inequality constraints.
        use_proximal: Boolean indicating whether to use proximal ALM or standard ALM.
        gamma0: Initial smoothing parameter for the proximal term.
        x_shapes: (Optional) Shapes of decision variable blocks (for structured variables).
        x_cumsizes: (Optional) Cumulative sizes of decision variable blocks (for structured variables).
        beta: Penalty update parameter (0 < beta < 1).
        alpha: Penalty update parameter (alpha > 1).
        delta: Proximal parameter update factor (delta > 0).
        xi1: Penalty update scaling factor for equality constraints (xi1 >= 1, xi1=1 for the paper algorithm).
        xi2: Penalty update scaling factor for inequality constraints (xi2 >= 1, xi2=1 for the paper algorithm).
        tol: Tolerance for convergence.
        fp_tol: Fixed-point tolerance for inner solver (if None, defaults to tol). Can be provided as a float or a function of iteration number k.
        max_iter: Maximum number of PBALM iterations.
        phase_I_tol: Tolerance for Phase I feasibility problem.
        start_feas: Boolean indicating whether to start with Phase I feasibility optimization.
        inner_solver: Name of the inner solver to use (only "PANOC" for now).
        pa_direction: (Optional) Direction method for PANOC inner solver. See PANOC documentation for details.
        pa_solver_opts: (Optional) Additional options for the PANOC inner solver.
        verbosity: Level of verbosity for logging (<=0: silent).
        max_runtime: Maximum runtime in hours.
        phi_strategy: Strategy for updating the minimum penalty parameter ("pow", "log", "linear").
        feas_reset_interval: Interval for resetting \hat{x} to a recent feasible point (step 1 in the algorithm).
        no_reset: Boolean indicating whether to disable resetting \hat{x}.
        adaptive_fp_tol: Boolean indicating whether to adaptively update the inner solver's fixed-point tolerance. If True, uses fp_tol if provided as a function (if not provided, uses fp_tol = a/(i+1)**b where a>0 and b>1).
        max_iter_inner: Maximum number of iterations for the inner solver.
    """
    
    lbda_sizes = []
    mu_sizes = []
    if problem.h is not None:
        lbda_sizes = [h(x0).flatten().shape[0] for h in problem.h]
        problem.h = get_constraint_function(problem.h, tol)
        problem.h = jax.jit(problem.h) if problem.jittable else problem.h
        if problem.jittable:
            problem.h_grad = GradEvalCounter(jax.jit(jacfwd(problem.h)))
        else:
            problem.h_grad = GradEvalCounter(jacfwd(problem.h))
        h0 = problem.h(x0)
        if lbda0 is None:
            rng = np.random.default_rng(1234)
            lbda0 = jnp.array(rng.standard_normal(h0.shape))
    if problem.g is not None:
        mu_sizes = [g(x0).flatten().shape[0] for g in problem.g]
        problem.g = get_constraint_function(problem.g, tol)
        problem.g = jax.jit(problem.g) if problem.jittable else problem.g
        if problem.jittable:
            problem.g_grad = GradEvalCounter(jax.jit(jacfwd(problem.g)))
        else:
            problem.g_grad = GradEvalCounter(jacfwd(problem.g))
        g0 = problem.g(x0)
        if mu0 is None:
            rng = np.random.default_rng(1234)
            mu0 = jnp.array(rng.standard_normal(g0.shape))
        mu0 = jnp.maximum(mu0, 0.0)
    problem.lbda_sizes = lbda_sizes
    problem.mu_sizes = mu_sizes
    solution = Solution(problem, x0, inner_solve_runner, lbda0, mu0, rho0, nu0, gamma0,
                        x_shapes, x_cumsizes, use_proximal=use_proximal, beta=beta, alpha=alpha, delta=delta,
                        xi1=xi1, xi2=xi2, tol=tol, fp_tol=fp_tol, max_iter=max_iter, phase_I_tol=phase_I_tol, start_feas=start_feas,
                        inner_solver=inner_solver, pa_direction=pa_direction, pa_solver_opts=pa_solver_opts, verbosity=verbosity,
                        max_runtime=max_runtime, phi_strategy=phi_strategy, feas_reset_interval=feas_reset_interval,
                        no_reset=no_reset, adaptive_fp_tol=adaptive_fp_tol, max_iter_inner=max_iter_inner)
    solution.pbalm()
    res = Result(solution.x, solution.fp_res, solution.kkt_res, solution.total_infeas, solution.f_hist, solution.rho_hist, solution.nu_hist, solution.gamma_hist, solution.prox_hist, solution.solve_status, solution.total_runtime, solution.solve_runtime, grad_evals=getattr(solution, 'grad_evals', None))
    
    return res

def get_constraint_function(constraints, tol):
    eps_tol = tol # accommodate numerical inexactness
    @jax.jit
    def con_fun(x):
        return jnp.concatenate([con(x).flatten()+eps_tol for con in constraints])
    return con_fun