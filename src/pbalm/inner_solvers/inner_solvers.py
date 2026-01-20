import numpy as np
import jax.numpy as jnp
import pbalm

np.NaN = np.nan
import alpaqa as pa
import jax


class PALMInnerTrainer:
    """
    Class to run PALM inner solver with Adam and L-BFGS optimization.
    """
    def __init__(self, train_fun):
        """
        Initialize the PALMInnerTrainer.

        Parameters
        ----------
        train_fun : callable
            Function to run the solver.
            train_fun(palm_obj_fun, x, lbfgs_iters, lbfgs_tol)
                where palm_obj_fun is a function that takes in x and returns the inner objective value.
        """
        self.train_fun = train_fun

class PaProblem(pa.BoxConstrProblem):
    """
    Internal problem class for PANOC solver.
    """
    def __init__(self, obj_fun, x0, f2=None, l1_lbda=None, solver_opts=None, tol=1e-9, max_iter=2000, direction=None, jittable=True):
        super().__init__(x0.shape[0], 0)
        self.jit_loss = jax.jit(obj_fun) if jittable else obj_fun
        self.init_x = x0
        if type(f2) == pbalm.Box:
            self.variable_bounds.lower = f2.lower
            self.variable_bounds.upper = f2.upper
        if l1_lbda is not None:
            if type(l1_lbda) in [float, int]:
                self.l1_reg = [l1_lbda]
            elif type(l1_lbda) != list:
                raise ValueError(f"Invalid L1 regularization type: {type(l1_lbda)}; expected float, int, or list.")
            elif type(l1_lbda) == list and len(l1_lbda) != x0.shape[0]:
                raise ValueError(f"Invalid L1 regularization list length: {len(l1_lbda)}; expected {x0.shape[0]}.")
            else:
                self.l1_reg = l1_lbda
        elif type(f2) == pbalm.L1Norm:
            self.l1_reg = [f2.λ]
        else:
            self.l1_reg = [0.0]
        self.pa_solver_opts = solver_opts
        self.pa_direction = direction
        self.pa_tol = tol
        self.pa_max_iter = max_iter
        if jittable:
            self.jit_grad_f = jax.jit(jax.grad(obj_fun))
        else:
            self.jit_grad_f = jax.grad(obj_fun)

    def eval_objective(self, x):
        return self.jit_loss(x)

    def eval_objective_gradient(self, x, grad_f):
        grad_f[:] = self.jit_grad_f(x)

def get_solver_run(f2=None, l1_lbda=None, solver_opts=None, direction=None, print_interval=0, jittable=True):
    """
    Internal function to get the solver run function for PALM inner trainer.
    """
    def solver_run(palm_obj_fun, x0, lbfgs_iters, lbfgs_tol):
        pa_prob = PaProblem(
            obj_fun=palm_obj_fun,
            x0=x0,
            f2=f2,
            l1_lbda=l1_lbda,
            solver_opts=solver_opts,
            tol=lbfgs_tol,
            max_iter=lbfgs_iters,
            direction=direction,
            jittable=jittable
        )
        if pa_prob.pa_direction is None:
            pa_prob.pa_direction = pa.LBFGSDirection({"memory": 20})
        if pa_prob.pa_solver_opts is None:
            pa_prob.pa_solver_opts = {
                    "print_interval": print_interval,
                    "max_iter": pa_prob.pa_max_iter,
                    "stop_crit": pa.ProjGradUnitNorm,
                    # "quadratic_upperbound_tolerance_factor": 1e-12,
                }
        pa_prob.pa_solver = pa.PANOCSolver(pa_prob.pa_solver_opts, pa_prob.pa_direction)
        cnt = pa.problem_with_counters(pa_prob)
        sol, stats = pa_prob.pa_solver(cnt.problem, {"tolerance": pa_prob.pa_tol}, pa_prob.init_x)
        return sol, {"obj_grad_evals": cnt.evaluations.objective_and_gradient,
                    "fp_res": stats['ε'], "obj_val": pa_prob.eval_objective(sol), "reg_val": stats['final_h'], "status": str(stats['status'])}
    
    return PALMInnerTrainer(solver_run)

def phase_I_optim(x0, h, g, f2, lbda0, mu0, alpha=20, gamma0=1e-8, tol=1e-7, max_iter=500, inner_solver="PANOC"):
    """
    Solves the Phase I feasibility problem to find an initial feasible point.
    """
    x_dim = x0.shape[0]
        
    if g is not None and h is None:
        feas_f = lambda z: jnp.sum(z[x_dim+1]**2)
        feas_g = lambda z: g(z[:x_dim]) - z[x_dim+1]
    elif g is not None and h is not None:
        feas_f = lambda z: 0.5*(jnp.sum(h(z[:x_dim])**2) + jnp.sum(z[x_dim+1]**2))
        feas_g = lambda z: g(z[:x_dim]) - z[x_dim+1]
    elif h is not None and g is None:
        feas_f = lambda z: 0.5*jnp.sum(h(z)**2)
        feas_g = None

    f2_0 = None

    feas_prob = pbalm.Problem(
                    f1=feas_f,
                    g=[feas_g] if feas_g is not None else None,
                    h=None,
                    f2=f2_0,
                    jittable=True
                )
    if g:
        z0 = jnp.concatenate([x0, jnp.array([0.0])])
    else:
        z0 = x0.copy()
    feas_res = pbalm.solve(feas_prob, z0, lbda0=lbda0, mu0=mu0, use_proximal=True, tol=tol, max_iter=max_iter,
                            max_iter_inner=10000, alpha=alpha, gamma0=gamma0,
                            start_feas=False, inner_solver=inner_solver, verbosity=0, max_runtime=0.8333)
    if h is not None and g is None:
        total_infeas = jnp.sum((h(feas_res.x[:x_dim]))**2)-tol
    else:
        total_infeas = feas_res.total_infeas[-1]
        if h is not None:
            total_infeas += jnp.sum((h(feas_res.x[:x_dim]))**2)-tol
    if total_infeas <= max(tol, 1e-5):
        print("Phase I optimization successful.")
    else:
        raise RuntimeError("Phase I optimization failed. Total infeasibility: {}".format(total_infeas))
    return feas_res.x[:x_dim]