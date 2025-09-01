from jaxopt import projection
from jaxopt import prox
from jaxopt import ProximalGradient, ScipyMinimize, ProjectedGradient

import proxop
import numpy as np
import jax.numpy as jnp
import pbalm

np.NaN = np.nan
import alpaqa as pa
import jax


def jaxopt_minimize(f, x0, reg=None, lbda=None, tol=1e-9, max_iter=2000, grad_fx=None, jittable=False, Lip_grad_est=None):
        low = reg.low if type(reg) == proxop.indicator.BoxConstraint else -jnp.inf
        high = reg.high if type(reg) == proxop.indicator.BoxConstraint else jnp.inf
        l1_reg = lbda if lbda is not None else None

        stepsize = -1 if Lip_grad_est is None else 0.95/Lip_grad_est


        def fun(x):
            return f(x) if grad_fx is None else (f(x), grad_fx(x))

        if type(reg) == proxop.indicator.BoxConstraint:
            hyperparams_proj = (low, high)

            proj_op = projection.projection_box
            solver = ProjectedGradient(fun=fun, projection=proj_op, tol=tol, stepsize=stepsize, maxiter=max_iter, acceleration=True, jit=jittable, value_and_grad=True if grad_fx is not None else False)
            sol = solver.run(x0, hyperparams_proj=hyperparams_proj)

        elif type(reg) == proxop.multi.L1Norm:
                prox_op = prox.prox_lasso
                hyperparams_prox = l1_reg
                solver = ProximalGradient(fun=fun, prox=prox_op, tol=tol, stepsize=stepsize, maxiter=max_iter, acceleration=True, jit=jittable, value_and_grad=True if grad_fx is not None else False)
                sol = solver.run(x0, hyperparams_prox=hyperparams_prox)
        elif reg == None:
            solver = ScipyMinimize(fun=fun, method="L-BFGS-B", tol=tol, maxiter=max_iter, jit=jittable, value_and_grad=True if grad_fx is not None else False, options={'maxcor': 20})
            sol = solver.run(x0)
        else:
            raise ValueError(f"Unsupported regularizer type: {type(reg)}. Supported types are BoxConstraint and L1Norm.")
        
        return sol

class PaProblem(pa.BoxConstrProblem):
    def __init__(self, f, x0, reg=None, lbda=None, solver_opts=None, tol=1e-9, max_iter=2000, direction=None, grad_fx=None, jittable=True, Lip_grad_est=None):
        super().__init__(x0.shape[0], 0)
        self.obj_f = jax.jit(f) if jittable else f
        self.init_x = x0
        if type(reg) == proxop.indicator.BoxConstraint:
            self.C.lowerbound[:] = reg.low
            self.C.upperbound[:] = reg.high
        if lbda is not None:
            if type(lbda) in [float, int]:
                self.l1_reg = [lbda]
            elif type(lbda) != list:
                raise ValueError(f"Invalid L1 regularization type: {type(lbda)}; expected float, int, or list.")
            elif type(lbda) == list and len(lbda) != x0.shape[0]:
                raise ValueError(f"Invalid L1 regularization list length: {len(lbda)}; expected {x0.shape[0]}.")
            else:
                self.l1_reg = lbda
        else:
            self.l1_reg = None
        self.pa_solver_opts = solver_opts
        self.pa_direction = direction
        self.pa_tol = tol
        self.pa_max_iter = max_iter
        self.L_0 = Lip_grad_est if Lip_grad_est is not None else -1
        self.jit_grad_f = jax.jit(jax.grad(self.obj_f)) if (grad_fx is None and jittable) else (grad_fx if grad_fx is not None else jax.grad(self.obj_f))

    def eval_f(self, x):
        return self.obj_f(x)

    def eval_grad_f(self, x, grad_f):
        grad_f[:] = self.jit_grad_f(x)

    def _run_pa_procedure(self):
        if self.pa_direction is None:
            self.pa_direction = pa.StructuredLBFGSDirection({"memory": 20})
        if self.pa_solver_opts is None:
            self.pa_solver_opts = {
                    # "print_interval": 4,
                    "max_iter": self.pa_max_iter,
                    "stop_crit": pa.ProjGradUnitNorm,
                    # "quadratic_upperbound_tolerance_factor": 1e-12,
                }
            if self.L_0 > 0:
                self.pa_solver_opts["L_max"] = self.L_0
                self.pa_solver_opts["Lipschitz"] = {"L_0": self.L_0}
        self.pa_solver = pa.PANOCSolver(self.pa_solver_opts, self.pa_direction)
        cnt = pa.problem_with_counters(self)
        sol, stats = self.pa_solver(cnt.problem, {"tolerance": self.pa_tol}, self.init_x)
        # print(self.eval_f(sol))
        return sol, stats, cnt.evaluations.grad_f

def phase_I_optim(x0, h, g, reg, lbda0, mu0, tol=1e-7, max_iter=500, inner_solver="PANOC"):
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

    reg_0 = None

    feas_prob = pbalm.Problem(
                    f=feas_f,
                    g=feas_g,
                    h=None,
                    reg=reg_0,
                    jittable=True
                )
    if g:
        z0 = jnp.concatenate([x0, jnp.array([0.0])])
    else:
        z0 = x0.copy()
    feas_res = pbalm.solve(feas_prob, z0, lbda0=lbda0, mu0=mu0, use_proximal=True, tol=tol, max_iter=max_iter, start_feas=False, inner_solver=inner_solver, verbosity=0, max_runtime=0.8333)
    if h is not None and g is None:
        total_infeas = jnp.sum(h(feas_res.x[:x_dim])**2)
    else:
        total_infeas = feas_res.total_infeas[-1]
        if h is not None:
            total_infeas += jnp.sum(h(feas_res.x[:x_dim])**2)
    if total_infeas <= tol:
        print("Phase I optimization successful.")
    else:
        raise RuntimeError("Phase I optimization failed.")
    return feas_res.x[:x_dim]