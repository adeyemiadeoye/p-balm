import pyscsopt as scs
from pyscsopt.algorithms import ProxGradient
from pyscsopt.algorithms import ProxLQNSCORE
from pyscsopt.regularizers import PHuberSmootherL1L2
from pyscsopt.regularizers import LogExpSmootherIndBox
from pyscsopt.regularizers import PHuberSmootherGL

from jaxopt import projection
from jaxopt import prox
from jaxopt import ProximalGradient, ScipyMinimize, ProjectedGradient

from ..utils.prox_utils import SGLPenalty
import proxop
import numpy as np
import jax.numpy as jnp
import pbalm

np.NaN = np.nan
import alpaqa as pa
import jax

def pyscsopt_minimize(x0, f, reg, lbda=1e-2, mu=1.0, C_set=None, group_lasso_P=None, m=10, algo="ProxLQNSCORE", grad_fx=None, ss_type=2, max_iter=100, tol=1e-6, verbose=0, jittable=True):
    if reg:
        if type(reg) == proxop.multi.L1Norm:
            reg_name = "l1"
            lbda = lbda if lbda != 0.0 else 1e-2
            hmu = PHuberSmootherL1L2(mu)
        elif type(reg) == proxop.multi.L2Norm:
            reg_name = "l2"
            lbda = lbda if lbda != 0.0 else 1e-2
            hmu = PHuberSmootherL1L2(mu)
        elif type(reg) == proxop.indicator.BoxConstraint:
            reg_name = "indbox"
            lbda = 1.0
            C_set = (reg.low, reg.high)
            hmu = LogExpSmootherIndBox(C_set, mu)
        elif type(reg) == SGLPenalty:
            reg_name = "gl"
            lbda = [reg.gamma1, reg.gamma2]
            group_lasso_P = reg.P
            hmu = PHuberSmootherGL(mu, lbda, group_lasso_P)
    else:
        raise ValueError("Please specify an optimizer from scipy minimize, e.g., L-BFGS-B")

    problem = scs.Problem(x0, f, lbda, C_set=C_set, P=group_lasso_P, grad_fx=jax.jit(grad_fx) if (grad_fx is not None and jittable) else grad_fx)
    if algo == "ProxLQNSCORE":
        method = ProxLQNSCORE(use_prox=True, ss_type=ss_type, m=m)
    elif algo == "ProxGradient":
        method = ProxGradient(use_prox=True, ss_type=1)
    else:
        raise ValueError("The following solvers from pyscsopt can be used: ProxLQNSCORE, ProxGradient")

    sol = scs.iterate(method, problem, reg_name, hmu, verbose=verbose, max_epoch=max_iter, x_tol=tol, f_tol=tol)

    return sol


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
            # self.pa_direction = pa.AndersonDirection()
            # self.pa_direction = pa.LBFGSDirection({"memory": 10})
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
    feas_res = pbalm.solve(feas_prob, z0, lbda0=lbda0, mu0=mu0, use_proximal=True, tol=tol, max_iter=max_iter,
                           start_feas=False, inner_solver=inner_solver, verbosity=0, max_runtime=0.8333)
    if h is not None and g is None:
        total_infeas = jnp.sum(h(feas_res.x[:x_dim])**2)
    else:
        total_infeas = feas_res.total_infeas[-1]
        if h is not None:
            total_infeas += jnp.sum(h(feas_res.x[:x_dim])**2)
    if total_infeas <= tol:
        print("Phase I optimization successful.")
    elif total_infeas <= 5e-3:
        print("Phase I optimization successful with an acceptable feasibility.")
    else:
        print("Phase I optimization failed. Retrying with a different Phase I inner solver...")
        if inner_solver == "JAXOPT":
            feas_res = pbalm.solve(feas_prob, z0, lbda0=lbda0, mu0=mu0, use_proximal=True, tol=tol, max_iter=max_iter, start_feas=False, inner_solver="PANOC", verbosity=0, max_runtime=0.008333)
        elif inner_solver == "PANOC":
            feas_res = pbalm.solve(feas_prob, z0, lbda0=lbda0, mu0=mu0, use_proximal=True, tol=tol, max_iter=max_iter, start_feas=False, inner_solver="JAXOPT", verbosity=0, max_runtime=0.008333)
        if h is not None and g is None:
            total_infeas = jnp.sum(h(feas_res.x[:x_dim])**2)
        else:
            total_infeas = feas_res.total_infeas[-1]
            if h is not None:
                total_infeas += jnp.sum(h(feas_res.x[:x_dim])**2)
        if total_infeas <= tol:
            print("Phase I optimization successful.")
        elif total_infeas <= 5e-3:
            print("Phase I optimization successful with an acceptable feasibility.")
        else:
            print("Phase I optimization failed, but algorithm MAY still find a feasible solution.")
    return feas_res.x[:x_dim]