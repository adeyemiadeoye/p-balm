import numpy as np
np.NaN = np.nan
import jax
from functools import partial
import jax.numpy as jnp
import alpaqa as pa
from .inner_solvers.inner_solvers import phase_I_optim, get_solver_run
from .utils.utils import params_flatten, update_penalties, GradEvalCounter
import time

reg_classes = [pa.functions.L1Norm, pa.Box, pa.functions.NuclearNorm]

class Solution:
    """
    Holds the solution process and data for PBALM.
    """
    def __init__(self, problem, x0, inner_solve_runner, lbda0, mu0, rho0, nu0, gamma0, x_shapes=None, x_cumsizes=None, use_proximal=False, beta=0.5, alpha=2.0, delta=1.0, xi1=1.0, xi2=1.0, tol=1e-6, fp_tol=None, max_iter=1000, phase_I_tol=1e-7, start_feas=True, inner_solver="PANOC", max_iter_inner=1000, pa_solver_opts=None, pa_direction=None, verbosity=1, max_runtime=24.0, phi_strategy="pow", feas_reset_interval=None, no_reset=False, adaptive_fp_tol=True):
        self.problem = problem
        self.x0 = x0
        self.x_shapes = x_shapes
        self.x_cumsizes = x_cumsizes
        self.is_not_flat_x = x_shapes is not None and x_cumsizes is not None
        self.lbda0 = lbda0
        self.mu0 = mu0
        self.rho0 = rho0
        self.nu0 = nu0
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.xi1 = xi1
        self.xi2 = xi2
        self.tol = tol
        self.fp_tol = fp_tol if fp_tol is not None else tol
        self.adaptive_fp_tol = adaptive_fp_tol
        self.max_iter_inner = max_iter_inner
        self.max_iter = max_iter
        self.verbosity = verbosity
        self.x = x0
        self.start_feas = start_feas
        self.inner_solver = inner_solver
        self.phase_I_tol = phase_I_tol
        self.fp_res = []
        self.kkt_res = []
        self.total_infeas = []
        self.f_hist = []
        self.rho_hist = []
        self.nu_hist = []
        self.gamma_hist = []
        self.prox_hist = []
        self.solve_status = None
        self.pa_direction = pa_direction
        self.pa_solver_opts = pa_solver_opts
        self.use_proximal = use_proximal
        self.gamma0 = gamma0
        self.gamma_k = gamma0 if use_proximal else jnp.nan
        self.phi_strategy = phi_strategy
        self.max_runtime = max_runtime * 3600 if max_runtime is not None else 24.0 * 3600
        self.total_runtime = None
        self.solve_runtime = None
        self.feas_reset_interval = feas_reset_interval
        self.reset_x0 = x0
        self.no_reset = no_reset
        self.alm_grad_fn = None
        self.inner_solve_runner = inner_solve_runner
        if self.inner_solve_runner is None:
            if self.problem.h is None and self.problem.g is None:
                print_interval = 5*self.verbosity
            else:
                print_interval = 0
            self.inner_solve_runner = get_solver_run(f2=self.problem.f2, l1_lbda=self.problem.f2_lbda, solver_opts=pa_solver_opts, direction=pa_direction, print_interval=print_interval, jittable=self.problem.jittable)
        else:
            self.inner_solve_runner = inner_solve_runner
        if self.problem.h is not None:
            h0 = self.problem.h(self.x0)
            self.rho_vec = jnp.ones_like(h0) * self.rho0
        else:
            self.rho_vec = None
        if self.problem.g is not None:
            g0 = self.problem.g(self.x0)
            self.nu_vec = jnp.ones_like(g0) * self.nu0
        else:
            self.nu_vec = None

        self.cons_present = 0
        if self.problem.h is not None:
            self.cons_present += 1
        if self.problem.g is not None:
            self.cons_present += 1
        self.num_eps = self.tol*self.cons_present


    def _get_phi_i(self, i):
        if self.phi_strategy == "log":
            return i*jnp.log(i)
        elif self.phi_strategy == "pow":
            return i**self.alpha
        elif self.phi_strategy == "linear":
            return 0
        else:
            raise ValueError(f"Unknown phi strategy: {self.phi_strategy}")

    def _is_feasible(self, x):
        h_x = self.problem.h(x) if self.problem.h else jnp.array([0])
        g_x = self.problem.g(x) if self.problem.g else jnp.array([0])
        h_feas = (self.problem.h is None or jnp.all(jnp.isclose(h_x, 0, atol=self.tol, rtol=self.tol)))
        g_feas = (self.problem.g is None or jnp.all(g_x <= 1e-16))
        return (h_feas and g_feas)
    
    def _eval_prox_term(self, x, x_prev, gamma_k):
        if self.use_proximal:
            return jnp.sum((1/(2*gamma_k))*(x - x_prev)**2)
        else:
            return 0.0

    def pbalm(self):
        total_start_time = time.time()
        start_time = time.time()
        warmup_end_time = None
        self.grad_evals = []
        last_grad_eval = 0
        if self.problem.h is None and self.problem.g is None:
            if self.verbosity > 0:
                print("Solving problem without constraints")
            self.use_proximal = False
            @jax.jit
            def obj_fun(x):
                return self._get_palm_obj_fun(x, x, self.lbda0, self.mu0, self.rho0, self.nu0, self.gamma_k)
            x = self.x0.copy()
            x_new, state = self.inner_solve_runner.train_fun(obj_fun, x, self.max_iter_inner, self.tol)
            if self.is_not_flat_x:
                x_new = params_flatten(x_new)

            self.x = x_new
            self.f_hist.append(self.problem.f1(self.x))
            self.total_runtime = time.time() - total_start_time
            self.solve_runtime = self.total_runtime
            if self.verbosity > 0:
                print(f"Unconstrained optimization completed after {self.solve_runtime:.6f} seconds.")
                print(f"{'Objective value:':<25} {self.f_hist[-1]:.6e}")
                self.solve_status = state['status']
                if self.solve_status.startswith("SolverStatus."): # match pbalm style
                    self.solve_status = self.solve_status[len("SolverStatus."):]
            return

        if self._is_feasible(self.x0):
            if self.verbosity > 0:
                print("Initial point is feasible.")
            x = self.x0.copy()
            self.start_feas = False
            warmup_end_time = time.time()
        else:
            if self.start_feas:
                if self.verbosity > 0:
                    print("Initial point is not feasible. Finding a feasible point...")
                self.x0 = phase_I_optim(self.x0, self.problem.h, self.problem.g, self.problem.f2, self.lbda0, self.mu0, tol=self.phase_I_tol, inner_solver=self.inner_solver)
                self.reset_x0 = self.x0.copy()
                warmup_end_time = time.time()
            else:
                if self.verbosity > 0:
                    print("Initial point is not feasible. Starting from infeasible point.")
                warmup_end_time = time.time()
            x = self.x0.copy()
        x_prev = x.copy()
        lbda = self.lbda0
        mu = self.mu0
        rho_vec = self.rho_vec.copy() if self.rho_vec is not None else None
        nu_vec = self.nu_vec.copy() if self.nu_vec is not None else None

        if self.verbosity > 0:
            print(f"{'iter':<5} | {'f':<10} | {'p. term':<10} | {'total infeas':<10} | {'rho':<10} | {'nu':<10} | {'gamma':<10}")
            print("-" * 90)

        prox_term_i = jnp.nan
        solve_start_time = warmup_end_time if warmup_end_time is not None else total_start_time
        
        grad_evals = 0
        if self.use_proximal:
            @jax.jit
            def L_aug(x):
                return self._get_prox_L_aug(x, x_prev, lbda, mu, rho_vec, nu_vec, self.gamma_k)
        else:
            @jax.jit
            def L_aug(x):
                return self._get_prox_L_aug(x, None, lbda, mu, rho_vec, nu_vec, self.gamma_k)
            
        self.alm_grad_fn = GradEvalCounter(jax.grad(L_aug)) if not self.problem.jittable else GradEvalCounter(jax.jit(jax.grad(L_aug)))

        ####### main outer ALM loop #######
        for i in range(self.max_iter):
            fp_tol = self.fp_tol
            phi_i = self._get_phi_i(i+1)
            if self.max_runtime is not None and (time.time() - start_time) > self.max_runtime:
                if self.verbosity > 0:
                    print("-" * 80)
                    print(f"Maximum runtime of {self.max_runtime:.2e} seconds reached. Exiting loop.")
                self.solve_status = "MaxRuntimeExceeded"
                break
            if self.problem.callback is not None:
                self.problem.callback(
                    iter=i,
                    x=x,
                    x_prev=x_prev,
                    lbda=lbda,
                    mu=mu,
                    rho=rho_vec,
                    nu=nu_vec,
                    gamma_k=self.gamma_k,
                    x0=self.x0
                )
            if i == 0:
                self.grad_evals.append(0)
                h_x = self.problem.h(x) if self.problem.h else jnp.array([0])
                g_x = self.problem.g(x) if self.problem.g else jnp.array([0])
                nrm_h = jnp.linalg.norm(h_x, jnp.inf) if self.problem.h else 0
                nrm_E = 0
                if self.problem.g:
                    E_x = jnp.minimum(-g_x, 1/nu_vec*mu)
                    nrm_E = jnp.linalg.norm(E_x, jnp.inf)
                L0_grad = self.alm_grad_fn(x)
                if self.grad_evals is not None:
                    grad_evals += self.alm_grad_fn.count
                if type(self.problem.f2) in reg_classes:
                    p_z = pa.prox(self.problem.f2, x - L0_grad)[1]
                    if self.problem.f2_lbda is not None and type(self.problem.f2) != pa.functions.L1Norm:
                        p_z = pa.prox(pa.functions.L1Norm(self.problem.f2_lbda), p_z - L0_grad)[1]
                    fp_res_i = jnp.linalg.norm(x - p_z, jnp.inf)
                elif type(self.problem.f2) == type(None): # this won't be the case though (case handled above)
                    fp_res_i = jnp.linalg.norm(L0_grad, jnp.inf)
                stopping_terms = jnp.array([fp_res_i, nrm_h, nrm_E])
                eps_kkt_res = jnp.max(stopping_terms[1:])
                f_x = self.problem.f1(x)
                self.kkt_res.append(eps_kkt_res)
                self.total_infeas.append(nrm_h + jnp.linalg.norm(jnp.maximum(g_x, 0), jnp.inf) if self.problem.g else nrm_h)
                self.fp_res.append(stopping_terms[0])
                self.f_hist.append(f_x)
                self.rho_hist.append(jnp.max(rho_vec) if rho_vec is not None else None)
                self.nu_hist.append(jnp.max(nu_vec) if nu_vec is not None else None)
                self.gamma_hist.append(self.gamma_k)
                self.prox_hist.append(prox_term_i)
                if self.start_feas:
                    num_eps = self.total_infeas[-1]
                else:
                    num_eps = self.num_eps
                if self.verbosity > 0:
                    print(
                        f"{i:<5} | {f_x:<10.4e} | {prox_term_i:<10.4e} | {abs(max(self.total_infeas[-1]-num_eps, self.total_infeas[-1])):<10.4e} | {jnp.max(rho_vec) if rho_vec is not None else 0:<10.4e} | {jnp.max(nu_vec) if nu_vec is not None else 0:<10.4e} | {self.gamma_k:<10.4e}")

                # Step 1: to reset xhat?
                L_aug_val = L_aug(x)
                if self.no_reset:
                    x_hat = x
                else:
                    f1_reset_x0 = self.problem.f1(self.reset_x0)
                    if self.problem.f2:
                        pz_val = pa.prox(self.problem.f2, x)[0]
                        L_aug_val += pz_val
                        if self.problem.f2_lbda is not None and type(self.problem.f2) != pa.functions.L1Norm:
                            pz_val = pa.prox(pa.functions.L1Norm(self.problem.f2_lbda), x)[0]
                            L_aug_val += pz_val
                        pz_reset_val = pa.prox(self.problem.f2, self.reset_x0)[0]
                        f1_reset_x0 += pz_reset_val
                        if self.problem.f2_lbda is not None and type(self.problem.f2) != pa.functions.L1Norm:
                            pz_reset_val = pa.prox(pa.functions.L1Norm(self.problem.f2_lbda), self.reset_x0)[0]
                            f1_reset_x0 += pz_reset_val
                    
                    prox_term = self._eval_prox_term(self.reset_x0, x, self.gamma_k)
                    L_func_val = L_aug_val + prox_term
                    if L_func_val > f1_reset_x0 + prox_term:
                        x_hat = self.reset_x0
                    else:
                        x_hat = x

            if self.adaptive_fp_tol:
                if callable(self.fp_tol):
                    fp_tol = self.fp_tol(i)
                else:
                    if self.start_feas:
                        # self.fp_tol = 0.1/(i+1)**(1.1)
                        fp_tol = 0.1/(i+1)**2
                    else:
                        fp_tol = 0.1/(i+1)**5

            def palm_obj_fun(x):
                return self._get_palm_obj_fun(x, x_prev, lbda, mu, rho_vec, nu_vec, self.gamma_k)
            
            x_new, state = self.inner_solve_runner.train_fun(palm_obj_fun, x_hat, self.max_iter_inner, fp_tol)
            if self.is_not_flat_x:
                x_new = params_flatten(x_new)

            isnanfpres = False if self.problem.f2 is None else (jnp.isnan(fp_res_i) or jnp.isinf(fp_res_i))
            isnannu = False if self.problem.g is None else (jnp.isnan(nu_vec).any() or jnp.isinf(nu_vec).any())
            isnanrho = False if self.problem.h is None else (jnp.isnan(rho_vec).any() or jnp.isinf(rho_vec).any())
            isnangamma = False if self.use_proximal is False else (jnp.isnan(self.gamma_k) or jnp.isinf(self.gamma_k))
            if jnp.isnan(self.problem.f1(x_new)) or jnp.isinf(self.problem.f1(x_new)) or isnanfpres or isnannu or isnanrho or isnangamma:
                if self.verbosity > 0:
                    print(
                    f"{i + 1:<5} | {f_x:<10.4e} | {prox_term_i:<10.4e} | {abs(max(self.total_infeas[-1]-self.num_eps, self.total_infeas[-1])):<10.4e} | {jnp.max(rho_vec) if rho_vec is not None else 0:<10.4e} | {jnp.max(nu_vec) if nu_vec is not None else 0:<10.4e} | {self.gamma_k:<10.4e}")
                    print("-" * 90)
                    print("One or more functions returned NaN or Inf. Stopping optimization.")
                    print(f"{'Objective value:':<25} {f_x:.6e}")
                    print(f"{'prox_term_i:':<25} {prox_term_i:.6e}")
                    print(f"{'eps-KKT residual:':<25} {eps_kkt_res:.6e}")
                    print(f"{'total infeas:':<25} {abs(max(self.total_infeas[-1]-self.num_eps, self.total_infeas[-1])):.6e}")
                    print(f"{'rho:':<25} {jnp.max(rho_vec) if rho_vec is not None else 0:.6e}")
                    print(f"{'nu:':<25} {jnp.max(nu_vec) if nu_vec is not None else 0:.6e}")
                    print(f"{'gamma:':<25} {self.gamma_k:.6e}")
                self.solve_status = "NaNOrInf"
                if self.problem.callback is not None:
                    self.problem.callback(
                        iter=i+1,
                        x=x,
                        x_prev=x_prev,
                        lbda=lbda,
                        mu=mu,
                        rho=rho_vec,
                        nu=nu_vec,
                        gamma_k=self.gamma_k,
                        x0=self.x0
                    )
                break

            h_x = self.problem.h(x_new) if self.problem.h else jnp.array([0])
            g_x = self.problem.g(x_new) if self.problem.g else jnp.array([0])

            if self.use_proximal:
                x_prev = x.copy()

            lbda = lbda + jnp.multiply(rho_vec, h_x) if self.problem.h else lbda
            mu_new = jnp.maximum(0, mu + jnp.multiply(nu_vec, g_x)) if self.problem.g else mu

            nrm_h = jnp.linalg.norm(h_x, jnp.inf) if self.problem.h else 0
            prev_h = self.problem.h(x) if self.problem.h else jnp.array([0])

            E_x = jnp.minimum(-g_x, jnp.divide(1, nu_vec)*mu) if self.problem.g else jnp.array([0])
            prev_g = self.problem.g(x) if self.problem.g else jnp.array([0])
            prev_E = jnp.minimum(-prev_g, jnp.divide(1, nu_vec) * mu) if self.problem.g else jnp.array([0])

            nrm_E = jnp.linalg.norm(E_x, jnp.inf)

            rho_vec, nu_vec = update_penalties(self.problem.lbda_sizes, self.problem.mu_sizes, rho_vec, nu_vec, self.rho0, self.nu0, E_x, prev_E, h_x, prev_h, self.beta, self.xi1, self.xi2, phi_i)

            x = x_new
            mu = mu_new

            if self.use_proximal:
                prox_term_i = jnp.sum(1/(2*self.gamma_k)*(x - x_prev)**2)
            else:
                prox_term_i = 0.0

            fp_res_i = state["fp_res"]
            stopping_terms = jnp.array([fp_res_i, nrm_h, nrm_E])
            if self.use_proximal:
                stopping_terms = jnp.concatenate([stopping_terms, jnp.array([prox_term_i])])

            eps_kkt_res = jnp.max(stopping_terms[1:])
            f_x = self.problem.f1(x)
            self.kkt_res.append(eps_kkt_res)
            self.total_infeas.append(nrm_h + jnp.linalg.norm(jnp.maximum(g_x, 0), jnp.inf) if self.problem.g else nrm_h)
            self.f_hist.append(f_x)
            self.fp_res.append(stopping_terms[0])
            self.rho_hist.append(jnp.max(rho_vec) if rho_vec is not None else None)
            self.nu_hist.append(jnp.max(nu_vec) if nu_vec is not None else None)
            self.gamma_hist.append(self.gamma_k)
            self.prox_hist.append(prox_term_i)

            if self.use_proximal:
                self.gamma_k = jnp.maximum(jnp.sum(self.delta*(self.x0 - x)**2), self.gamma0*phi_i)

            if self.verbosity > 0 and ((i + 1) % int(20/self.verbosity) == 0 or (i + 1) == self.max_iter):
                print(
                    f"{i + 1:<5} | {f_x:<10.4e} | {prox_term_i:<10.4e} | {abs(max(self.total_infeas[-1]-self.num_eps, self.total_infeas[-1])):<10.4e} | {jnp.max(rho_vec) if rho_vec is not None else 0:<10.4e} | {jnp.max(nu_vec) if nu_vec is not None else 0:<10.4e} | {self.gamma_k:<10.4e}")

            if (eps_kkt_res <= self.tol):
                if self.verbosity > 0:
                    print(
                        f"{i + 1:<5} | {f_x:<10.4e} | {prox_term_i:<10.4e} | {abs(max(self.total_infeas[-1]-self.num_eps, self.total_infeas[-1])):<10.4e} | {jnp.max(rho_vec) if rho_vec is not None else 0:<10.4e} | {jnp.max(nu_vec) if nu_vec is not None else 0:<10.4e} | {self.gamma_k:<10.4e}")
                    print("-" * 90)
                    print(f"Convergence achieved after {i + 1} iterations and {(time.time() - solve_start_time):.2f} seconds.")
                    print(f"{'Optimal f value found:':<25} {f_x:.6e}")
                    print(f"{'eps-KKT residual:':<25} {eps_kkt_res:.6e}")
                    print(f"{'total infeas:':<25} {abs(max(self.total_infeas[-1]-self.num_eps, self.total_infeas[-1])):.6e}")
                    print(f"{'rho:':<25} {jnp.max(rho_vec) if rho_vec is not None else 0:.6e}")
                    print(f"{'nu:':<25} {jnp.max(nu_vec) if nu_vec is not None else 0:.6e}")
                    print(f"{'gamma:':<25} {self.gamma_k:.6e}")
                    print(f"{'prox_term_i:':<25} {prox_term_i:.6e}")
                    self.solve_status = "Converged"
                if self.problem.callback is not None:
                    self.problem.callback(
                        iter=i+1,
                        x=x,
                        x_prev=x_prev,
                        lbda=lbda,
                        mu=mu,
                        rho=rho_vec,
                        nu=nu_vec,
                        gamma_k=self.gamma_k,
                        x0=self.x0
                    )
                break
            elif (i + 1) == self.max_iter and self.verbosity > 0:
                if self.verbosity > 0:
                    print("-" * 90)
                    print("Maximum iterations reached without convergence.")
                    print(f"{'Objective value:':<25} {f_x:.6e}")
                    print(f"{'prox_term_i:':<25} {prox_term_i:.6e}")
                    print(f"{'eps-KKT residual:':<25} {eps_kkt_res:.6e}")
                    print(f"{'total infeas:':<25} {abs(max(self.total_infeas[-1]-self.num_eps, self.total_infeas[-1])):.6e}")
                    print(f"{'rho:':<25} {jnp.max(rho_vec) if rho_vec is not None else 0:.6e}")
                    print(f"{'nu:':<25} {jnp.max(nu_vec) if nu_vec is not None else 0:.6e}")
                    print(f"{'gamma:':<25} {self.gamma_k:.6e}")
                self.solve_status = "Stopped"
                if self.problem.callback is not None:
                    self.problem.callback(
                        iter=i+1,
                        x=x,
                        x_prev=x_prev,
                        lbda=lbda,
                        mu=mu,
                        rho=rho_vec,
                        nu=nu_vec,
                        gamma_k=self.gamma_k,
                        x0=self.x0
                    )

            if self.feas_reset_interval is not None and self.feas_reset_interval > 0 and (i + 1) % self.feas_reset_interval == 0:
                if self._is_feasible(x):
                    self.reset_x0 = x.copy() if hasattr(x, 'copy') else jnp.array(x)

            grad_evals += state["obj_grad_evals"]
            self.grad_evals.append(grad_evals)
            last_grad_eval = grad_evals

            # to reset xhat?
            if self.no_reset:
                x_hat = x
            else:
                f1_reset_x0 = self.problem.f1(self.reset_x0)
                L_func_val = state["obj_val"] + state["reg_val"]
                if self.problem.f2:
                    pz_reset_val = pa.prox(self.problem.f2, self.reset_x0)[0]
                    f1_reset_x0 += pz_reset_val
                    if self.problem.f2_lbda is not None and type(self.problem.f2) != pa.functions.L1Norm:
                        pz_reset_val = pa.prox(pa.functions.L1Norm(self.problem.f2_lbda), self.reset_x0)[0]
                        f1_reset_x0 += pz_reset_val

                prox_term = self._eval_prox_term(self.reset_x0, x, self.gamma_k)
                if L_func_val > f1_reset_x0 + prox_term:
                    x_hat = self.reset_x0
                else:
                    x_hat = x

        if self.grad_evals is not None:
            target_len = len(self.f_hist)
            while len(self.grad_evals) < target_len:
                self.grad_evals.append(last_grad_eval)

        self.x = x
        self.total_runtime = time.time() - total_start_time
        self.solve_runtime = time.time() - solve_start_time
        return

    # @partial(jax.jit, static_argnums=0)
    def _get_prox_L_aug(self, x, x_prev, lbda, mu, rho, nu, gamma_k):
        h_x = self.problem.h(x) if self.problem.h else jnp.array([0])
        g_x = self.problem.g(x) if self.problem.g else jnp.array([0])

        L_aug_val = self.problem.f1(x)

        if self.problem.h:
            eq_term = jnp.sum(rho*0.5*h_x**2) + jnp.dot(lbda,h_x)
            L_aug_val += eq_term

        if self.problem.g:
            nu_g_x_mu = nu*g_x + mu
            ineq_term = jnp.sum((1/(2*nu)) * jnp.where(nu_g_x_mu > 0, nu_g_x_mu**2, 0)) - jnp.sum((1/(2*nu))*mu**2)
            L_aug_val += ineq_term

        prox_term = self._eval_prox_term(x, x_prev, gamma_k)
        L_aug_val += prox_term

        return L_aug_val
    
    @partial(jax.jit, static_argnums=0)
    def _get_palm_obj_fun(self, x_curr, x_prev, lbda, mu, rho, nu, gamma_k):
        if self.is_not_flat_x:
            x = params_flatten(x_curr)
        else:
            x = x_curr.copy()
        h_x = self.problem.h(x) if self.problem.h else jnp.array([0])
        g_x = self.problem.g(x) if self.problem.g else jnp.array([0])

        L_aug_val = self.problem.f1(x)

        if self.problem.h:
            eq_term = jnp.sum(rho*0.5*h_x**2) + jnp.dot(lbda,h_x)
            L_aug_val += eq_term

        if self.problem.g:
            nu_g_x_mu = nu*g_x + mu
            ineq_term = jnp.sum((1/(2*nu)) * jnp.where(nu_g_x_mu > 0, nu_g_x_mu**2, 0)) - jnp.sum((1/(2*nu))*mu**2)
            L_aug_val += ineq_term

        prox_term = self._eval_prox_term(x, x_prev, gamma_k)
        L_aug_val += prox_term

        return L_aug_val




