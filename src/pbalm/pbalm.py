import numpy as np
np.NaN = np.nan
import jax
from jax import grad, jacfwd
import jax.numpy as jnp
import proxop
from .inner_solvers.inner_solvers import PaProblem, phase_I_optim
import time

class GradEvalCounter:
    def __init__(self, fn):
        self.fn = fn
        self.count = 0
    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.fn(*args, **kwargs)
    def reset(self):
        self.count = 0

class Problem:
    def __init__(self, f, reg=None, reg_lbda=0.0, h=None, g=None, f_grad=None, h_grad=None, g_grad=None, jittable=False, callback=None):
        self.f = jax.jit(f) if jittable else f
        self.reg = reg
        self.reg_lbda = reg_lbda
        self.h = jax.jit(h) if (h and jittable) else h
        self.g = jax.jit(g) if (g and jittable) else g
        if jittable:
            self.f_grad = GradEvalCounter(jax.jit(f_grad) if f_grad else jax.jit(grad(self.f)))
            self.h_grad = GradEvalCounter(jax.jit(h_grad) if (h and h_grad) else (jax.jit(jacfwd(self.h)) if h else None)) if h else None
            self.g_grad = GradEvalCounter(jax.jit(g_grad) if (g and g_grad) else (jax.jit(jacfwd(self.g)) if g else None)) if g else None
        else:
            self.f_grad = GradEvalCounter(f_grad if f_grad else grad(self.f))
            self.h_grad = GradEvalCounter(h_grad if h and h_grad else (jacfwd(self.h) if h else None))
            self.g_grad = GradEvalCounter(g_grad if g and g_grad else (jacfwd(self.g) if g else None))
        self.jittable = jittable
        self.callback = callback
    def reset_counters(self):
        if hasattr(self, 'f_grad') and hasattr(self.f_grad, 'reset'):
            self.f_grad.reset()
        if hasattr(self, 'h_grad') and self.h_grad is not None and hasattr(self.h_grad, 'reset'):
            self.h_grad.reset()
        if hasattr(self, 'g_grad') and self.g_grad is not None and hasattr(self.g_grad, 'reset'):
            self.g_grad.reset()

class Solution:
    def __init__(self, problem, x0, lbda0, mu0, rho0, nu0, gamma0, use_proximal=True, beta=0.5, alpha=2.0, delta=1.0, xi1=1.0, xi2=1.0, tol=1e-6, fp_tol=None, max_iter=1000, phase_I_tol=1e-7, start_feas=True, inner_solver="PANOC", pa_solver_opts=None, pa_direction=None, verbosity=1, max_runtime=24.0, phi_strategy="pow", patience=None, feas_reset_interval=None, uniform_pen=False, no_reset=False, Lip_grad_est=None, use_autodiff_alm=True, adaptive_fp_tol=True):
        self.problem = problem
        self.x0 = x0
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
        self.max_iter = max_iter
        self.verbosity = verbosity
        self.x = x0
        self.start_feas = start_feas
        self.inner_solver = inner_solver
        self.phase_I_tol = phase_I_tol
        self.prox_grad_res = []
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
        self.patience = patience # THIS is a future feature
        self.feas_reset_interval = feas_reset_interval
        self.reset_x0 = x0
        self.no_reset = no_reset
        self.Lip_grad_est = Lip_grad_est
        self.uniform_pen = uniform_pen
        self.use_autodiff_alm = use_autodiff_alm
        self.alm_grad_fn = None
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
        h_x = self.problem.h(x) if self.problem.h else jnp.array([])
        g_x = self.problem.g(x) if self.problem.g else jnp.array([])
        h_feas = (self.problem.h is None or jnp.all(jnp.isclose(h_x, 0, atol=self.tol)))
        g_feas = (self.problem.g is None or jnp.all(g_x <= 0))
        return h_feas and g_feas

    def pbalm(self):
        total_start_time = time.time()
        start_time = time.time()
        warmup_end_time = None
        self.grad_evals = [] if hasattr(self.problem, 'jittable') and self.problem.jittable else None
        last_grad_eval = 0
        if self.problem.h is None and self.problem.g is None:
            if self.verbosity > 0:
                print("Solving problem without constraints")
            pa_prob = PaProblem(self.problem.f, self.x0, reg=self.problem.reg, lbda=self.problem.reg_lbda, solver_opts=self.pa_solver_opts, direction=self.pa_direction, tol=self.tol, grad_fx=self.problem.f_grad if not self.problem.jittable else jax.jit(self.problem.f_grad), jittable=self.problem.jittable, Lip_grad_est=self.Lip_grad_est)
            x_opt, _, cnt_grad_f = pa_prob._run_pa_procedure()
            self.x = x_opt
            self.total_runtime = time.time() - total_start_time
            self.solve_runtime = self.total_runtime
            return

        if self._is_feasible(self.x0):
            if self.verbosity > 0:
                print("Initial point is feasible.")
            x = self.x0.copy()
            warmup_end_time = time.time()
        else:
            if self.start_feas:
                if self.verbosity > 0:
                    print("Initial point is not feasible. Finding a feasible point...")
                self.x0 = phase_I_optim(self.x0, self.problem.h, self.problem.g, self.problem.reg, self.lbda0, self.mu0, tol=self.phase_I_tol, inner_solver=self.inner_solver)
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
        
        best_f = None
        best_kkt = None
        patience_counter = 0
        patience_x = None
        grad_evals = 0
        if self.use_proximal:
            def L_aug(x):
                return self._get_prox_L_aug(x, x_prev=x_prev, lbda=lbda, mu=mu, rho=rho_vec, nu=nu_vec, gamma_k=self.gamma_k)
        else:
            def L_aug(x):
                return self._get_prox_L_aug(x, lbda=lbda, mu=mu, rho=rho_vec, nu=nu_vec)

        ####### main outer ALM loop #######
        for i in range(self.max_iter):
            phi_i = self._get_phi_i(i+1)
            if self.max_runtime is not None and (time.time() - start_time) > self.max_runtime:
                if self.verbosity > 0:
                    print("-" * 80)
                    print(f"Maximum runtime of {self.max_runtime:.2e} seconds reached. Exiting loop.")
                self.solve_status = "MaxRuntimeExceeded"
                break
            self.alm_grad_fn = GradEvalCounter(jax.grad(L_aug)) if not self.problem.jittable else GradEvalCounter(jax.jit(jax.grad(L_aug)))
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
                h_x = self.problem.h(x) if self.problem.h else jnp.array([])
                g_x = self.problem.g(x) if self.problem.g else jnp.array([])
                nrm_h = jnp.linalg.norm(h_x, jnp.inf) if self.problem.h else 0
                nrm_E = 0
                if self.problem.g:
                    E_x = jnp.minimum(-g_x, 1/nu_vec*mu)
                    nrm_E = jnp.linalg.norm(E_x, jnp.inf)
                if self.use_autodiff_alm:
                    L0_grad = self.alm_grad_fn(x)
                    if self.grad_evals is not None:
                        grad_evals += self.alm_grad_fn.count
                else:
                    if self.use_proximal:
                        L0_grad = self._get_prox_L_aug_grad(x, x_prev=x_prev, lbda=lbda, mu=mu, rho=rho_vec, nu=nu_vec, gamma_k=self.gamma_k, aug=True)
                    else:
                        L0_grad = self._get_prox_L_aug_grad(x, lbda=lbda, mu=mu, rho=rho_vec, nu=nu_vec, gamma_k=self.gamma_k, aug=True)
                if type(self.problem.reg) in [proxop.multi.L1Norm, proxop.indicator.BoxConstraint]:
                    prox_grad_res_i = jnp.linalg.norm(x - self.problem.reg.prox(x - L0_grad), jnp.inf)
                elif type(self.problem.reg) == type(None): # this won't be the case though (case handled above)
                    prox_grad_res_i = jnp.linalg.norm(L0_grad, jnp.inf)
                stopping_terms = jnp.array([prox_grad_res_i, nrm_h, nrm_E])
                eps_kkt_res = jnp.max(stopping_terms[1:])
                f_x = self.problem.f(x)
                self.kkt_res.append(eps_kkt_res)
                self.total_infeas.append(nrm_h + jnp.linalg.norm(jnp.maximum(g_x, 0), jnp.inf) if self.problem.g else nrm_h)
                self.prox_grad_res.append(stopping_terms[0])
                self.f_hist.append(f_x)
                self.rho_hist.append(jnp.max(rho_vec) if rho_vec is not None else None)
                self.nu_hist.append(jnp.max(nu_vec) if nu_vec is not None else None)
                self.gamma_hist.append(self.gamma_k)
                self.prox_hist.append(prox_term_i)
                if self.verbosity > 0:
                    print(
                        f"{i:<5} | {f_x:<10.4e} | {prox_term_i:<10.4e} | {self.total_infeas[-1]:<10.4e} | {jnp.max(rho_vec) if rho_vec is not None else 0:<10.4e} | {jnp.max(nu_vec) if nu_vec is not None else 0:<10.4e} | {self.gamma_k:<10.4e}")
                best_f = f_x
                best_kkt = eps_kkt_res
                patience_counter = 0

            # store x at patience//2 if needed
            if self.patience is not None and self.patience >= 2 and patience_counter == self.patience // 2:
                patience_x = x.copy() if hasattr(x, 'copy') else jnp.array(x)

            L_aug_val = L_aug(x)
            if self.no_reset:
                x_hat = x
            else:
                f1_reset_x0 = self.problem.f(self.reset_x0)
                if self.problem.reg:
                    L_aug_val += self.problem.reg(x)
                    f1_reset_x0 += self.problem.reg(self.reset_x0)
                if self.use_proximal:
                    prox_term = jnp.sum((1/(2*self.gamma_k))*(self.reset_x0 - x)**2)
                    L_func_val = L_aug_val + prox_term
                    if L_func_val > f1_reset_x0 + prox_term:
                        x_hat = self.reset_x0
                    else:
                        x_hat = x
                else:
                    L_func_val = L_aug_val
                    if L_func_val > f1_reset_x0:
                        x_hat = self.reset_x0
                    else:
                        x_hat = x

            if callable(self.Lip_grad_est):
                if self.use_proximal:
                    gam = self.gamma_k
                else:
                    gam = 0.0
                Lip_grad_est = self.Lip_grad_est(x_hat, lbda, mu, rho_vec, nu_vec, gam)
            else:
                Lip_grad_est = self.Lip_grad_est

            if self.adaptive_fp_tol:
                self.fp_tol = 0.1/(i+1)**(1.1)
            if self.use_proximal:
                x_new, cnt_grad_f = self._minimize_prox_L_aug(x_hat, x_prev, lbda, mu, rho_vec, nu_vec, self.gamma_k, Lip_grad_est=Lip_grad_est)
                grad_evals += cnt_grad_f
            else:
                x_new, cnt_grad_f = self._minimize_L_aug(x_hat, lbda, mu, rho_vec, nu_vec, Lip_grad_est=Lip_grad_est)
                grad_evals += cnt_grad_f

            isnanfpres = False if self.problem.reg is None else (jnp.isnan(prox_grad_res_i) or jnp.isinf(prox_grad_res_i))
            isnannu = False if self.problem.g is None else (jnp.isnan(nu_vec).any() or jnp.isinf(nu_vec).any())
            isnanrho = False if self.problem.h is None else (jnp.isnan(rho_vec).any() or jnp.isinf(rho_vec).any())
            isnangamma = False if self.use_proximal is False else (jnp.isnan(self.gamma_k) or jnp.isinf(self.gamma_k))
            if jnp.isnan(self.problem.f(x_new)) or jnp.isinf(self.problem.f(x_new)) or isnanfpres or isnannu or isnanrho or isnangamma:
                if self.verbosity > 0:
                    print(
                    f"{i + 1:<5} | {f_x:<10.4e} | {prox_term_i:<10.4e} | {self.total_infeas[-1]:<10.4e} | {jnp.max(rho_vec) if rho_vec is not None else 0:<10.4e} | {jnp.max(nu_vec) if nu_vec is not None else 0:<10.4e} | {self.gamma_k:<10.4e}")
                    print("-" * 90)
                    print("One or more functions returned NaN or Inf. Stopping optimization.")
                    print(f"{'Objective value:':<25} {f_x:.6e}")
                    print(f"{'prox_term_i:':<25} {prox_term_i:.6e}")
                    print(f"{'eps-KKT residual:':<25} {eps_kkt_res:.6e}")
                    print(f"{'total infeas:':<25} {self.total_infeas[-1]:.6e}")
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

            h_x = self.problem.h(x_new) if self.problem.h else jnp.array([])
            g_x = self.problem.g(x_new) if self.problem.g else jnp.array([])

            if self.use_proximal:
                x_prev = x.copy()

            if self.use_autodiff_alm:
                L0_grad = self.alm_grad_fn(x_new)
                if self.grad_evals is not None:
                    grad_evals += self.alm_grad_fn.count
            else:
                if self.use_proximal:
                    L0_grad = self._get_prox_L_aug_grad(x_new, x_prev=x_prev, lbda=lbda, mu=mu, rho=rho_vec, nu=nu_vec, gamma_k=self.gamma_k, aug=True)
                else:
                    L0_grad = self._get_prox_L_aug_grad(x_new, lbda=lbda, mu=mu, rho=rho_vec, nu=nu_vec, gamma_k=self.gamma_k, aug=True)

            lbda = lbda + jnp.multiply(rho_vec, h_x) if self.problem.h else lbda
            mu_new = jnp.maximum(0, mu + jnp.multiply(nu_vec, g_x)) if self.problem.g else mu

            nrm_h = jnp.linalg.norm(h_x, jnp.inf) if self.problem.h else 0
            if self.problem.h:
                if self.uniform_pen:
                    nrm_h = jnp.linalg.norm(h_x, jnp.inf)
                    prev_nrm_h = jnp.linalg.norm(self.problem.h(x), jnp.inf)
                    if nrm_h > self.beta * prev_nrm_h:
                        rho_vec = jnp.maximum(self.xi1*rho_vec, jnp.full_like(rho_vec, self.rho0*phi_i))
                else:
                    prev_h = self.problem.h(x)
                    mask = jnp.abs(h_x) > self.beta * jnp.abs(prev_h)
                    rho_candidate = jnp.maximum(self.xi1 * rho_vec, self.rho0 * phi_i)
                    rho_vec = jnp.where(mask, rho_candidate, rho_vec)

            E_x = jnp.minimum(-g_x, jnp.divide(1, nu_vec)*mu) if self.problem.g else jnp.array([])
            prev_g = self.problem.g(x) if self.problem.g else jnp.array([])
            prev_E = jnp.minimum(-prev_g, jnp.divide(1, nu_vec) * mu) if self.problem.g else jnp.array([])
            if self.problem.g:
                if self.uniform_pen:
                    nrm_E = jnp.linalg.norm(E_x, jnp.inf)
                    prev_nrm_E = jnp.linalg.norm(prev_E, jnp.inf)
                    if nrm_E > self.beta * prev_nrm_E:
                        nu_vec = jnp.maximum(self.xi2*nu_vec, jnp.full_like(nu_vec, self.nu0*phi_i))
                else:
                    nrm_E = jnp.max(jnp.abs(E_x))
                    mask_E = jnp.abs(E_x) > self.beta * jnp.abs(prev_E)
                    nu_candidate = jnp.maximum(self.xi2 * nu_vec, self.nu0 * phi_i)
                    nu_vec = jnp.where(mask_E, nu_candidate, nu_vec)

            x = x_new
            mu = mu_new

            if self.use_proximal:
                prox_term_i = jnp.sum(1/(2*self.gamma_k)*(x - x_prev)**2)
            else:
                prox_term_i = 0.0

            if type(self.problem.reg) in [proxop.multi.L1Norm, proxop.indicator.BoxConstraint]:
                prox_grad_res_i = jnp.linalg.norm(x - self.problem.reg.prox(x - L0_grad), jnp.inf)
            elif type(self.problem.reg) == type(None): # this won't be the case though (case handled above)
                prox_grad_res_i = jnp.linalg.norm(L0_grad, jnp.inf)
            stopping_terms = jnp.array([prox_grad_res_i, nrm_h, nrm_E])
            if self.use_proximal:
                stopping_terms = jnp.concatenate([stopping_terms, jnp.array([prox_term_i])])

            eps_kkt_res = jnp.max(stopping_terms[1:])
            f_x = self.problem.f(x)
            self.kkt_res.append(eps_kkt_res)
            self.total_infeas.append(nrm_h + jnp.linalg.norm(jnp.maximum(g_x, 0), jnp.inf) if self.problem.g else nrm_h)
            self.f_hist.append(f_x)
            self.prox_grad_res.append(stopping_terms[0])
            self.rho_hist.append(jnp.max(rho_vec) if rho_vec is not None else None)
            self.nu_hist.append(jnp.max(nu_vec) if nu_vec is not None else None)
            self.gamma_hist.append(self.gamma_k)
            self.prox_hist.append(prox_term_i)

            if self.use_proximal:
                self.gamma_k = jnp.maximum(jnp.sum(self.delta*(self.x0 - x)**2), self.gamma0*phi_i)

            if self.verbosity > 0 and ((i + 1) % int(20/self.verbosity) == 0 or (i + 1) == self.max_iter):
                print(
                    f"{i + 1:<5} | {f_x:<10.4e} | {prox_term_i:<10.4e} | {self.total_infeas[-1]:<10.4e} | {jnp.max(rho_vec) if rho_vec is not None else 0:<10.4e} | {jnp.max(nu_vec) if nu_vec is not None else 0:<10.4e} | {self.gamma_k:<10.4e}")

            # patience logic
            current_f = f_x
            current_kkt = eps_kkt_res
            if self.patience is not None:
                if (current_f > best_f) and (current_kkt > best_kkt):
                    patience_counter += 1
                else:
                    if current_f < best_f:
                        best_f = current_f
                    if current_kkt < best_kkt:
                        best_kkt = current_kkt
                    patience_counter = 0
                if patience_counter >= self.patience:
                    if patience_x is not None:
                        x = patience_x
                        if self.verbosity > 0:
                            print("-" * 80)
                            print(f"Patience level {self.patience} reached: reverting to solution at patience//2 (iteration {i+1-self.patience//2}).")
                    elif self.verbosity > 0:
                        print("-" * 80)
                        print(f"Patience level {self.patience} reached: objective and KKT residual both worsened for {self.patience} consecutive iterations.")
                    self.solve_status = "PatienceExceeded"
                    break

            if (eps_kkt_res <= self.tol):
                if self.verbosity > 0:
                    print(
                        f"{i + 1:<5} | {f_x:<10.4e} | {prox_term_i:<10.4e} | {self.total_infeas[-1]:<10.4e} | {jnp.max(rho_vec) if rho_vec is not None else 0:<10.4e} | {jnp.max(nu_vec) if nu_vec is not None else 0:<10.4e} | {self.gamma_k:<10.4e}")
                    print("-" * 90)
                    print(f"Convergence achieved after {i + 1} iterations.")
                    print(f"{'Optimal f value found:':<25} {f_x:.6e}")
                    print(f"{'eps-KKT residual:':<25} {eps_kkt_res:.6e}")
                    print(f"{'total infeas:':<25} {self.total_infeas[-1]:.6e}")
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
                    print(f"{'total infeas:':<25} {self.total_infeas[-1]:.6e}")
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

            if self.grad_evals is not None:
                if hasattr(self.problem.f_grad, 'count'):
                    grad_evals += self.problem.f_grad.count
                if hasattr(self.problem.h_grad, 'count'):
                    grad_evals += self.problem.h_grad.count
                if hasattr(self.problem.g_grad, 'count'):
                    grad_evals += self.problem.g_grad.count
                self.grad_evals.append(grad_evals)
                last_grad_eval = grad_evals

        if self.grad_evals is not None:
            target_len = len(self.f_hist)
            while len(self.grad_evals) < target_len:
                self.grad_evals.append(last_grad_eval)

        self.x = x
        self.total_runtime = time.time() - total_start_time
        self.solve_runtime = time.time() - solve_start_time
        return

    def _get_prox_L_aug(self, x, x_prev=None, lbda=None, mu=None, rho=None, nu=None, gamma_k=None):
        h_x = self.problem.h(x) if self.problem.h else jnp.array([])
        g_x = self.problem.g(x) if self.problem.g else jnp.array([])

        L_aug_val = self.problem.f(x)

        if self.problem.h:
            eq_term = jnp.sum(rho*0.5*h_x**2) + jnp.dot(lbda,h_x)
            L_aug_val += eq_term

        if self.problem.g:
            nu_g_x_mu = nu*g_x + mu
            ineq_term = jnp.sum((1/(2*nu)) * jnp.where(nu_g_x_mu > 0, nu_g_x_mu**2, 0)) - jnp.sum((1/2*nu)*mu**2)
            L_aug_val += ineq_term

        if self.use_proximal:
            prox_term = jnp.sum((1/(2*gamma_k))*(x - x_prev)**2)
            L_aug_val += prox_term

        return L_aug_val
    
    def _get_prox_L_aug_grad(self, x, x_prev=None, lbda=None, mu=None, rho=None, nu=None, gamma_k=None, aug=True):
        h_x = self.problem.h(x) if self.problem.h else jnp.array([])
        g_x = self.problem.g(x) if self.problem.g else jnp.array([])
        h_grad_x = self.problem.h_grad(x) if self.problem.h else jnp.array([])
        g_grad_x = self.problem.g_grad(x) if self.problem.g else jnp.array([])
        con_term_grad = 0.0

        if self.problem.h:
            con_term_grad += jnp.dot(lbda, h_grad_x)

        if aug:
            if self.problem.h:
                con_term_grad += jnp.dot(rho*h_x, h_grad_x)
            if self.problem.g:
                nu_g_x_mu = nu*g_x + mu
                mask = nu_g_x_mu > 0
                g_grad_x_mod = jnp.where(mask[:, None], nu[:, None]*g_grad_x, 0)
                con_term_grad += jnp.dot(jnp.maximum(0, nu_g_x_mu)/nu, g_grad_x_mod)
        else:
            if self.problem.g:
                con_term_grad += jnp.dot(mu, g_grad_x)

        L_aug_grad = self.problem.f_grad(x) + con_term_grad

        if self.use_proximal:
            L_aug_grad += (1/gamma_k)*(x - x_prev)

        return L_aug_grad

    def _minimize_L_aug(self, x0, lbda, mu, rho, nu, Lip_grad_est=None):
        def L_aug(x):
            return self._get_prox_L_aug(x, lbda=lbda, mu=mu, rho=rho, nu=nu)

        def dL_aug(x):
            return self._get_prox_L_aug_grad(x, lbda=lbda, mu=mu, rho=rho, nu=nu, aug=True)
        
        if self.use_autodiff_alm:
            self.alm_grad_fn = GradEvalCounter(jax.grad(L_aug)) if not self.problem.jittable else GradEvalCounter(jax.jit(jax.grad(L_aug)))
            grad_fx = self.alm_grad_fn
        else:
            grad_fx = dL_aug if not self.problem.jittable else jax.jit(dL_aug)
        
        pa_prob = PaProblem(L_aug, x0, reg=self.problem.reg, lbda=self.problem.reg_lbda, solver_opts=self.pa_solver_opts, tol=self.fp_tol,
                            direction=self.pa_direction, grad_fx=grad_fx, jittable=self.problem.jittable, Lip_grad_est=Lip_grad_est)
        x_opt, _, cnt_grad_f = pa_prob._run_pa_procedure()
        return x_opt, cnt_grad_f

    def _minimize_prox_L_aug(self, x0, x_prev, lbda, mu, rho, nu, gamma_k, Lip_grad_est=None):
        """
        Minimize the proximal augmented Lagrangian with a proximal step for nonsmooth regularizers.
        """
        def prox_L_aug(x):
            return self._get_prox_L_aug(x, x_prev=x_prev, lbda=lbda, mu=mu, rho=rho, nu=nu, gamma_k=gamma_k)

        def dprox_L_aug(x):
            return self._get_prox_L_aug_grad(x, x_prev=x_prev, lbda=lbda, mu=mu, rho=rho, nu=nu, gamma_k=gamma_k, aug=True)
        
        if self.use_autodiff_alm:
            self.alm_grad_fn = GradEvalCounter(jax.grad(prox_L_aug)) if not self.problem.jittable else GradEvalCounter(jax.jit(jax.grad(prox_L_aug)))
            grad_fx = self.alm_grad_fn
        else:
            grad_fx = dprox_L_aug if not self.problem.jittable else jax.jit(dprox_L_aug)
        
        pa_prob = PaProblem(prox_L_aug, x0, reg=self.problem.reg, lbda=self.problem.reg_lbda,
                            solver_opts=self.pa_solver_opts, tol=self.fp_tol, direction=self.pa_direction,
                            grad_fx=grad_fx, jittable=self.problem.jittable, Lip_grad_est=Lip_grad_est)
        x_opt, _, cnt_grad_f = pa_prob._run_pa_procedure()
        return x_opt, cnt_grad_f

class Result:
    def __init__(self, x, prox_grad_res, kkt_res, total_infeas, f_hist, rho_hist, nu_hist, gamma_hist, prox_hist, solve_status, total_runtime, solve_runtime, grad_evals=None):
        self.x = x
        self.prox_grad_res = prox_grad_res
        self.kkt_res = kkt_res
        self.total_infeas = total_infeas
        self.f_hist = f_hist
        self.rho_hist = rho_hist
        self.nu_hist = nu_hist
        self.gamma_hist = gamma_hist
        self.prox_hist = prox_hist
        self.solve_status = solve_status
        self.total_runtime = total_runtime
        self.solve_runtime = solve_runtime
        self.grad_evals = grad_evals

def solve(problem, x0, lbda0=None, mu0=None, rho0=1e-3, nu0=1e-3, use_proximal=True,
            gamma0=1e-1, beta=0.5, alpha=3, delta=1e-6, xi1=1.0, xi2=1.0, tol=1e-6, fp_tol=None, max_iter=1000, phase_I_tol=1e-7,
            start_feas=True, inner_solver=None, pa_direction=None, pa_solver_opts=None, verbosity=1, max_runtime=24.0,
            phi_strategy="pow", patience=None, feas_reset_interval=None, uniform_pen=True, no_reset=False, Lip_grad_est=None, use_autodiff_alm=True, adaptive_fp_tol=False):
    if inner_solver is None:
        inner_solver = "PANOC"
    else:
        if inner_solver == "PANOC" and type(problem.reg) not in [proxop.multi.L1Norm, proxop.indicator.BoxConstraint, type(None)]:
            print("PANOC solver is only available for L1, BoxConstraint and NoneType regularizers.")
            raise ValueError("Incompatible regularizer for PANOC solver.")
            
    if problem.h is not None and lbda0 is None:
        h0 = problem.h(x0)
        rng = np.random.default_rng(1234)
        lbda0 = jnp.array(rng.standard_normal(h0.shape))
    if problem.g is not None and mu0 is None:
        g0 = problem.g(x0)
        rng = np.random.default_rng(1234)
        mu0 = jnp.array(rng.standard_normal(g0.shape))
    solution = Solution(problem, x0, lbda0, mu0, rho0, nu0, gamma0, use_proximal=use_proximal, beta=beta, alpha=alpha, delta=delta,
                        xi1=xi1, xi2=xi2, tol=tol, fp_tol=fp_tol, max_iter=max_iter, phase_I_tol=phase_I_tol, start_feas=start_feas,
                        inner_solver=inner_solver, pa_direction=pa_direction, pa_solver_opts=pa_solver_opts, verbosity=verbosity,
                        max_runtime=max_runtime, phi_strategy=phi_strategy, patience=patience, feas_reset_interval=feas_reset_interval,
                        uniform_pen=uniform_pen, no_reset=no_reset, Lip_grad_est=Lip_grad_est, use_autodiff_alm=use_autodiff_alm, adaptive_fp_tol=adaptive_fp_tol)
    solution.pbalm()
    res = Result(solution.x, solution.prox_grad_res, solution.kkt_res, solution.total_infeas, solution.f_hist, solution.rho_hist, solution.nu_hist, solution.gamma_hist, solution.prox_hist, solution.solve_status, solution.total_runtime, solution.solve_runtime, grad_evals=getattr(solution, 'grad_evals', None))
    return res