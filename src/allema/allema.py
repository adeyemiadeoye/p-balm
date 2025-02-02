import numpy as np
from scipy.optimize import minimize
from jax import grad, jacfwd
import jax.numpy as jnp

class Problem:
    def __init__(self, obj, eq_con=None, ineq_con=None, obj_grad=None, eq_con_grad=None, ineq_con_grad=None):
        self.obj = obj
        self.eq_con = eq_con
        self.ineq_con = ineq_con
        self.obj_grad = obj_grad if obj_grad else grad(obj)
        self.eq_con_grad = eq_con_grad if (eq_con and eq_con_grad) else (jacfwd(eq_con) if eq_con else None)
        self.ineq_con_grad = ineq_con_grad if (ineq_con and ineq_con_grad) else (jacfwd(ineq_con) if ineq_con else None)

class Solution:
    def __init__(self, problem, x0, lbda0, mu0, rho0, nu0, beta=0.5, alpha=2.0, xi1=2.0, xi2=2.0, tol=1e-6, max_iter=1000, start_feas=True, inner_solver="BFGS", lbfgs_options=None, verbosity=1):
        self.problem = problem
        self.x0 = x0
        self.lbda0 = lbda0
        self.mu0 = mu0
        self.rho0 = rho0
        self.nu0 = nu0
        self.beta = beta
        self.alpha = alpha
        self.xi1 = xi1
        self.xi2 = xi2
        self.tol = tol
        self.max_iter = max_iter
        self.verbosity = verbosity
        self.x = x0
        self.start_feas = start_feas
        self.inner_solver = inner_solver
        self.grad_norm = []
        self.kkt_res = []
        self.obj_hist = []
        self.rho_hist = []
        self.nu_hist = []
        self.solve_status = None
        self.lbfgs_options = {'maxls': 20, 'gtol': self.tol, 'eps': 1.e-8, 'ftol': self.tol, 'maxfun': self.max_iter, 'maxcor': 10} if lbfgs_options is None else lbfgs_options

    def allema(self):
        if self.problem.eq_con is None and self.problem.ineq_con is None:
            if self.verbosity > 0:
                print("Solving problem without constraints")
            options = {'disp': self.verbosity > 0}
            res = minimize(self.problem.obj, self.x0, method=self.inner_solver, jac=self.problem.obj_grad, options=options)
            return res

        eq_con_x0 = self.problem.eq_con(self.x0) if self.problem.eq_con else np.array([])
        ineq_con_x0 = self.problem.ineq_con(self.x0) if self.problem.ineq_con else np.array([])
        if (self.problem.eq_con is None or np.all(np.isclose(eq_con_x0, 0, atol=self.tol))) and (self.problem.ineq_con is None or np.all(ineq_con_x0 <= 0)):
            if self.verbosity > 0:
                print("Initial point is feasible.")
            x = self.x0
        else:
            if self.start_feas:
                if self.verbosity > 0:
                    print("Initial point is not feasible. Finding a feasible point...")
                self.x0 = self._find_feas_x0()
            else:
                if self.verbosity > 0:
                    print("Initial point is not feasible. Starting from infeasible point.")
            x = self.x0
        lbda = self.lbda0
        mu = self.mu0
        rho = self.rho0
        nu = self.nu0

        if self.verbosity > 0:
            print(f"{'iter':<5} | {'obj':<10} | {'|L-grad|':<10} | {'kkt res':<10} | {'rho':<10} | {'nu':<10}")
            print("-"*70)

        for i in range(self.max_iter):
            if i == 0 and self.verbosity > 0:
                eq_con_x = self.problem.eq_con(x) if self.problem.eq_con else np.array([])
                ineq_con_x = self.problem.ineq_con(x) if self.problem.ineq_con else np.array([])
                nrm_h = np.linalg.norm(eq_con_x, np.inf) if self.problem.eq_con else 0
                E_x = np.minimum(-ineq_con_x, 1/nu*mu) if self.problem.ineq_con else np.array([])
                nrm_E = np.linalg.norm(E_x, np.inf) if self.problem.ineq_con else 0
                L0_grad = self._get_L_aug_grad(x, lbda, mu, rho, nu, aug=False)
                stopping_terms = [np.linalg.norm(L0_grad, np.inf), nrm_h, nrm_E]
                eps_kkt_res = np.max(stopping_terms[1:])
                self.kkt_res.append(eps_kkt_res)
                self.grad_norm.append(stopping_terms[0])
                self.obj_hist.append(self.problem.obj(x))
                self.rho_hist.append(rho)
                self.nu_hist.append(nu)
                print(f"{i:<5} | {self.problem.obj(x):<10.4e} | {stopping_terms[0]:<10.4e} | {eps_kkt_res:<10.4e} | {rho:<10.4e} | {nu:<10.4e}")

            L_aug = self._get_L_aug(x, lbda, mu, rho, nu, aug=True)
            if L_aug > self.problem.obj(self.x0):
                x = self.x0

            x_new = self._minimize_L_aug(x, lbda, mu, rho, nu)
            eq_con_x = self.problem.eq_con(x_new) if self.problem.eq_con else np.array([])
            ineq_con_x = self.problem.ineq_con(x_new) if self.problem.ineq_con else np.array([])
            lbda = lbda + rho*eq_con_x if self.problem.eq_con else lbda
            mu_new = np.maximum(0, mu + nu*ineq_con_x) if self.problem.ineq_con else mu

            nrm_h = np.linalg.norm(eq_con_x, np.inf) if self.problem.eq_con else 0
            if self.problem.eq_con and nrm_h > self.beta*np.linalg.norm(self.problem.eq_con(x), np.inf):
                rho = np.maximum(self.xi1*rho, self.rho0*(i + 1)**self.alpha)

            E_x = np.minimum(-ineq_con_x, 1/nu*mu_new) if self.problem.ineq_con else np.array([])
            nrm_E = np.linalg.norm(E_x, np.inf) if self.problem.ineq_con else 0
            if self.problem.ineq_con and nrm_E > self.beta*np.linalg.norm(np.minimum(-self.problem.ineq_con(x), 1/nu*mu), np.inf):
                nu = np.maximum(self.xi2*nu, self.nu0*(i + 1)**self.alpha)

            x = x_new
            mu = mu_new

            L0_grad = self._get_L_aug_grad(x, lbda, mu, rho, nu, aug=False)

            stopping_terms = [np.linalg.norm(L0_grad, np.inf), nrm_h, nrm_E]
            violations = (ineq_con_x < -self.tol) & ~np.isclose(mu_new, 0) if self.problem.ineq_con else np.array([])
            eps_kkt_res = np.max(stopping_terms[1:])
            self.kkt_res.append(eps_kkt_res)
            self.obj_hist.append(self.problem.obj(x))
            self.rho_hist.append(rho)
            self.nu_hist.append(nu)

            if self.verbosity > 0 and ((i+1) % 20 == 0 or (i+1) == self.max_iter):
                print(f"{i+1:<5} | {self.problem.obj(x):<10.4e} | {stopping_terms[0]:<10.4e} | {eps_kkt_res:<10.4e} | {rho:<10.4e} | {nu:<10.4e}")

            if (~np.any(violations) if self.problem.ineq_con else True) and eps_kkt_res <= self.tol:
                if self.verbosity > 0:
                    print(f"{i+1:<5} | {self.problem.obj(x):<10.4e} | {stopping_terms[0]:<10.4e} | {eps_kkt_res:<10.4e} | {rho:<10.4e} | {nu:<10.4e}")
                    print("-"*70)
                    print(f"Convergence achieved after {i+1} iterations.")
                    print(f"{'Optimal obj value found:':<25} {self.problem.obj(x):.6e}")
                    print(f"{'Lagrangian gradient norm:':<25} {stopping_terms[0]:.6e}")
                    print(f"{'eps-KKT residual:':<25} {eps_kkt_res:.6e}")
                    print(f"{'rho:':<25} {rho:.6e}")
                    print(f"{'nu:':<25} {nu:.6e}")
                    self.solve_status = "Converged"
                break
            elif (i+1) == self.max_iter and self.verbosity > 0:
                print("-"*70)
                print("Maximum iterations reached without convergence.")
                print(f"{'Objective value:':<25} {self.problem.obj(x):.6e}")
                print(f"{'Lagrangian gradient norm:':<25} {stopping_terms[0]:.6e}")
                print(f"{'eps-KKT residual:':<25} {eps_kkt_res:.6e}")
                print(f"{'rho:':<25} {rho:.6e}")
                print(f"{'nu:':<25} {nu:.6e}")
                self.solve_status = "Stopped"
        
        self.x = x
        return
    
    def _find_feas_x0(self):
        def feas_obj(x):
            eq_con_x = self.problem.eq_con(x) if self.problem.eq_con else np.array([])
            ineq_con_x = self.problem.ineq_con(x) if self.problem.ineq_con else np.array([])
            return np.sum(np.square(eq_con_x)) + np.sum(np.square(np.minimum(0, ineq_con_x)))

        res = minimize(feas_obj, self.x0, method=self.inner_solver)
        return res.x
    
    def _get_L_aug_grad(self, x, lbda, mu, rho, nu, aug=True):
        eq_con_grad_x = self.problem.eq_con_grad(x) if self.problem.eq_con else np.array([])
        ineq_con_grad_x = self.problem.ineq_con_grad(x) if self.problem.ineq_con else np.array([])
        
        con_term_grad = np.dot(lbda, eq_con_grad_x) if self.problem.eq_con else 0
        
        if aug:
            if self.problem.eq_con:
                con_term_grad += rho*np.dot(self.problem.eq_con(x), eq_con_grad_x)
            if self.problem.ineq_con:
                nu_ineq_con_x_mu = nu*self.problem.ineq_con(x) + mu
                ineq_con_grad_x = np.where((nu_ineq_con_x_mu)[:, np.newaxis] > 0, nu*ineq_con_grad_x, 0)
                con_term_grad += 1/nu*np.dot(np.maximum(0, nu_ineq_con_x_mu), ineq_con_grad_x)
        else:
            if self.problem.ineq_con:
                con_term_grad += np.dot(mu, ineq_con_grad_x)
        
        return self.problem.obj_grad(x) + con_term_grad
        # return grad(lambda x: self._get_L_aug(x, lbda, mu, rho, nu, aug))(x)
    
    def _get_L_aug(self, x, lbda, mu, rho, nu, aug=True):
        eq_term = np.sum(lbda*self.problem.eq_con(x)) if self.problem.eq_con else 0
        ineq_term = np.sum(mu*self.problem.ineq_con(x)) if self.problem.ineq_con else 0

        if aug:
            if self.problem.eq_con:
                eq_term += rho/2*np.sum(self.problem.eq_con(x)**2)
            if self.problem.ineq_con:
                nu_ineq_con_x_mu = nu*self.problem.ineq_con(x) + mu
                ineq_term += 1/(2*nu)*np.sum(jnp.where(nu_ineq_con_x_mu > 0, nu_ineq_con_x_mu**2, 0))

        return self.problem.obj(x) + eq_term + ineq_term

    def _minimize_L_aug(self, x0, lbda, mu, rho, nu):
        def L_aug(x):
            return self._get_L_aug(x, lbda, mu, rho, nu, aug=True)
        def dL_aug(x):
            return self._get_L_aug_grad(x, lbda, mu, rho, nu, aug=True)
        if self.inner_solver == "L-BFGS-B":
            res = minimize(L_aug, x0, method=self.inner_solver, jac=dL_aug, options=self.lbfgs_options)
        else:
            res = minimize(L_aug, x0, method=self.inner_solver, jac=dL_aug)
        return res.x

class Result:
    def __init__(self, x, grad_norm, kkt_res, obj_hist, rho_hist, nu_hist, solve_status):
        self.x = x
        self.grad_norm = grad_norm
        self.kkt_res = kkt_res
        self.obj_hist = obj_hist
        self.rho_hist = rho_hist
        self.nu_hist = nu_hist
        self.solve_status = solve_status

def solve(problem, x0, lbda0, mu0, rho0, nu0, beta=0.5, alpha=1.5, xi1=1.5, xi2=1.5, tol=1e-6, max_iter=1000, start_feas=True, inner_solver="BFGS", lbfgs_options=None, verbosity=1):
    solution = Solution(problem, x0, lbda0, mu0, rho0, nu0, beta=beta, alpha=alpha, xi1=xi1, xi2=xi2, tol=tol, max_iter=max_iter, start_feas=start_feas, inner_solver=inner_solver, lbfgs_options=lbfgs_options, verbosity=verbosity)
    solution.allema()
    res = Result(solution.x, solution.grad_norm, solution.kkt_res, solution.obj_hist, solution.rho_hist, solution.nu_hist, solution.solve_status)
    return res