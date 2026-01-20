class Result:
    """
    Class to store the results of the PBALM solver.
    """
    def __init__(self, x, fp_res, kkt_res, total_infeas, f_hist, rho_hist, nu_hist, gamma_hist, prox_hist, solve_status, total_runtime, solve_runtime, grad_evals=None):
        """
        Initializes the Result object.

        Attributes:
            x: The solution found by the solver.
            fp_res: History of fixed-point residuals.
            kkt_res: History of KKT residuals.
            total_infeas: Total infeasibility at the solution.
            f_hist: History of objective function values.
            rho_hist: History of penalty parameter rho (for equality constraints).
            nu_hist: History of penalty parameter nu (for inequality constraints).
            gamma_hist: History of the proximal parameter gamma.
            prox_hist: History of proximal term.
            solve_status: Status of the solver ('Converged', 'Stopped', 'MaxRuntimeExceeded', 'NanOrInf').
            total_runtime: Total runtime of the solver.
            solve_runtime: Runtime of the solving phase.
            grad_evals: Number of gradient evaluations.
        """
        self.x = x
        self.fp_res = fp_res
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