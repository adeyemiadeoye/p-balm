import allema as alm
import numpy as np
import pycutest
from prettytable import PrettyTable

__author__ = "Adeyemi Damilare Adeoye"
__copyright__ = "Adeyemi Damilare Adeoye"
__license__ = "MIT"

np.random.seed(1234)

constraints = ['linear', 'quadratic']
nvar_range = [2, 1000]

probnames = pycutest.find_problems(objective='sum of squares', constraints=constraints, n=nvar_range)

cutest_problems = []
for probname in probnames:
    props = pycutest.problem_properties(probname)
    try:
        if props['n'] >= nvar_range[0] and props['n'] <= nvar_range[1]:
            cutest_problems.append(probname)
    except:
        pass

print("Number of problems: ", len(cutest_problems))

results = []

for probname in cutest_problems:
    prob = pycutest.import_problem(probname)
    x0 = prob.x0
    nvar = prob.n
    m_eq = prob.cons(x0)[prob.is_eq_cons].shape[0]
    m_ineq = prob.cons(x0)[~prob.is_eq_cons].shape[0]

    def f(x):
        return prob.obj(x)
    
    if prob.cons(x0)[prob.is_eq_cons].shape[0] == 0:
        h = None
        dh = None
    else:
        def h(x):
            return prob.cons(x)[prob.is_eq_cons]
        def dh(x):
            return prob.cons(x, gradient=True)[1][prob.is_eq_cons]
        
    if prob.cons(x0)[~prob.is_eq_cons].shape[0] == 0:
        g = None
        dg = None
    else:
        def g(x):
            return prob.cons(x)[~prob.is_eq_cons]
        def dg(x):
            return prob.cons(x, gradient=True)[1][~prob.is_eq_cons]
    
    def df(x):
        return prob.grad(x)
    
    problem = alm.Problem(
        obj=f,
        eq_con=h,
        ineq_con=g,
        obj_grad=df,
        eq_con_grad=dh,
        ineq_con_grad=dg
    )

    lbda = np.random.randn(m_eq)
    mu0 = np.random.randn(m_ineq)
    rho = 1
    nu = 1
    tol = 1e-4
    max_iter = 1000
    sol = alm.solve(problem, x0, lbda, mu0, rho, nu, tol=tol, max_iter=max_iter, start_feas=True, inner_solver="L-BFGS-B")
    
    results.append((probname, nvar, m_eq, m_ineq, sol.solve_status))

table = PrettyTable()
table.field_names = ["prob name", "n. vars", "n. eq cons", "n. ineq cons", "solve status"]

total_solved = 0
for probname, nvar, m_eq, m_ineq, status in results:
    table.add_row([probname, nvar, m_eq, m_ineq, status])
    if status == "Converged":
        total_solved += 1

print(table)
print(f"Total Solved: {total_solved} out of {len(results)}")