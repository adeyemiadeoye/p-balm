import allema as alm
import numpy as np
import jax.numpy as jnp
import jax.random as random

# problem as seen in:
# Kiessling, David, Sven Leyffer, and Charlie Vanaret. "A Unified Funnel Restoration SQP Algorithm." arXiv preprint arXiv:2409.09208 (2024).

__author__ = "Adeyemi Damilare Adeoye"
__copyright__ = "Adeyemi Damilare Adeoye"
__license__ = "MIT"

np.random.seed(1234)


def f(x):
    return 0.0

def h(x):
    return jnp.array([-1 + 10000*x[0]*x[1], -1.0001 + jnp.exp(-x[0]) + jnp.exp(-x[1])])


from scipy.optimize import minimize
def constraint_1(x):
    return -1 + 10000 * x[0] * x[1]
def constraint_2(x):
    return -1.0001 + np.exp(-x[0]) + np.exp(-x[1])
constraints = [
    {'type': 'eq', 'fun': constraint_1},
    {'type': 'eq', 'fun': constraint_2}
]
x0 = np.array([0.1, 0.1])
result = minimize(f, x0, method='SLSQP', constraints=constraints)
print("Optimization Result:", result)
print("Optimal x:", result.x)

x0 = jnp.array([0.5, 0.1])
lbda = jnp.array([1, 1])
mu0 = jnp.array([])
rho = 10
nu = 1
tol = 1e-6
max_iter = 100

problem = alm.Problem(
    obj=f,
    eq_con=h,
    ineq_con=None
)

sol = alm.solve(problem, x0, lbda, mu0, rho, nu, tol=tol, max_iter=max_iter, start_feas=False, inner_solver="BFGS")

print(sol.x)