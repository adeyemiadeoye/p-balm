import allema as alm
import numpy as np
import jax.numpy as jnp
import jax.random as random

__author__ = "Adeyemi Damilare Adeoye"
__copyright__ = "Adeyemi Damilare Adeoye"
__license__ = "MIT"

np.random.seed(1234)


def f(x):
    return x[0]**2 + x[1]**2

def h(x):
    return jnp.array([x[0]**2 + x[1]**2 - 1])

g = None

nvar = 2
x0 = random.uniform(random.PRNGKey(42), (nvar,))
lbda = random.uniform(random.PRNGKey(42), (1,))
mu0 = jnp.array([])
rho = 1
nu = 1
tol = 1e-6
max_iter = 1000

problem = alm.Problem(
    obj=f,
    eq_con=h,
    ineq_con=g
)

sol = alm.solve(problem, x0, lbda, mu0, rho, nu, tol=tol, max_iter=max_iter, start_feas=True, inner_solver="BFGS")