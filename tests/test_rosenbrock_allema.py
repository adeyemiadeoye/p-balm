import allema as alm
import numpy as np
import jax.numpy as jnp
import jax.random as random

__author__ = "Adeyemi Damilare Adeoye"
__copyright__ = "Adeyemi Damilare Adeoye"
__license__ = "MIT"

np.random.seed(1234)

def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def h(x):
    return jnp.array([jnp.sum(x) - 1])

g = None

x0 = jnp.array([-1.2, 1]) # random.uniform(random.PRNGKey(42), (nvar,))
lbda = random.uniform(random.PRNGKey(42), (1,))
mu0 = jnp.array([])
rho = 1
nu = 1
tol = 1e-6
max_iter = 1000

problem = alm.Problem(
    obj=f,
    eq_con=h,
    ineq_con=None
)

sol = alm.solve(problem, x0, lbda, mu0, rho, nu, tol=tol, max_iter=max_iter, start_feas=False, inner_solver="L-BFGS-B")

print(jnp.sum(sol.x))