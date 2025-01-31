import allema as alm
import numpy as np
import jax.numpy as jnp
import jax.random as random

__author__ = "Adeyemi Damilare Adeoye"
__copyright__ = "Adeyemi Damilare Adeoye"
__license__ = "MIT"

np.random.seed(1234)

nvar = 100
n_eq_cons = 30
n_ineq_cons = 40

Q = jnp.tril(random.uniform(random.PRNGKey(42), (nvar, nvar)))
Q = Q + Q.T - jnp.diag(Q.diagonal())
Q = Q + nvar * jnp.eye(nvar)
c = jnp.ones(nvar)
A = jnp.ones((n_eq_cons, nvar))
b1 = jnp.ones(n_eq_cons) * 2
G = jnp.eye(n_ineq_cons, nvar)
b2 = jnp.ones(n_ineq_cons) * 0.5

def f(x):
    return 0.5 * jnp.dot(x, jnp.dot(Q, x)) + jnp.dot(c, x)

def h(x):
    return jnp.dot(A, x) - b1

def g(x):
    return jnp.dot(G, x) - b2  # x >= 0.5

problem = alm.Problem(
        obj=f,
        eq_con=h,
        ineq_con=g
    )
x0 = random.uniform(random.PRNGKey(42), (nvar,))
lbda = random.uniform(random.PRNGKey(42), (n_eq_cons,))
mu0 = random.uniform(random.PRNGKey(42), (n_ineq_cons,))
rho = 1
nu = 1
tol = 1e-6
max_iter = 1000
sol = alm.solve(problem, x0, lbda, mu0, rho, nu, tol=tol, max_iter=max_iter, start_feas=True, inner_solver="BFGS")