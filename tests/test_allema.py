import allema as alm
import numpy as np
import jax.numpy as jnp

__author__ = "Adeyemi Damilare Adeoye"
__copyright__ = "Adeyemi Damilare Adeoye"
__license__ = "MIT"

np.random.seed(1234)

problem = alm.Problem(
        obj=lambda x: x[0]**2 + x[1]**2,
        eq_con=lambda x: 3*x - 1,
        ineq_con=lambda x: 2*x**2 - 2
    )
x0 = np.array([1, 0])
lbda = np.array([1.2, 0.5])
mu0 = np.array([1, 1.5])
rho = 1
nu = 1
tol = 1e-6
max_iter = 1000
sol = alm.solve(problem, x0, lbda, mu0, rho, nu, tol=tol, max_iter=max_iter)

problem = alm.Problem(
    obj=lambda x: np.sum(x**2),
    eq_con=lambda x: jnp.array([[1, 1, 1, 1, 1], 
                                [1, -1, 0, 0, 0], 
                                [0, 0, 1, -1, 0]]).dot(x) - 0.1*jnp.ones(3),
    ineq_con=lambda x: jnp.array([[1, 0, 0, 0, 0], 
                                  [0, 1, 0, 0, 0], 
                                  [0, 0, 1, 0, 0], 
                                  [0, 0, 0, 1, 0], 
                                  [0, 0, 0, 0, 1]]).dot(x) - 0.2*jnp.ones(5)
    )
x0 = np.ones(5)
lbda = 0.1*np.ones(3)
mu0 = 0.5*np.ones(5)
rho = 1
nu = 1
tol = 1e-4
max_iter = 1000
sol = alm.solve(problem, x0, lbda, mu0, rho, nu, tol=tol, max_iter=max_iter, start_feas=True, inner_solver="BFGS")