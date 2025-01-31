import allema as alm
import numpy as np
import jax.numpy as jnp
import jax.random as random


# maratos problem as described in:
# Fletcher, Roger, Sven Leyffer, and Ph L. Toint. "A brief history of filter methods." Preprint ANL/MCS-P1372-0906, Argonne National Laboratory, Mathematics and Computer Science Division 36 (2006).

__author__ = "Adeyemi Damilare Adeoye"
__copyright__ = "Adeyemi Damilare Adeoye"
__license__ = "MIT"

np.random.seed(1234)


def f(x):
    return 2*(x[0]**2 + x[1]**2 - 1) - x[0]

def h(x):
    return jnp.array([x[0]**2 + x[1]**2 - 1])

nvar = 2
x0 = jnp.array([jnp.sqrt(2)/2, jnp.sqrt(2)/2])
lbda = jnp.array([3/2])
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

sol = alm.solve(problem, x0, lbda, mu0, rho, nu, tol=tol, max_iter=max_iter, start_feas=False, inner_solver="BFGS")

print(sol.x)


# import matplotlib.pyplot as plt
# plt.plot(sol.obj_hist)
# plt.xlabel('Iteration')
# plt.ylabel('Objective Value')
# plt.title('Objective Value History')
# plt.grid(True)
# plt.show()