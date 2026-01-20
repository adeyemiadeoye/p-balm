Constrained Optimization Examples
==================================

This page demonstrates various constrained optimization problems that can be solved with **pbalm**.

Example 1: Quadratic Program with Linear Constraints
----------------------------------------------------

Consider minimizing a quadratic objective subject to linear equality and inequality constraints:

.. math::

   \min_{x} \quad & \frac{1}{2} x^T Q x + c^T x \\
   \text{s.t.} \quad & Ax = b \\
                     & Gx \leq h

Implementation
^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   import pbalm
   import numpy as np

   # Configure JAX
   import jax
   jax.config.update('jax_platform_name', 'cpu')
   jax.config.update("jax_enable_x64", True)

   # Problem data
   n = 10
   rng = np.random.default_rng(42)

   # Positive definite Q matrix
   M = rng.standard_normal((n, n))
   Q = jnp.array(M.T @ M + 0.1 * np.eye(n))
   c = jnp.array(rng.standard_normal(n))

   # Equality constraint: sum(x) = 1
   A = jnp.ones((1, n))
   b_eq = jnp.array([1.0])

   # Inequality constraint: x >= 0 (i.e., -x <= 0)
   G = -jnp.eye(n)
   h_ineq = jnp.zeros(n)

   # Define functions
   def f1(x):
       return 0.5 * x @ Q @ x + c @ x

   def h(x):
       return A @ x - b_eq

   def g(x):
       return G @ x - h_ineq

   # Create and solve problem
   problem = pbalm.Problem(f1=f1, h=[h], g=[g], jittable=True)
   x0 = jnp.ones(n) / n  # Start on simplex
   
   result = pbalm.solve(problem, x0, use_proximal=True, tol=1e-6)
   
   print(f"Optimal x: {result.x}")
   eq_con = h(result.x)
   ineq_con = g(result.x)
   print(f"Equality constraint: {eq_con}")
   print(f"Inequality constraint: {ineq_con}")

Example 2: Nonlinear Least Squares with Constraints
---------------------------------------------------

Fit a nonlinear model with parameter constraints:

.. math::

   \min_{\theta} \quad & \sum_{i=1}^{m} (y_i - f(x_i; \theta))^2 \\
   \text{s.t.} \quad & \theta_1 + \theta_2 \leq 1 \\
                     & \theta \geq 0

Implementation
^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   import pbalm
   import numpy as np

   # Configure JAX
   import jax
   jax.config.update('jax_platform_name', 'cpu')
   jax.config.update("jax_enable_x64", True)

   # Generate synthetic data
   rng = np.random.default_rng(123)
   n_samples = 100
   x_data = jnp.linspace(0, 1, n_samples)
   
   # True model: y = theta1 * exp(-theta2 * x) + noise
   theta_true = jnp.array([0.5, 2.0])
   y_data = theta_true[0] * jnp.exp(-theta_true[1] * x_data) + 0.05 * rng.standard_normal(n_samples)

   # Model prediction
   def model(x, theta):
       return theta[0] * jnp.exp(-theta[1] * x)

   # Objective: sum of squared residuals
   def f1(theta):
       predictions = model(x_data, theta)
       residuals = y_data - predictions
       return jnp.sum(residuals**2)

   # Constraints
   def g1(theta):
       return theta[0] + theta[1] - 5.0  # theta1 + theta2 <= 5

   def g2(theta):
       return -theta  # theta >= 0 (element-wise)

   # Create problem
   problem = pbalm.Problem(
       f1=f1,
       g=[g1, g2],
       jittable=True
   )

   # Initial guess
   theta0 = jnp.array([0.3, 1.0])

   # Solve
   result = pbalm.solve(
       problem, 
       theta0, 
       use_proximal=True, 
       tol=1e-6,
       max_iter=200
   )

   print(f"True parameters: {theta_true}")
   print(f"Estimated parameters: {result.x}")
   print(f"Final objective: {f1(result.x):.6f}")

Example 3: Optimization on a Sphere
-----------------------------------

Minimize a function subject to a spherical constraint :math:`\|x\|^2 = 1`:

.. math::

   \min_{x} \quad & c^T x \\
   \text{s.t.} \quad & \|x\|^2 = 1

This finds the point on the unit sphere that minimizes the linear objective.

Implementation
^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   import pbalm
   import numpy as np

   # Configure JAX
   import jax
   jax.config.update('jax_platform_name', 'cpu')
   jax.config.update("jax_enable_x64", True)

   # Problem data
   n = 5
   rng = np.random.default_rng(456)
   c = jnp.array(rng.standard_normal(n))

   # Objective
   def f1(x):
       return c @ x

   # Sphere constraint: ||x||^2 = 1
   def h(x):
       return jnp.sum(x**2) - 1.0

   # Create problem
   problem = pbalm.Problem(f1=f1, h=[h], jittable=True)

   # Initial point (will be projected to feasible)
   x0 = jnp.ones(n) / jnp.sqrt(n)

   # Solve
   result = pbalm.solve(
       problem, 
       x0, 
       use_proximal=True, 
       tol=1e-8,
       start_feas=False  # Start from normalized point
   )

   # Analytical solution: x* = -c / ||c||
   x_analytical = -c / jnp.linalg.norm(c)

   print(f"PBALM solution: {result.x}")
   print(f"Analytical solution: {x_analytical}")
   print(f"Solution norm: {jnp.linalg.norm(result.x)}")
   print(f"Error: {jnp.linalg.norm(result.x - x_analytical):.2e}")

Example 4: Multiple Equality Constraints
----------------------------------------

A problem with multiple nonlinear equality constraints:

.. math::

   \min_{x} \quad & x_1^2 + x_2^2 + x_3^2 \\
   \text{s.t.} \quad & x_1 x_2 = 1 \\
                     & x_2 x_3 = 2

Implementation
^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   import pbalm

   def f1(x):
       return jnp.sum(x**2)

   def h1(x):
       return x[0] * x[1] - 1.0

   def h2(x):
       return x[1] * x[2] - 2.0

   # Create problem with multiple equality constraints
   problem = pbalm.Problem(
       f1=f1,
       h=[h1, h2],
       jittable=True
   )

   x0 = jnp.array([1.0, 1.0, 2.0])

   result = pbalm.solve(
       problem, 
       x0, 
       use_proximal=True, 
       tol=1e-9
   )

   print(f"Solution: {result.x}")
   print(f"h1(x) = x1*x2 - 1 = {result.x[0] * result.x[1] - 1:.2e}")
   print(f"h2(x) = x2*x3 - 2 = {result.x[1] * result.x[2] - 2:.2e}")
   print(f"Objective: {f1(result.x):.6f}"))

Example 5: Mixed Constraints with Regularization
------------------------------------------------

Combining equality constraints, inequality constraints, and L1 regularization:

.. math::

   \min_{x} \quad & \frac{1}{2}\|Ax - b\|^2 + \lambda \|x\|_1 \\
   \text{s.t.} \quad & \mathbf{1}^T x = 1 \\
                     & x \geq 0

Implementation
^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   import pbalm
   import numpy as np

   # Configure JAX
   import jax
   jax.config.update('jax_platform_name', 'cpu')
   jax.config.update("jax_enable_x64", True)

   # Problem data
   m, n = 50, 100
   rng = np.random.default_rng(789)
   
   A = jnp.array(rng.standard_normal((m, n)))
   b = jnp.array(rng.standard_normal(m))

   # Smooth part of objective
   def f1(x):
       residual = A @ x - b
       return 0.5 * jnp.sum(residual**2)

   # Equality: sum(x) = 1
   def h(x):
       return jnp.sum(x) - 1.0

   # Inequality: x >= 0
   def g(x):
       return -x

   # L1 regularization
   f2_lbda = 0.1
   f2 = pbalm.L1Norm(f2_lbda)

   # Create problem
   problem = pbalm.Problem(
       f1=f1,
       h=[h],
       g=[g],
       f2=f2,
       jittable=True
   )

   x0 = jnp.ones(n) / n

   result = pbalm.solve(
       problem, 
       x0, 
       use_proximal=True, 
       tol=1e-5,
       max_iter=500
   )

   print(f"Sum of x: {jnp.sum(result.x):.6f}")
   print(f"Min of x: {jnp.min(result.x):.6f}")
   print(f"Number of zeros: {jnp.sum(jnp.abs(result.x) < 1e-4)}")
   print(f"Objective: {f1(result.x) + f2_lbda * jnp.sum(jnp.abs(result.x)):.6f}")
