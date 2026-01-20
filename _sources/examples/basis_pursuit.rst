Basis Pursuit
=============

This example demonstrates how to solve a **basis pursuit** problem, which seeks the sparsest solution to an underdetermined linear system.

Problem Formulation
-------------------

Given a matrix :math:`B \in \mathbb{R}^{m \times n}` with :math:`m < n` and a vector :math:`b \in \mathbb{R}^m`, the basis pursuit problem seeks a sparse solution :math:`z` such that :math:`Bz = b`:

.. math::

   \min_{z} \quad & \|z\|_1 \\
   \text{s.t.} \quad & Bz = b

By introducing a change of variables :math:`z = u_1^2 - u_2^2` where :math:`u_1, u_2 \geq 0`, we can reformulate the problem as follows [BP1]_:

.. math::

   \min_{u_1, u_2} \quad & \|u_1\|_2^2 + \|u_2\|_2^2 \\
   \text{s.t.} \quad & [B, -B] \begin{bmatrix} u_1^2 \\ u_2^2 \end{bmatrix} = b

Let :math:`x = [u_1; u_2]` and :math:`B_{\text{big}} = [B, -B]`. The problem becomes:

.. math::

   \min_{x} \quad & \|x\|_2^2 \\
   \text{s.t.} \quad & B_{\text{big}} (x \circ x) = b

where :math:`\circ` denotes element-wise multiplication.

Implementation
--------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import pbalm
   import numpy as np

   # Configure JAX
   jax.config.update('jax_platform_name', 'cpu')
   jax.config.update("jax_enable_x64", True)

   # Generate basis pursuit data
   def get_basis_pursuit_data(m, n, k, key=1234):
       """
       Generate random data for basis pursuit problem.
       
       Parameters:
           m: Number of measurements
           n: Signal dimension
           k: Sparsity level (number of nonzero entries)
           key: Random seed
       
       Returns:
           B: Measurement matrix (m x n)
           b: Observation vector (m,)
           z_star: Ground truth sparse signal
           x_star: Ground truth sparse signal
           B_big: Extended matrix [B, -B]
       """
       rng = np.random.default_rng(key)
       B = jnp.array(rng.standard_normal((m, n)))
       
       # Generate sparse ground truth
       z_star = jnp.zeros(n)
       support = rng.choice(n, size=k, replace=False)
       amplitudes = jnp.array(rng.standard_normal(k))
       z_star = z_star.at[support].set(amplitudes)
       
       # Compute observations
       b = B @ z_star
       
       # Extended matrix for squared formulation
       B_big = jnp.concatenate([B, -B], axis=1)
       
       # Compute optimal x from z_star
       z_star_pos = jnp.maximum(z_star, 0.0)
       z_star_neg = jnp.maximum(-z_star, 0.0)
       u1_star = jnp.sqrt(z_star_pos)
       u2_star = jnp.sqrt(z_star_neg)
       x_star = jnp.concatenate([u1_star, u2_star], axis=0)
       
       return B, b, z_star, x_star, B_big

   # Problem dimensions
   m, n, k = 200, 512, 10  # 200 measurements, 512 variables, 10 nonzeros

   # Generate data
   B, b, z_star, x_star, B_big = get_basis_pursuit_data(m, n, k)

   # Define objective function: ||x||^2
   def f1(x):
       return jnp.sum(x**2)

   # Define equality constraint: B_big @ (x^2) = b
   def h(x):
       return B_big @ (x**2) - b

   # Check optimal value
   f_star = f1(x_star)
   print(f"Optimal objective value: {f_star}")

   # Initial point
   rng = np.random.default_rng(1234)
   x0 = jnp.array(rng.standard_normal(2*n))
   f0 = f1(x0)
   print(f"Initial objective value: {f0}")

   # Define problem
   problem = pbalm.Problem(
       f1=f1,
       h=[h],
       jittable=True  # Enable JIT compilation
   )

   # Solve using PBALM
   tol = 1e-9   # small tolerance for high accuracy
   result = pbalm.solve(
       problem, 
       x0, 
       use_proximal=True,  # Use proximal ALM
       tol=tol, 
       max_iter=300, 
       alpha=10, 
       delta=1.0
   )

   # Results
   x_pbalm = result.x
   print(f"Solver status: {result.solve_status}")
   print(f"Final objective: {f1(x_pbalm)}")
   print(f"Relative error: {(f1(x_pbalm) - f_star) / (f0 - f_star):.6e}")

Running this example produces output similar to:

.. code-block:: text

    Optimal objective value: 9.393346945133704
    Initial objective value: 1045.3144493189802
    Initial point is not feasible. Finding a feasible point...
    Phase I optimization successful.
    iter  | f          | p. term    | total infeas | rho        | nu         | gamma     
    ------------------------------------------------------------------------------------------
    0     | 8.3323e+02 | nan        | 2.1452e-09 | 1.0000e-03 | 0.0000e+00 | 1.0000e-01
    19    | 9.3933e+00 | 5.1580e-21 | 8.2671e-10 | 6.1311e+09 | 0.0000e+00 | 6.1311e+11
    ------------------------------------------------------------------------------------------
    ...
    Solver status: Converged
    Final objective: 9.393346955103313
    Relative error: 9.623908e-12

Key Observations
----------------

1. **Phase I**: Since the random initial point is typically infeasible, PBALM first solves a Phase I problem to find a feasible starting point. User may skip this step by setting `start_feas=False` in `pbalm.solve()`.

2. **Nonconvex formulation**: The squared reformulation introduces nonconvexity, which PBALM handles effectively.

3. **Proximal ALM**: Using ``use_proximal=True`` improves convergence for this nonconvex problem.

4. **Sparse recovery**: The solution accurately recovers the sparse ground truth signal.

After solving, recover the original sparse signal:

.. code-block:: python

   # Split solution into u1 and u2
   u1 = x_pbalm[:n]
   u2 = x_pbalm[n:]

   # Recover z = u1^2 - u2^2
   z_recovered = u1**2 - u2**2

   # Compute recovery error
   z_error = jnp.linalg.norm(z_recovered - z_star) / jnp.linalg.norm(z_star)
   print(f"Signal recovery error: {z_error:.6e}")

   # Check sparsity
   threshold = 1e-5
   nnz = jnp.sum(jnp.abs(z_recovered) > threshold)
   print(f"Number of nonzeros in recovered signal: {nnz}")

References
----------
.. [BP1] Sahin, M. F., Alacaoglu, A., Latorre, F., & Cevher, V. (2019). An inexact augmented Lagrangian framework for nonconvex optimization with nonlinear constraints. Advances in Neural Information Processing Systems, 32.