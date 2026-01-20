Sparse Regression
=================

This example demonstrates solving sparse regression problems with **pbalm**, including LASSO-type formulations with additional constraints.

Problem Background
------------------

In sparse regression, we seek a sparse coefficient vector :math:`\beta` that explains the relationship between features :math:`X` and response :math:`y`:

.. math::

   \min_{\beta} \quad \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1

The :math:`\ell_1` penalty encourages sparsity in the solution.

Constrained LASSO
-----------------

Sometimes we need additional constraints on the coefficients. Consider:

.. math::

   \min_{\beta} \quad & \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1 \\
   \text{s.t.} \quad & \mathbf{1}^T \beta = 1 \quad \text{(coefficients sum to 1)} \\
                     & \beta \geq 0 \quad \text{(non-negativity)}

This constrained formulation is useful in applications like portfolio optimization or mixture models.

Implementation
--------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import pbalm
   import numpy as np

   jax.config.update('jax_platform_name', 'cpu')
   jax.config.update("jax_enable_x64", True)

   # Generate synthetic regression data
   def generate_data(n_samples, n_features, n_nonzero, noise_std=0.1, seed=42):
       """Generate sparse regression data."""
       rng = np.random.default_rng(seed)
       
       # Design matrix
       X = rng.standard_normal((n_samples, n_features))
       X = jnp.array(X)
       
       # True sparse coefficients (non-negative, sum to 1)
       beta_true = jnp.zeros(n_features)
       support = rng.choice(n_features, size=n_nonzero, replace=False)
       values = rng.uniform(0.1, 1.0, size=n_nonzero)
       values = values / values.sum()  # Normalize to sum to 1
       beta_true = beta_true.at[support].set(jnp.array(values))
       
       # Response with noise
       y = X @ beta_true + noise_std * jnp.array(rng.standard_normal(n_samples))
       
       return X, y, beta_true

   # Problem dimensions
   n_samples = 200
   n_features = 100
   n_nonzero = 10
   lmbda = 0.01  # Regularization parameter

   # Generate data
   X, y, beta_true = generate_data(n_samples, n_features, n_nonzero)

   print(f"True number of nonzeros: {jnp.sum(beta_true > 0)}")
   print(f"True coefficients sum: {jnp.sum(beta_true):.4f}")

   # Define objective (smooth part)
   def f1(beta):
       residual = y - X @ beta
       return 0.5 / n_samples * jnp.sum(residual**2)

   # Equality constraint: sum(beta) = 1
   def h(beta):
       return jnp.sum(beta) - 1.0

   # Inequality constraint: beta >= 0
   def g(beta):
       return -beta

   # L1 regularization
   f2 = pbalm.L1Norm(lmbda)

   # Create problem
   problem = pbalm.Problem(
       f1=f1,
       h=[h],
       g=[g],
       f2=f2,
       jittable=True
   )

   # Initial point (uniform)
   beta0 = jnp.ones(n_features) / n_features

   # Solve
   result = pbalm.solve(
       problem,
       beta0,
       use_proximal=True,
       tol=1e-6,
       max_iter=500,
       alpha=5,
       verbosity=1
   )

   # Analyze solution
   beta_hat = result.x
   threshold = 1e-5

   print("\n" + "="*50)
   print("Results")
   print("="*50)
   print(f"Solver status: {result.solve_status}")
   print(f"Coefficients sum: {jnp.sum(beta_hat):.6f}")
   print(f"Min coefficient: {jnp.min(beta_hat):.6e}")
   print(f"Number of nonzeros: {jnp.sum(jnp.abs(beta_hat) > threshold)}")
   print(f"True nonzeros: {jnp.sum(beta_true > 0)}")

   # Prediction error
   y_pred = X @ beta_hat
   mse = jnp.mean((y - y_pred)**2)
   print(f"MSE: {mse:.6f}")

   # Support recovery
   support_true = set(jnp.where(beta_true > 0)[0].tolist())
   support_hat = set(jnp.where(jnp.abs(beta_hat) > threshold)[0].tolist())
   
   precision = len(support_true & support_hat) / max(len(support_hat), 1)
   recall = len(support_true & support_hat) / len(support_true)
   
   print(f"Support precision: {precision:.4f}")
   print(f"Support recall: {recall:.4f}")

Elastic Net with Box Constraints
--------------------------------

An elastic net variant with box constraints on the coefficients:

.. math::

   \min_{\beta} \quad & \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \frac{\lambda_2}{2} \|\beta\|_2^2 \\
   \text{s.t.} \quad & -1 \leq \beta_i \leq 1

Implementation
^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np
   import pbalm

   # Configure JAX
   import jax
   jax.config.update('jax_platform_name', 'cpu')
   jax.config.update("jax_enable_x64", True)

   # Problem parameters
   lambda1 = 0.1  # L1 penalty
   lambda2 = 0.05  # L2 penalty

   # Objective (smooth part including L2)
   def f1_elastic(beta):
       residual = y - X @ beta
       return 0.5 / n_samples * jnp.sum(residual**2) + 0.5 * lambda2 * jnp.sum(beta**2)

   # Box constraints via pbalm.Box (must use numpy arrays)
   n = n_features
   f2_box = pbalm.Box(
       lower=-np.ones(n),
       upper=np.ones(n)
   )

   # Create problem with L1 regularization and box constraints
   problem_elastic = pbalm.Problem(
       f1=f1_elastic,
       f2=f2_box,  # Box constraints as regularizer
       f2_lbda=lambda1, # L1 penalty
       jittable=True
   )

   beta0 = jnp.zeros(n_features)
   
   result_elastic = pbalm.solve(
       problem_elastic,
       beta0,
       tol=1e-6,
       max_iter=300
   )

   print(f"Solution range: [{jnp.min(result_elastic.x):.4f}, {jnp.max(result_elastic.x):.4f}]")

Group Sparse Regression
-----------------------

When features are organized into groups, we may want group-level sparsity:

.. math::

   \min_{\beta} \quad & \frac{1}{2n} \|y - X\beta\|_2^2 \\
   \text{s.t.} \quad & \sum_{g=1}^{G} \mathbb{1}[\|\beta_g\| > 0] \leq k

This can be approximated using squared reformulations similar to basis pursuit.

Example with Squared Reformulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   import pbalm
   import numpy as np

   # Configure JAX
   import jax
   jax.config.update('jax_platform_name', 'cpu')
   jax.config.update("jax_enable_x64", True)

   # Generate grouped data
   n_samples = 100
   n_groups = 20
   group_size = 5
   n_features = n_groups * group_size
   n_active_groups = 3

   rng = np.random.default_rng(123)
   X = jnp.array(rng.standard_normal((n_samples, n_features)))

   # True coefficients with group sparsity
   beta_true = jnp.zeros(n_features)
   active_groups = rng.choice(n_groups, size=n_active_groups, replace=False)
   for g in active_groups:
       start = g * group_size
       end = start + group_size
       beta_true = beta_true.at[start:end].set(
           jnp.array(rng.standard_normal(group_size))
       )

   y = X @ beta_true + 0.1 * jnp.array(rng.standard_normal(n_samples))

   # Squared reformulation: beta = u^2 - v^2 for signed coefficients
   # For simplicity, assume non-negative coefficients: beta = u^2

   def f1_group(u):
       beta = u**2
       residual = y - X @ beta
       return 0.5 / n_samples * jnp.sum(residual**2) + 0.01 * jnp.sum(u**2)

   problem_group = pbalm.Problem(f1=f1_group, jittable=True)

   u0 = jnp.ones(n_features) * 0.1
   
   result_group = pbalm.solve(
       problem_group,
       u0,
       tol=1e-5,
       max_iter=200
   )

   beta_recovered = result_group.x**2
   beta_hat = jnp.where(beta_recovered > 1e-5, beta_recovered, 0.0)
   print(f"Number of near-zero coefficients: {jnp.sum(beta_hat < 1e-5)}")

Visualization
-------------

Plot the comparison between true and estimated coefficients:

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.figure(figsize=(12, 4))

   plt.subplot(1, 2, 1)
   plt.stem(beta_true, linefmt='b-', markerfmt='bo', basefmt='k-', label='True')
   plt.xlabel('Feature index')
   plt.ylabel('Coefficient')
   plt.title('True Coefficients')
   plt.legend()

   plt.subplot(1, 2, 2)
   plt.stem(beta_hat, linefmt='r-', markerfmt='ro', basefmt='k-', label='Estimated')
   plt.xlabel('Feature index')
   plt.ylabel('Coefficient')
   plt.title('Estimated Coefficients')
   plt.legend()

   plt.tight_layout()
   plt.savefig('sparse_regression_comparison.png', dpi=150)
   plt.show()
