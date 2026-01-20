User Guide
==========

This guide covers the detailed usage of **pbalm** for solving constrained optimization problems.

Defining Optimization Problems
------------------------------

The ``Problem`` Class
^^^^^^^^^^^^^^^^^^^^^

The core of **pbalm** is the ``Problem`` class, which defines the optimization problem:

.. code-block:: python

   import pbalm

   problem = pbalm.Problem(
       f1,             # Smooth objective function
       f2=None,        # Nonsmooth regularizer (optional)
       f2_lbda=0.0,    # Regularization parameter (float or list)
       h=None,         # List of equality constraints
       g=None,         # List of inequality constraints
       f1_grad=None,   # Custom gradient of f1 (optional)
       jittable=False, # Enable JAX JIT compilation
       callback=None   # Callback function (optional)
   )

Objective Function
^^^^^^^^^^^^^^^^^^

The objective function must accept a JAX array and return a scalar:

.. code-block:: python

   import jax.numpy as jnp

   def f1(x):
       return jnp.sum(x**2) + jnp.sin(x[0])

For best performance with JAX, avoid using NumPy functions directly; use ``jax.numpy`` instead.

Constraints
^^^^^^^^^^^

Constraints are defined as lists of functions:

**Equality constraints** :math:`h_i(x) = 0`:

.. code-block:: python

   def h1(x):
       return x[0] + x[1] - 1  # x[0] + x[1] = 1

   def h2(x):
       return x[0] * x[1] - 0.5  # x[0] * x[1] = 0.5

   problem = pbalm.Problem(f1=f1, h=[h1, h2])

**Inequality constraints** :math:`g_i(x) \leq 0`:

.. code-block:: python

   def g1(x):
       return x[0]**2 + x[1]**2 - 1  # x[0]^2 + x[1]^2 <= 1

   def g2(x):
       return -x[0]  # x[0] >= 0

   problem = pbalm.Problem(f1=f1, g=[g1, g2])

**Both equality and inequality constraints**:

.. code-block:: python

   problem = pbalm.Problem(f1=f1, h=[h1], g=[g1, g2])

Regularization
^^^^^^^^^^^^^^

For composite objectives :math:`f_1(x) + f_2(x)` where :math:`f_2` is nonsmooth:

.. code-block:: python

   # L1 regularization: f_2(x) = lambda * ||x||_1
   f2 = pbalm.L1Norm()
   f2_lbda = 0.1

   problem = pbalm.Problem(
       f1=f1, 
       h=[h1],
       f2=f2,
       f2_lbda=f2_lbda
   )

**Supported regularizers** from ``pbalm`` include:

- ``pbalm.L1Norm(f2_lbda)``: L1 norm (induces sparsity)
- ``pbalm.NuclearNorm(f2_lbda, rows, cols)``: Nuclear norm (induces low-rank)
- ``pbalm.Box(lower, upper)``: Box constraints (must use numpy arrays, e.g., ``pbalm.Box(lower=np.asarray([0.1]), upper=np.asarray([1.0]))``).

.. note::

   When using L1 regularization, ``pbalm.L1Norm()`` must be set as ``f2``. If ``f2_lbda`` is however provided, its value is prioritized for the inner iterations. This is useful when you want to specify ``f2_lbda`` as a list (same length as the optimization variable) for element-wise L1 weights, or combine L1 with other regularizers like box constraints.

Solving Problems
----------------

The ``solve`` Function
^^^^^^^^^^^^^^^^^^^^^^

Once a problem is defined, solve it using ``pbalm.solve``:

.. code-block:: python

   result = pbalm.solve(
       problem,           # Problem instance
       x0,                # Initial point
       tol=1e-6,          # Convergence tolerance
       max_iter=1000,     # Maximum outer iterations
       verbosity=1        # Output level
   )

Key Solver Parameters
^^^^^^^^^^^^^^^^^^^^^

**Basic parameters:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``x0``
     - Initial guess for decision variables (JAX array)
   * - ``tol``
     - Convergence tolerance for KKT conditions
   * - ``max_iter``
     - Maximum number of outer ALM iterations
   * - ``verbosity``
     - Output level (0: silent, 1: normal)

**Proximal ALM parameters:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``use_proximal``
     - Enable proximal ALM (default: ``True``)
   * - ``gamma0``
     - Initial proximal parameter (default: ``0.1``)
   * - ``delta``
     - Proximal parameter update factor

**Penalty parameters:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``rho0``
     - Initial penalty for equality constraints
   * - ``nu0``
     - Initial penalty for inequality constraints
   * - ``alpha``
     - Penalty growth exponent
   * - ``beta``
     - Constraint satisfaction threshold

**Inner solver parameters:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``max_iter_inner``
     - Maximum iterations for inner solver
   * - ``fp_tol``
     - Fixed-point tolerance for inner solver
   * - ``adaptive_fp_tol``
     - Adaptively decrease inner tolerance
   * - ``pa_solver_opts``
     - Options passed to PANOC solver
   * - ``pa_direction``
     - Direction method for PANOC

**Feasibility parameters:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``start_feas``
     - Find feasible point first (default: ``True``)
   * - ``phase_I_tol``
     - Tolerance for Phase I problem
   * - ``no_reset``
     - Disable feasibility reset

Working with Results
--------------------

The ``Result`` Class
^^^^^^^^^^^^^^^^^^^^

The ``solve`` function returns a ``Result`` object:

.. code-block:: python

   result = pbalm.solve(problem, x0)

   # Solution
   x_opt = result.x

   # Solver status
   status = result.solve_status
   # Possible values: 'Converged', 'Stopped', 'MaxRuntimeExceeded', 'NaNOrInf'

   # History
   f_values = result.f_hist           # Objective values per iteration
   infeas = result.total_infeas       # Constraint violation per iteration
   prox_res = result.fp_res    # Fixed-point residual

   # Timing
   total_time = result.total_runtime  # Total runtime in seconds
   solve_time = result.solve_runtime  # Solving phase runtime

Analyzing Convergence
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt

   # Plot objective value convergence
   plt.figure(figsize=(10, 4))
   
   plt.subplot(1, 2, 1)
   plt.semilogy(result.f_hist)
   plt.xlabel('Iteration')
   plt.ylabel('Objective value')
   plt.title('Objective Convergence')
   
   plt.subplot(1, 2, 2)
   plt.semilogy(result.total_infeas[1:])  # Skip first value (if phase I was used)
   plt.xlabel('Iteration')
   plt.ylabel('Infeasibility')
   plt.title('Constraint Satisfaction')
   
   plt.tight_layout()
   plt.show()

Advanced Features
-----------------

JIT Compilation
^^^^^^^^^^^^^^^

For better performance, enable JIT compilation with JAX-compatible functions:

.. code-block:: python

   problem = pbalm.Problem(
       f=f,
       h=[h],
       jittable=True  # Enable JIT compilation
   )

.. note::

   When ``jittable=True``, all functions (``f``, ``h``, ``g``) must be compatible
   with JAX transformations. Avoid side effects and ensure all operations use
   ``jax.numpy`` instead of ``numpy``.

Custom Gradient
^^^^^^^^^^^^^^^

If you have an efficient custom gradient implementation:

.. code-block:: python

   def f1(x):
       return jnp.sum(x**2)

   def f1_grad(x):
       return 2 * x

   problem = pbalm.Problem(f1=f1, f1_grad=f1_grad, jittable=True)

Callback Function
^^^^^^^^^^^^^^^^^

Monitor optimization progress with a callback:

.. code-block:: python

   def my_callback(iter, x, x_prev, lbda, mu, rho, nu, gamma_k, x0):
       print(f"Iteration {iter}: ||x|| = {jnp.linalg.norm(x):.6f}")

   problem = pbalm.Problem(f1=f1, h=[h], callback=my_callback)

Custom Inner Solver
^^^^^^^^^^^^^^^^^^^

For advanced users, a custom inner solver can be provided:

.. code-block:: python

   from pbalm.inner_solvers.inner_solvers import PALMInnerTrainer

   def custom_train_fun(palm_obj_fun, x, max_iter, tol):
       # Your custom optimization logic
       # Must return (x_optimal, state),
       # where state is a dict with keys: "obj_grad_evals", "fp_res", "obj_val", "reg_val", "status"
       ...
       return x_new, state

   custom_runner = PALMInnerTrainer(custom_train_fun)
   
   result = pbalm.solve(
       problem, 
       x0, 
       inner_solve_runner=custom_runner
   )

Structured Variables
^^^^^^^^^^^^^^^^^^^^

For problems with structured decision variables (e.g., matrices):

.. code-block:: python

   import numpy as np
   from pbalm.utils.utils import params_flatten, params_shape

   # Define variables as a list
   A = jnp.zeros((3, 3))
   b = jnp.zeros(3)
   params = [A, b]

   # Get shape information
   shapes, cumsizes = params_shape(params)

   # Flatten for the solver
   x0 = params_flatten(params)

   # Solve with shape information
   result = pbalm.solve(
       problem, 
       x0, 
       x_shapes=shapes, 
       x_cumsizes=cumsizes
   )

Tips and Best Practices
-----------------------

1. **Start with ``use_proximal=True``** for nonconvex problems
2. **Enable ``jittable=True``** when functions are JAX-compatible for faster execution
3. **Use ``start_feas=True``** (default) for highly constrained problems
4. **Tune ``alpha``** for better penalty parameter growth
5. **Monitor convergence** using the history attributes in ``Result``
6. **Set ``verbosity=0``** for production runs to disable output
