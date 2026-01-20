Quick Start Guide
=================

This guide will help you get started with **pbalm** by walking through a simple example.

Basic Workflow
--------------

The typical workflow for using **pbalm** consists of three steps:

1. **Define the problem**: Create objective and constraint functions
2. **Create a Problem instance**: Use ``pbalm.Problem`` to define the optimization problem
3. **Solve**: Call ``pbalm.solve`` to find the solution

Minimal Example
---------------

Here's a simple example of solving a constrained optimization problem:

.. code-block:: python

   import jax.numpy as jnp
   import pbalm

   # Step 1: Define the objective function
   def f1(x):
       return (x[0] - 1)**2 + (x[1] - 2)**2

   # Step 2: Define constraints
   # Equality constraint: x[0] + x[1] = 2
   def h(x):
       return x[0] + x[1] - 2

   # Step 3: Create the problem
   problem = pbalm.Problem(f1=f1, h=[h], jittable=True)

   # Step 4: Set initial point and solve
   x0 = jnp.array([0.0, 0.0])
   result = pbalm.solve(problem, x0, tol=1e-6)

   # Access the solution
   print(f"Solution: {result.x}")
   print(f"Objective value: {f1(result.x)}")

Understanding the Output
------------------------

The ``solve`` function returns a ``Result`` object containing:

- ``x``: The solution vector
- ``f_hist``: History of objective values
- ``total_infeas``: Total constraint infeasibility at each iteration
- ``solve_status``: Status of the solver (e.g., 'Converged', 'MaxRuntimeExceeded')
- ``solve_runtime``: Time taken to solve

.. code-block:: python

   # Check solver status
   print(f"Status: {result.solve_status}")
   print(f"Runtime: {result.solve_runtime:.4f} seconds")
   print(f"Final infeasibility: {result.total_infeas[-1]:.2e}")

Adding Inequality Constraints
-----------------------------

To add inequality constraints of the form :math:`g(x) \leq 0`:

.. code-block:: python

   import jax.numpy as jnp
   import pbalm

   def f1(x):
       return x[0]**2 + x[1]**2

   # Inequality constraint: x[0] + x[1] >= 1 --> -(x[0] + x[1] - 1) <= 0
   def g(x):
       return -(x[0] + x[1] - 1)

   problem = pbalm.Problem(f1=f1, g=[g], jittable=True)
   x0 = jnp.array([0.0, 0.0])
   
   result = pbalm.solve(problem, x0, tol=1e-6)
   print(f"Solution: {result.x}")

Using Proximal ALM
------------------

For better convergence on some problems, enable the proximal ALM variant:

.. code-block:: python

   result = pbalm.solve(
       problem, 
       x0, 
       use_proximal=True,  # Enable proximal ALM
       gamma0=0.1,         # Initial proximal parameter
       tol=1e-6
   )

Adding Regularization
---------------------

You can add nonsmooth regularization terms using regularziers from **alpaqa**, accessible via ``pbalm``:

.. code-block:: python

   import jax.numpy as jnp
   import pbalm

   def f1(x):
       return jnp.sum(x**2)

   def h(x):
       return jnp.sum(x) - 1

   # L1 regularization with lambda = 0.1
   f2 = pbalm.L1Norm()
   f2_lbda = 0.1  # can be a list for element-wise weights (same length as x)

   problem = pbalm.Problem(
       f1=f1, 
       h=[h], 
       f2=f2,
       f2_lbda=f2_lbda,
       jittable=True
   )

   x0 = jnp.ones(10)
   result = pbalm.solve(problem, x0, tol=1e-6)

Box Constraints
---------------

Box constraints can be handled via ``pbalm.Box``. Note that ``pbalm.Box`` requires numpy arrays:

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np
   import pbalm

   def f1(x):
       return jnp.sum((x - 1)**2)

   # Box constraints: 0 <= x <= 2 (must use numpy arrays)
   n = 5
   f2 = pbalm.Box(
       lower=np.zeros(n), 
       upper=2*np.ones(n)
   )

   problem = pbalm.Problem(f1=f1, f2=f2, jittable=True)
   x0 = jnp.zeros(n)
   
   result = pbalm.solve(problem, x0, tol=1e-6)

Controlling Verbosity
---------------------

Control the amount of output with the ``verbosity`` parameter:

.. code-block:: python

   # Silent mode
   result = pbalm.solve(problem, x0, verbosity=0)

   # Normal output (default)
   result = pbalm.solve(problem, x0, verbosity=1)

Next Steps
----------

- See :doc:`usage` for more detailed usage information
- Check out :doc:`examples/index` for more comprehensive examples
- Refer to :doc:`api/index` for the complete API reference
