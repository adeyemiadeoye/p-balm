Utilities
=========

The utilities module provides helper functions for working with structured variables and tracking evaluations.

.. module:: pbalm.utils.utils
   :synopsis: Utility functions for PBALM

Structured Variable Utilities
-----------------------------

These functions help manage decision variables that have structure (e.g., matrices, multiple blocks).

params_flatten
^^^^^^^^^^^^^^

.. function:: params_flatten(params)

   Flatten a list of arrays into a single 1D vector.

   :param params: List of arrays (can be scalars, vectors, or matrices)
   :type params: list
   :returns: Flattened 1D array
   :rtype: jax.numpy.ndarray

   **Example:**

   .. code-block:: python

      import jax.numpy as jnp
      from pbalm.utils.utils import params_flatten

      # Multiple variable blocks
      A = jnp.zeros((3, 3))  # 9 elements
      b = jnp.zeros(3)       # 3 elements
      c = 1.0                # 1 element

      params = [A, b, c]
      x = params_flatten(params)
      print(x.shape)  # (13,)

params_shape
^^^^^^^^^^^^

.. function:: params_shape(params)

   Get shapes and cumulative sizes for a list of parameters.

   :param params: List of arrays
   :type params: list
   :returns: Tuple of (shapes, cumsizes) where shapes is a list of shapes
             and cumsizes is a numpy array of cumulative sizes
   :rtype: tuple

   **Example:**

   .. code-block:: python

      from pbalm.utils.utils import params_shape

      params = [A, b, c]
      shapes, cumsizes = params_shape(params)
      
      print(shapes)    # [(3, 3), (3,), ()]
      print(cumsizes)  # [ 9 12 13]

params_unflatten
^^^^^^^^^^^^^^^^

.. function:: params_unflatten(params_flattened, shapes, cumsizes)

   Unflatten a 1D vector back into structured parameters.

   :param params_flattened: Flattened parameter vector
   :type params_flattened: jax.numpy.ndarray
   :param shapes: List of shapes (from ``params_shape``)
   :type shapes: list
   :param cumsizes: Cumulative sizes (from ``params_shape``)
   :type cumsizes: numpy.ndarray
   :returns: List of arrays with original shapes
   :rtype: list

   **Example:**

   .. code-block:: python

      from pbalm.utils.utils import params_unflatten

      # After optimization
      x_opt = result.x
      
      # Recover structured parameters
      params_opt = params_unflatten(x_opt, shapes, cumsizes)
      A_opt, b_opt, c_opt = params_opt

Complete Workflow Example
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   import pbalm
   from pbalm.utils.utils import params_flatten, params_shape, params_unflatten

   # Define structured variables
   W = jnp.zeros((5, 3))  # Weight matrix
   b = jnp.zeros(5)       # Bias vector

   # Get structure info
   params = [W, b]
   shapes, cumsizes = params_shape(params)

   # Flatten for the solver
   x0 = params_flatten(params)

   # Define objective using flattened variables
   def f(x):
       # Unflatten inside objective
       W, b = params_unflatten(x, shapes, cumsizes)
       return jnp.sum(W**2) + jnp.sum(b**2)

   # Solve
   problem = pbalm.Problem(f=f, jittable=True)
   result = pbalm.solve(
       problem, 
       x0,
       x_shapes=shapes,
       x_cumsizes=cumsizes
   )

   # Recover structured solution
   W_opt, b_opt = params_unflatten(result.x, shapes, cumsizes)

Gradient Evaluation Counter
---------------------------

GradEvalCounter
^^^^^^^^^^^^^^^

.. class:: GradEvalCounter(fn)

   Wrapper class to count function evaluations.

   This is used internally to track the number of gradient evaluations
   during optimization.

   :param fn: Function to wrap
   :type fn: callable

   **Attributes:**

   .. attribute:: fn
      :type: callable

      The wrapped function.

   .. attribute:: count
      :type: int

      Number of times the function has been called.

   **Methods:**

   .. method:: __call__(*args, **kwargs)

      Call the wrapped function and increment counter.

   .. method:: reset()

      Reset the evaluation counter to zero.

   **Example:**

   .. code-block:: python

      import jax
      from pbalm.utils.utils import GradEvalCounter

      def my_function(x):
          return x**2

      # Wrap with counter
      counted_fn = GradEvalCounter(my_function)

      # Use the function
      for i in range(10):
          y = counted_fn(i)

      print(f"Function was called {counted_fn.count} times")  # 10

      # Reset counter
      counted_fn.reset()
      print(f"After reset: {counted_fn.count}")  # 0

Penalty Update
--------------

update_penalties
^^^^^^^^^^^^^^^^

.. function:: update_penalties(lbda_sizes, mu_sizes, rho, nu, rho0, nu0, E_x, prev_E, h_x, prev_h, beta, xi1, xi2, phi_i)

   Update penalty parameters based on constraint satisfaction progress.

   This function implements the adaptive penalty update rule from the PBALM algorithm.
   Penalties are increased when constraints are not being satisfied sufficiently.

   :param lbda_sizes: Sizes of equality constraint multipliers
   :type lbda_sizes: list
   :param mu_sizes: Sizes of inequality constraint multipliers
   :type mu_sizes: list
   :param rho: Current equality penalty parameters
   :type rho: jax.numpy.ndarray
   :param nu: Current inequality penalty parameters
   :type nu: jax.numpy.ndarray
   :param rho0: Initial equality penalty
   :type rho0: float
   :param nu0: Initial inequality penalty
   :type nu0: float
   :param E_x: Current inequality constraint term
   :type E_x: jax.numpy.ndarray
   :param prev_E: Previous inequality constraint term
   :type prev_E: jax.numpy.ndarray
   :param h_x: Current equality constraint values
   :type h_x: jax.numpy.ndarray
   :param prev_h: Previous equality constraint values
   :type prev_h: jax.numpy.ndarray
   :param beta: Satisfaction threshold (0 < beta < 1)
   :type beta: float
   :param xi1: Equality penalty scaling factor
   :type xi1: float
   :param xi2: Inequality penalty scaling factor
   :type xi2: float
   :param phi_i: Minimum penalty floor value
   :type phi_i: float
   :returns: Updated penalty parameters (rho_new, nu_new)
   :rtype: tuple

   The update rule is:

   .. math::

      \rho_i^{k+1} = \begin{cases}
         \max(\xi_1 \rho_i^k, \rho_0 \cdot \phi(k)) & \text{if } \|h_i(x^{k+1})\| > \beta \|h_i(x^k)\| \\
         \rho_i^k & \text{otherwise}
      \end{cases}
