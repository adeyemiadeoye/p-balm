Problem Class
=============

The ``Problem`` class defines the optimization problem to be solved by PBALM.

.. module:: pbalm.problem
   :synopsis: Problem definition for PBALM

Class Definition
----------------

.. class:: Problem(f1, f2=None, f2_lbda=0.0, h=None, g=None, f1_grad=None, jittable=True, callback=None)

   Defines a general optimization problem for PBALM.

   The problem has the form:

   .. math::

      \min_{x} \quad & f_1(x) + \text{f2\_lbda} \cdot f_2(x) \\
      \text{s.t.} \quad & h_i(x) = 0, \quad i = 1, \ldots, p \\
                        & g_j(x) \leq 0, \quad j = 1, \ldots, m

   :param f1: The smooth objective function :math:`f_1: \mathbb{R}^n \to \mathbb{R}`.
   :type f1: callable
   :param f2: Nonsmooth regularization function with a proximal operator.
               Uses regularizers from alpaqa, accessible via ``pbalm``. Examples include
               ``pbalm.L1Norm()``, ``pbalm.NuclearNorm()``, or ``pbalm.Box(lower, upper)``.
   :type f2: alpaqa regularizer or None
   :param f2_lbda: Regularization parameter :math:`\lambda` multiplying the
                    regularizer. Default is ``None``. When set alongside ``pbalm.L1Norm()``,
                    this value is prioritized for inner iterations. Can be a float or
                    a list (same length as optimization variable) for element-wise weights.
   :type f2_lbda: float or list
   :param h: List of equality constraint functions. Each function :math:`h_i`
             should return a scalar or array, and the constraint is :math:`h_i(x) = 0`.
   :type h: list of callables or None
   :param g: List of inequality constraint functions. Each function :math:`g_j`
             should return a scalar or array, and the constraint is :math:`g_j(x) \leq 0`.
   :type g: list of callables or None
   :param f1_grad: Custom gradient function for the smooth objective. If not provided,
                  automatic differentiation via JAX is used.
   :type f1_grad: callable or None
   :param jittable: If ``True``, enable JAX JIT compilation for all functions.
                    All provided functions must be JAX-compatible.
   :type jittable: bool
   :param callback: Optional callback function called at each outer iteration.
   :type callback: callable or None

Attributes
----------

.. attribute:: f1
   :type: callable

   The (JIT-compiled if ``jittable=True``) smooth objective function.

.. attribute:: f2
   :type: alpaqa regularizer or None

   The nonsmooth regularizer.

.. attribute:: f2_lbda
   :type: float or list

   The regularization parameter.

.. attribute:: h
   :type: callable or None

   The combined equality constraint function (set by ``solve()``).

.. attribute:: g
   :type: callable or None

   The combined inequality constraint function (set by ``solve()``).

.. attribute:: f1_grad
   :type: GradEvalCounter

   Wrapped gradient function with evaluation counting.

.. attribute:: h_grad
   :type: GradEvalCounter or None

   Jacobian of equality constraints (set by ``solve()``).

.. attribute:: g_grad
   :type: GradEvalCounter or None

   Jacobian of inequality constraints (set by ``solve()``).

.. attribute:: jittable
   :type: bool

   Whether JIT compilation is enabled.

.. attribute:: callback
   :type: callable or None

   The callback function.

.. attribute:: lbda_sizes
   :type: list

   Sizes of equality constraint multipliers (set by ``solve()``).

.. attribute:: mu_sizes
   :type: list

   Sizes of inequality constraint multipliers (set by ``solve()``).

Methods
-------

.. method:: reset_counters()

   Reset gradient evaluation counters for ``f_grad``, ``h_grad``, and ``g_grad``.

Example Usage
-------------

Basic problem with equality constraint:

.. code-block:: python

   import jax.numpy as jnp
   import pbalm

   def f1(x):
       return jnp.sum(x**2)

   def h(x):
       return jnp.sum(x) - 1.0

   problem = pbalm.Problem(f1=f1, h=[h], jittable=True)

Problem with L1 regularization:

.. code-block:: python

   def f1(x):
       return jnp.sum((x - 1)**2)

   # L1 regularization
   f2 = pbalm.L1Norm()
   
   problem = pbalm.Problem(
       f1=f1,
       f2=f2,
       f2_lbda=0.1,
       jittable=True
   )

Problem with callback:

.. code-block:: python

   def my_callback(iter, x, x_prev, lbda, mu, rho, nu, gamma_k, x0):
       print(f"Iteration {iter}: f1(x) = {f1(x):.6f}")

   problem = pbalm.Problem(
       f1=f1,
       h=[h],
       callback=my_callback,
       jittable=True
   )

Callback Signature
------------------

The callback function is called at each outer iteration with the following arguments:

.. code-block:: python

   def callback(iter, x, x_prev, lbda, mu, rho, nu, gamma_k, x0):
       """
       Callback function signature.

       Parameters:
           iter: Current iteration number (int)
           x: Current solution (jax.numpy.ndarray)
           x_prev: Previous solution (jax.numpy.ndarray)
           lbda: Current equality constraint multipliers (jax.numpy.ndarray or None)
           mu: Current inequality constraint multipliers (jax.numpy.ndarray or None)
           rho: Current equality penalty parameters (jax.numpy.ndarray or None)
           nu: Current inequality penalty parameters (jax.numpy.ndarray or None)
           gamma_k: Current proximal parameter (float)
           x0: Initial point (jax.numpy.ndarray)
       """
       pass
