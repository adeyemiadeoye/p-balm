Inner Solvers
=============

The inner solvers module provides the optimization routines used to solve the augmented Lagrangian subproblems.

.. module:: pbalm.inner_solvers.inner_solvers
   :synopsis: Inner solver implementations for PBALM

Overview
--------

At each outer iteration, PBALM needs to (approximately) minimize the augmented Lagrangian:

.. math::

   x^{k+1} \approx \arg\min_x \mathcal{L}_{\rho}(x, \lambda^k, \mu^k)

This is handled by inner solvers that support composite optimization (smooth + nonsmooth terms).

PALMInnerTrainer
----------------

.. class:: PALMInnerTrainer(train_fun)

   Class to run PALM inner solver.

   :param train_fun: Function to run the inner optimization.
   :type train_fun: callable

   The ``train_fun`` should have the signature:

   .. code-block:: python

      def train_fun(palm_obj_fun, x, max_iter, tol):
          """
          Parameters:
              palm_obj_fun: Augmented Lagrangian objective function
              x: Initial point
              max_iter: Maximum iterations
              tol: Convergence tolerance
          
          Returns:
              x_new: Optimized point
              state: Solver state/statistics (a dict with keys: "obj_grad_evals", "fp_res", "obj_val", "reg_val", "status")
          """
          ...
          return x_new, state

PaProblem
---------

.. class:: PaProblem(f, x0, reg=None, lbda=None, solver_opts=None, tol=1e-9, max_iter=2000, direction=None, jittable=True)

   Internal problem class for the PANOC solver from Alpaqa.

   :param f: Objective function
   :type f: callable
   :param x0: Initial point (used for problem dimensions)
   :type x0: jax.numpy.ndarray
   :param reg: Regularizer (supports ``pbalm.L1Norm()``, ``pbalm.NuclearNorm()``, ``pbalm.Box(lower, upper)``)
   :type reg: alpaqa regularizer or None
   :param lbda: L1 regularization weights
   :type lbda: float, list, or None
   :param solver_opts: Options for PANOC solver
   :type solver_opts: dict or None
   :param tol: Convergence tolerance
   :type tol: float
   :param max_iter: Maximum iterations
   :type max_iter: int
   :param direction: Direction method for PANOC
   :type direction: alpaqa direction or None
   :param jittable: Enable JIT compilation
   :type jittable: bool

   **Methods:**

   .. method:: eval_objective(x)

      Evaluate the objective function at ``x``.

   .. method:: eval_objective_gradient(x, grad_f)

      Evaluate the gradient and store in ``grad_f``.

get_solver_run
--------------

.. function:: get_solver_run(reg=None, lbda=None, solver_opts=None, direction=None, jittable=True)

   Factory function to create the default inner solver runner.

   :param reg: Regularizer for the problem
   :type reg: alpaqa regularizer or None
   :param lbda: L1 regularization weights
   :type lbda: float, list, or None
   :param solver_opts: Options for PANOC solver
   :type solver_opts: dict or None
   :param direction: Direction method (default: L-BFGS with memory 20)
   :type direction: alpaqa direction or None
   :param jittable: Enable JIT compilation
   :type jittable: bool
   :returns: A ``PALMInnerTrainer`` instance
   :rtype: PALMInnerTrainer

phase_I_optim
-------------

.. function:: phase_I_optim(x0, h, g, reg, lbda0, mu0, alpha=20, gamma0=1e-8, tol=1e-7, max_iter=500, inner_solver="PANOC")

   Solve the Phase I feasibility problem to find an initial feasible point.

   When starting from an infeasible point, this function finds a point that
   satisfies (or nearly satisfies) the constraints.

   :param x0: Initial (infeasible) point
   :type x0: jax.numpy.ndarray
   :param h: Equality constraint functions (list)
   :type h: list of callables or None
   :param g: Inequality constraint functions (list)
   :type g: list of callables or None
   :param reg: Regularizer
   :type reg: alpaqa regularizer or None
   :param lbda0: Initial equality multipliers
   :type lbda0: jax.numpy.ndarray or None
   :param mu0: Initial inequality multipliers
   :type mu0: jax.numpy.ndarray or None
   :param alpha: Penalty growth parameter
   :type alpha: float
   :param gamma0: Initial proximal parameter
   :type gamma0: float
   :param tol: Feasibility tolerance
   :type tol: float
   :param max_iter: Maximum iterations
   :type max_iter: int
   :param inner_solver: Inner solver name
   :type inner_solver: str
   :returns: Feasible point
   :rtype: jax.numpy.ndarray
   :raises RuntimeError: If feasibility cannot be achieved

Custom Inner Solver Example
---------------------------

To use a custom inner solver:

.. code-block:: python

   from pbalm.inner_solvers.inner_solvers import PALMInnerTrainer
   import jax
   import jax.numpy as jnp

   def custom_gradient_descent(palm_obj_fun, x0, max_iter, tol):
       """Simple gradient descent inner solver."""
       grad_fn = jax.grad(palm_obj_fun)
       x = x0.copy()
       step_size = 0.01
       
       for i in range(max_iter):
           grad = grad_fn(x)
           x_new = x - step_size * grad
           
           if jnp.linalg.norm(grad) < tol:
               break
           x = x_new
       
       return x, {
                  "obj_grad_evals": i + 1,
                  "fp_res": jnp.linalg.norm(grad),
                  "obj_val": palm_obj_fun(x),
                  "reg_val": 0.0,  # assuming no regularization in this simple example
                  "status": "Converged" if jnp.linalg.norm(grad) < tol else "MaxIterReached"
                  }

   # Create custom trainer
   custom_runner = PALMInnerTrainer(custom_gradient_descent)

   # Use with solve
   result = pbalm.solve(
       problem, 
       x0, 
       inner_solve_runner=custom_runner
   )

PANOC Solver Options
--------------------

The default inner solver uses PANOC from Alpaqa. Common options:

.. code-block:: python

   import alpaqa as pa

   # Custom direction (L-BFGS with more memory)
   direction = pa.LBFGSDirection({"memory": 50})

   # Custom solver options
   solver_opts = {
       "max_iter": 5000,
       "stop_crit": pa.ProjGradUnitNorm,
   }

   result = pbalm.solve(
       problem,
       x0,
       pa_direction=direction,
       pa_solver_opts=solver_opts
   )

For more details on PANOC options, see the `Alpaqa documentation <https://github.com/kul-optec/alpaqa>`_.
