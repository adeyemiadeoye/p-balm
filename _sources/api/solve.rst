Solve Function
==============

The ``solve`` function is the main entry point for solving optimization problems with PBALM.

.. module:: pbalm.solution
   :synopsis: PBALM solver interface

Function Definition
-------------------

.. function:: solve(problem, x0, inner_solve_runner=None, lbda0=None, mu0=None, rho0=1e-3, nu0=1e-3, use_proximal=False, gamma0=1e-1, x_shapes=None, x_cumsizes=None, beta=0.5, alpha=10, delta=1.0, xi1=1.0, xi2=1.0, tol=1e-6, fp_tol=None, max_iter=1000, phase_I_tol=1e-7, start_feas=True, inner_solver=None, pa_direction=None, pa_solver_opts=None, verbosity=1, max_runtime=24.0, phi_strategy="pow", feas_reset_interval=None, no_reset=False, adaptive_fp_tol=False, max_iter_inner=1000)

   Solve the optimization problem using the Proximal Augmented Lagrangian Method (PBALM).

   :param problem: An instance of the ``Problem`` class defining the optimization problem.
   :type problem: pbalm.Problem
   :param x0: Initial guess for the decision variables.
   :type x0: jax.numpy.ndarray
   :returns: A ``Result`` object containing the solution and solver information.
   :rtype: pbalm.Result

Parameters
----------

**Required Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``problem``
     - Problem
     - The optimization problem to solve
   * - ``x0``
     - ndarray
     - Initial guess for decision variables

**Lagrange Multiplier Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``lbda0``
     - ndarray or None
     - Initial multipliers for equality constraints. If ``None``, randomly initialized.
   * - ``mu0``
     - ndarray or None
     - Initial multipliers for inequality constraints. If ``None``, randomly initialized.

**Penalty Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``rho0``
     - float
     - Initial penalty parameter for equality constraints. Default: ``1e-3``
   * - ``nu0``
     - float
     - Initial penalty parameter for inequality constraints. Default: ``1e-3``
   * - ``alpha``
     - float
     - Penalty growth exponent. Controls how quickly penalties increase. Default: ``10``
   * - ``beta``
     - float
     - Constraint satisfaction threshold (0 < beta < 1). Penalties increase when 
       constraint violation doesn't decrease by this factor. Default: ``0.5``
   * - ``xi1``
     - float
     - Penalty update scaling for equality constraints. Default: ``1.0``
   * - ``xi2``
     - float
     - Penalty update scaling for inequality constraints. Default: ``1.0``
   * - ``phi_strategy``
     - str
     - Strategy for minimum penalty floor: ``"pow"`` (default), ``"log"``, or ``"linear"``

**Proximal ALM Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``use_proximal``
     - bool
     - Enable proximal ALM variant. Recommended for nonconvex problems. Default: ``False``
   * - ``gamma0``
     - float
     - Initial proximal parameter. Default: ``0.1``
   * - ``delta``
     - float
     - Proximal parameter update factor. Default: ``1.0``

**Convergence Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``tol``
     - float
     - Convergence tolerance for KKT conditions. Default: ``1e-6``
   * - ``max_iter``
     - int
     - Maximum number of outer ALM iterations. Default: ``1000``
   * - ``max_runtime``
     - float
     - Maximum runtime in hours. Default: ``24.0``

**Inner Solver Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``inner_solve_runner``
     - object or None
     - Custom inner solver runner. If ``None``, uses default PANOC solver.
   * - ``inner_solver``
     - str or None
     - Name of inner solver (only ``"PANOC"`` currently supported)
   * - ``max_iter_inner``
     - int
     - Maximum iterations for inner solver. Default: ``1000``
   * - ``fp_tol``
     - float, callable, or None
     - Fixed-point tolerance for inner solver. Can be a constant or function of iteration.
   * - ``adaptive_fp_tol``
     - bool
     - Adaptively decrease inner solver tolerance. Default: ``False``
   * - ``pa_direction``
     - object or None
     - Direction method for PANOC (e.g., L-BFGS)
   * - ``pa_solver_opts``
     - dict or None
     - Additional options for PANOC solver

**Feasibility Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``start_feas``
     - bool
     - Solve Phase I to find feasible starting point. Default: ``True``
   * - ``phase_I_tol``
     - float
     - Tolerance for Phase I feasibility problem. Default: ``1e-7``
   * - ``feas_reset_interval``
     - int or None
     - Interval for resetting to a feasible point
   * - ``no_reset``
     - bool
     - Disable feasibility resets. Default: ``False``

**Structured Variable Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``x_shapes``
     - list or None
     - Shapes of decision variable blocks for structured problems
   * - ``x_cumsizes``
     - list or None
     - Cumulative sizes of decision variable blocks

**Output Parameters:**

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``verbosity``
     - int
     - Output level. 0: silent, ≥1: show progress. Default: ``1``

Example Usage
-------------

Basic usage:

.. code-block:: python

   import jax.numpy as jnp
   import pbalm

   # Define problem
   def f1(x):
       return jnp.sum(x**2)

   def h(x):
       return jnp.sum(x) - 1.0

   problem = pbalm.Problem(f1=f1, h=[h], jittable=True)
   x0 = jnp.zeros(10)

   # Solve with default settings
   result = pbalm.solve(problem, x0)

With proximal ALM and custom parameters:

.. code-block:: python

   result = pbalm.solve(
       problem,
       x0,
       use_proximal=True,     # Enable proximal ALM
       gamma0=0.1,            # Initial proximal parameter
       tol=1e-8,              # Tighter tolerance
       max_iter=500,          # More iterations
       alpha=5,               # Slower penalty growth
       verbosity=1            # Show progress
   )

With adaptive inner solver tolerance:

.. code-block:: python

   # Custom tolerance schedule
   def tol_schedule(k):
       return 0.1 / (k + 1)**2

   result = pbalm.solve(
       problem,
       x0,
       fp_tol=tol_schedule,
       adaptive_fp_tol=True
   )

Silent mode:

.. code-block:: python

   result = pbalm.solve(problem, x0, verbosity=0)

Solver Output
-------------

When ``verbosity=1``, the solver prints a table showing progress:

.. code-block:: text

   iter  | f          | p. term    | total infeas | rho        | nu         | gamma     
   ------------------------------------------------------------------------------------------
   0     | 1.0000e+00 | nan        | 5.0000e-01   | 1.0000e-03 | 1.0000e-03 | 1.0000e-01
   1     | 5.5000e-01 | 1.2345e-02 | 1.2000e-01   | 1.0000e-03 | 1.0000e-03 | 1.0000e-01
   ...

Columns:

- **iter**: Outer iteration number
- **f**: Objective function value
- **p. term**: Proximal term value
- **total infeas**: Total constraint infeasibility
- **rho**: Maximum penalty parameter for equality constraints
- **nu**: Maximum penalty parameter for inequality constraints  
- **gamma**: Proximal parameter
