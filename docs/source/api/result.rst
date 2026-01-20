Result Class
============

The ``Result`` class stores the output of the PBALM solver.

.. module:: pbalm.result
   :synopsis: Result object for PBALM solver

Class Definition
----------------

.. class:: Result(x, fp_res, kkt_res, total_infeas, f_hist, rho_hist, nu_hist, gamma_hist, prox_hist, solve_status, total_runtime, solve_runtime, grad_evals=None)

   Class to store the results of the PBALM solver.

   This object is returned by ``pbalm.solve()`` and contains the solution,
   convergence history, and solver diagnostics.

Attributes
----------

**Solution:**

.. attribute:: x
   :type: jax.numpy.ndarray

   The solution vector found by the solver. This is the optimal (or final)
   value of the decision variables.

**Solver Status:**

.. attribute:: solve_status
   :type: str or None

   Status indicating how the solver terminated. Possible values:

   - ``"Converged"``: Solver converged to tolerance
   - ``"Stopped"``: Solver stopped (user-defined condition)
   - ``"MaxRuntimeExceeded"``: Maximum runtime limit reached
   - ``"NaNOrInf"``: Numerical issues encountered (NaN or Inf values)
   - ``None``: Solver still running or status not set

**Convergence History:**

.. attribute:: f_hist
   :type: list of float

   History of objective function values at each outer iteration.

.. attribute:: fp_res
   :type: list of float

   History of fixed-point residuals. This measures the optimality
   condition for the composite problem.

.. attribute:: kkt_res
   :type: list of float

   History of KKT residuals, measuring overall optimality.

.. attribute:: total_infeas
   :type: list of float

   History of total constraint infeasibility at each iteration.
   Includes both equality and inequality constraint violations.

**Parameter History:**

.. attribute:: rho_hist
   :type: list of float or None

   History of penalty parameters :math:`\rho` for equality constraints.

.. attribute:: nu_hist
   :type: list of float or None

   History of penalty parameters :math:`\nu` for inequality constraints.

.. attribute:: gamma_hist
   :type: list of float

   History of proximal parameter :math:`\gamma` values.

.. attribute:: prox_hist
   :type: list of float

   History of proximal term values.

**Timing:**

.. attribute:: total_runtime
   :type: float

   Total runtime of the solver in seconds, including Phase I (if applicable).

.. attribute:: solve_runtime
   :type: float

   Runtime of the main solving phase in seconds (excluding Phase I).

**Evaluation Counts:**

.. attribute:: grad_evals
   :type: list of int or None

   Number of gradient evaluations at each iteration (if tracked).

Example Usage
-------------

Accessing basic solution information:

.. code-block:: python

   import pbalm

   result = pbalm.solve(problem, x0)

   # Get the solution
   x_opt = result.x
   print(f"Optimal x: {x_opt}")

   # Check solver status
   if result.solve_status == "Converged":
       print("Solver converged successfully!")
   else:
       print(f"Solver status: {result.solve_status}")

Analyzing convergence:

.. code-block:: python

   # Objective value convergence
   print(f"Initial objective: {result.f_hist[0]:.6f}")
   print(f"Final objective: {result.f_hist[-1]:.6f}")

   # Constraint satisfaction
   print(f"Initial infeasibility: {result.total_infeas[0]:.2e}")
   print(f"Final infeasibility: {result.total_infeas[-1]:.2e}")

   # Number of iterations
   print(f"Iterations: {len(result.f_hist)}")

Plotting convergence:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(2, 2, figsize=(12, 8))

   # Objective value
   axes[0, 0].semilogy(result.f_hist)
   axes[0, 0].set_xlabel('Iteration')
   axes[0, 0].set_ylabel('Objective')
   axes[0, 0].set_title('Objective Convergence')
   axes[0, 0].grid(True)

   # Infeasibility
   axes[0, 1].semilogy(result.total_infeas)
   axes[0, 1].set_xlabel('Iteration')
   axes[0, 1].set_ylabel('Infeasibility')
   axes[0, 1].set_title('Constraint Satisfaction')
   axes[0, 1].grid(True)

   # Fixed-point residual
   axes[1, 0].semilogy(result.fp_res)
   axes[1, 0].set_xlabel('Iteration')
   axes[1, 0].set_ylabel('Residual')
   axes[1, 0].set_title('Fixed-Point Residual')
   axes[1, 0].grid(True)

   # Penalty parameters
   if result.rho_hist[0] is not None:
       axes[1, 1].semilogy(result.rho_hist, label='rho (equality)')
   if result.nu_hist[0] is not None:
       axes[1, 1].semilogy(result.nu_hist, label='nu (inequality)')
   axes[1, 1].set_xlabel('Iteration')
   axes[1, 1].set_ylabel('Penalty')
   axes[1, 1].set_title('Penalty Parameters')
   axes[1, 1].legend()
   axes[1, 1].grid(True)

   plt.tight_layout()
   plt.savefig('convergence.png', dpi=150)
   plt.show()

Timing analysis:

.. code-block:: python

   print(f"Total runtime: {result.total_runtime:.4f} seconds")
   print(f"Solve runtime: {result.solve_runtime:.4f} seconds")
   
   if result.total_runtime > result.solve_runtime:
       phase_I_time = result.total_runtime - result.solve_runtime
       print(f"Phase I time: {phase_I_time:.4f} seconds")
