.. pbalm documentation master file, created by
   sphinx-quickstart on Wed Jan 14 19:43:06 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pbalm
=====

**pbalm** implements a proximal augmented Lagrangian method for solving (nonconvex) nonlinear programming (NLP) problems of the general form:

.. math::

   \min_{x} \quad & f_1(x) + f_2(x) \\
   \text{s.t.} \quad & g(x) \leq 0 \\
                     & h(x) = 0

where:

- :math:`f_1: \mathbb{R}^n \to \mathbb{R}` is a smooth (possibly nonconvex) function
- :math:`f_2: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}` is a possibly nonsmooth but prox-friendly function
- :math:`g: \mathbb{R}^n \to \mathbb{R}^m` defines smooth inequality constraints
- :math:`h: \mathbb{R}^n \to \mathbb{R}^p` defines smooth equality constraints

The algorithm is based on the paper [PBALM]_, and uses JAX for automatic differentiation and JIT compilation.

Key Features
------------

- **Nonconvex optimization**: Handles nonconvex objective functions and constraints
- **Composite objectives**: Supports smooth + nonsmooth composite objective functions
- **Flexible constraints**: Both equality and inequality constraints
- **JAX-based**: Uses JAX for automatic differentiation and JIT compilation

Quick Example
-------------

.. code-block:: python

   import jax.numpy as jnp
   import pbalm

   # smooth part of the objective
   def f1(x):
      return jnp.sum(x**2)

   # L1 regularization (nonsmooth)
   lbda = 0.1
   f2 = pbalm.L1Norm(lbda)

   # inequality constraint g_j(x) <= 0; j=1
   def g_1(x):
      return x[0] - 0.8

   # equality constraints h_i(x) = 0; i=1,2
   def h_1(x):
      return x[0] + x[1] - 1.0

   def h_2(x):
      return x[1] * x[2] - 2.0

   x0 = jnp.array([1.0, 1.0, 2.0])

   # create problem and solve
   problem = pbalm.Problem(f1=f1, f2=f2, g=[g_1], h=[h_1, h_2])
   result = pbalm.solve(problem, x0=x0, tol=1e-6)

   print(f"Solution: {result.x}")


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
----------

.. [PBALM] Adeoye, A. D., Latafat, P., & Bemporad, A. (2025). `A proximal augmented Lagrangian method for nonconvex optimization with equality and inequality constraints <https://arxiv.org/abs/2509.02894>`_. arXiv preprint arXiv:2509.02894.