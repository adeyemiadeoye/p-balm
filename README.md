pbalm
======

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2509.02894" alt="arXiv" target="_blank"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
</p>

This repository implements a proximal augmented Lagrangian method for solving nonconvex structured nonlinear programming problems as described in the paper [*A proximal augmented Lagrangian method for nonconvex optimization with equality and inequality constraints*](https://arxiv.org/abs/2509.02894) by Adeyemi D. Adeoye, Puya Latafat and Alberto Bemporad (2025).

To run the examples, first install `pbalm` in editable mode:

    pip install -e .

Note:
- **Gurobi Optimizer**: Required for the MM example. Obtain a license from the [Gurobi website](https://www.gurobi.com/downloads/gurobi-software/). Academic users can request a free license. The Gurobi Optimizer is a registered trademark of Gurobi Optimization, LLC. We acknowledge its use in our research.

- **Alpaqa**: The code also depends on [Alpaqa](https://github.com/kul-optec/alpaqa). Ensure compliance with the LGPL terms.

See `pyproject.toml` for other dependencies.