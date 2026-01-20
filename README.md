# pbalm

<p align="center">
    <img src="https://github.com/adeyemiadeoye/p-balm/blob/main/docs/source/_static/pbalm_logo.png" alt="pbalm logo"/>
</p>
<p align="center">
    <a href="https://adeyemiadeoye.github.io/p-balm/"><img src="https://img.shields.io/badge/docs-Sphinx-blue" alt="Documentation"/></a>
    <a href="https://arxiv.org/abs/2509.02894"><img src="https://img.shields.io/badge/paper-arXiv-red" alt="arXiv"/></a>
    <a href="https://pypi.org/project/pbalm/"><img src="https://img.shields.io/pypi/v/pbalm" alt="PyPI"/></a>
    <a href="https://github.com/adeyemiadeoye/p-balm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/adeyemiadeoye/p-balm" alt="License"/></a>
</p>

A Python package providing a **proximal augmented Lagrangian method** for solving nonlinear programming problems with equality and inequality constraints.

## Problem structure

**pbalm** solves optimization problems of the form:

$$
\begin{aligned}
\min_{x} \quad & f_1(x) + f_2(x) \\
\text{s.t.} \quad & g(x) \leq 0 \\
& h(x) = 0
\end{aligned}
$$

where $f_1$ is smooth (possibly nonconvex), $f_2$ is possibly nonsmooth but prox-friendly, and $g$, $h$ define smooth inequality and equality constraints, respectively.

## Key Features

- **Nonconvex optimization**: handles nonconvex objectives and constraints
- **Composite objectives**: supports smooth + nonsmooth terms
- **Flexible constraints**: both equality and inequality constraints
- **JAX-powered**: automatic differentiation and JIT compilation

## Quick Example

```python
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
```

## Installation

```bash
python3 -m pip install pbalm
```

For development:

```bash
git clone https://github.com/adeyemiadeoye/p-balm.git
cd p-balm
python3 -m pip install -e .
```

## Documentation

Full documentation is available at **[adeyemiadeoye.github.io/p-balm](https://adeyemiadeoye.github.io/p-balm/)**.

## Citation

If you use **pbalm** in your research, please cite:

```bibtex
@article{adeoye2025pbalm,
  title={A proximal augmented Lagrangian method for nonconvex optimization with equality and inequality constraints},
  author={Adeoye, Adeyemi D. and Latafat, Puya and Bemporad, Alberto},
  journal={arXiv preprint arXiv:2509.02894},
  year={2025}
}
```

## License

See [LICENSE](LICENSE) for details.

## Acknowledgements

This work was funded by the European Union (ERC Advanced Research Grant COMPACT, No. 101141351). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

<p align="center">
    <img src="https://github.com/adeyemiadeoye/p-balm/blob/main/src/other_media/erc-logo.png" alt="ERC logo" width="400"/>
</p>

**pbalm** depends on the efficient implementation of [PANOC](https://ieeexplore.ieee.org/document/8263933) provided by [alpaqa](https://github.com/kul-optec/alpaqa), as well as its regularizers module.