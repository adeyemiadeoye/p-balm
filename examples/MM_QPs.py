import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.plotting import setup_matplotlib
import jax
import jax.numpy as jnp
import pbalm
import proxop
import numpy as np

import gurobipy as gp
from gurobipy import GRB

from pathlib import Path
import contextlib
import gc

import alpaqa as pa

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)


def safe_for_plot(arr, max_finite=1e10):
    arr = np.asarray(arr)
    arr = np.where(np.isposinf(arr), max_finite, arr)
    arr = np.where(np.isneginf(arr), -max_finite, arr)
    arr = np.where(arr > max_finite, max_finite, arr)
    arr = np.where(arr < -max_finite, -max_finite, arr)
    return arr

def compare_alm_algs(prob_name, f_star=None):

    # here I assume you have the compiled MM problem in this dir
    # with the .so and OUTSDIF.d files
    mm_qps_dir = Path.home() / "opt" / "CUTEST" / "compiled_QP_MM"

    with contextlib.suppress(NameError):
        del prob
    gc.collect()

    prob_path = str(mm_qps_dir / prob_name / prob_name) + ".so"
    prob_outsdiff_filename = str(mm_qps_dir / prob_name / "OUTSDIF.d")

    prob = pa.CUTEstProblem(prob_path, prob_outsdiff_filename)
    n = prob.n
    m = prob.m

    Q, _ = prob.eval_hess_L(jnp.zeros(n), jnp.zeros(m))
    c, q = prob.eval_f_grad_f(jnp.zeros(n))
    x_lb = prob.C.lowerbound
    x_ub = prob.C.upperbound
    A, _ = prob.eval_jac_g(jnp.zeros(n))
    g = prob.eval_g(jnp.zeros(n))
    g_lb = prob.D.lowerbound - g
    g_ub = prob.D.upperbound - g

    bmin = jnp.where(jnp.isneginf(g_lb), -1e32, g_lb)
    bmax = jnp.where(jnp.isposinf(g_ub), 1e32, g_ub)

    def f(x):
        return 0.5*jnp.dot(x, jnp.dot(Q, x)) + jnp.dot(q, x) + c

    def f_grad(x):
        return jnp.dot(Q, x) + q

    def g(x):
        g_ub_part = jnp.dot(A, x) - bmax
        g_lb_part = -jnp.dot(A, x) + bmin
        return jnp.concatenate([g_ub_part, g_lb_part], axis=0)

    reg = proxop.BoxConstraint(jnp.array(x_lb), jnp.array(x_ub))
    problem = pbalm.Problem(
        f=jax.jit(f),
        g=jax.jit(g),
        reg=reg,
        f_grad=jax.jit(f_grad),
        jittable=True
    )
    rng = np.random.default_rng(1234)
    x0 = jnp.array(rng.standard_normal(n))
    g0 = g(x0)
    mu0 = jnp.array(rng.standard_normal(g0.shape))

    tol = 1e-5

    max_iter = 2000

    cond_Q = np.linalg.cond(np.array(Q))
    
    def solve_gurobi():
        model = gp.Model("qp")
        model.Params.OutputFlag = 0
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        x_vars = model.addMVar(shape=n, lb=np.array(x_lb), ub=np.array(x_ub), name="x")
        obj = 0.5*x_vars@np.array(Q)@x_vars + np.array(q)@x_vars + float(c)
        model.setObjective(obj, GRB.MINIMIZE)
        model.addConstr(np.array(A)@x_vars >= np.array(bmin), name="ineq_lb")
        model.addConstr(np.array(A)@x_vars <= np.array(bmax), name="ineq_ub")
        model.optimize()
        f_star = model.ObjVal
        return f_star

    if f_star is None:
        f_star = solve_gurobi()

    print(f"f_star: {f_star}")

    f0 = f(x0)
    print(f"f0: {f0}")
    f0_f_star = f0 - f_star
    alpha = 12
    xi_alm = 10

    phi_strategy = "pow"
    xi = 1
    delta = 1

    feas_meas_results = []
    grad_evals_results = []
    legends = []
    legends_nu_1 = []
    f_hist_results = []

    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>', 'p', 'H', 'h', '1', '2', '3', '4']
    num_lines = max(len(grad_evals_results), len(markers))
    n_dark2 = plt.cm.Dark2.N
    n_set1 = plt.cm.Set1.N
    colors_dark2 = plt.cm.Dark2(np.linspace(0, 1, n_dark2))
    colors_set1 = plt.cm.Set1(np.linspace(0, 1, n_set1))
    colors_ext = np.concatenate([colors_dark2, colors_set1], axis=0)
    if num_lines > len(colors_ext):
        colors = np.tile(colors_ext, (int(np.ceil(num_lines / len(colors_ext))), 1))[:num_lines]
    else:
        colors = colors_ext[:num_lines]

    marker_size = 10

    # ALM
    sol_alm = pbalm.solve(
        problem, x0, mu0=mu0, use_proximal=False, tol=tol, max_iter=max_iter, start_feas=True, no_reset=True, inner_solver="PANOC", phi_strategy="linear", xi1=xi_alm, xi2=xi_alm, beta=0.5, rho0=1e-3, nu0=1e-3
    )
    feas_meas_alm = np.array(sol_alm.total_infeas)
    grad_evals_alm = sol_alm.grad_evals
    f_hist_alm = np.array(sol_alm.f_hist) - f_star
    if len(f_hist_alm) < 1:
        legends.append(r"\texttt{ALM}")
    else:
        if xi_alm >= 100:
            legend_xi = rf'${{{str(10)}}}^{{{str(int(np.log10(xi_alm)))}}}$'
            legends.append(r"\texttt{ALM}-" + legend_xi)
        else:
            legends.append(r"\texttt{ALM}-" + str(xi_alm))
    legends_nu_1.append(r"\texttt{ALM}")
    feas_meas_results.append(feas_meas_alm)
    grad_evals_results.append(grad_evals_alm)
    f_hist_results.append(f_hist_alm)
    problem.reset_counters()

    # P-BALM
    sol_pbalm = pbalm.solve(
        problem, x0, mu0=mu0, use_proximal=True, tol=tol, max_iter=max_iter, start_feas=True,
        inner_solver="PANOC", phi_strategy=phi_strategy, xi1=xi, xi2=xi, alpha=alpha, delta=delta, beta=0.5, rho0=1e-3, nu0=1e-3, gamma0=1e-1
    )
    feas_meas_pbalm = np.array(sol_pbalm.total_infeas)
    grad_evals_pbalm = sol_pbalm.grad_evals
    f_hist_pbalm = np.array(sol_pbalm.f_hist) - f_star
    if len(f_hist_pbalm) < 1:
        legends.append(r"\texttt{P-BALM}")
    else:
        legends.append(r"\texttt{P-BALM}-" + str(alpha))
    legends_nu_1.append(r"\texttt{P-BALM}")
    feas_meas_results.append(feas_meas_pbalm)
    grad_evals_results.append(grad_evals_pbalm)
    f_hist_results.append(f_hist_pbalm)
    problem.reset_counters()

    # BALM
    sol_balm = pbalm.solve(
        problem, x0, mu0=mu0, use_proximal=False, tol=tol, max_iter=max_iter, start_feas=True,
        inner_solver="PANOC", phi_strategy=phi_strategy, xi1=xi, xi2=xi, alpha=alpha, beta=0.5, rho0=1e-3, nu0=1e-3
    )
    feas_meas_balm = np.array(sol_balm.total_infeas)
    grad_evals_balm = sol_balm.grad_evals
    f_hist_balm = np.array(sol_balm.f_hist) - f_star
    if len(f_hist_balm) < 1:
        legends.append(r"\texttt{BALM}")
    else:
        legends.append(r"\texttt{BALM}-" + str(alpha))
    legends_nu_1.append(r"\texttt{BALM}")
    feas_meas_results.append(feas_meas_balm)
    grad_evals_results.append(grad_evals_balm)
    f_hist_results.append(f_hist_balm)
    problem.reset_counters()

    legends = legends_nu_1
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7,5), dpi=300)
    for idx, (grad_evals, tot_inf, legend) in enumerate(zip(grad_evals_results, feas_meas_results, legends)):
        tot_inf = safe_for_plot(tot_inf)
        label = legend
        plt.plot(grad_evals, np.maximum(tot_inf, 1e-6), label=label, marker=markers[idx % len(markers)],
                markevery=0.1, markerfacecolor='none',
                color=colors[idx % len(colors)],
                linestyle='dashdot',
                markersize=marker_size
                )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\textbf{grad evals}$')
    plt.ylabel(r'$\textbf{total infeas}$')
    plt.legend(fontsize=18, loc='lower left')
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()
    fname = f"alm_algs_{prob_name}_phi_{phi_strategy}.pdf"
    plt.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(7,5), dpi=300)
    for idx, (grad_evals, fx_minus_fxstar, legend) in enumerate(zip(grad_evals_results, f_hist_results, legends)):
        label = legend
        fx_minus_fxstar = safe_for_plot(fx_minus_fxstar)
        plt.plot(grad_evals, np.maximum(np.abs(fx_minus_fxstar)/np.abs(f0_f_star), 1e-7), label=label,
                marker=markers[idx % len(markers)], markevery=0.1, markerfacecolor='none',
                color=colors[idx % len(colors)],
                linestyle='dashdot',
                markersize=marker_size
                )
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\frac{|f_1(x^k) - f_1^\star|}{|f_1(x^0) - f_1^\star|}$')
    plt.title(rf'\texttt{{{prob_name}}} ($\kappa(Q)=\textrm{{\textbf{{{cond_Q:.2E}}}}}$)')
    plt.legend(fontsize=18, loc='lower left')
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout()
    fname = f"alm_algs_fx_minus_fxstar_{prob_name}_phi_{phi_strategy}.pdf"
    plt.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close()

compare_alm_algs("CVXQP2_M")