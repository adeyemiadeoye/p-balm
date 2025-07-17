import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from plotting import setup_matplotlib
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


def compare_alm_algs(prob_name, f_star=None):

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
    fp_tol = 5e-3
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
        if model.status == GRB.OPTIMAL:
            f_star = model.ObjVal
        else:
            print("Gurobi did not solve to optimality. Status:", model.status)
            f_star = None
        return f_star

    if f_star is None:
        # get f_star
        f_star = solve_gurobi()

    print(f"f_star: {f_star}")

    #################### MAIN ALGORITHM COMPARISON ####################

    ### this is used with pbalm_org
    if prob_name in ["GENHS28", "DUALC1"]:
        alpha_vals_alm = [6, 9, 12]
        xi_vals_alm = [4, 10]
    else:
        alpha_vals_alm = [12]
        xi_vals_alm = [10]

    phi_strategy = "pow"
    xi = 1
    delta = 1

    nrmQ = jnp.linalg.norm(Q)
    A_ineq = jnp.concatenate([A, -A], axis=0)
    nrmATA = jnp.linalg.norm(A_ineq.T@A_ineq)
    def Lip_grad_est(x_hat, lbda, mu, rho, nu, gamma):
        if gamma in [None, jnp.nan]:
            gam = 0
        else:
            gam = gamma
        return float(nrmQ + jnp.max(nu)*nrmATA + gam)

    Lip_grad_est = None # allow linesearch to estimate Lipschitz constant

    adaptive_fp_tol = True

    feas_meas_results = []
    grad_evals_results = []
    legends = []
    legends_nu = []
    f_hist_results = []
    f_hist_nu = []
    feas_meas_nu = []
    nu_hist_list = []

    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>']

    # ALM
    for xi_alm in xi_vals_alm:
        sol_alm = pbalm.solve(
            problem, x0, mu0=mu0, use_proximal=False, tol=tol, fp_tol=fp_tol, max_iter=max_iter, start_feas=False, no_reset=True, inner_solver="PANOC", phi_strategy="linear", xi1=xi_alm, xi2=xi_alm, beta=0.5, Lip_grad_est=Lip_grad_est, adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3
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
        if prob_name == "DUALC1":
            nu_hist_list.append(np.array(sol_alm.nu_hist))
            legends_nu.append(legends[-1])
            f_hist_nu.append(f_hist_alm)
            feas_meas_nu.append(feas_meas_alm)
        feas_meas_results.append(feas_meas_alm)
        grad_evals_results.append(grad_evals_alm)
        f_hist_results.append(f_hist_alm)
        problem.reset_counters()

    # P-BALM
    for alpha in alpha_vals_alm:
        sol_pbalm = pbalm.solve(
            problem, x0, mu0=mu0, use_proximal=True, tol=tol, fp_tol=fp_tol, max_iter=max_iter, start_feas=True,
            inner_solver="PANOC", phi_strategy=phi_strategy, xi1=xi, xi2=xi, alpha=alpha, delta=delta, beta=0.5, Lip_grad_est=Lip_grad_est, adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3, gamma0=1e-1
        )
        feas_meas_pbalm = np.array(sol_pbalm.total_infeas)
        grad_evals_pbalm = sol_pbalm.grad_evals
        f_hist_pbalm = np.array(sol_pbalm.f_hist) - f_star
        if len(f_hist_pbalm) < 1:
            legends.append(r"\texttt{P-BALM}")
        else:
            legends.append(r"\texttt{P-BALM}-" + str(alpha))
        if prob_name == "DUALC1" and alpha == alpha_vals_alm[0]:
            nu_hist_list.append(np.array(sol_pbalm.nu_hist))
            legends_nu.append(legends[-1])
            f_hist_nu.append(f_hist_pbalm)
            feas_meas_nu.append(feas_meas_pbalm)
        feas_meas_results.append(feas_meas_pbalm)
        grad_evals_results.append(grad_evals_pbalm)
        f_hist_results.append(f_hist_pbalm)
        problem.reset_counters()

    # BALM
    for alpha in alpha_vals_alm:
        sol_balm = pbalm.solve(
            problem, x0, mu0=mu0, use_proximal=False, tol=tol, fp_tol=fp_tol, max_iter=max_iter, start_feas=True,
            inner_solver="PANOC", phi_strategy=phi_strategy, xi1=xi, xi2=xi, alpha=alpha, beta=0.5, Lip_grad_est=Lip_grad_est, adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3
        )
        feas_meas_balm = np.array(sol_balm.total_infeas)
        grad_evals_balm = sol_balm.grad_evals
        f_hist_balm = np.array(sol_balm.f_hist) - f_star
        if len(f_hist_balm) < 1:
            legends.append(r"\texttt{BALM}")
        else:
            legends.append(r"\texttt{BALM}-" + str(alpha))
        if prob_name == "DUALC1" and alpha == alpha_vals_alm[0]:
            nu_hist_list.append(np.array(sol_balm.nu_hist))
            legends_nu.append(legends[-1])
            f_hist_nu.append(f_hist_balm)
            feas_meas_nu.append(feas_meas_balm)
        feas_meas_results.append(feas_meas_balm)
        grad_evals_results.append(grad_evals_balm)
        f_hist_results.append(f_hist_balm)
        problem.reset_counters()

    setup_matplotlib(24, 24)
    plt.figure(figsize=(7,5), dpi=300)
    for idx, (grad_evals, tot_inf, legend) in enumerate(zip(grad_evals_results, feas_meas_results, legends)):
        if len(alpha_vals_alm) > 1 or len(xi_vals_alm) > 1:
            label = None
        else:
            label = legend
        plt.plot(grad_evals, np.maximum(tot_inf, 1e-6), label=label, marker=markers[idx % len(markers)], markevery=0.1, markerfacecolor='none')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'grad evals')
    # plt.xlabel('iter')
    # plt.ylabel('tot inf')
    # plt.ylabel(r'$\max\{\|h(x^k)\|_{\infty}, \|E^k\|_{\infty}\}$')
    plt.ylabel(r'total infeas')
    plt.grid(True, which='major', ls='--')
    # plt.title(rf'\texttt{{{prob_name}}} ($\kappa(Q)=\textrm{{\textbf{{{cond_Q:.2E}}}}}$)')
    if len(alpha_vals_alm) == 1 or len(xi_vals_alm) == 1:
        plt.legend(fontsize=18, loc='lower left')
    ax = plt.gca()
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()
    fname = f"alm_algs_{prob_name}_phi_{phi_strategy}.pdf"
    plt.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7,5), dpi=300)
    for idx, (grad_evals, fx_minus_fxstar, legend) in enumerate(zip(grad_evals_results, f_hist_results, legends)):
        if len(alpha_vals_alm) > 1 or len(xi_vals_alm) > 1:
            label = None
        else:
            label = legend
        plt.plot(grad_evals, np.maximum(np.abs(fx_minus_fxstar)/np.abs(f_star), 5e-6), label=label, marker=markers[idx % len(markers)], markevery=0.1, markerfacecolor='none')
    plt.xscale('log')
    plt.yscale('log')
    # plt.xlabel(r'grad evals')
    plt.ylabel(r'$\frac{|f(x^k) - f^\star|}{|f^\star|}$')
    plt.grid(True, which='major', ls='--')
    plt.title(rf'\texttt{{{prob_name}}} ($\kappa(Q)=\textrm{{\textbf{{{cond_Q:.2E}}}}}$)')
    if len(alpha_vals_alm) == 1 or len(xi_vals_alm) == 1:
        plt.legend(fontsize=18, loc='lower left')
    ax = plt.gca()
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout()
    fname = f"alm_algs_fx_minus_fxstar_{prob_name}_phi_{phi_strategy}.pdf"
    plt.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close()


    fig_leg, ax_leg = plt.subplots(figsize=(7, 1), dpi=300)
    handles = []
    for idx, legend in enumerate(legends):
        handle, = ax_leg.plot([], [], marker=markers[idx % len(markers)], markerfacecolor='none')
        handles.append(handle)
    ax_leg.legend(handles=handles, labels=legends, fontsize=14, loc='center', ncol=len(legends))
    ax_leg.axis('off')
    # plt.tight_layout()
    if len(alpha_vals_alm) > 1 or len(xi_vals_alm) > 1:
        fname_leg = f"legend_MM_0.pdf"
    else:
        fname_leg = f"legend_MM.pdf"
    plt.savefig(fname_leg, format='pdf', bbox_inches='tight')
    plt.close(fig_leg)



    ##### PLOT nu_hist #####
    if prob_name == "DUALC1":

        plt.figure(figsize=(7,5), dpi=300)
        for idx, (nu_hist, tot_inf, legend) in enumerate(zip(nu_hist_list, feas_meas_nu, legends_nu)):
            plt.plot(nu_hist, np.maximum(tot_inf, 1e-6), label=legend, marker=markers[idx % len(markers)], markevery=0.1, markerfacecolor='none')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\nu$')
        plt.ylabel(r'total infeas')
        plt.grid(True, which='major', ls='--')
        # plt.title(rf'\texttt{{{prob_name}}} ($\kappa(Q)=\textrm{{\textbf{{{cond_Q:.2E}}}}}$)')
        plt.legend(fontsize=18, loc='lower left')
        plt.tight_layout()
        fname = f"nu_hist_{prob_name}_phi_{phi_strategy}.pdf"
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        plt.close()


        plt.figure(figsize=(7,5), dpi=300)
        for idx, (nu_hist, f_hist, legend) in enumerate(zip(nu_hist_list, f_hist_nu, legends_nu)):
            plt.plot(nu_hist, np.maximum(np.abs(f_hist)/np.abs(f_star), 5e-6), label=legend, marker=markers[idx % len(markers)], markevery=0.1, markerfacecolor='none')
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlabel(r'$\nu$')
        plt.ylabel(r'$\frac{|f(x^k) - f^\star|}{|f^\star|}$')
        plt.grid(True, which='major', ls='--')
        plt.title(rf'\texttt{{{prob_name}}} ($\kappa(Q)=\textrm{{\textbf{{{cond_Q:.2E}}}}}$)')
        plt.legend(fontsize=18, loc='lower left')
        plt.tight_layout()
        fname = f"nu_hist_fx_minus_fxstar_{prob_name}_phi_{phi_strategy}.pdf"
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        plt.close()

# compare_alm_algs("GENHS28")
# compare_alm_algs("DUALC1")
compare_alm_algs("LOTSCHD")
# compare_alm_algs("CVXQP2_M")
# compare_alm_algs("AUG3D")