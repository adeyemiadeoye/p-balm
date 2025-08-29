import jax
import jax.numpy as jnp
import pbalm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from plotting import setup_matplotlib

import gurobipy as gp
from gurobipy import GRB

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

def get_basis_pursuit_data(m, n, k, key=1234):
    rng = np.random.default_rng(key)
    B = jnp.array(rng.standard_normal((m, n)))
    z_star = jnp.zeros(n)
    support = rng.choice(n, size=k, replace=False)
    amplitudes = jnp.array(rng.standard_normal(k))
    z_star = z_star.at[support].set(amplitudes)
    b = B @ z_star
    B_big = jnp.concatenate([B, -B], axis=1)
    z_star_pos = jnp.maximum(z_star, 0.0)
    z_star_neg = jnp.maximum(-z_star, 0.0)
    u1_star = jnp.sqrt(z_star_pos)
    u2_star = jnp.sqrt(z_star_neg)
    x_star = jnp.concatenate([u1_star, u2_star], axis=0)
    return B, b, z_star, x_star, B_big



def run_basis_pursuit_benchmark(p, n, k):
    B, b, z_star, x_star, B_big = get_basis_pursuit_data(p, n, k)


    def solve_gurobi():
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        x_vars = model.addMVar(shape=2*n, name="x")
        obj = gp.quicksum(x_vars[i]*x_vars[i] for i in range(2*n))
        model.setObjective(obj, GRB.MINIMIZE)
        eq_exprs = np.array(B_big)@(x_vars*x_vars)
        for i in range(b.shape[0]):
            model.addConstr(eq_exprs[i] == b[i], name=f"eq_{i}")
        model.optimize()
        if model.status == GRB.OPTIMAL:
            f_star = model.ObjVal
        else:
            print("Gurobi did not solve to optimality. Status:", model.status)
            f_star = None
        return f_star

    def f(x):
        return jnp.sum(x**2)
    def h(x):
        return B_big@(x**2) - b
    def h_grad(x):
        return 2*jnp.dot(B_big,x)
    problem = pbalm.Problem(
        f=f,
        h=h,
        h_grad=h_grad,
        jittable=True
    )
    rng = np.random.default_rng(1234)
    x0 = jnp.array(rng.standard_normal(2*n))
    h0 = h(x0)
    lbda0 = jnp.array(rng.standard_normal(h0.shape))

    tol = 1e-5
    # tol = 1e-11

    if tol < 1e-5:
        max_iter = 120
    else:
        max_iter = 2000
    
    f_star = f(x_star)
    # f_star = solve_gurobi()
    ## for the default setup (p, n, k) and default seed,
    ## gurobi finds approximately the same f_star as the analytical solution
    ## but it takes a long time to solve

    print(f"f_star: {f_star}")
    f0 = f(x0)
    print(f"f0: {f0}")
    f0_f_star = f0 - f_star

    # alpha_vals_alm = [3]
    # xi_vals_alm = [4]

    alpha_vals_alm = [4]
    xi_vals_alm = [4]

    # alpha_vals_alm = [4, 12]
    # xi_vals_alm = [4, 10]
    #
    # alpha_vals_alm = [4, 6, 9, 12]
    # xi_vals_alm = [2, 4, 7, 10]
    
    phi_strategy = "pow"
    xi = 1
    delta = 1e-6
    adaptive_fp_tol = True

    feas_meas_results = []
    grad_evals_results = []
    grad_evals_results_0 = []
    legends = []
    legends_rho_0 = []
    f_hist_results = []
    rho_hist_list = []

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

    linestyles = [
                    #   (0, (1, 5)),
                    #   (0, (1, 1)),
                      (5, (10, 3)),
                      (0, (5, 1)),
                      (0, (3, 1, 1, 1)),
                      (0, (3, 1, 1, 1, 1, 1)),
                      (0, (3, 5, 1, 5, 1, 5)),
                      '--', '-', '-.', ':']
    marker_size = 10

    # ALM
    for xi_alm in xi_vals_alm:
        sol_alm = pbalm.solve(
            problem, x0, lbda0=lbda0, use_proximal=False, tol=tol, max_iter=max_iter, start_feas=True, no_reset=True, inner_solver="PANOC", phi_strategy="linear", xi1=xi_alm, xi2=xi_alm, beta=0.5, adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3
        )
        feas_meas_alm = np.array(sol_alm.total_infeas)
        grad_evals_alm = sol_alm.grad_evals
        f_hist_alm = np.array(sol_alm.f_hist) - f_star
        legends.append(r"\texttt{ALM}-" + str(xi_alm))
        rho_hist_list.append(np.array(sol_alm.rho_hist))
        legends_rho_0.append(r"\texttt{ALM}")
        feas_meas_results.append(feas_meas_alm)
        grad_evals_results.append(grad_evals_alm)
        grad_evals_results_0.append(grad_evals_alm)
        f_hist_results.append(f_hist_alm)
        problem.reset_counters()

    # P-BALM
    for alpha in alpha_vals_alm:
        sol_pbalm = pbalm.solve(
            problem, x0, lbda0=lbda0, use_proximal=True, tol=tol, max_iter=300, start_feas=True,
            inner_solver="PANOC", phi_strategy=phi_strategy, xi1=xi, xi2=xi, alpha=alpha, delta=delta, beta=0.5, adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3, gamma0=1e-1
        )
        feas_meas_pbalm = np.array(sol_pbalm.total_infeas)
        grad_evals_pbalm = sol_pbalm.grad_evals
        f_hist_pbalm = np.array(sol_pbalm.f_hist) - f_star
        legends.append(r"\texttt{P-BALM}-" + str(alpha))
        rho_hist_list.append(np.array(sol_pbalm.rho_hist))
        legends_rho_0.append(r"\texttt{P-BALM}")
        feas_meas_results.append(feas_meas_pbalm)
        grad_evals_results.append(grad_evals_pbalm)
        grad_evals_results_0.append(grad_evals_pbalm)
        f_hist_results.append(f_hist_pbalm)
        problem.reset_counters()

    # BALM
    for alpha in alpha_vals_alm:
        sol_balm = pbalm.solve(
            problem, x0, lbda0=lbda0, use_proximal=False, tol=tol, max_iter=300, start_feas=True,
            inner_solver="PANOC", phi_strategy=phi_strategy, xi1=xi, xi2=xi, alpha=alpha, beta=0.5, adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3
        )
        feas_meas_balm = np.array(sol_balm.total_infeas)
        grad_evals_balm = sol_balm.grad_evals
        f_hist_balm = np.array(sol_balm.f_hist) - f_star
        legends.append(r"\texttt{BALM}-" + str(alpha))
        rho_hist_list.append(np.array(sol_balm.rho_hist))
        legends_rho_0.append(r"\texttt{BALM}")
        feas_meas_results.append(feas_meas_balm)
        grad_evals_results.append(grad_evals_balm)
        grad_evals_results_0.append(grad_evals_balm)
        f_hist_results.append(f_hist_balm)
        problem.reset_counters()

    if len(alpha_vals_alm) == 1 and len(xi_vals_alm) == 1:
        legends = legends_rho_0

    if tol < 1e-5: # for illustration purposes
        setup_matplotlib(style_name="bmh", font_scale=3)
        colors = ['dimgray', 'red', 'black', 'darkred', 'darkgoldenrod', 'royalblue', 'rebeccapurple', 'saddlebrown', 'darkslategray', 'darkorange', 'steelblue', 'lightcoral']
        plt.figure(figsize=(7,5), dpi=300)
        for idx, (fx_minus_fxstar, rho_hist, legend) in enumerate(zip(f_hist_results, rho_hist_list, legends_rho_0)):
            plt.plot(np.abs(fx_minus_fxstar)/np.abs(f0_f_star), rho_hist, label=legend,
                     marker=markers[idx % len(markers)], markevery=0.1, markerfacecolor='none',
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    markersize=marker_size, linewidth=3.5
                     )
        plt.xscale('log')
        plt.yscale('log')
        plt.gca().invert_xaxis()
        plt.xlabel(r'$\textbf{suboptimality gap}$')
        # plt.ylabel(r'$\rho_k$')
        plt.legend(fontsize=18, loc='upper right')
        plt.tight_layout()
        fname = f"rho_fhist_basis_pursuit_p{p}_n{n}_k{k}.pdf"
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(7,5), dpi=300)
        for idx, (rho_hist, legend) in enumerate(zip(rho_hist_list, legends_rho_0)):
            plt.plot(rho_hist, label=legend, marker=markers[idx % len(markers)],
                     markevery=0.1, markerfacecolor='none',
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    markersize=marker_size, linewidth=3.5
                     )
        plt.yscale('log')
        plt.xlabel(r'$\textbf{n. iterations}$')
        plt.ylabel(r'$\rho_k$')
        plt.legend(fontsize=18, loc='upper left')
        plt.tight_layout()
        fname = f"rho_iter_basis_pursuit_p{p}_n{n}_k{k}.pdf"
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(7,5), dpi=300)
        for idx, (tot_inf, rho_hist, legend) in enumerate(zip(feas_meas_results, rho_hist_list, legends_rho_0)):
            plt.plot(tot_inf[1:], rho_hist[1:], label=legend,
                     marker=markers[idx % len(markers)], markevery=0.1, markerfacecolor='none',
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    markersize=marker_size, linewidth=3.5
                     )
        plt.xscale('log')
        plt.yscale('log')
        plt.gca().invert_xaxis()
        plt.xlabel(r'$\textbf{total infeas}$')
        # plt.ylabel(r'$\rho_k$')
        plt.legend(fontsize=18, loc='upper right')
        plt.tight_layout()
        fname = f"rho_totinf_basis_pursuit_p{p}_n{n}_k{k}.pdf"
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        plt.close()
    else:
        setup_matplotlib()
        fig, ax = plt.subplots(figsize=(7,5), dpi=300)
        for idx, (grad_evals, tot_inf, legend) in enumerate(zip(grad_evals_results_0, feas_meas_results, legends)):
            if len(alpha_vals_alm) > 1 or len(xi_vals_alm) > 1:
                label = None
            else:
                label = legend
            plt.plot(grad_evals, np.maximum(tot_inf, 1e-6), label=label,
                    marker=markers[idx % len(markers)], markevery=0.1, markerfacecolor='none',
                    color=colors[idx % len(colors)],
                    #  linestyle=linestyles[idx % len(linestyles)],
                    linestyle='dashdot',
                    markersize=marker_size)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\textbf{grad evals}$')
        plt.ylabel(r'$\textbf{total infeas}$')
        if len(alpha_vals_alm) == 1 or len(xi_vals_alm) == 1:
            plt.legend(fontsize=18, loc='lower left')
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(formatter)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.tight_layout()
        if len(alpha_vals_alm) == 1 or len(xi_vals_alm) == 1:
            fname = f"alm_algs_basis_pursuit_p{p}_n{n}_k{k}.pdf"
        else:
            fname = f"alm_algs_basis_pursuit_p{p}_n{n}_k{k}_multi.pdf"
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(7,5), dpi=300)
        for idx, (grad_evals, fx_minus_fxstar, legend) in enumerate(zip(grad_evals_results, f_hist_results, legends)):
            if len(alpha_vals_alm) > 1 or len(xi_vals_alm) > 1:
                label = None
            else:
                label = legend
            plt.plot(grad_evals, np.maximum(np.abs(fx_minus_fxstar)/np.abs(f0_f_star), 1e-7),
                    label=label, marker=markers[idx % len(markers)], markevery=0.1, markerfacecolor='none',
                    color=colors[idx % len(colors)],
                    #  linestyle=linestyles[idx % len(linestyles)],
                    linestyle='dashdot',
                    markersize=marker_size)
        plt.xscale('log')
        plt.yscale('log')
        if len(alpha_vals_alm) > 1 or len(xi_vals_alm) > 1:
            plt.xlabel(r'$\textbf{grad evals}$')
        plt.ylabel(r'$\frac{|f_1(x^k) - f_1^\star|}{|f_1(x^0) - f_1^\star|}$')
        if len(alpha_vals_alm) == 1 or len(xi_vals_alm) == 1:
            plt.legend(fontsize=18, loc='lower left')
            plt.title(rf'$p = {p}, n = {n}$')
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(formatter)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.tight_layout()
        if len(alpha_vals_alm) == 1 or len(xi_vals_alm) == 1:
            fname = f"alm_algs_fx_minus_fxstar_basis_pursuit_p{p}_n{n}_k{k}.pdf"
        else:
            fname = f"alm_algs_fx_minus_fxstar_basis_pursuit_p{p}_n{n}_k{k}_multi.pdf"
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        plt.close()

        if len(alpha_vals_alm) > 1 or len(xi_vals_alm) > 1:
            fig_leg, ax_leg = plt.subplots(figsize=(7, 1), dpi=300)
            handles = []
            for idx, legend in enumerate(legends):
                handle, = ax_leg.plot([], [], marker=markers[idx % len(markers)], markerfacecolor='none',
                                    color=colors[idx % len(colors)],
                                    #   linestyle=linestyles[idx % len(linestyles)],
                                    linestyle='dashdot',
                                    markersize=marker_size
                                    )
                handles.append(handle)
            ax_leg.legend(handles=handles, labels=legends, fontsize=14, loc='center', ncol=len(legends)//2)
            # ax_leg.legend(handles=handles, labels=legends, fontsize=14, loc='center', ncol=len(legends))
            ax_leg.axis('off')
            # plt.tight_layout()
            fname_leg = f"legend_pb_0.pdf"
            plt.savefig(fname_leg, format='pdf', bbox_inches='tight')
            plt.close(fig_leg)

run_basis_pursuit_benchmark(200, 512, 10)