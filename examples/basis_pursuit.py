import jax
import jax.numpy as jnp
import pbalm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.plotting import setup_matplotlib

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

    max_iter = 2000
    
    f_star = f(x_star)

    print(f"f_star: {f_star}")
    f0 = f(x0)
    print(f"f0: {f0}")
    f0_f_star = f0 - f_star

    alpha = 4
    xi_alm = 4
    
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
    marker_size = 10

    # ALM
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

    legends = legends_rho_0
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7,5), dpi=300)
    for idx, (grad_evals, tot_inf, legend) in enumerate(zip(grad_evals_results_0, feas_meas_results, legends)):
        label = legend
        plt.plot(grad_evals, np.maximum(tot_inf, 1e-6), label=label,
                marker=markers[idx % len(markers)], markevery=0.1, markerfacecolor='none',
                color=colors[idx % len(colors)],
                linestyle='dashdot',
                markersize=marker_size)
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
    fname = f"alm_algs_basis_pursuit_p{p}_n{n}_k{k}.pdf"
    plt.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(7,5), dpi=300)
    for idx, (grad_evals, fx_minus_fxstar, legend) in enumerate(zip(grad_evals_results, f_hist_results, legends)):
        label = legend
        plt.plot(grad_evals, np.maximum(np.abs(fx_minus_fxstar)/np.abs(f0_f_star), 1e-7),
                label=label, marker=markers[idx % len(markers)], markevery=0.1, markerfacecolor='none',
                color=colors[idx % len(colors)],
                linestyle='dashdot',
                markersize=marker_size)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\frac{|f_1(x^k) - f_1^\star|}{|f_1(x^0) - f_1^\star|}$')
    plt.legend(fontsize=18, loc='lower left')
    plt.title(rf'$p = {p}, n = {n}$')
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()
    fname = f"alm_algs_fx_minus_fxstar_basis_pursuit_p{p}_n{n}_k{k}.pdf"
    plt.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close()

run_basis_pursuit_benchmark(200, 512, 10)