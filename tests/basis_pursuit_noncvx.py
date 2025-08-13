import jax
import jax.numpy as jnp
import pbalm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from plotting import setup_matplotlib
from matplotlib.axis import Tick

import gurobipy as gp
from gurobipy import GRB

from mpl_toolkits.mplot3d import Axes3D  # Add this import at the top if not present

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

import numpy as np
import matplotlib.pyplot as plt

# lbda_prev = None
# rho_prev = None

def make_L_aug_callback(
    f, h, reg=None, plot_iters=[1], alpha=None, xi=None
):
    setup_matplotlib(style_name="bmh", grid=False)
    plotted = set()
    fig = plt.figure(figsize=(7, 5), dpi=300)
    ax = fig.gca()
    colors = [
        'dimgray', 'red', 'black', 'darkred', 'darkgoldenrod', 'royalblue',
        'rebeccapurple', 'saddlebrown', 'darkslategray', 'darkorange',
        'steelblue', 'lightcoral'
    ]
    linestyles = [
                    #   (0, (1, 5)),
                    #   (0, (1, 1)),
                      (5, (10, 3)),
                      (0, (5, 1)),
                      (0, (3, 1, 1, 1)),
                      (0, (3, 1, 1, 1, 1, 1)),
                      (0, (3, 5, 1, 5, 1, 5)),
                      '--', '-', '-.', ':']
    if alpha is not None:
        fname="L_aug_plot_ours_" + str(alpha) + ".pdf"
    elif xi is not None:
        fname="L_aug_plot_xi_alm_" + str(xi) + ".pdf"

    def L_aug(x, x_prev, lbda, mu, rho, nu, gamma_k):
        L = f(x)
        h_x = h(x)
        L += np.sum(rho * 0.5 * h_x**2) + np.dot(lbda, h_x)
        if gamma_k > 0:
            L += np.sum((1/(2*gamma_k)) * (x - x_prev)**2)
        if reg is not None:
            L += reg(x)
        return L

    def callback(iter, x, x_prev, lbda, mu, rho, nu, gamma_k, x0=None):
        # global lbda_prev, rho_prev
        if gamma_k > 0:
            ylabel = r"$\mathcal{L}_{\rho_k, \nu_k, \gamma_k}(x,\lambda^k, \mu^k)$"
        else:
            ylabel = r"$\mathcal{L}_{\rho_k, \nu_k}(x,\lambda^k, \mu^k)$"
        if iter in plot_iters and iter not in plotted:
            plotted.add(iter)
            x0 = np.atleast_1d(x)
            x_prev0 = np.atleast_1d(x_prev)
            # 2D grid for surface plot (first two coordinates)
            x1_vals = np.linspace(np.maximum(float(x0[0]) - 2, np.min(x0)), np.minimum(float(x0[0]) + 2, np.max(x0)), 100)
            x2_vals = np.linspace(np.maximum(float(x0[1]) - 2, np.min(x0)), np.minimum(float(x0[1]) + 2, np.max(x0)), 100)
            X1, X2 = np.meshgrid(x1_vals, x2_vals)
            Z = np.zeros_like(X1)
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    x_mod = x0.copy()
                    x_mod[0] = X1[i, j]
                    x_mod[1] = X2[i, j]
                    Z[i, j] = L_aug(x_mod, x_prev0, lbda, mu, rho, nu, gamma_k)
            Z_floor = np.nanmin(Z)
            fig_3d = plt.figure(figsize=(8, 6), dpi=300)
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.plot_surface(X1, X2, Z, cmap='bone', edgecolor='none', alpha=0.9)
            ax_3d.contour(X1, X2, Z, zdir='z', offset=Z_floor, cmap='bone', levels=20)
            ax_3d.set_xticklabels([])
            ax_3d.set_yticklabels([])
            ax_3d.set_zticklabels([])
            for axis in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
                axis._axinfo['tick']['inward_factor'] = 0
                axis._axinfo['tick']['outward_factor'] = 0
            ax_3d.set_title(rf"$k = {iter}$")
            ax_3d.set_zlim(Z_floor, np.nanmax(Z))
            fname_iter = f"L_aug_surface_iter_alpha{alpha}_k{iter}.pdf" if alpha is not None else f"L_aug_surface_iter_xi{xi}_k{iter}.pdf"
            # plt.tight_layout()
            plt.savefig(fname_iter, format='pdf', bbox_inches='tight')
            plt.close(fig_3d)
            x_vals = np.linspace(np.maximum(float(x0[0]) - 2, np.min(x0)), np.minimum(float(x0[0]) + 2, np.max(x0)), 400)
            y_vals = []
            for x_i in x_vals:
                x_mod = x0.copy()
                x_mod[0] = x_i
                y_vals.append(L_aug(x_mod, x_prev0, lbda, mu, rho, nu, gamma_k))
            color = colors[len(plotted) % len(colors)]
            linestyle = linestyles[len(plotted) % len(linestyles)]
            ax.plot(x_vals, y_vals, label=rf"$k= {iter}$", color=color, linestyle=linestyle)
            ax.axvline(float(x0[0]), color=color, linestyle=':', alpha=0.5)
            ax.set_yscale('symlog', linthresh=1)
            if xi is not None:
                ax.set_yticks([1e1, 1e10, 1e21])
            elif alpha is not None:
                if alpha == 3:
                    ax.set_yticks([1e0, 1e3, 1e6])
                else:
                    ax.set_yticks([-1e5, 0, 1e9])
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(ylabel)
            plt.tight_layout()
            if set(plot_iters).issubset(plotted):
                ax.legend(fontsize=20, loc='center left', bbox_to_anchor=(1.01, 0.5))
                # ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=len(plot_iters), frameon=False)
                plt.savefig(fname, format='pdf', bbox_inches='tight')
                plt.close(fig)
        lbda_prev = lbda
        rho_prev = rho

    return callback


def run_basis_pursuit_benchmark(p, n, k, alpha=None, xi=None, plot_iters=[1]):
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
    
    callback = make_L_aug_callback(f=f, h=h, alpha=alpha, xi=xi, plot_iters=plot_iters)
    problem = pbalm.Problem(
        f=f,
        h=h,
        h_grad=h_grad,
        jittable=True,
        callback=callback
    )
    rng = np.random.default_rng(1234)
    x0 = jnp.array(rng.standard_normal(2*n))
    h0 = h(x0)
    lbda0 = jnp.array(rng.standard_normal(h0.shape))

    tol = 1e-6
    max_iter = 50

    fp_tol = 5e-3
    
    f_star = f(x_star)
    # f_star = solve_gurobi()
    ## for the default setup (p, n, k) and default seed,
    ## gurobi finds approximately the same f_star as the analytical solution
    ## but it takes a long time to solve

    print(f"f_star: {f_star}")
    f0 = f(x0)
    print(f"f0: {f0}")

    delta = 1e-6
    adaptive_fp_tol = True

    if alpha is not None:
        sol_pbalm = pbalm.solve(
            problem, x0, lbda0=lbda0, use_proximal=True, tol=tol, fp_tol=fp_tol, max_iter=max_iter, start_feas=True,
            inner_solver="PANOC", phi_strategy="pow", xi1=1, xi2=1, alpha=alpha, delta=delta, beta=0.5, adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3, gamma0=1e-1
        )
    elif xi is not None:
        sol_alm = pbalm.solve(
                problem, x0, lbda0=lbda0, use_proximal=False, tol=tol, fp_tol=fp_tol, max_iter=max_iter, start_feas=True, no_reset=True, inner_solver="PANOC", phi_strategy="linear", xi1=xi, xi2=xi, beta=0.5, adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3
            )

alpha = 9
if alpha in [9, 12]:
    plot_iters = [1, 2, 3, 5, 7]
elif alpha in [2, 3, 4]:
    # plot_iters = [1, 3, 6, 10, 15]
    plot_iters = [1, 3, 10, 20, 50]
run_basis_pursuit_benchmark(400, 1024, 10, alpha=alpha, plot_iters=plot_iters)

# xi = 4
# # plot_iters = [1, 3, 6, 10, 15]
# if xi in [2, 4]:
#     plot_iters = [1, 3, 10, 20, 43]
# elif xi == 7:
#     plot_iters = [1, 3, 10, 20, 30]
# run_basis_pursuit_benchmark(400, 1024, 10, xi=xi, plot_iters=plot_iters)