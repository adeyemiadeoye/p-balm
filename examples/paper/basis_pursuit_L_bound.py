import jax
import jax.numpy as jnp
import pbalm
import numpy as np
import matplotlib.pyplot as plt
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

import numpy as np
import matplotlib.pyplot as plt

lbda_prev = None
rho_prev = None

def make_L_aug_callback(
    f, h, reg=None, alpha=None, xi=None
):
    L_aug_values = []
    ref_values = []
    iter_values = []

    def L_aug(x, x_prev, lbda, mu, rho, nu, gamma_k):
        L = f(x)
        h_x = h(x)
        L += np.sum(rho * 0.5 * h_x**2) + np.dot(lbda, h_x)
        if gamma_k > 0:
            L += (1/(2*gamma_k)) * np.sum((x - x_prev)**2)
        if reg is not None:
            L += reg(x)
        return L

    def callback(iter, x, x_prev, lbda, mu, rho, nu, gamma_k, x0):
        global lbda_prev, rho_prev
        if iter > 0:
            L_val = L_aug(x, x_prev, lbda_prev, mu, rho_prev, nu, gamma_k)
            ref_val = f(x0) + (1/(2*gamma_k)) * np.sum((x0 - x_prev)**2)
            L_aug_values.append(L_val)
            ref_values.append(ref_val)
            iter_values.append(iter)
        lbda_prev = lbda
        rho_prev = rho

    return callback, L_aug_values, ref_values, iter_values

def run_basis_pursuit_benchmark(p, n, k, alpha=None, xi=None):
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
    
    rng = np.random.default_rng(1234)
    x0 = jnp.array(rng.standard_normal(2*n))
    h0 = h(x0)
    lbda0 = jnp.array(rng.standard_normal(h0.shape))

    tol = 1e-9
    max_iter = 50
    fp_tol = 5e-3
    
    f_star = f(x_star)
    print(f"f_star: {f_star}")
    f0 = f(x0)
    print(f"f0: {f0}")

    phi_strategy = "pow"
    xi = 1
    delta = 1e-6
    adaptive_fp_tol = True

    callback, L_aug_values, ref_values, iter_values = make_L_aug_callback(
        f=f, h=h, alpha=alpha, xi=xi
    )

    problem = pbalm.Problem(
        f=f,
        h=h,
        h_grad=h_grad,
        jittable=True,
        callback=callback
    )

    sol_pbalm = pbalm.solve(
        problem, x0, lbda0=lbda0, use_proximal=True, tol=tol, fp_tol=fp_tol, max_iter=max_iter, start_feas=True,
        inner_solver="PANOC", phi_strategy=phi_strategy, xi1=xi, xi2=xi, alpha=alpha, delta=delta, beta=0.5, adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3, gamma0=1e-1, 
    )

    # Plot after optimization
    setup_matplotlib(font_scale=2.5, style_name="bmh")
    colors = ['red', 'black']
    markers = ['o', 's']
    linestyles = [(5, (10, 3)), (0, (5, 1))]
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(ref_values, label=r"$f(x^0) + \frac{1}{2\gamma_k}\|x^0 - x^k\|^2$", color=colors[0], marker=markers[0], linestyle=linestyles[0], markersize=10, markevery=0.1, markerfacecolor='none', linewidth=3.5)
    plt.plot(L_aug_values, label=r"$\mathcal{L}_{\rho_k,\nu_k,\gamma_k}(x^{k+1},\lambda^k,\mu^k;x^k) + f_2(x^{k+1})$", color=colors[1], marker=markers[1], linestyle=linestyles[1], markersize=10, markevery=0.1, markerfacecolor='none', linewidth=3.5)
    plt.yscale('symlog', linthresh=1)
    plt.xlabel(r'$k$')
    plt.legend(fontsize=16, loc='lower right')
    plt.tight_layout()
    plt.savefig("L_aug_vs_ref_curve.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(sol_pbalm.prox_hist, color=colors[1], marker=markers[1], linestyle=linestyles[1], markersize=10, markevery=0.1, markerfacecolor='none', linewidth=3.5)
    # plt.yscale('symlog', linthresh=1)
    plt.yscale('log')
    plt.xlabel(r'$k$')
    plt.ylabel(r"$\frac{1}{2\gamma_k}\|x^{k+1} - x^k\|^2$")
    plt.tight_layout()
    plt.savefig("prox_term_hist.pdf", format='pdf', bbox_inches='tight')
    plt.close()

run_basis_pursuit_benchmark(400, 1024, 10, alpha=4)