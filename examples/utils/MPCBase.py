import random
import jax
import jax.numpy as jnp
import numpy as np
import pbalm
import proxop
import os

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update('jax_enable_x64', True)

SEED = 1234
np.random.seed(SEED)
random.seed(SEED)
jax_key = jax.random.PRNGKey(SEED)

# closed-loop simulation
def solve_mpc(system, algo, sim_steps, callback=None, xi=10, alpha=12, delta=1e-6, tol=1e-5, max_iter=2000,
                verbose=False, callback_steps=[], save_metrics=False):
    x = system.x0
    Np = system.Np_default
    traj_x = [np.array(x)]
    traj_u = []
    traj_ys = []
    if save_metrics:
        tol_key = str(tol).replace('.', 'p').replace('-', 'm')
        grad_key = f"{algo}_tol{tol_key}_grad"
        fhist_key = f"{algo}_tol{tol_key}_fhist"
        grad_list = []
        fhist_list = []
    for k in range(sim_steps):
        if k not in callback_steps:
            callback_fn = None
        else:
            callback_fn = callback
        problem, z0, unpack = build_problem(system, x, Np, callback=callback_fn)
        if algo == 'alm':
            sol = pbalm.solve(problem, z0, use_proximal=False, tol=tol, max_iter=max_iter, start_feas=True,
                                no_reset=True, inner_solver="PANOC", phi_strategy="linear", xi1=xi, xi2=xi,
                                beta=0.5, adaptive_fp_tol=True, rho0=1e-3, nu0=1e-3, phase_I_tol=1e-6)
        elif algo == 'balm':
            sol = pbalm.solve(problem, z0, use_proximal=False, tol=tol, max_iter=max_iter, start_feas=True,
                                inner_solver="PANOC", phi_strategy="pow", xi1=1, xi2=1, alpha=alpha, beta=0.5,
                                adaptive_fp_tol=True, rho0=1e-3, nu0=1e-3, phase_I_tol=1e-6)
        else:
            if algo == 'pbalm':
                uniform_pen = True
            elif algo == 'pbalm-star':
                uniform_pen = False
            else:
                raise ValueError(f"Unknown algorithm {algo}")
            sol = pbalm.solve(problem, z0, use_proximal=True, tol=tol, max_iter=max_iter, start_feas=True,
                                inner_solver="PANOC", phi_strategy="pow", xi1=1, xi2=1, alpha=alpha, delta=delta,
                                uniform_pen=uniform_pen, beta=0.5, adaptive_fp_tol=True, rho0=1e-3, nu0=1e-3,
                                gamma0=1e-1, phase_I_tol=1e-6)

        xs, us, y_s = unpack(sol.x)
        if verbose:
            print(f"k={k:02d} infeas={sol.total_infeas[-1]:.2e} f={sol.f_hist[-1]:.3e} y_s={np.array(y_s)}")
        u_apply = np.array(us[0])
        traj_u.append(u_apply)
        traj_ys.append(np.array(y_s))
        x = system.step(x, us[0])
        traj_x.append(np.array(x))
        if save_metrics:
            last_eval = sol.grad_evals[-1]
            last_f = sol.f_hist[-1]
            grad_list.append(float(last_eval))
            fhist_list.append(float(last_f))

    if save_metrics:
        archive_name = "metrics_archive"
        if hasattr(system, 'init_state') and system.init_state is not None and isinstance(system.init_state, str):
            safe_state = system.init_state.replace(' ', '_').replace('/', '_')
            archive_name = f"{archive_name}_{safe_state}"
        archive_path = archive_name + ".npz"
        if os.path.exists(archive_path):
            existing = dict(np.load(archive_path, allow_pickle=True))
        else:
            existing = {}
        existing[grad_key] = np.array(grad_list)
        existing[fhist_key] = np.array(fhist_list)
        np.savez_compressed(archive_path, **existing)

    return np.array(traj_x), np.array(traj_u), np.array(traj_ys)


# one-shot MPC problem builder
def build_problem(system, x_current, Np, callback=None):
    state_dim = system.state_dim
    ctrl_dim = system.ctrl_dim
    N = Np
    z_dim = (N+1)*state_dim + N*ctrl_dim + system.y_dim

    def unpack(z):
        xs = z[: (N+1)*state_dim].reshape(N+1, state_dim)
        us = z[(N+1)*state_dim : (N+1)*state_dim + N*ctrl_dim].reshape(N, ctrl_dim)
        y_s = z[-system.y_dim:]
        return xs, us, y_s

    def f(z):
        xs, us, y_s = unpack(z)
        x_s = system.g_x_of_ys(y_s)
        u_s = system.g_u_of_ys(y_s)
        stage = 0.0
        for k in range(N):
            dx = xs[k] - x_s
            du = us[k] - u_s
            stage += jnp.dot(dx, jnp.dot(system.Q_pos, dx)) + jnp.dot(du, jnp.dot(system.R_u, du))
        offset = jnp.dot(y_s - system.y_ref, jnp.dot(system.T_off, y_s - system.y_ref))
        return stage + offset

    def f_grad(z):
        xs, us, y_s = unpack(z)
        x_s = system.g_x_of_ys(y_s)
        u_s = system.g_u_of_ys(y_s)
        dstage_dx = jnp.zeros_like(xs)
        dstage_du = jnp.zeros_like(us)
        for k in range(N):
            dx = xs[k] - x_s
            du = us[k] - u_s
            dstage_dx[k] = 2.0 * jnp.dot(dx, system.Q_pos)
            dstage_du[k] = 2.0 * jnp.dot(du, system.R_u)
        doffset_dy = 2.0 * jnp.dot(y_s - system.y_ref, system.T_off)
        return jnp.concatenate([dstage_dx, dstage_du, doffset_dy])

    def h(z):
        xs, us, y_s = unpack(z)
        cons = [xs[0] - x_current]
        for k in range(N):
            cons.append(xs[k+1] - system.step(xs[k], us[k]))
        cons.append(xs[-1] - system.g_x_of_ys(y_s))
        return jnp.concatenate(cons)

    def g(z):
        xs, us, y_s = unpack(z)
        vals = []
        for k in range(N+1):
            yk = system.output(xs[k])
            vals.append(jnp.array([system.F_union(yk[0], yk[1])]))
        vals.append(jnp.array([system.F_union(y_s[0], y_s[1])]))
        return jnp.concatenate(vals)

    # box constraints:
    low = -jnp.inf * jnp.ones(z_dim)
    high = jnp.inf * jnp.ones(z_dim)
    for k in range(N+1):
        base = k*state_dim
        low = low.at[base:base+state_dim].set(system.state_low)
        high = high.at[base:base+state_dim].set(system.state_high)
    ctrl_start = (N+1)*state_dim
    for k in range(N):
        for i in range(ctrl_dim):
            idx = ctrl_start + k*ctrl_dim + i
            low = low.at[idx].set(system.ctrl_low[i])
            high = high.at[idx].set(system.ctrl_high[i])
    reg = proxop.BoxConstraint(low=low, high=high)
    problem = pbalm.Problem(
        f=jax.jit(f),
        h=jax.jit(h),
        g=jax.jit(g),
        reg=reg,
        f_grad=jax.jit(f_grad),
        jittable=True,
        callback=None if callback is None else callback(f, h, g, reg)
    )

    xs_list = [x_current]
    x_tmp = x_current
    u_zero = jnp.zeros((2,))
    for _ in range(N):
        x_tmp = system.step(x_tmp, u_zero)
        xs_list.append(x_tmp)
    xs0 = jnp.stack(xs_list)
    us0 = jnp.zeros((N, ctrl_dim))
    y_s0 = xs0[-1, 0:system.y_dim]
    z0 = jnp.concatenate([xs0.reshape(-1), us0.reshape(-1), y_s0])
    return problem, z0, unpack