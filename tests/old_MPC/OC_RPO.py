import jax
import jax.numpy as jnp
import numpy as np
import pbalm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from plotting import setup_matplotlib
import proxop

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update('jax_enable_x64', True)

"""
Rendezvous and Proximity Operations (RPO) optimal control problem.
(Updated) We replace the physical thrust 2-norm integral with an L1 model acting on the thrust components.
Reason: (i) L1 still promotes fuel / impulse minimization while encouraging componentwise sparsity in thrust
commands; (ii) it yields a separable proximal operator (soft-threshold) efficiently handled by the solver;
(iii) removes the need for a smoothing epsilon used earlier for differentiability of ||T||.
The smooth objective f is set to zero; the solver receives reg_lbda to weight the L1 term internally.
Nonconvexity arises only from the concave KOZ constraints.

Normalized continuous-time dynamics (dimensionless form referenced):
    r' = V
    V' = -(1/||r||^3) r + T / m
    m' = - (1 / v_ex) ||T||
where r is scaled by R0, V by sqrt(g0 R0), T by m0 g0, m by m0, and time by sqrt(R0 / g0). Here we keep
existing dimensional model for relative motion but adopt the mass depletion form using the effective exhaust
velocity v_ex directly (v_ex = Isp * g0 in dimensional settings). Thus m_dot = -||T|| / v_ex.
"""

# Physical / problem constants
mu_earth = 3.986004418e14  # m^3/s^2
R_earth = 6378e3           # m
g0 = 9.80665               # m/s^2
# Effective exhaust velocity (can be Isp * g0 if Isp given). Using value consistent with prior Isp=200s.
v_ex = 200.0 * g0          # m/s (was Isp * g0 with Isp=200 s)
Tmax = 30.0                # N (component-wise bound after norm→abs conversion)

# Orbit parameters
perigee_alt = 350e3
rp = R_earth + perigee_alt
e = 0.3
a = rp / (1 - e)

# Time parameters
tf = 4000.0
# Use a time mesh of 400 nodes for discretization (N_nodes nodes => N=N_nodes-1 intervals)
N_nodes = 40
N = N_nodes - 1  # number of intervals
dt = tf / N

# Event times
t1 = 2000.0
t2 = 3200.0

# KOZ ellipsoid semi-axes (meters)
koz_ax = 200.0  # along V-bar (x)
koz_ay = 100.0  # along H-bar (y)
koz_az = 100.0  # along R-bar (z)

# Acquisition target relative position at t1
acq_pos = jnp.array([200.0, 0.0, 0.0])

# Initial relative state (given)
r_rel0 = jnp.array([-6000.0, 0.0, 300.0])
V_rel0 = jnp.array([-0.5, 0.0, 0.2])
m0 = 1500.0

# Velocity bounds schedule (per component after norm→abs conversion)
vel_bound_acq = 0.2
vel_bound_mid = 0.3
vel_bound_final = 0.1

# Small epsilon for smooth 1-norm surrogate
fuel_eps = 1e-8

# Build problem

def kepler_E(M, e, tol=1e-12, max_iter=50):
    def body(E, M):
        return E - e * jnp.sin(E) - M
    def bodyp(E):
        return 1 - e * jnp.cos(E)
    E0 = M
    def newton(carry):
        E, i = carry
        f = body(E, M)
        fp = bodyp(E)
        E_new = E - f / fp
        return (E_new, i + 1)
    def cond(carry):
        E, i = carry
        return jnp.logical_and(jnp.abs(body(E, M)) > tol, i < max_iter)
    E_final, _ = jax.lax.while_loop(cond, newton, (E0, 0))
    return E_final

@jax.jit
def target_state(t):
    n = jnp.sqrt(mu_earth / a**3)
    f0 = jnp.deg2rad(10.0)
    E0 = 2 * jnp.arctan(jnp.tan(f0/2) * jnp.sqrt((1 - e) / (1 + e)))
    M0 = E0 - e * jnp.sin(E0)
    M = M0 + n * t
    E = kepler_E(M, e)
    f = 2 * jnp.arctan(jnp.tan(E/2) * jnp.sqrt((1 + e) / (1 - e)))
    r_mag = a * (1 - e * jnp.cos(E))
    r_pf = r_mag * jnp.array([jnp.cos(f), jnp.sin(f), 0.0])
    v_pf = jnp.sqrt(mu_earth * a) / r_mag * jnp.array([-jnp.sin(E), jnp.sqrt(1 - e**2) * jnp.cos(E), 0.0])
    r_I = r_pf
    v_I = v_pf
    return r_I, v_I

@jax.jit
def lvlh_dcm(r_I, v_I):
    r_hat = r_I / jnp.linalg.norm(r_I)
    h = jnp.cross(r_I, v_I)
    h_hat = h / jnp.linalg.norm(h)
    z_L = -r_hat
    y_L = h_hat
    x_L = jnp.cross(y_L, z_L)
    return jnp.stack([x_L, y_L, z_L], axis=1)


def build_rpo_problem(thrust_l1_weight: float = 1.0):
    # times etc.
    times = jnp.linspace(0.0, tf, N + 1)
    r_t_list = []
    v_t_list = []
    C_list = []
    for t in times:
        rI, vI = target_state(t)
        C = lvlh_dcm(rI, vI)
        r_t_list.append(rI)
        v_t_list.append(vI)
        C_list.append(C)
    r_t = jnp.stack(r_t_list)
    v_t = jnp.stack(v_t_list)

    k1 = int(np.round(t1 / dt))
    k2 = int(np.round(t2 / dt))

    state_dim = 7
    ctrl_dim = 3

    def unpack_z(z):
        x_traj = z[: (N + 1) * state_dim].reshape(N + 1, state_dim)
        u_traj = z[(N + 1) * state_dim :].reshape(N, ctrl_dim)
        return x_traj, u_traj

    def dynamics(xk, uk, rt, vt):
        r_rel = xk[0:3]
        v_rel = xk[3:6]
        m = xk[6]
        rt_norm = jnp.linalg.norm(rt)
        rc = rt + r_rel
        rc_norm = jnp.linalg.norm(rc)
        a_rel_grav = -mu_earth * rc / rc_norm**3 + mu_earth * rt / rt_norm**3
        a_control = uk / jnp.maximum(m, 1.0)
        r_rel_dot = v_rel
        v_rel_dot = a_rel_grav + a_control
        # Mass depletion uses effective exhaust velocity v_ex (dimensional) directly per normalized form m'=-||T||/v_ex
        m_dot = - jnp.linalg.norm(uk) / v_ex
        return jnp.concatenate([r_rel_dot, v_rel_dot, jnp.array([m_dot])])

    def rk4_step(xk, uk, rt, vt):
        k1_vec = dynamics(xk, uk, rt, vt)
        k2_vec = dynamics(xk + 0.5 * dt * k1_vec, uk, rt, vt)
        k3_vec = dynamics(xk + 0.5 * dt * k2_vec, uk, rt, vt)
        k4_vec = dynamics(xk + dt * k3_vec, uk, rt, vt)
        return xk + dt * (k1_vec + 2*k2_vec + 2*k3_vec + k4_vec) / 6.0

    def f(z):
        # Discrete-time approximation of the integral ∫ ||T(t)||^2 dt using rectangle rule
        _, u_traj = unpack_z(z)
        return dt * jnp.sum(u_traj**2)

    r_rel0_vec = r_rel0
    V_rel0_vec = V_rel0

    C_r0 = jnp.hstack([jnp.eye(3), jnp.zeros((3,3)), jnp.zeros((3,1))])          # 3x7
    d_r0 = -r_rel0
    C_v0 = jnp.hstack([jnp.zeros((3,3)), jnp.eye(3), jnp.zeros((3,1))])          # 3x7
    d_v0 = -V_rel0
    C_m0 = jnp.hstack([jnp.zeros((1,6)), jnp.array([[1.0]])])                    # 1x7
    d_m0 = -jnp.array([m0])
    C_acq = C_r0
    d_acq = -acq_pos
    Cf = jnp.block([
        [jnp.eye(3), jnp.zeros((3,3)), jnp.zeros((3,1))],
        [jnp.zeros((3,3)), jnp.eye(3), jnp.zeros((3,1))]
    ])  # 6x7
    df = jnp.zeros(6)

    def h(z):
        x_traj, u_traj = unpack_z(z)
        cons = []
        # Initial (linear)
        cons.append(C_r0 @ x_traj[0] + d_r0)
        cons.append(C_v0 @ x_traj[0] + d_v0)
        cons.append(C_m0 @ x_traj[0] + d_m0)
        # Dynamics (nonlinear)
        for k in range(N):
            x_next_pred = rk4_step(x_traj[k], u_traj[k], r_t[k], v_t[k])
            cons.append(x_traj[k+1] - x_next_pred)
        # Acquisition (linear)
        cons.append(C_acq @ x_traj[k1] + d_acq)
        # Terminal (linear)
        cons.append(Cf @ x_traj[-1] + df)
        return jnp.concatenate(cons)

    def g(z):
        x_traj, _ = unpack_z(z)
        vals = []
        for k in range(k1+1):
            r_rel = x_traj[k,0:3]
            s_koz = 1.0 - (r_rel[0]/koz_ax)**2 - (r_rel[1]/koz_ay)**2 - (r_rel[2]/koz_az)**2
            vals.append(jnp.array([s_koz]))
        if len(vals) == 0:
            return jnp.zeros(1)
        return jnp.concatenate(vals)

    z_dim = (N + 1) * state_dim + N * ctrl_dim
    low = np.full(z_dim, -np.inf)
    high = np.full(z_dim, np.inf)

    def state_index(k):
        return k * state_dim

    for k in range(N + 1):
        base = state_index(k)
        if k == k1:
            vbound = vel_bound_acq
        elif k1 < k <= k2:
            vbound = vel_bound_mid
        elif k > k2:
            vbound = vel_bound_final
        else:
            vbound = 1e6
        for i in range(3):
            low[base + 3 + i] = -vbound
            high[base + 3 + i] = vbound
        low[base + 6] = 100.0
        high[base + 6] = m0

    ctrl_start = (N + 1) * state_dim
    for k in range(N):
        for i in range(ctrl_dim):
            idx = ctrl_start + k * ctrl_dim + i
            low[idx] = -Tmax
            high[idx] = Tmax

    reg = proxop.BoxConstraint(low=jnp.array(low), high=jnp.array(high))

    # Vector reg_lbda_list: zero for all state variables, dt*thrust_l1_weight for each control component (Python list expected)
    reg_lbda_list = jnp.zeros(z_dim).at[ctrl_start:].set(thrust_l1_weight*dt).tolist()

    problem = pbalm.Problem(f=f, h=h, g=g, reg=reg, reg_lbda=reg_lbda_list, jittable=True)
    # problem = pbalm.Problem(f=f, h=h, g=g, reg=reg, jittable=True)

    # ---------- Build a feasible initial point by shooting with three piecewise-constant thrust phases ----------
    # Phases: [0, k1) with u1; [k1, kmid) with u2; [kmid, N) with u3
    kmid = (k1 + N) // 2
    kmid = max(kmid, k1 + 1)  # ensure at least one step in last phase

    x_init = np.hstack([np.array(r_rel0), np.array(V_rel0), np.array([m0])])

    def unroll_to_k1(u1):
        xk = jnp.array(x_init)
        for kk in range(0, k1):
            xk = rk4_step(xk, jnp.array(u1), r_t[kk], v_t[kk])
        return np.array(xk)

    def unroll_with_three(u1, u2, u3, store=False):
        xk = jnp.array(x_init)
        X = None
        if store:
            X = np.zeros((N + 1, state_dim))
            X[0] = np.array(xk)
        # phase 1
        for kk in range(0, k1):
            xk = rk4_step(xk, jnp.array(u1), r_t[kk], v_t[kk])
            if store:
                X[kk + 1] = np.array(xk)
        # phase 2
        for kk in range(k1, kmid):
            xk = rk4_step(xk, jnp.array(u2), r_t[kk], v_t[kk])
            if store:
                X[kk + 1] = np.array(xk)
        # phase 3
        for kk in range(kmid, N):
            xk = rk4_step(xk, jnp.array(u3), r_t[kk], v_t[kk])
            if store:
                X[kk + 1] = np.array(xk)
        return (np.array(xk) if not store else X)

    # Solve for u1 so that r(k1) = acq_pos
    u1 = np.zeros(3)
    for _ in range(100):
        xk1 = unroll_to_k1(u1)
        r_err = xk1[0:3] - np.array(acq_pos)
        if np.linalg.norm(r_err) < 1e-8:
            break
        # Finite-difference Jacobian of r(k1) w.r.t u1 (3x3)
        eps = 1e-6
        J = np.zeros((3, 3))
        for i in range(3):
            du = np.zeros(3); du[i] = eps
            r_plus = unroll_to_k1(u1 + du)[0:3]
            r_base = xk1[0:3]
            J[:, i] = (r_plus - r_base) / eps
        # Solve J * delta = -r_err
        try:
            delta = np.linalg.solve(J, -r_err)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, -r_err, rcond=None)[0]
        u1 = u1 + delta

    # Solve for u2, u3 so that r(tf)=0 and v(tf)=0
    u2 = np.zeros(3)
    u3 = np.zeros(3)
    for _ in range(12):
        xN = unroll_with_three(u1, u2, u3, store=False)
        rv_err = np.hstack([xN[0:3], xN[3:6]])  # desire zeros
        if np.linalg.norm(rv_err) < 1e-8:
            break
        # Finite-difference Jacobian of [r(tf); v(tf)] w.r.t [u2; u3] (6x6)
        eps = 1e-4
        J = np.zeros((6, 6))
        # columns 0..2: u2, columns 3..5: u3
        for i in range(3):
            du = np.zeros(3); du[i] = eps
            xN_plus = unroll_with_three(u1, u2 + du, u3, store=False)
            J[:, i] = (np.hstack([xN_plus[0:3], xN_plus[3:6]]) - rv_err) / eps
        for i in range(3):
            du = np.zeros(3); du[i] = eps
            xN_plus = unroll_with_three(u1, u2, u3 + du, store=False)
            J[:, 3 + i] = (np.hstack([xN_plus[0:3], xN_plus[3:6]]) - rv_err) / eps
        # Solve J * delta = -rv_err
        try:
            delta = np.linalg.solve(J, -rv_err)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, -rv_err, rcond=None)[0]
        u2 = u2 + delta[0:3]
        u3 = u3 + delta[3:6]

    # Build full trajectories with solved (u1, u2, u3)
    x0_traj = unroll_with_three(u1, u2, u3, store=True)
    u0_traj = np.zeros((N, ctrl_dim))
    if k1 > 0:
        u0_traj[0:k1, :] = u1
    if kmid > k1:
        u0_traj[k1:kmid, :] = u2
    if N > kmid:
        u0_traj[kmid:N, :] = u3

    # Clip controls to box (keeps z0 inside bounds); dynamics equalities remain satisfied since x already built from these controls
    u0_traj = np.clip(u0_traj, -Tmax, Tmax)

    z0 = jnp.concatenate([jnp.array(x0_traj).reshape(-1), jnp.array(u0_traj).reshape(-1)])

    aux = {'times': times, 'k1': k1, 'k2': k2, 'dt': dt, 'r_t': r_t, 'v_t': v_t,
           'C_r0': C_r0, 'C_v0': C_v0, 'C_m0': C_m0, 'C_acq': C_acq, 'Cf': Cf}
    return problem, z0, aux


def safe_for_plot(arr, max_finite=1e10):
    arr = np.asarray(arr)
    arr = np.where(np.isposinf(arr), max_finite, arr)
    arr = np.where(np.isneginf(arr), -max_finite, arr)
    arr = np.where(arr > max_finite, max_finite, arr)
    arr = np.where(arr < -max_finite, -max_finite, arr)
    return arr


def run_rpo(thrust_l1_weight: float = 1.0):
    problem, z0, aux = build_rpo_problem(thrust_l1_weight=thrust_l1_weight)
    # print(z0)
    # print(z0.shape, len(problem.reg_lbda))
    # print(problem.g(z0).shape, problem.h(z0).shape)
    f_star = float(problem.f(z0))
    f0 = f_star
    f0_f_star = 1.0
    alpha_vals = [6]
    xi_vals_alm = [10]
    xi = 1
    delta = 1e-6
    tol = 1e-5
    max_iter = 400
    adaptive_fp_tol = True
    feas_meas_results = []
    grad_evals_results = []
    f_hist_results = []
    legends = []
    markers = ['o','s','D','^','v','P','*','X']
    colors = plt.cm.Dark2(np.linspace(0,1,8))
    marker_size = 7
    for xi_alm in xi_vals_alm:
        sol_alm = pbalm.solve(problem, z0, use_proximal=False, tol=tol, max_iter=max_iter, start_feas=True,
                               no_reset=True, inner_solver="PANOC", phi_strategy="linear", xi1=xi_alm, xi2=xi_alm,
                               beta=0.5, adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3)
        feas_meas_results.append(np.array(sol_alm.total_infeas))
        grad_evals_results.append(sol_alm.grad_evals)
        f_hist_results.append(np.array(sol_alm.f_hist) - f_star)
        legends.append(r"\texttt{ALM}")
        problem.reset_counters()
    for alpha in alpha_vals:
        sol_pbalm = pbalm.solve(problem, z0, use_proximal=True, tol=tol, max_iter=max_iter, start_feas=True,
                                 inner_solver="PANOC", phi_strategy="pow", xi1=xi, xi2=xi, alpha=alpha, delta=delta,
                                 beta=0.5, adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3, gamma0=1e-1)
        feas_meas_results.append(np.array(sol_pbalm.total_infeas))
        grad_evals_results.append(sol_pbalm.grad_evals)
        f_hist_results.append(np.array(sol_pbalm.f_hist) - f_star)
        legends.append(r"\texttt{P-BALM}")
        problem.reset_counters()
    for alpha in alpha_vals:
        sol_balm = pbalm.solve(problem, z0, use_proximal=False, tol=tol, max_iter=max_iter, start_feas=True,
                                inner_solver="PANOC", phi_strategy="pow", xi1=xi, xi2=xi, alpha=alpha, beta=0.5,
                                adaptive_fp_tol=adaptive_fp_tol, rho0=1e-3, nu0=1e-3)
        feas_meas_results.append(np.array(sol_balm.total_infeas))
        grad_evals_results.append(sol_balm.grad_evals)
        f_hist_results.append(np.array(sol_balm.f_hist) - f_star)
        legends.append(r"\texttt{BALM}")
        problem.reset_counters()
    setup_matplotlib()
    plt.figure(figsize=(7,5), dpi=200)
    for idx,(ge, infeas, leg) in enumerate(zip(grad_evals_results, feas_meas_results, legends)):
        infeas = safe_for_plot(infeas)
        plt.plot(ge, np.maximum(infeas, 1e-8), label=leg, marker=markers[idx%len(markers)],
                 markevery=0.15, markerfacecolor='none', color=colors[idx%len(colors)], linestyle='dashdot',
                 markersize=marker_size)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r'$\textbf{grad evals}$')
    plt.ylabel(r'$\textbf{total infeas}$')
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('rpo_infeas.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(7,5), dpi=200)
    for idx,(ge, fh, leg) in enumerate(zip(grad_evals_results, f_hist_results, legends)):
        fh = safe_for_plot(fh)
        plt.plot(ge, np.maximum(np.abs(fh)/max(np.abs(f0_f_star),1.0), 1e-8), label=leg,
                 marker=markers[idx%len(markers)], markevery=0.15, markerfacecolor='none',
                 color=colors[idx%len(colors)], linestyle='dashdot', markersize=marker_size)
    plt.xscale('log'); plt.yscale('log')
    plt.ylabel(r'$|f(z^k)-f^*| / |f(z^0)-f^*|$')
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('rpo_obj_gap.pdf', format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_rpo(thrust_l1_weight=1.0)