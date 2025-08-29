import argparse, random
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import diffrax as dfx
import pbalm
import proxop
from plotting import setup_matplotlib

jax.config.update('jax_platform_name', 'cpu')
if not jax.config.jax_enable_x64:
    jax.config.update('jax_enable_x64', True)

SEED = 1234
np.random.seed(SEED)
random.seed(SEED)
jax_key = jax.random.PRNGKey(SEED)

# =============================================================
# Ball-on-Plate MPC (full second-order model with plate dynamics)
# State x in R^8 ordered as:
#   x = [x1, v1, theta1, w1, x2, v2, theta2, w2]
# where: x_i ball position; v_i = \dot x_i; theta_i plate angle; w_i = \dot theta_i.
# Control u = [u1, u2] = [\ddot theta1, \ddot theta2] (plate angular accelerations).
# Ball accelerations depend on angular rates w1, w2.
# Output y = [x1, x2]; terminal equality enforces x_N = [y_s, 0, 0, 0, y_s, 0, 0, 0].
# =============================================================

g0 = 9.80665
bop_dt = 0.25
Np_default = 4

# Physical parameters (generic; yield 5/7 factor for solid sphere)
m = 0.05
r = 0.01
Ib = (2.0/5.0) * m * r**2
c_b = m / (m + Ib / r**2)  # = 5/7 for solid sphere

# Cost weights (penalize position errors only in state part)
Q_pos = jnp.eye(8)
R_u = 10.0 * jnp.eye(2)
T_off = 1e5 * jnp.eye(2)

y_ref = jnp.array([1.0, -0.8])

# Initial state: [x1, v1, theta1, w1, x2, v2, theta2, w2]
# x0_bop = jnp.array([-0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
x0_bop = jnp.array([-0.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

# Only bounded inputs per reference (no explicit state box bounds besides output set Y)
u_max = 0.1  # |u_i| <= 0.1 (here u_i = \ddot theta_i)
# State / steady output bounding box (for numeric safety)
vel_box = 1.5
ang_box = jnp.pi/4.0  # 45 degrees
# Plot range for visualization only
pos_plot_box = 1.5

# Two ellipses defining output set Y = E1 ∪ E2
P1 = jnp.array([[16.0, 0.0], [0.0, 0.5]])
P2 = jnp.array([[5.8551, 7.3707], [7.3707, 10.6449]])

p11_1, p12_1, p22_1 = P1[0,0], P1[0,1], P1[1,1]
p11_2, p12_2, p22_2 = P2[0,0], P2[0,1], P2[1,1]

@jax.jit
def bop_f1(x, y):
    return p11_1 * x**2 + 2.0 * p12_1 * x * y + p22_1 * y**2 - 1.0

@jax.jit
def bop_f2(x, y):
    return p11_2 * x**2 + 2.0 * p12_2 * x * y + p22_2 * y**2 - 1.0

@jax.jit
def bop_F_union(x, y):
    F1 = bop_f1(x, y)
    F2 = bop_f2(x, y)
    return F1 + F2 - jnp.sqrt(F1**2 + F2**2)

# Continuous-time dynamics
# x = [x1, v1, theta1, w1, x2, v2, theta2, w2]; u = [u1, u2] = [\ddot theta1, \ddot theta2]
@jax.jit
def bop_ct_dynamics(t, x, u):
    x1, v1, phi1, w1, x2, v2, phi2, w2 = x
    u1, u2 = u
    a1 = c_b * (x1 * w1**2 + x2 * w1 * w2 + g0 * jnp.sin(phi1))
    a2 = c_b * (x2 * w2**2 + x1 * w1 * w2 + g0 * jnp.sin(phi2))
    return jnp.array([
        v1,      # x1_dot
        a1,      # v1_dot
        w1,      # theta1_dot
        u1,      # w1_dot = theta1_ddot
        v2,      # x2_dot
        a2,      # v2_dot
        w2,      # theta2_dot
        u2       # w2_dot = theta2_ddot
    ])

@jax.jit
def bop_step(x, u, rtol=1e-6, atol=1e-6):
    term = dfx.ODETerm(lambda t, y, args: bop_ct_dynamics(t, y, args))
    solver = dfx.Tsit5()
    sol = dfx.diffeqsolve(term, solver, t0=0.0, t1=bop_dt, dt0=bop_dt,
                          y0=x, args=u, saveat=dfx.SaveAt(t1=True),
                          stepsize_controller=dfx.ConstantStepSize())
    return sol.ys[-1]

# Steady-state mappings
@jax.jit
def g_x_of_ys(y_s):
    # Equilibrium with zero velocities, angles and angular rates
    return jnp.array([y_s[0], 0.0, 0.0, 0.0, y_s[1], 0.0, 0.0, 0.0])

@jax.jit
def g_u_of_ys(y_s):
    return jnp.array([0.0, 0.0])

# Build one-shot MPC problem
def build_bop_problem(x_current, Np: int, prev=None):
    state_dim = 8
    ctrl_dim = 2
    N = Np
    z_dim = (N+1)*state_dim + N*ctrl_dim + 2

    def unpack(z):
        xs = z[: (N+1)*state_dim].reshape(N+1, state_dim)
        us = z[(N+1)*state_dim : (N+1)*state_dim + N*ctrl_dim].reshape(N, ctrl_dim)
        y_s = z[-2:]
        return xs, us, y_s

    def f(z):
        xs, us, y_s = unpack(z)
        x_s = g_x_of_ys(y_s)
        u_s = g_u_of_ys(y_s)
        stage = 0.0
        for k in range(N):
            dx = xs[k] - x_s
            du = us[k] - u_s
            stage += jnp.dot(dx, jnp.dot(Q_pos, dx)) + jnp.dot(du, jnp.dot(R_u, du))
        offset = jnp.dot(y_s - y_ref, jnp.dot(T_off, y_s - y_ref))
        return stage + offset
    
    def f_grad(z):
        xs, us, y_s = unpack(z)
        x_s = g_x_of_ys(y_s)
        u_s = g_u_of_ys(y_s)
        dstage_dx = jnp.zeros_like(xs)
        dstage_du = jnp.zeros_like(us)
        for k in range(N):
            dx = xs[k] - x_s
            du = us[k] - u_s
            dstage_dx[k] = 2.0 * jnp.dot(dx, Q_pos)
            dstage_du[k] = 2.0 * jnp.dot(du, R_u)
        doffset_dy = 2.0 * jnp.dot(y_s - y_ref, T_off)
        return jnp.concatenate([dstage_dx, dstage_du, doffset_dy])

    def h(z):
        xs, us, y_s = unpack(z)
        cons = [xs[0] - x_current]
        for k in range(N):
            cons.append(xs[k+1] - bop_step(xs[k], us[k]))
        cons.append(xs[-1] - g_x_of_ys(y_s))
        return jnp.concatenate(cons)

    def g(z):
        xs, us, y_s = unpack(z)
        vals = []
        for k in range(N+1):
            yk = bop_output(xs[k])
            vals.append(jnp.array([bop_F_union(yk[0], yk[1])]))
        vals.append(jnp.array([bop_F_union(y_s[0], y_s[1])]))
        return jnp.concatenate(vals)

    # Box constraints (states, controls, steady output)
    low = -jnp.inf * jnp.ones(z_dim)
    high = jnp.inf * jnp.ones(z_dim)
    for k in range(N+1):
        base = k*state_dim
        low = low.at[base + 1].set(-vel_box); high = high.at[base + 1].set(vel_box)
        low = low.at[base + 2].set(-ang_box); high = high.at[base + 2].set(ang_box)
        low = low.at[base + 5].set(-vel_box); high = high.at[base + 5].set(vel_box)
        low = low.at[base + 6].set(-ang_box); high = high.at[base + 6].set(ang_box)
    ctrl_start = (N+1)*state_dim
    for k in range(N):
        for i in range(ctrl_dim):
            idx = ctrl_start + k*ctrl_dim + i
            low = low.at[idx].set(-u_max)
            high = high.at[idx].set(u_max)
    # y_s unbounded except by union-of-ellipses constraint
    reg = proxop.BoxConstraint(low=low, high=high)
    problem = pbalm.Problem(
        f=jax.jit(f),
        h=jax.jit(h),
        g=jax.jit(g),
        reg=reg,
        f_grad=jax.jit(f_grad),
        jittable=True
    )

    # Simple warm start: zero controls rollout
    xs_list = [x_current]
    x_tmp = x_current
    u_zero = jnp.zeros((2,))
    for _ in range(N):
        x_tmp = bop_step(x_tmp, u_zero)
        xs_list.append(x_tmp)
    xs0 = jnp.stack(xs_list)
    us0 = jnp.zeros((N, ctrl_dim))
    y_s0 = bop_output(xs0[-1])
    # y_s0 = project_onto_union(y_s0)
    # y_s0 = xs0[-1, 0:2]
    z0 = jnp.concatenate([xs0.reshape(-1), us0.reshape(-1), y_s0])
    return problem, z0, unpack

# Output: ball position
@jax.jit
def bop_output(x):
    # Output: ball position (x1, x2) from 8D state [x1, v1, phi1, w1, x2, v2, phi2, w2]
    return jnp.array([x[0], x[4]])

@jax.jit
def project_onto_union(y):
    # If already feasible
    Fv = bop_F_union(y[0], y[1])
    def proj(_):
        # Projection onto each ellipse centered at origin: scale to boundary if outside
        val1 = jnp.dot(y, jnp.dot(P1, y))
        val2 = jnp.dot(y, jnp.dot(P2, y))
        y1 = jnp.where(val1 <= 1.0, y, y / jnp.sqrt(val1))
        y2 = jnp.where(val2 <= 1.0, y, y / jnp.sqrt(val2))
        # Choose closer in Euclidean distance
        d1 = jnp.linalg.norm(y - y1)
        d2 = jnp.linalg.norm(y - y2)
        return jnp.where(d1 <= d2, y1, y2)
    return jax.lax.cond(Fv <= 0.0, lambda _: y, proj, operand=None)

# Closed-loop simulation
def solve_bop_mpc(mode: str, sim_steps: int, Np: int, verbose: bool=False):
    x = x0_bop
    traj_x = [np.array(x)]
    traj_u = []
    traj_ys = []
    max_iter = 2000
    tol = 1e-5
    xi = 7
    alpha = 9
    delta = 1e-6
    for k in range(sim_steps):
        problem, z0, unpack = build_bop_problem(x, Np, prev=None)
        if mode == 'alm':
            sol = pbalm.solve(problem, z0, use_proximal=False, tol=tol, max_iter=max_iter, start_feas=True,
                               no_reset=True, inner_solver="PANOC", phi_strategy="linear", xi1=xi, xi2=xi,
                               beta=0.5, adaptive_fp_tol=True, rho0=1e-3, nu0=1e-3, phase_I_tol=1e-9)
        elif mode == 'balm':
            sol = pbalm.solve(problem, z0, use_proximal=False, tol=tol, max_iter=max_iter, start_feas=True,
                               inner_solver="PANOC", phi_strategy="pow", xi1=1, xi2=1, alpha=alpha, beta=0.5,
                               adaptive_fp_tol=True, rho0=1e-3, nu0=1e-3, phase_I_tol=1e-9)
        else:
            sol = pbalm.solve(problem, z0, use_proximal=True, tol=tol, max_iter=max_iter, start_feas=True,
                               inner_solver="PANOC", phi_strategy="pow", xi1=1, xi2=1, alpha=alpha, delta=delta,
                               beta=0.5, adaptive_fp_tol=True, rho0=1e-3, nu0=1e-3, gamma0=1e-1, phase_I_tol=1e-9)
        xs, us, y_s = unpack(sol.x)
        if verbose:
            print(f"k={k:02d} infeas={sol.total_infeas[-1]:.2e} f={sol.f_hist[-1]:.3e} y_s={np.array(y_s)}")
        u_apply = np.array(us[0])
        traj_u.append(u_apply)
        traj_ys.append(np.array(y_s))
        x = bop_step(x, us[0])
        traj_x.append(np.array(x))
    return np.array(traj_x), np.array(traj_u), np.array(traj_ys)

# Plotting
def plot_bop_results(traj_x, traj_u, traj_ys, mode):
    setup_matplotlib()
    colors = ['dimgray', 'red', 'black', 'darkred', 'darkgoldenrod', 'royalblue', 'rebeccapurple', 'saddlebrown',
              'darkslategray', 'darkorange', 'steelblue', 'lightcoral']
    linestyles = [
        (5, (10, 3)),
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (3, 1, 1, 1, 1, 1)),
        (0, (3, 5, 1, 5, 1, 5)),
        '--', '-', '-.', ':'
    ]
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>', 'p', 'H', 'h', '1', '2', '3', '4']
    marker_size = 8
    y_traj = np.stack([bop_output(traj_x[i]) for i in range(traj_x.shape[0])], axis=0)
    grid = np.linspace(-pos_plot_box, pos_plot_box, 400)
    Xg, Yg = np.meshgrid(grid, grid, indexing='xy')
    Fgrid = bop_F_union(Xg, Yg)
    plt.figure(figsize=(4.6,4.2), dpi=300)
    plt.contour(Xg, Yg, Fgrid, levels=[0.0], colors='black')
    plt.plot(y_traj[:,0], y_traj[:,1], label=r"$y(t)$", marker=markers[6], markevery=0.1, markerfacecolor='none',
             color=colors[2], markersize=marker_size)
    plt.plot(traj_ys[:,0], traj_ys[:,1],  label=r"$y_s(t)$", color=colors[0], linestyle=linestyles[1],
             marker=markers[0], markevery=0.1, markerfacecolor='none', markersize=marker_size)
    plt.plot(y_ref[0], y_ref[1], label=r"$y_r$", marker=markers[3], color=colors[3])
    plt.xlabel(r"$y_1$"); plt.ylabel(r"$y_2$")
    plt.legend(fontsize=10)
    plt.gca().set_aspect('equal','box')
    plt.tight_layout()
    plt.savefig(f'bop_traj_{mode}.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    t = np.arange(traj_u.shape[0]) * bop_dt
    plt.figure(figsize=(6.0,2.8), dpi=300)
    plt.plot(t, traj_u[:,0], label=r"$\ddot{\theta}_1$")
    plt.plot(t, traj_u[:,1], label=r"$\ddot{\theta}_2$")
    plt.xlabel(r'$\textbf{Time [s]}$'); plt.ylabel(r'$\ddot{\theta}$ $\textbf{[rad/s^2]}$')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'bop_control_{mode}.pdf', format='pdf', bbox_inches='tight')
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['alm','balm','pbalm'], default='pbalm')
parser.add_argument('--steps', type=int, default=20, help='simulation steps')
parser.add_argument('--horizon', type=int, default=Np_default, help='prediction horizon Np')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()
X, U, YS = solve_bop_mpc(mode=args.mode, sim_steps=args.steps, Np=args.horizon, verbose=args.verbose)
plot_bop_results(X, U, YS, args.mode)