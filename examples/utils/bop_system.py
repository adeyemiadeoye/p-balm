import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from utils.plotting import setup_matplotlib


# ball-on-plate system: dynamics,etc
class BOPSystem:

    def __init__(self, init_state=None):
        # phy constants
        self.g0 = 9.80665
        self.dt = 0.25

        # dimensions
        self.state_dim = 8
        self.ctrl_dim = 2
        self.y_dim = 2
        self.Np_default = 15

        m = 0.05
        r = 0.01
        Ib = (2.0/5.0)*m*r**2
        self.c_b = m/(m+Ib/r**2) # = 5/7 for solid sphere (as in paper)

        # cost weights
        self.Q_pos = jnp.diag(jnp.array([10.0, 0.05, 0.05, 0.05, 10.0, 0.05, 0.05, 0.05]))
        self.R_u = 10.0*jnp.eye(2)
        self.T_off = 1e5*jnp.eye(2)

        self.y_ref = jnp.array([1.0, -0.8])

        # initial state
        self.init_state = init_state
        if self.init_state is not None:
            if self.init_state == "s_NW":
                self.x0 = jnp.array([-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif self.init_state == "s_NE":
                self.x0 = jnp.array([-0.1, 0.0, 0.0, 0.0, 1, 0.0, 0.0, 0.0])
            elif self.init_state == "s_SW":
                self.x0 = jnp.array([-0.18, 0.0, 0.0, 0.0, -0.6, 0.0, 0.0, 0.0])
            else:
                raise ValueError(f"Unknown init_state {self.init_state}")
        else:
            self.x0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # actuation limits
        self.u_max = 0.5

        # box constraints on velocities and angles
        self.vel_box = 0.4
        self.ang_box = jnp.pi/4

        s_low = -jnp.inf * jnp.ones(self.state_dim)
        s_high = jnp.inf * jnp.ones(self.state_dim)
        s_low = s_low.at[1].set(-self.vel_box); s_high = s_high.at[1].set(self.vel_box)
        s_low = s_low.at[2].set(-self.ang_box); s_high = s_high.at[2].set(self.ang_box)
        s_low = s_low.at[5].set(-self.vel_box); s_high = s_high.at[5].set(self.vel_box)
        s_low = s_low.at[6].set(-self.ang_box); s_high = s_high.at[6].set(self.ang_box)
        self.state_low = s_low
        self.state_high = s_high

        # ctrl bounds
        self.ctrl_low = -self.u_max * jnp.ones(self.ctrl_dim)
        self.ctrl_high = self.u_max * jnp.ones(self.ctrl_dim)

        # just for plotting..
        self.pos_plot_box = 1.5

        # output set Y = E1 ∪ E2 (ellipses)
        self.P1 = jnp.array([[16.0, 0.0], [0.0, 0.5]])
        self.P2 = jnp.array([[5.8551, 7.3707], [7.3707, 10.6449]])

        # system matrices (linearzed)
        A_lin, B_lin = self.linearize_ball_plate()
        self.Ad, self.Bd = self.discretize_system(A_lin, B_lin, self.dt)

    def bop_f1(self, x, y):
        p11_1, p12_1, p22_1 = self.P1[0, 0], self.P1[0, 1], self.P1[1, 1]
        return p11_1 * x**2 + 2.0 * p12_1 * x * y + p22_1 * y**2 - 1.0

    def bop_f2(self, x, y):
        p11_2, p12_2, p22_2 = self.P2[0, 0], self.P2[0, 1], self.P2[1, 1]
        return p11_2 * x**2 + 2.0 * p12_2 * x * y + p22_2 * y**2 - 1.0

    def f1(self, x, y):
        return self.bop_f1(x, y)

    def f2(self, x, y):
        return self.bop_f2(x, y)

    def F_union(self, x, y):
        F1 = self.bop_f1(x, y)
        F2 = self.bop_f2(x, y)
        return F1 + F2 - jnp.sqrt(F1**2 + F2**2)

    def linearize_ball_plate(self):
        A = jnp.zeros((8, 8))
        A = A.at[0, 1].set(1.0)
        A = A.at[1, 2].set(self.g0 * self.c_b)
        A = A.at[2, 3].set(1.0)
        A = A.at[4, 5].set(1.0)
        A = A.at[5, 6].set(self.g0 * self.c_b)
        A = A.at[6, 7].set(1.0)

        B = jnp.zeros((8, 2))
        B = B.at[3, 0].set(1.0)
        B = B.at[7, 1].set(1.0)

        return A, B

    def discretize_system(self, A, B, Ts):
        n = A.shape[0]
        m = B.shape[1]
        M = jnp.zeros((n + m, n + m))
        M = M.at[:n, :n].set(A)
        M = M.at[:n, n:].set(B)
        E = jsp_linalg.expm(M * Ts)
        Ad = E[:n, :n]
        Bd = E[:n, n:]
        return Ad, Bd

    def step(self, x, u):
        return jnp.dot(self.Ad, x) + jnp.dot(self.Bd, u)

    def g_x_of_ys(self, y_s):
        return jnp.array([y_s[0], 0.0, 0.0, 0.0, y_s[1], 0.0, 0.0, 0.0])

    def g_u_of_ys(self, y_s):
        return jnp.array([0.0, 0.0])

    def output(self, x):
        return jnp.array([x[0], x[4]])

    def plot_results(self, X, U, YS, mode, xi=None, alpha=None, name_append=None):
        setup_matplotlib(font_scale=3.5)
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
        marker_size = 10
        star = mpath.Path.unit_regular_star(6)
        circle = mpath.Path.unit_circle()
        cut_star = mpath.Path(
            vertices=np.concatenate([circle.vertices, star.vertices]),
            codes=np.concatenate([circle.codes, star.codes])
        )
        y_traj = np.stack([self.output(X[i]) for i in range(X.shape[0])], axis=0)
        ys_traj = np.stack(YS, axis=0)
        grid = np.linspace(-self.pos_plot_box, self.pos_plot_box, 400)
        Xg, Yg = np.meshgrid(grid, grid, indexing='xy')
        Fgrid = self.F_union(Xg, Yg)
        if mode == "alm":
            suff = f"{xi}"
        else:
            if mode == "pbalm-star":
                suff = "star_"+f"{alpha}"
            else:
                suff = f"{alpha}"
        if name_append:
            suff += f"_{name_append}"
        plt.figure(figsize=(7,6), dpi=300)
        plt.contour(Xg, Yg, Fgrid, levels=[0.0], colors='black')
        plt.plot(y_traj[:, 0], y_traj[:, 1], label=r"$y(t)$", marker=cut_star, markevery=0.02,
                 color=colors[5], markersize=marker_size+1, markerfacecolor='white')
        plt.plot(ys_traj[:,0], ys_traj[:,1],  label=r"$y_s(t)$", color=colors[0], linestyle=linestyles[1],
                 marker=markers[0], markevery=0.02, markerfacecolor='none', markersize=marker_size)
        plt.plot(self.y_ref[0], self.y_ref[1], label=r"$y_r$", marker=markers[7], color=colors[3], linestyle='None', markersize=marker_size+1)
        plt.plot(self.x0[0], self.x0[4], label=r"$y(0)$", marker=markers[10], color=colors[4], linestyle='None',
                 markersize=marker_size+1)
        plt.xlabel(r"$y_1$"+r" $\bf{[m]}$"); plt.ylabel(r"$y_2$"+r" $\bf{[m]}$")
        plt.legend(fontsize=18)
        plt.gca()
        plt.tight_layout()
        plt.savefig(f'bop_traj_{mode}_{suff}.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        t = np.arange(U.shape[0]) * self.dt
        plt.figure(figsize=(8,4), dpi=300)
        plt.plot(t, U[:,0], label=r'$\ddot{\theta}_1$')
        plt.plot(t, U[:,1], label=r'$\ddot{\theta}_2$')
        plt.xlabel(r'$\bf{time [s]}$'); plt.ylabel(r'$\ddot{\theta}$'+r' $\bf{[rad/s^2]}$')
        plt.legend(fontsize=18)
        plt.tight_layout()
        plt.savefig(f'bop_control_{mode}_{suff}.pdf', format='pdf', bbox_inches='tight')
        plt.close()
