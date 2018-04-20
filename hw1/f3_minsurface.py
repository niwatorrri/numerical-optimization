import newton as nt
import numpy as np
from numpy import log
from scipy.optimize import broyden2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class minimalSurface(object):
    """ Finite element approximation to Enneper surface problem """
    def __init__(self, N):
        self.N = N

        def boundary(x):
            # Boundary values obtained by nonlinear solver
            x = np.array(x) / N - 0.5
            def fun(v):
                return [v[0] + v[0] * v[1] ** 2 - v[0] ** 3 / 3 - x[0],
                        -v[1] - v[1] * v[0] ** 2 + v[1] ** 3 / 3 - x[1]]
            v = broyden2(fun, [0, 0])
            return v[0] ** 2 - v[1] ** 2

        # Store boundary values for future use
        u = {}
        for i in range(N + 1):
            u[-1, i] = boundary([0, i])
            u[i, -1] = boundary([i, 0])
            u[N - 1, i] = boundary([N, i])
            u[i, N - 1] = boundary([i, N])
        self.u = u


    def dx(self, u, i, j): # helper routine 1
        u_0 = u[i - 1, j] if i > 0 else self.u[-1, j]
        u_2 = u[i + 1, j] if i < N - 2 else self.u[N - 1, j]
        u_1 = u[i, j]
        return (u_0 + u_2 - 2 * u_1)

    def dy(self, u, i, j): # helper routine 2
        u_0 = u[i, j - 1] if j > 0 else self.u[i, -1]
        u_2 = u[i, j + 1] if j < N - 2 else self.u[i, N - 1]
        u_1 = u[i, j]
        return (u_0 + u_2 - 2 * u_1)


    def f(self, x): # compute f
        N = self.N
        fx = 0
        u = np.reshape(x, (N - 1, N - 1))

        for i in range(N - 1):
            for j in range(N - 1):
                fx += np.sqrt(N ** (-4) + self.dx(u, i, j) ** 2
                                        + self.dy(u, i, j) ** 2)
        return fx

    def g(self, x): # compute g
        N = self.N
        gx = np.zeros((N - 1, N - 1))
        u = np.reshape(x, (N - 1, N - 1))

        for i in range(N - 1):
            for j in range(N - 1):
                dx = self.dx(u, i, j)
                dy = self.dy(u, i, j)
                f = np.sqrt(N ** (-4) + dx ** 2 + dy ** 2)
                if i > 0:
                    gx[i - 1, j] += dx / f
                if i < N - 2:
                    gx[i + 1, j] += dx / f
                if j > 0:
                    gx[i, j - 1] += dy / f
                if j < N - 2:
                    gx[i, j + 1] += dy / f
                gx[i, j] += -2 * (dx + dy) / f

        gx = np.reshape(gx, (N - 1) ** 2)
        return gx


for N in [10, 20, 30]:
    # Compute numerical solution
    x0 = np.zeros((N - 1) ** 2)
    H0 = np.eye((N - 1) ** 2)
    fun = minimalSurface(N)
    x = nt.quasiNewton(fun, x0, H0, method='sr1')
    print('& %.3e & %.1f & %d & %d'%(x[1], -log(x[2]) / log(10), x[3], x[4]))

    # Plot surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(-0.5 + 1 / N, 0.5, 1 / N)
    Y = np.arange(-0.5 + 1 / N, 0.5, 1 / N)
    X, Y = np.meshgrid(X, Y)
    U = np.reshape(x[0], (N - 1, N - 1))
    surf = ax.plot_surface(X, Y, U, cmap=plt.cm.coolwarm)
    # plt.show()
    # plt.savefig('surface_bfgs_%d.png'%N)
