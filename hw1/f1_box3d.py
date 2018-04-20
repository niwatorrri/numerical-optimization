import newton as nt
import numpy as np
from numpy import exp, log


class box3d(object):
    """ Box three-dimensional function """
    def __init__(self, m):
        self.m = m
        
    def f(self, x): # compute f
        fx = 0
        for k in range(self.m):
            tk = 0.1 * k
            rk = exp(-tk * x[0]) - exp(-tk * x[1]) - x[2] * (exp(-tk) - exp(-k))
            fx += rk ** 2
        return fx

    def g(self, x): # compute g
        gx = np.zeros(3)
        for k in range(self.m):
            tk = 0.1 * k
            rk = exp(-tk * x[0]) - exp(-tk * x[1]) - x[2] * (exp(-tk) - exp(-k))
            gk = np.array([-tk * exp(-tk * x[0]),
                            tk * exp(-tk * x[1]),
                            exp(-k) - exp(-tk)])
            gx = gx + 2 * rk * gk
        return gx

    def G(self, x): # compute G
        Gx = np.zeros((3, 3))
        for k in range(self.m):
            tk = 0.1 * k
            rk = exp(-tk * x[0]) - exp(-tk * x[1]) - x[2] * (exp(-tk) - exp(-k))
            gk = np.array([-tk * exp(-tk * x[0]),
                           tk * exp(-tk * x[1]),
                           exp(-k) - exp(-tk)])
            Gk = np.diag([(tk ** 2) * exp(-tk * x[0]),
                          (-tk ** 2) * exp(-tk * x[1]), 0])
            Gx = Gx + np.outer(gk, 2 * gk) + (2 * rk) * Gk
        return Gx


n = 3
x0 = np.array([0, 10, 20])
H0 = np.eye(3)

for m in [3, 5, 10, 15, 20]:
    fun = box3d(m)
    # x = nt.Newton(fun, x0, method='damped')
    # x = nt.modifiedNewton(fun, x0, method='lm')
    # x = nt.quasiNewton(fun, x0, H0, method='bfgs')
    print('& %.3e & %.1f & %d & %d'%(x[1], -log(x[2]) / log(10), x[3], x[4]))
