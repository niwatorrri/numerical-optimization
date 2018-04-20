import newton as nt
import numpy as np
from numpy import exp, log


class penalty2(object):
    """ Penalty II function """
    def __init__(self, n):
        self.n = n
        
    def f(self, x): # compute f
        fx = (x[0] - 0.2) ** 2
        tmp = 0
        a = 1e-5

        for k in range(self.n):
            if k < n - 1:
                yk = exp((k + 2) / 10) + exp((k + 1) / 10)
                fx += a * (exp(x[k + 1] / 10) + exp(x[k] / 10) - yk) ** 2
                fx += a * (exp(x[k + 1] / 10)- exp(-1 / 10)) ** 2            
            tmp += (n - k) * (x[k] ** 2)
        fx += (tmp - 1) ** 2
        return fx

    def g(self, x): # compute g
        gx = np.zeros(self.n)
        gx[0] = 1
        gx = 2 * (x[0] - 0.2) * gx

        tmpr = 0
        tmpg = np.zeros(self.n)
        a = 1e-5

        for k in range(self.n):
            if k < n - 1:
                rk = exp(x[k + 1] / 10)- exp(-1 / 10)
                gx[k + 1] += 2 * a * rk * exp(x[k + 1] / 10) / 10

                yk = exp((k + 2) / 10) + exp((k + 1) / 10)
                rk = exp(x[k + 1] / 10) + exp(x[k] / 10) - yk
                gx[k + 1] += 2 * a * rk * exp(x[k + 1] / 10) / 10
                gx[k] += 2 * a * rk * exp(x[k] / 10) / 10

            tmpr += (n - k) * (x[k] ** 2)
            tmpg[k] = 2 * (n - k) * x[k]

        gx = gx + 2 * (tmpr - 1) * tmpg
        return gx

    def G(self, x): # compute G
        Gx = np.zeros((self.n, self.n))
        Gx[0, 0] = 1

        tmpr = 0
        tmpg = np.zeros(self.n)
        a = 1e-5

        for k in range(self.n):
            if k < n - 1:
                rk = exp(x[k + 1] / 10)- exp(-1 / 10)
                Gx[k + 1, k + 1] += a * exp(x[k + 1] / 5) / 100
                Gx[k + 1, k + 1] += a * rk * exp(x[k + 1] / 10) / 100

                yk = exp((k + 2) / 10) + exp((k + 1) / 10)
                rk = exp(x[k + 1] / 10) + exp(x[k] / 10) - yk
                Gx[k + 1, k + 1] += a * exp(x[k + 1] / 5) / 100
                Gx[k, k] += a * exp(x[k] / 5) / 100  
                diag = a * exp((x[k + 1] + x[k]) / 10) / 100
                Gx[k + 1, k] += diag
                Gx[k, k + 1] += diag
              
                Gx[k + 1, k + 1] += a * rk * exp(x[k + 1] / 10) / 100
                Gx[k, k] += a * rk * exp(x[k] / 10) / 100

            tmpr += (n - k) * (x[k] ** 2)
            tmpg[k] = 2 * (n - k) * x[k]

        Gx += np.outer(tmpg, tmpg)
        Gx += 2 * (tmpr - 1) * np.diag(np.arange(n, 0, -1))
        Gx = Gx * 2
        return Gx


for n in [2, 4, 6, 8, 10]:
    x0 = np.ones(n) / 2
    fun = penalty2(n)
    # x = nt.Newton(fun, x0, method='damped')
    # x = nt.modifiedNewton(fun, x0, method='lm')
    # x = nt.quasiNewton(fun, x0, method='bfgs')
    print('& %.3e & %.1f & %d & %d'%(x[1], -log(x[2]) / log(10), x[3], x[4]))

