import cg, bb
import numpy as np
from numpy import log


class GENPOWSG(object):
    """ Generalized Powell singular function """
    def __init__(self, n, scale=1):
        self.n = n
        self.scale = scale
        
    def f(self, x): # compute f
        x1 = x[0:-2:2]
        x2 = x[1:-2:2]
        x3 = x[2::2]
        x4 = x[3::2]
        fx = np.sum((x1 + 10 * x2) ** 2) + 5 * np.sum((x3 - x4) ** 2) \
             + np.sum((x2 - 2 * x3) ** 4) + 10 * np.sum((x1 - x4) ** 4)
        return fx * self.scale

    def g(self, x): # compute g
        gx = np.zeros(self.n)
        x1 = x[0:-2:2]
        x2 = x[1:-2:2]
        x3 = x[2::2]
        x4 = x[3::2]
        gx[0:-2:2] += 2 * (x1 + 10 * x2) + 40 * (x1 - x4) ** 3
        gx[1:-2:2] += 20 * (x1 + 10 * x2) + 4 * (x2 - 2 * x3) ** 3
        gx[2::2] += 10 * (x3 - x4) - 8 * (x2 - 2 * x3) ** 3
        gx[3::2] += - 10 * (x3 - x4) - 40 * (x1 - x4) ** 3
        return gx * self.scale


n_list = [1000, 2000, 5000, 10000]
for n in n_list:
    x = np.tile(np.array([3, -1]), n // 2)
    scale = 0.1
    eps = 1e-8 * scale
    fun = GENPOWSG(n, scale)

    for method in ['fr', 'prp', 'prp+', 'hs', 'cd', 'dy', 'bb', 'sd']:
        if method == 'bb':
            res = bb.bb(fun, x, eps=eps)
        elif method == 'sd':
            res = bb.sd(fun, x, eps=eps)
        else:
            res = cg.cg(fun, x, method=method, eps=eps)

        print(res[0][:6])
        print('& %.3e & %.1f & %d & %d '%(
            res[1] / scale, -log(res[2] / scale) / log(10), res[3], res[4]))
