import cg, bb
import numpy as np
from numpy import log


class GENSIN(object):
    """ Generalized SINEVAL function """
    def __init__(self, n, c1=1e4, c2=0.25, scale=1):
        self.n = n
        self.c1 = c1
        self.c2 = c2
        self.scale = scale
        
    def f(self, x): # compute f
        tmp = x[1:] - np.sin(x[:-1])
        fx = np.sum(tmp ** 2) * self.c1
        fx += np.sum(x[:-1] ** 2) * self.c2
        return fx * self.scale

    def g(self, x): # compute g
        gx = np.zeros(self.n)
        tmp = x[1:] - np.sin(x[:-1])
        gx[1:] += (2 * self.c1) * tmp
        gx[:-1] -= (2 * self.c1) * tmp * np.cos(x[:-1])
        gx[:-1] += (2 * self.c2) * x[:-1]
        return gx * self.scale


n_list = [1000, 2000, 5000, 10000]
for n in n_list:
    x = -np.ones(n)
    x[0] = 4.712389
    scale = 1e-5
    eps = 1e-8 * scale
    fun = GENSIN(n, scale=scale)

    for method in ['fr', 'prp', 'prp+', 'hs', 'cd', 'dy', 'bb', 'sd']:
        if method == 'bb':
            res = bb.bb(fun, x, eps=eps)
        elif method == 'sd':
            res = bb.sd(fun, x, eps=eps)
        else:
            res = cg.cg(fun, x, method=method, eps=eps)

        print(res[0][:10])
        print('& %.3e & %.1f & %d & %d '%(
            res[1] / scale - 2.455, -log(res[2] / scale) / log(10), res[3], res[4]))
