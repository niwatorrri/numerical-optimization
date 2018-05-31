import cg, bb
import numpy as np
from numpy import log


class FLETCHCR(object):
    """ FLETCHCR function (from CUTEr) """
    def __init__(self, n, scale=1):
        self.n = n
        self.scale = scale
        
    def f(self, x): # compute f
        tmp = x[1:] - x[:-1] + 1 - x[:-1] ** 2
        fx = np.dot(tmp, tmp)
        return fx * self.scale

    def g(self, x): # compute g
        gx = np.zeros(self.n)
        tmp = x[1:] - x[:-1] + 1 - x[:-1] ** 2
        gx[:-1] -= 2 * tmp * (1 + 2 * x[:-1])
        gx[1:] += 2 * tmp
        return gx * self.scale


n_list = [1000, 2000, 5000, 10000]
for n in n_list:
    x = np.zeros(n)
    scale = 1e-1
    kwargs = {'eps': 1e-7 * scale, 'maxiter': 50000}
    fun = FLETCHCR(n, scale)

    for method in ['fr', 'prp', 'prp+', 'hs', 'cd', 'dy', 'bb', 'sd']:
        if method == 'bb':
            res = bb.bb(fun, x, **kwargs)
        elif method == 'sd':
            res = bb.sd(fun, x, **kwargs)
        else:
            res = cg.cg(fun, x, method=method, **kwargs)

        print(res[0][:10])
        print('& %.3e & %.1f & %d & %d '%(
            res[1] / scale, -log(res[2] / scale) / log(10), res[3], res[4]))
