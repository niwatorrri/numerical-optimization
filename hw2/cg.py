#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues May 1 22:14:02 2018

@author: Niwatori
"""

"""
# Non-linear Conjugate Gradient Methods
# - FR (Fletcher-Reeves)
# - PRP / PRP+ (Polak-Ribiere-Polyak)
# - HS (Hestenes-Stiefel)
# - CD (Conjugate Descent)
# - DY (Dai-Yuan)
"""

import numpy as np
from numpy.linalg import norm
import linesearch as ls


def cg(fun, x0, method='prp+', search='inexact', 
       eps=1e-8, maxiter=10000, nu=0.2, debug=False, **kwargs):
    """Non-linear conjugate gradient methods

    Parameters
    ----------
    fun: object
        objective function, with callable method f and g
    x0: ndarray
        initial point
    method: string, optional
        options: 'fr' for FR, 'prp' for PRP, 'prp+' for PRP+, 'hs' for HS,
                 'cd' for conjugate descent, 'dy' for Dai-Yuan
    search: string, optional
        'exact' for exact line search, 'inexact' for inexact line search
    eps: float, optional
        tolerance, used for stopping criterion
    maxiter: int, optional
        maximum number of iterations
    nv: float, optional
        parameter for restart by orthogonality test
    debug: boolean, optional
        output information for every iteration if set to True
    kwargs: dict, optional
        other arguments to pass down

    Returns
    -------
    x: ndarray
        optimal point
    f: float
        optimal function value
    gnorm: float
        norm of gradient at optimal point
    niter: int
        number of iterations
    neval: int
        number of function evaluations (f and g)
    """
    
    x = x0
    n = x.size
    f0 = -np.inf
    f1 = fun.f(x)
    g0 = np.zeros(n)
    g1 = fun.g(x)
    d = -g1
    niter = 0
    neval = 2

    while (abs(f1 - f0) > eps) or (norm(g1) > eps):
        if abs(np.dot(g1, g0)) > 0.2 * np.dot(g1, g1):
            d = -g1

        if search == 'inexact':
            alpha, v = ls.inexact(fun, x, d, fx=f1, gx=g1, **kwargs)
        elif search == 'exact':
            alpha, v = ls.exact(fun, x, d, **kwargs)
        else:
            raise ValueError('Invalid search type')

        x = x + alpha * d

        g0 = g1
        g1 = fun.g(x)
        y = g1 - g0
        
        f0 = f1
        f1 = fun.f(x)
        neval += (v + 2)

        if debug:
            print('iter:', niter, alpha)

        if method == 'fr':
            beta = np.dot(g1, g1) / np.dot(g0, g0)
        elif method == 'prp':
            beta = np.dot(g1, y) / np.dot(g0, g0)
        elif method == 'prp+':
            beta = max(np.dot(g1, y) / np.dot(g0, g0), 0)
        elif method == 'hs':
            beta = np.dot(g1, y) / np.dot(d, y)
        elif method == 'cd':
            beta = -np.dot(g1, g1) / np.dot(g0, d)
        elif method == 'dy':
            beta = np.dot(g1, g1) / np.dot(d, y)
        else:
            raise ValueError('Invalid method name')

        d = beta * d - g1

        niter += 1
        if niter == maxiter:
            break

    return x, f1, norm(g1), niter, neval

