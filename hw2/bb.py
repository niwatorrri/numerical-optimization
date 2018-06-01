#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues May 22 20:58:23 2018

@author: Niwatori
"""

"""
# Negative Gradient Methods
# - BB (Barzilai-Borwein) gradient method
# - SD (Steepest descent)
"""

import numpy as np
import linesearch as ls
from numpy.linalg import norm
from collections import deque


def bb(fun, x0, eps=1e-8, maxiter=10000, debug=False,
       delta=1e-10, rho=1e-4, mu=0.5, lam=1, maxlen=10):
    """BB (Barzilai-Borwein) gradient method

    Parameters
    ----------
    fun: object
        objective function, with callable method f and g
    x0: ndarray
        initial point
    eps: float, optional
        tolerance, used for stopping criterion
    maxiter: int, optional
        maximum number of iterations
    debug: boolean, optional
        output information for every iteration if set to True
    delta, rho, mu, lam: float, optional
        parameters for non-monotonic line search
    maxlen: int, optional
        number of previous steps considered in non-monotonic line search

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
    f0 = -np.inf
    f1 = fun.f(x)
    g0 = 0
    g1 = fun.g(x)
    gnorm = norm(g1)
    queue = deque([f1])
    niter = 0
    neval = 2

    while (abs(f1 - f0) > eps) or (gnorm > eps):
        if lam < delta or lam > 1 / delta:
            lam = max(1, min(1e5, 1 / gnorm))

        # Non-monotonic line search
        alpha = 1 / lam
        fmax = max(queue)
        while fun.f(x - alpha * g1) > fmax - rho * alpha * (gnorm ** 2):
            alpha = mu * alpha
            neval += 1
        x = x - alpha * g1

        if debug:
            print('iter:', niter, alpha)

        g0 = g1
        g1 = fun.g(x)
        y = g1 - g0

        lam = -np.dot(g0, y) / (alpha * gnorm ** 2)
        gnorm = norm(g1)

        f0 = f1
        f1 = fun.f(x)
        queue.append(f1)
        if len(queue) > maxlen:
            queue.popleft()

        neval += 2
        niter += 1
        if niter == maxiter:
            break

    return x, f1, gnorm, niter, neval


def sd(fun, x0, search='inexact', eps=1e-8, maxiter=10000, **kwargs):
    """Steepest descent

    Parameters
    ----------
    fun: object
        objective function, with callable method f, g and G
    x0: ndarray
        initial point
    search: string, optional
        'exact' for exact line search, 'inexact' for inexact line search
    eps: float, optional
        tolerance, used for convergence criterion
    maxiter: int, optional
        maximum number of iterations
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
        number of function evaluations (f, g and G)
    """
    x = x0
    f0 = -np.inf
    f1 = fun.f(x0)
    g1 = fun.g(x0)
    niter = 0
    neval = 2

    while (abs(f1 - f0) > eps) or (norm(g1) > eps):
        d = -g1

        if search == 'inexact':
            alpha, v = ls.inexact(fun, x, d, fx=f1, gx=g1, **kwargs)
        elif search == 'exact':
            alpha, v = ls.exact(fun, x, d, **kwargs)
        else:
            raise ValueError('Invalid search type')

        x = x + alpha * d

        f0 = f1
        f1 = fun.f(x)
        g1 = fun.g(x)
        niter += 1
        neval += (v + 2)
        if niter == maxiter:
            break

    return x, f1, norm(g1), niter, neval
