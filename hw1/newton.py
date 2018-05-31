#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:26:37 2018

@author: Niwatori
"""

"""
# Newton-type Methods
# - Normal Newton / Damped Newton
# - Modified Newton: mixed direction / LM method
# - Quasi-Newton: SR1 / DFP / BFGS
"""

import numpy as np
from numpy.linalg import norm
import linesearch as ls


def Newton(fun, x0, method='damped', search='inexact',
           eps=1e-8, maxiter=1000, **kwargs):
    """Newton's method: normal or damped

    Parameters
    ----------
    fun: object
        objective function, with callable method f, g and G
    x0: ndarray
        initial point
    method: string, optional
        'normal' for Normal Newton, 'damped' for damped Newton
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
    f0 = 0
    f1 = fun.f(x0)
    g1 = fun.g(x0)
    niter = 0
    neval = 2
    errflag = 0

    while (abs(f1 - f0) > eps) or (norm(g1) > eps):
        G = fun.G(x)
        try: # test if positive definite
            L = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            errflag = 1

        d = np.linalg.solve(G, -g1)

        if method == 'normal':
            alpha, v = 1, 0

        elif method == 'damped':
            if search == 'inexact':
                alpha, v = ls.inexact(fun, x, d, fx=f1, gx=g1, **kwargs)
            elif search == 'exact':
                alpha, v = ls.exact(fun, x, d, **kwargs)
            else:
                raise ValueError('Invalid search type')
        else:
            raise ValueError('Invalid method name')

        x = x + alpha * d

        f0 = f1
        f1 = fun.f(x)
        g1 = fun.g(x)
        niter += 1
        neval += (v + 3)
        if niter == maxiter:
            break

    if errflag == 1:
        print('Warning: Non-positive-definite Hessian encountered.')
    return x, f1, norm(g1), niter, neval


def modifiedNewton(fun, x0, method='mix', search='inexact',
                   eps=1e-8, maxiter=1000, **kwargs):
    """Modified Newton's method: mixed direction or LM method

    Parameters
    ----------
    fun: object
        objective function, with callable method f, g and G
    x0: ndarray
        initial point
    method: string, optional
        'mix' for mixed direction method, 'lm' for Levenberg-Marquardt method
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
    f0 = 0
    f1 = fun.f(x0)
    g1 = fun.g(x0)
    niter = 0
    neval = 2

    while (abs(f1 - f0) > eps) or (norm(g1) > eps):
        if method == 'mix':
            try: # test if singular
                d = np.linalg.solve(fun.G(x), -g1)
                if abs(np.dot(g1, d)) < eps * norm(g1) * norm(d): # orthogonal
                    d = -g1
                if np.dot(g1, d) > eps * norm(g1) * norm(d): # non-descent
                    d = -d
            except np.linalg.LinAlgError:
                d = -g1

        elif method == 'lm':
            G = fun.G(x)
            v = 0
            while True:
                try: # test if positive definite
                    L = np.linalg.cholesky(G + v * np.eye(x.size))
                    break
                except np.linalg.LinAlgError:
                    if v == 0:
                        v = norm(G) / 2 # Frobenius norm
                    else:
                        v *= 2
            y = np.linalg.solve(L, -g1)
            d = np.linalg.solve(L.T, y)
        else:
            raise ValueError('Invalid method name')

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
        neval += (v + 3)
        if niter == maxiter:
            break

    return x, f1, norm(g1), niter, neval


def quasiNewton(fun, x0, H0=None, method='bfgs', search='inexact', 
                eps=1e-8, maxiter=1000, **kwargs):
    """Quasi-Newton methods: SR1 / DFP / BFGS

    Parameters
    ----------
    fun: object
        objective function, with callable method f and g
    x0: ndarray
        initial point
    H0: ndarray, optional
        initial Hessian inverse, identity by default
    method: string, optional
        'sr1' for SR1, 'dfp' for DFP, 'bfgs' for BFGS
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
        number of function evaluations (f and g)
    """
    x = x0
    if H0 is not None:
        H = H0
    else:
        H = np.eye(x.size)

    f0 = 0
    f1 = fun.f(x)
    g0 = np.zeros(x.size)
    g1 = fun.g(x)
    niter = 0
    neval = 2

    while (abs(f1 - f0) > eps) or (norm(g1) > eps):
        d = -(H @ g1)
        if search == 'inexact':
            alpha, v = ls.inexact(fun, x, d, fx=f1, gx=g1, **kwargs)
        elif search == 'exact':
            alpha, v = ls.exact(fun, x, d, **kwargs)
        else:
            raise ValueError('Invalid search type')

        s = alpha * d
        x = x + s

        g0 = g1
        g1 = fun.g(x)
        y = g1 - g0

        if f0 == 0 and H0 is None: # initial scaling
            H = (np.dot(y, s) / np.dot(y, y)) * H
        
        f0 = f1
        f1 = fun.f(x)
        neval += (v + 2)

        if method == 'sr1':
            z = s - H @ y
            if abs(np.dot(z, y)) >= eps * norm(z) * norm(y):
                H = H + np.outer(z, z / np.dot(z, y))

        elif method == 'dfp':
            z = H @ y
            H = H + np.outer(s, s / np.dot(s, y)) - np.outer(z, z / np.dot(y, z))

        elif method == 'bfgs':
            r = 1 / np.dot(s, y)
            z = r * (H @ y)
            H = H + r * (1 + np.dot(y, z)) * np.outer(s, s) \
                  - np.outer(s, z) - np.outer(z, s)
        else:
            raise ValueError('Invalid method name')

        niter += 1
        if niter == maxiter:
            break

    return x, f1, norm(g1), niter, neval
