#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 21:47:09 2018

@author: Niwatori
"""

"""
# Algorithms for Non-linear Least Squares Problem
# - Gauss-Newton method
# - Levenberg-Marquardt method (options: LM-Fletcher / LM-Nielsen)
# - Dogleg method (options: dogleg / double dogleg)
# - Trust-region reflective algorithm (implemented in SciPy)
# - Dennis-Gay-Welsch method for large-residual problems
"""

import numpy as np
import linesearch
import scipy.optimize
from numpy import sqrt
from numpy.linalg import norm


def GaussNewton(fun, x0, method=None, search='inexact',
                eps=1e-8, maxiter=1000, **kwargs):
    """Gauss-Newton (GN) method

    Parameters
    ----------
    fun: object
        objective function, with callable method eval (J, r, f, g)
    x0: ndarray
        initial point
    method: string, optional
        ignored
    search: string, optional
        'exact' for exact line search, 'inexact' for inexact line search
    eps: float, optional
        tolerance, used for stopping criterion
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
        number of function evaluations
    """
    x = x0
    f0 = -np.inf
    J, r, f1, g1 = fun.eval(x)
    niter = 0
    neval = 1

    while (abs(f1 - f0) >= eps * abs(f0)):
        d = np.linalg.lstsq(J, -r, rcond=None)[0]

        if search == 'inexact':
            alpha, v = linesearch.inexact(fun, x, d, fx=f1, gx=g1, **kwargs)
        elif search == 'exact':
            alpha, v = linesearch.exact(fun, x, d, **kwargs)
        else:
            raise ValueError('Invalid search type')

        x = x + alpha * d

        f0 = f1
        J, r, f1, g1 = fun.eval(x)

        niter += 1
        neval += (v + 1)
        if niter == maxiter:
            break

    return x, f1, norm(g1), niter, neval


def LevenbergMarquardt(fun, x0, method='lmf', search=None,
                       eps=1e-8, maxiter=1000, **kwargs):
    """Levenberg-Marquardt (LM) method: LMF or LMN

    Parameters
    ----------
    fun: object
        objective function, with callable method eval (J, r, f, g)
    x0: ndarray
        initial point
    method: string, optional
        'lmf' for LM-Fletcher, 'lmn' for LM-Nielsen
    search: string, optional
        ignored
    eps: float, optional
        tolerance, used for stopping criterion
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
        number of function evaluations
    """
    x = x0
    f0 = -np.inf
    J, r, f1, g1 = fun.eval(x)
    n = x.size
    nu = 1
    c = 2
    niter = 0
    neval = 1

    while (abs(f1 - f0) >= eps * abs(f0)):
        J1 = np.concatenate((J, sqrt(nu) * np.identity(n)))
        r1 = np.concatenate((r, np.zeros(n)))
        d = np.linalg.lstsq(J1, -r1, rcond=None)[0]

        delta_f = f1 - fun.f(x + d)
        delta_q = np.dot(d, nu * d - g1) / 2
        gamma = delta_f / delta_q

        if method == 'lmf':
            if gamma < .25:
                nu = nu * 4
            if gamma > .75:
                nu = nu / 2
            if gamma > 0:
                x = x + d
                f0 = f1
                J, r, f1, g1 = fun.eval(x)
                neval += 1

        elif method == 'lmn':
            if gamma < 0:
                nu = c * nu
                c = c * 2
            else:
                nu = max(1 / 3, 1 - (2 * gamma - 1) ** 3) * nu
                x = x + d
                f0 = f1
                J, r, f1, g1 = fun.eval(x)
                neval += 1
        else:
            raise ValueError('Invalid method name')

        niter += 1
        if niter == maxiter:
            break

    return x, f1, norm(g1), niter, neval


def Dogleg(fun, x0, method='single', search=None,
           eps=1e-8, maxiter=1000, **kwargs):
    """Dogleg method: dogleg or double dogleg

    Parameters
    ----------
    fun: object
        objective function, with callable method eval (J, r, f, g)
    x0: ndarray
        initial point
    method: string, optional
        'single' for dogleg, 'double' for double dogleg
    search: string, optional
        ignored
    eps: float, optional
        tolerance, used for stopping criterion
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
        number of function evaluations
    """
    x = x0
    f0 = -np.inf
    J, r, f1, g1 = fun.eval(x)
    delta = 1 # radius of trust-region
    niter = 0
    neval = 1

    while (abs(f1 - f0) >= eps * abs(f0)):
        d_SD = -g1
        d_GN = np.linalg.lstsq(J, -r, rcond=None)[0]
        alpha = (norm(d_SD) / norm(np.dot(J, d_SD))) ** 2

        # Dogleg routine
        if norm(d_GN) < delta:
            d = d_GN
        elif alpha * norm(d_SD) > delta:
            d = delta * (d_SD / norm(d_SD))
        else:
            if method == 'single':
                eta = 1
            elif method == 'double':
                v = np.linalg.solve(np.dot(J.T, J), g1)
                rho = norm(g1) ** 4 / norm(np.dot(J, g1)) ** 2 / np.dot(g1, v)
                eta = .8 * rho + .2
            else:
                raise ValueError('Invalid method name')

            D = norm(eta * d_GN - alpha * d_SD) ** 2
            E = 2 * alpha * np.dot(d_SD, eta * d_GN - alpha * d_SD)
            F = (alpha * norm(d_SD)) ** 2 - delta ** 2
            det = E ** 2 - 4 * D * F

            beta = (-E + sqrt(det)) / (2 * D)
            d = (1 - beta) * alpha * d_SD + beta * eta * d_GN

        # Normal trust-region routine
        delta_f = f1 - fun.f(x + d)
        delta_q = -np.dot(d, g1) - norm(np.dot(J, d)) ** 2 / 2
        gamma = delta_f / delta_q

        if gamma < .25:
            delta = delta / 4
        if gamma > .75 and abs(norm(d) - delta) < 1e-8 * delta:
            delta = delta * 2
        if gamma > 0:
            x = x + d
            f0 = f1
            J, r, f1, g1 = fun.eval(x)
            neval += 1

        niter += 1
        if niter == maxiter:
            break

    return x, f1, norm(g1), niter, neval


def ScipyMethod(fun, x0, method='trf', search=None,
                eps=1e-8, maxiter=1000, **kwargs):
    """Trust region reflexive algorithm (Scipy implementation)

    Parameters
    ----------
    fun: object
        objective function, with callable method eval (J, r, f, g)
    x0: ndarray
        initial point
    method: string, optional
        'trf' for trust region reflexive algorithm
    search: string, optional
        ignored
    eps: float, optional
        tolerance, used for stopping criterion
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
        number of function evaluations
    """
    res = scipy.optimize.least_squares(fun.r, x0, fun.J, method=method,
                                       ftol=eps, gtol=eps, max_nfev=maxiter)
    x, f1, gnorm, niter, neval = res.x, res.cost, norm(res.grad), res.nfev, res.njev
    return x, f1, gnorm, niter, neval


def DennisGayWelsch(fun, x0, B0=None, method='ls', search='inexact',
                    eps=1e-8, maxiter=1000, **kwargs):
    """Dennis-Gay-Welsch method: line search or trust region

    Parameters
    ----------
    fun: object
        objective function, with callable method eval (J, r, f, g)
    x0: ndarray
        initial point
    B0: ndarray, optional
        initial approximation of matrix S, half the identity by default
    method: string, optional
        'ls' for DGW with line search, 'tr' for DGW with trust region
    search: string, optional
        used only when method == 'ls'
        'exact' for exact line search, 'inexact' for inexact line search
    eps: float, optional
        tolerance, used for stopping criterion
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
        number of function evaluations
    """
    def dogleg(G, g1):
        d_SD = -g1
        d_GN = np.linalg.solve(G, -g1)
        alpha = norm(d_SD) ** 2 / np.dot(d_SD, G @ d_SD)

        if norm(d_GN) < delta:
            d = d_GN
        elif alpha * norm(d_SD) > delta:
            d = delta * (d_SD / norm(d_SD))
        else:
            D = norm(d_GN - alpha * d_SD) ** 2
            E = 2 * alpha * np.dot(d_SD, d_GN - alpha * d_SD)
            F = (alpha * norm(d_SD)) ** 2 - delta ** 2
            det = E ** 2 - 4 * D * F

            beta = (-E + sqrt(det)) / (2 * D)
            d = (1 - beta) * alpha * d_SD + beta * d_GN
        return d

    x = x0
    f0 = -np.inf
    J, r, f1, g1 = fun.eval(x)
    B = .5 * np.identity(x.size) if B0 is None else B0

    delta = 1 # radius of trust-region
    niter = 0
    neval = 1
    uflag = True

    while (abs(f1 - f0) >= eps * abs(f0)):
        # DGW with line search routine
        if method == 'ls':
            d = np.linalg.solve(np.dot(J.T, J) + B, -g1)
            if np.dot(g1, d) > 0:
                d = np.linalg.solve(np.dot(J.T, J), -g1)

            if search == 'inexact':
                alpha, v = linesearch.inexact(fun, x, d, fx=f1, gx=g1, **kwargs)
            elif search == 'exact':
                alpha, v = linesearch.exact(fun, x, d, **kwargs)
            else:
                raise ValueError('Invalid search type')
            s = alpha * d

        # DGW with trust region (dogleg) routine
        elif method == 'tr':
            G = np.dot(J.T, J) + B
            d = dogleg(G, g1)
            delta_f = f1 - fun.f(x + d)

            if delta_f < 0:
                G = np.dot(J.T, J)
                d = dogleg(G, g1)
                delta_f = f1 - fun.f(x + d)

            delta_q = -np.dot(d, g1) - np.dot(d, G @ d) / 2
            gamma = delta_f / delta_q

            uflag = False
            if gamma < .25:
                delta = delta / 4
            if gamma > .75 and abs(norm(d) - delta) < 1e-8 * delta:
                delta = delta * 2
            if gamma > 0:
                s = d
                uflag = True
        else:
            raise ValueError('Invalid method name')

        # Update of point x and matrix approximation B
        if uflag == True:
            x = x + s
            g0 = g1
            f0 = f1
            J0 = J
            J, r, f1, g1 = fun.eval(x)
            neval += 1

            y = g1 - g0
            y_hat = g1 - np.dot(J0.T, r)

            # Scaling for faster convergence
            tau = min(1, abs(np.dot(s, y_hat) / np.dot(s, B @ s)))
            B = tau * B

            z = y_hat - B @ s
            t = np.dot(y, s)
            B = B + np.outer(z, y) / t + np.outer(y, z) / t \
                  - np.outer(y * np.dot(z, s) / t, y / t)

        niter += 1
        if niter == maxiter:
            break

    return x, f1, norm(g1), niter, neval

