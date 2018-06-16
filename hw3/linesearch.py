#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:59:37 2018

@author: Niwatori
"""

"""
# Line Search Algorithms
# - Exact line searchL 0.618 method
# - Inexact line search: strong Wolfe condition
"""

import numpy as np


def exact(fun, x, d, eps=1e-2, a_min=0.001, a_max=10):
    """Exact line search: 0.618 method

    Parameters
    ----------
    fun: object
        objective function, with callable method f
    x: ndarray
        current point
    d: ndarray
        current search direction
    eps: float, optional
        tolerance, used for convergence criterion
    a_min: float, optional
        lower bound for step length
    a_max: float, optional
        upper bound for step length

    Returns
    -------
    alpha: float
        step length that minimizes f(x + alpha * d)
    neval: int
        number of function evaluations (f)
    """
    def phi(a):
        return fun.f(x + a * d)

    a, b = a_min, a_max
    r = (np.sqrt(5) - 1) / 2
    lflag, rflag = 0, 0
    neval = 0

    while b - a > eps:
        if lflag == 0:
            a_l = a + (1 - r) * (b - a)
            phi_l = phi(a_l)
            neval += 1

        if rflag == 0:
            a_r = a + r * (b - a)
            phi_r = phi(a_r)
            neval += 1

        if phi_l < phi_r:
            b = a_r
            a_r = a_l
            phi_r = phi_l
            lflag, rflag = 0, 1
        else:
            a = a_l
            a_l = a_r
            phi_l = phi_r
            lflag, rflag = 1, 0

    return (a + b) / 2, neval


def inexact(fun, x, d, a_min=0.001, a_max=10,
            rho=1e-4, sigma=0.5, fx=None, gx=None):
    """Inexact line search: strong Wolfe condition

    Parameters
    ----------
    fun: object
        objective function, with callable method f and g
    x: ndarray
        current point
    d: ndarray
        current search direction
    a_min: float, optional
        lower bound for step length
    a_max: float, optional
        upper bound for step length
    rho: float, optional
        parameter for Armijo condition
    sigma: float, optional
        parameter for curvature condition
    fx: float, optional
        f at x, used for saving function evaluations
    gx: ndarray, optional
        g at x, used for saving function evaluations

    Returns
    -------
    alpha: float
        step length that satisfies strong Wolfe condition
    neval: int
        number of function evaluations (f and g)
    """
    a_0 = a_min
    a_1 = 1

    def phi(a):
        return fun.f(x + a * d)

    def phip(a): # derivative of phi
        return np.dot(fun.g(x + a * d), d)

    phi_0 = fx or phi(0)
    phip_0 = phip(0) if gx is None else np.dot(gx, d)
    phi_a0 = phi_0
    phi_max = phi(a_max)
    neval = 1
    maxiter = 10

    while True:
        phi_a1 = phi(a_1)
        if (phi_a1 > phi_0 + rho * a_1 * phip_0) \
            or (a_0 > 0 and phi_a1 >= phi_a0):
            zoom, v = _zoom(a_0, a_1, phi, phip,
                            phi_0, phip_0, phi_a0, rho, sigma)
            return zoom, neval + v

        phip_a1 = phip(a_1)
        if abs(phip_a1) <= -sigma * phip_0:
            return a_1, neval
        if phip_a1 >= 0:
            zoom, v = _zoom(a_1, a_0, phi, phip,
                            phi_0, phip_0, phi_a1, rho, sigma)
            return zoom, neval + v

        # Choose an alpha from (a_1, a_max)
        a_0 = a_1
        anew = _cubicInterplt(a_1, phi_a1, phip_a1, a_0, phi_a0, a_max, phi_max)
        anew_lo = a_1 + 0.1 * (a_max - a_1)
        anew_hi = a_max - 0.1 * (a_max - a_1)
        if (anew is not None) and (anew > anew_lo) and (anew < anew_hi):
            a_1 = anew
        else:
            anew = _quadInterplt(a_1, phi_a1, phip_a1, a_max, phi_max)
            if (anew is not None) and (anew > anew_lo) and (anew < anew_hi):
                a_1 = anew
            else:
                a_1 = (a_1 + a_max) / 2

        neval += 2
        maxiter -= 1
        if maxiter == 0:
            break

    return a_1, neval


def _zoom(a_lo, a_hi, phi, phip,
          phi_0, phip_0, phi_lo, rho, sigma):
    """Selection phase of inexact line search

    Parameters
    ----------
    a_lo, a_hi: float
        interval to search within
    phi: function
        phi = f(x + alpha * d)
    phip: function
        derivative of phi
    phi_0: float
        phi at 0
    phip_0: float
        derivative of phi at 0
    phi_lo: float
        phi at a_lo
    rho: float
        parameter for Armijo condition
    sigma: float
        parameter for curvature condition

    Returns
    -------
    alpha: float
        step length that satisfies strong Wolfe condition
    neval: int
        number of function evaluations (f and g)
    """
    neval = 0
    maxiter = 10

    while True:
        # bisection
        a = (a_hi + a_lo) / 2
        phi_a = phi(a)

        if (phi_a > phi_0 + rho * a * phip_0) or (phi_a >= phi_lo):
            a_hi = a
        else:
            phip_a = phip(a)
            if abs(phip_a) <= -sigma * phip_0:
                return a, neval
            if phip_a * (a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = a
            phi_lo = phi_a
            neval += 1

        neval += 1
        maxiter -= 1
        if maxiter == 0:
            break

    # backtrack if alpha is still not small enough
    while phi_a > phi_0 + rho * a * phip_0:
        a = a / 2
        phi_a = phi(a)

    return a, neval


def _cubicInterplt(a, fa, fpa, b, fb, c, fc):
    """Cubic interpolation

    Finds the minimizer for a cubic polynomial that goes through the
    points (a, fa), (b, fb) and (c, fc) with derivative at a (fpa).

    If no minimizer can be found return None
    """
    # f(x) = A * (x - a) ^ 3 + B * (x - a) ^ 2 + C * (x - a) + D

    C = fpa
    db = b - a
    dc = c - a
    if (db == 0) or (dc == 0) or (b == c):
        return None

    denom = (db * dc) ** 2 * (db - dc)
    d1 = np.array([[dc ** 2, -db ** 2], [-dc ** 3, db ** 3]])
    [A, B] = np.dot(d1, np.array([fb - fa - C * db, fc - fa - C * dc]))
    A /= denom
    B /= denom
    r = B * B - 3 * A * C
    if r < 0 or A == 0:
        return None
    xmin = a + (-B + np.sqrt(r)) / (3 * A)
    return xmin


def _quadInterplt(a, fa, fpa, b, fb):
    """Quadratic interpolation

    Finds the minimizer for a quadratic polynomial that goes through
    the points (a, fa) and (b, fb) with derivative at a (fpa),

    If no minimizer can be found return None
    """
    # f(x) = B * (x - a) ^ 2 + C * (x - a) + D
    D = fa
    C = fpa
    h = b - a

    B = (fb - D - C * h) / (h * h)
    if B <= 0:
        return None
    xmin = a - C / (2 * B)
    return xmin
