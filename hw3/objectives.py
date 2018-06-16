#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 1 19:01:54 2018

@author: Niwatori
"""

"""
# Test functions for non-linear least squares algorithms
# - Osborne function 1
# - Osborne function 2
# - Chebyshev quadrature function
# - Jennrich and Sampson function
# - Meyer function
# - Brown and Dennis function
"""
import numpy as np
from numpy import sqrt, sin, cos, arccos


class ObjFunc(object):
    """
    Objective function class

    Every non-linear least squares test function should inherit
    this class, and implement (at least) two methods including
    r (residual) and J (Jacobian). If not, a NotImplementedError
    will be raised.
    """
    def __init__(self):
        pass

    def r(self, x): # residual r
        raise NotImplementedError("Residual method not implemented.")

    def J(self, x): # Jacobian J
        raise NotImplementedError("Jacobian method not implemented.")

    def f(self, x): # objective f
        r = self.r(x)
        return np.dot(r, r) / 2

    def g(self, x): # gradient g
        r = self.r(x)
        J = self.J(x)
        return np.dot(J.T, r)

    def eval(self, x):
        r = self.r(x)
        J = self.J(x)
        f = np.dot(r, r) / 2
        g = np.dot(J.T, r)
        return J, r, f, g


class Osborne1(ObjFunc):
    """ Osborne function 1 """
    def __init__(self):
        super().__init__()
        self.n = 5
        self.m = 33
        self.t = np.arange(self.m) * 10
        self.y = np.array([.844, .908, .932, .936, .925, .908, .881, .850, .818,
                           .784, .751, .718, .685, .658, .628, .603, .580, .558,
                           .538, .522, .506, .490, .478, .467, .457, .448, .438,
                           .431, .424, .420, .414, .411, .406])

    def r(self, x): # residual r
        rx = self.y - (x[0] + x[1] * np.exp(-self.t * x[3])
                            + x[2] * np.exp(-self.t * x[4]))
        return rx

    def J(self, x): # Jacobian J
        Jx = np.zeros((self.m, self.n))
        t = self.t
        Jx[:, 0] = -1
        Jx[:, 1] = -np.exp(-t * x[3])
        Jx[:, 2] = -np.exp(-t * x[4])
        Jx[:, 3] = t * x[1] * np.exp(-t * x[3])
        Jx[:, 4] = t * x[2] * np.exp(-t * x[4])
        return Jx


class Osborne2(ObjFunc):
    """ Osborne function 2 """
    def __init__(self):
        super().__init__()
        self.n = 11
        self.m = 65
        self.t = np.arange(self.m) / 10
        self.y = np.array([1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831,
                           0.847, 0.786, 0.725, 0.746, 0.679, 0.608, 0.655,
                           0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649,
                           0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558,
                           0.533, 0.495, 0.500, 0.423, 0.395, 0.375, 0.372,
                           0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562,
                           0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645,
                           0.632, 0.591, 0.559, 0.597, 0.625, 0.739, 0.710,
                           0.729, 0.720, 0.636, 0.581, 0.428, 0.292, 0.162,
                           0.098, 0.054])

    def r(self, x): # residual r
        t = self.t
        rx = self.y - (x[0] * np.exp(-t * x[4]) \
                       + x[1] * np.exp(-(t - x[8]) ** 2 * x[5])
                       + x[2] * np.exp(-(t - x[9]) ** 2 * x[6])
                       + x[3] * np.exp(-(t - x[10]) ** 2 * x[7]))
        return rx

    def J(self, x): # Jacobian J
        Jx = np.zeros((self.m, self.n))
        t = self.t
        Jx[:, 0] = -np.exp(-t * x[4])
        Jx[:, 1] = -np.exp(-(t - x[8]) ** 2 * x[5])
        Jx[:, 2] = -np.exp(-(t - x[9]) ** 2 * x[6])
        Jx[:, 3] = -np.exp(-(t - x[10]) ** 2 * x[7])
        Jx[:, 4] = t * x[0] * np.exp(-t * x[4])
        Jx[:, 5] = (t - x[8]) ** 2 * x[1] * np.exp(-(t - x[8]) ** 2 * x[5])
        Jx[:, 6] = (t - x[9]) ** 2 * x[2] * np.exp(-(t - x[9]) ** 2 * x[6])
        Jx[:, 7] = (t - x[10]) ** 2 * x[3] * np.exp(-(t - x[10]) ** 2 * x[7])
        Jx[:, 8] = 2 * (x[8] - t) * x[5] * x[1] * np.exp(-(t - x[8]) ** 2 * x[5])
        Jx[:, 9] = 2 * (x[9] - t) * x[6] * x[2] * np.exp(-(t - x[9]) ** 2 * x[6])
        Jx[:, 10] = 2 * (x[10] - t) * x[7] * x[3] * np.exp(-(t - x[10]) ** 2 * x[7])
        return Jx


class ChebyQuad(ObjFunc):
    """ Chebyshev quadrature function """
    def __init__(self, n, m=0):
        super().__init__()
        self.n = n
        self.m = max(m, n)
        self.I = np.zeros(n)
        self.I[1::2] = 1 / (1 - np.arange(2, n + 1, 2) ** 2)

    def cheb1(self, x, n): # Chebyshev polynomial of the first kind
        for k in range(x.size):
            if abs(x[k]) < 1:
                x[k] = cos(n * arccos(x[k]))
            else:
                root = sqrt(x[k] ** 2 - 1)
                x[k] = ((x[k] - root) ** n + (x[k] + root) ** n) / 2
        return x

    def cheb2(self, x, n): # Chebyshev polynomial of the second kind
        for k in range(x.size):
            if abs(x[k]) < 1:
                x[k] = sin(n * arccos(x[k])) / sin(arccos(x[k]))
            else:
                root = sqrt(x[k] ** 2 - 1)
                x[k] = ((x[k] + root) ** n - (x[k] - root) ** n) / 2 / root
        return x

    def r(self, x): # residual r
        rx = np.zeros(self.m)
        for k in range(self.m):
            rx[k] = np.sum(self.cheb1(2 * x - 1, k + 1)) / self.n
        rx = rx - self.I
        return rx

    def J(self, x): # Jacobian J
        Jx = np.zeros((self.m, self.n))
        for k in range(self.m):
            Jx[k:] = 2 * (k + 1) * self.cheb2(2 * x - 1, k + 1) / self.n
        return Jx


class JennrichSampson(ObjFunc):
    """ Jennrich and Sampson function """
    def __init__(self, m=10):
        super().__init__()
        self.n = 2
        self.m = m

    def r(self, x): # residual r
        t = np.arange(1, self.m + 1)
        rx = 2 + 2 * t - np.exp(t * x[0]) - np.exp(t * x[1])
        return rx

    def J(self, x): # Jacobian J
        Jx = np.zeros((self.m, self.n))
        t = np.arange(1, self.m + 1)
        Jx[:, 0] = -t * np.exp(t * x[0])
        Jx[:, 1] = -t * np.exp(t * x[1])
        return Jx


class Meyer(ObjFunc):
    """ Meyer function """
    def __init__(self):
        super().__init__()
        self.n = 3
        self.m = 16
        self.t = 45 + 5 * np.arange(1, self.m + 1)
        self.y = np.array([34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744,
                           8261, 7030, 6005, 5147, 4427, 3820, 3307, 2872])

    def r(self, x): # residual r
        rx = x[0] * np.exp(x[1] / (self.t + x[2])) - self.y
        return rx

    def J(self, x): # Jacobian J
        Jx = np.zeros((self.m, self.n))
        t = self.t
        Jx[:, 0] = np.exp(x[1] / (t + x[2]))
        Jx[:, 1] = x[0] * np.exp(x[1] / (t + x[2])) / (t + x[2])
        Jx[:, 2] = -x[0] * x[1] * np.exp(x[1] / (t + x[2])) / (t + x[2]) ** 2
        return Jx


class BrownDennis(ObjFunc):
    """ Brown and Dennis function """
    def __init__(self, m=20):
        super().__init__()
        self.n = 4
        self.m = m

    def r(self, x): # residual r
        t = np.arange(1, self.m + 1) / 5
        rx = (x[0] + t * x[1] - np.exp(t)) ** 2 \
             + (x[2] + x[3] * np.sin(t) - np.cos(t)) ** 2
        return rx

    def J(self, x): # Jacobian J
        Jx = np.zeros((self.m, self.n))
        t = np.arange(1, self.m + 1) / 5
        Jx[:, 0] = 2 * (x[0] + t * x[1] - np.exp(t))
        Jx[:, 1] = 2 * t * (x[0] + t * x[1] - np.exp(t))
        Jx[:, 2] = 2 * (x[2] + x[3] * np.sin(t) - np.cos(t))
        Jx[:, 3] = 2 * np.sin(t) * (x[2] + x[3] * np.sin(t) - np.cos(t))
        return Jx

