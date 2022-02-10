"""
Author: Rayla Kurosaki
File: root_solver.py
Description: This file contains functions to find the roots to linear and
             nonlinear equations.
"""

import numpy as np
import sympy as sym

MAX = 1000000


################################################################################


def bisection(f, a, b, TOL):
    """
    The Bisection method

    :param f: A continuous function of one variable
    :param a: Left end point
    :param b: Right end point
    :param TOL: Accuracy
    :return: A root of f in the interval [a, b]
    """
    if f(a) * f(b) > 0:
        raise Exception("f does not have a root in the interval [" + str(a) +
                        "," + str(b) + "].")
    i = 0
    while ((b - a) / 2 > TOL) and (i < MAX):
        i += 1
        c = (a + b) / 2
        if f(c) == 0:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2


def fpi(g, x0, TOL):
    """
    Fixed-point iteration is a method of computing fixed points of a function.

    :param g:
    :param x0:
    :param TOL:
    :return:
    """
    i = 1
    xn = g(x0)
    while (np.abs(xn - x0) >= TOL) and (i < MAX):
        i += 1
        x0 = xn
        xn = g(x0)
    return xn


def newton(f, x0, TOL):
    i = 1
    x = sym.symbols('x')
    df = sym.diff(f(x), x)
    xn = x0 - (f(x0) / df.evalf(subs={x: x0}))
    while (np.abs(xn - x0) >= TOL) and (i < MAX):
        i += 1
        x0 = xn
        xn = x0 - (f(x0) / df.evalf(subs={x: x0}))
    return xn


def secant(f, x0, x1, TOL):
    i = 1
    xn = x1 - (f(x1) * (x1 - x0) / (f(x1) - f(x0)))
    while (np.abs(xn - x0) >= TOL) and (i < MAX):
        i += 1
        x0 = x1
        x1 = xn
        xn = x1 - (f(x1) * (x1 - x0) / (f(x1) - f(x0)))
    return xn


if __name__ == '__main__':
    a, b = 0, 1
    p = 7
    TOL = 1 * 10 ** (-p)
    x0 = 0
    x1 = 1


    def f(x):
        return x ** 3 + x - 1


    # x = sym.symbols('x')
    # df = sym.diff(f(x), x)
    # print(df)

    print(secant(f, x0, x1, TOL))
    pass
