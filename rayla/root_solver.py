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
    The Bisection Method

    :param f: A continuous function of one variable
    :param a: Left end point
    :param b: Right end point
    :param TOL: Accuracy
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
    if i == MAX:
        raise Exception("The Bisection Method failed.")
    else:
        return (a + b) / 2
    pass


def fpi(g, x0, TOL):
    """
    Fixed-Point Iteration

    :param g: A continuous function of one variable.
    :param x0: Initial guess
    :param TOL: Accuracy
    """
    i = 0
    root = g(x0)
    while (np.abs(root - x0) >= TOL) and (i < MAX):
        i += 1
        x0 = root
        root = g(x0)
    if i == MAX:
        raise Exception("The Fixed-Point Iteration Method failed.")
    else:
        return root
    pass


def newton(f, x0, TOL):
    """
    Newton's Method

    :param f: A continuous function of one variable
    :param x0: Initial guess
    :param TOL: Accuracy
    """
    i = 0
    x = sym.symbols('x')
    df = sym.diff(f(x), x)
    root = x0 - (f(x0) / df.evalf(subs={x: x0}))
    while (np.abs(root - x0) >= TOL) and (i < MAX):
        i += 1
        x0 = root
        root = x0 - (f(x0) / df.evalf(subs={x: x0}))
    if i == MAX:
        raise Exception("Newton's Method failed.")
    else:
        return root
    pass


def secant(f, x0, x1, TOL):
    """
    Secant Method

    :param f: A continuous function of one variable
    :param x0: Initial guess 1
    :param x1: Initial guess 2
    :param TOL: Accuracy
    """
    i = 0
    root = x1 - (f(x1) * (x1 - x0) / (f(x1) - f(x0)))
    while (np.abs(root - x0) >= TOL) and (i < MAX):
        i += 1
        x0 = x1
        x1 = root
        root = x1 - (f(x1) * (x1 - x0) / (f(x1) - f(x0)))
    if i == MAX:
        raise Exception("The Secant Method failed.")
    else:
        return root
    pass


def false_position(f, a, b, TOL):
    """
    The False Position Method

    :param f: A continuous function of one variable
    :param a: Left end point
    :param b: Right end point
    :param TOL: Accuracy
    """
    if f(a) * f(b) > 0:
        raise Exception("f does not have a root in the interval [" + str(a) +
                        "," + str(b) + "].")
    i = 0
    while ((b - a) / 2 > TOL) and (i < MAX):
        i += 1
        c = (b * f(a) - a * f(b)) / (f(a) - f(b))
        if f(c) == 0:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    if i == MAX:
        raise Exception("The False Position Method failed.")
    else:
        return (b * f(a) - a * f(b)) / (f(a) - f(b))
    pass


def muller(f, x0, x1, x2, TOL):
    """
    Muller's method

    :param f: A continuous function of one variable
    :param x0: Initial guess 1
    :param x1: Initial guess 2
    :param x2: Initial guess 3
    :param TOL: Accuracy
    """
    f0, f1, f2 = f(x0), f(x1), f(x2)
    h1, h2 = x1 - x0, x2 - x1
    d1, d2 = (f1 - f0) / h1, (f2 - f1) / h2
    d = (d2 - d1) / (h2 + h1)
    i = 2
    while i < MAX:
        i += 1
        b = d2 + h2 * d
        discriminant = b ** 2 - 4 * f2 * d
        if discriminant < 0:
            raise Exception("Requires Complex Arithmetic")
        discriminant = discriminant ** (1 / 2)
        E1, E2 = b + discriminant, b - discriminant
        if np.abs(E2) < np.abs(E1):
            E = E1
        else:
            E = E2
        h = -2 * f2 / E
        root = f2 + h
        if np.abs(h) < TOL:
            return root
        x0, x1, x2 = x1, x2, root
        h1, h2 = x1 - x0, x2 - x1
        d1, d2 = (f1 - f0) / h1, (f2 - f1) / h2
        d = (d2 - d1) / (h2 + h1)
    raise Exception("Muller's Method failed. ")
    pass


def steffensen(f, x0, TOL):
    """
    Steffensen's Method

    :param f: A continuous function of one variable
    :param x0: Initial guess
    :param TOL: Accuracy
    """
    i = 0
    while i < MAX:
        x1 = f(x0)
        x2 = f(x1)
        root = x0 - ((x1 - x0) ** 2 / (x2 - 2 * x1 + x0))
        if np.abs(root - x0) < TOL:
            return root
        x0 = root
    raise Exception("Steffensen's method Method failed.")
    pass


def itp(f, a, b, k1, k2, n0, TOL):
    """
    The ITP (Interpolate Truncate and Project) method

    :param f: A continuous function of one variable
    :param a: Left end point
    :param b: Right end point
    :param k1: Hyper-parameter 1 (0 < k1 < ∞)
    :param k2: Hyper-parameter 2 (1 <= k2 < 2.618034)
    :param n0: Hyper-parameter 3 (0 <= n0 < ∞)
    :param TOL: Accuracy
    """
    # Check to make sure all parameters are valid
    if not (0 < k1 < np.infty):
        raise Exception("k1 must be positive.")
    if not (1 <= k2 < 1 + ((1 + np.sqrt(5)) / 2)):
        raise Exception("k1 must satisfy [1, 1+ϕ), or [1,2.618034).")
    if not (0 <= n0 < np.infty):
        raise Exception("n0 must be non-negative.")
    n_b = np.ceil(np.log2((b - a) / (2 * TOL)))
    n_MAX = n_b + n0
    i = 0
    while ((b - a) / 2 > TOL) and (i < MAX):
        # Calculating Parameters
        x_b = (a + b) / 2
        r = TOL * 2 ** (n_MAX - i) - ((b - a) / 2)
        delta = k1 * (b - a) ** k2
        # Interpolation
        x_fp = (b * f(a) - a * f(b)) / (f(a) - f(b))
        # Truncation
        sigma = np.sign(x_b - x_fp)
        if delta <= np.abs(x_b - x_fp):
            x_t = x_fp + sigma * delta
        else:
            x_t = x_b
        # Projection
        if np.abs(x_t - x_b) <= r:
            x = x_t
        else:
            x = x_b - sigma * r
        # Updating Interval
        y = f(x)
        if y > 0:
            b = x
        elif y < 0:
            a = x
        else:
            a, b = x, y
        i += 1
    if i == MAX:
        raise Exception("The ITP Method failed.")
    else:
        return (a + b) / 2
    pass


def halley(f, x0, TOL):
    """
    Halley's Method

    :param f: A continuous function of one variable
    :param x0: Initial guess
    :param TOL: Accuracy
    """
    i = 0
    x = sym.symbols('x')
    f_x = f(x0)
    df = sym.diff(f(x), x)
    df_x = df.evalf(subs={x: x0})
    ddf = sym.diff(f(x), x, 2)
    ddf_x = ddf.evalf(subs={x: x0})
    root = x0 - (2 * f_x * df_x / (2 * df_x ** 2 - f_x * ddf_x))
    while (np.abs(root - x0) >= TOL) and (i < MAX):
        print(root)
        i += 1
        x0 = root
        f_x = f(x0)
        df_x = df.evalf(subs={x: x0})
        ddf_x = ddf.evalf(subs={x: x0})
        root = x0 - (2 * f_x * df_x / (2 * df_x ** 2 - f_x * ddf_x))
    if i == MAX:
        raise Exception("Halley's Method failed.")
    else:
        return root
    pass