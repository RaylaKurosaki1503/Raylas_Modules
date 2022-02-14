"""
Author: Rayla Kurosaki
File: root_solver.py
Description: This file contains functions to find the roots to linear and
             nonlinear equations.
"""

import numpy as np

import sympy as sym

MAX = 1000000


def bisection(f, a, b, TOL):
    """
    The Bisection Method

    :param f: A continuous function of one variable
    :param a: Left end point
    :param b: Right end point
    :param TOL: Accuracy
    """
    # If the Intermediate Value Theorem is satisfied
    if f(a) * f(b) > 0:
        # Exits the program if the Intermediate Value Theorem is not satisfied
        raise Exception("f does not have a root in the interval [" + str(a) +
                        "," + str(b) + "].")
    # Set a counter for the number of iterations
    i = 0
    # Continue until you reached the max number of iterations
    while i < MAX:
        # Increment iteration counter
        i += 1
        # Compute the midpoint of the interval
        root = (a + b) / 2
        # If the root at f is small enough
        if np.abs(f(root)) < TOL:
            # Return the root
            return root
        # Determine the new interval if the solution is not found.
        if f(a) * f(root) < 0:
            b = root
        else:
            a = root
    # Exits the program if the algorithm reached the max number of iterations.
    raise Exception("The Bisection Method failed.")


def fpi(g, x0, TOL):
    """
    Fixed-Point Iteration

    :param g: A continuous function of one variable.
    :param x0: Initial guess
    :param TOL: Accuracy
    """
    # Set a counter for the number of iterations
    i = 0
    # Continue until you reached the max number of iterations
    while i < MAX:
        # Increment iteration counter
        i += 1
        # Compute the new root
        root = g(x0)
        # If the magnitude of the difference of the old and new root is small
        # enough
        if np.abs(root - x0) < TOL:
            # Return the root
            return root
        # Set the initial guess as the new root
        x0 = root
    # Exits the program if the algorithm reached the max number of iterations.
    raise Exception("The Fixed-Point Iteration Method failed.")


def newton(f, x0, TOL):
    """
    Newton's Method

    :param f: A continuous function of one variable
    :param x0: Initial guess
    :param TOL: Accuracy
    """
    # Set a counter for the number of iterations
    i = 0
    # Compute f', the derivative of f
    x = sym.symbols('x')
    df = sym.diff(f(x), x)
    # Continue until you reached the max number of iterations
    while i < MAX:
        # Increment iteration counter
        i += 1
        # Compute f'(x0)
        df_x = df.evalf(subs={x: x0})
        # Compute the root
        root = x0 - (f(x0) / df_x)
        # If the magnitude of the difference of the old and new root is small
        # enough
        if np.abs(root - x0) < TOL:
            # Return the root
            return float(root)
        # Set the initial guess as the new root
        x0 = root
    # Exits the program if the algorithm reached the max number of iterations.
    raise Exception("Newton's Method failed.")


def secant(f, x0, x1, TOL):
    """
    Secant Method

    :param f: A continuous function of one variable
    :param x0: Initial guess 1
    :param x1: Initial guess 2
    :param TOL: Accuracy
    """
    # Set a counter for the number of iterations
    i = 0
    # Continue until you reached the max number of iterations
    while i < MAX:
        # Increment iteration counter
        i += 1
        # Compute the new root
        root = x1 - (f(x1) * (x1 - x0) / (f(x1) - f(x0)))
        # If the magnitude of the difference of root and the first initial
        # guess is small enough
        if np.abs(root - x0) < TOL:
            # Return the root
            return root
        # Set the previous values equal to the current values
        x0, x1 = x1, root
    # Exits the program if the algorithm reached the max number of iterations.
    raise Exception("The Secant Method failed.")


def false_position(f, a, b, TOL):
    """
    The False Position Method

    :param f: A continuous function of one variable
    :param a: Left end point
    :param b: Right end point
    :param TOL: Accuracy
    """
    # If the Intermediate Value Theorem is satisfied
    if f(a) * f(b) > 0:
        # Exits the program if the Intermediate Value Theorem is not satisfied
        raise Exception("f does not have a root in the interval [" + str(a) +
                        "," + str(b) + "].")
    # Set a counter for the number of iterations
    i = 0
    # Continue until you reached the max number of iterations
    while i < MAX:
        # Increment iteration counter
        i += 1
        # Compute the new root
        root = (b * f(a) - a * f(b)) / (f(a) - f(b))
        # If the root at f is small enough
        if np.abs(f(root)) < TOL:
            # Return the root
            return root
        # Determine the new interval
        if f(a) * f(root) < 0:
            b = root
        else:
            a = root
    # Exits the program if the algorithm reached the max number of iterations.
    raise Exception("The False Position Method failed.")


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


def steffensen(f, x0, TOL):
    """
    Steffensen's Method

    :param f: A continuous function of one variable
    :param x0: Initial guess
    :param TOL: Accuracy
    """
    # Set a counter for the number of iterations
    i = 0
    # Continue until you reached the max number of iterations
    while i < MAX:
        # Compute some values
        x1 = f(x0)
        x2 = f(x1)
        # Compute the root
        root = x0 - ((x1 - x0) ** 2 / (x2 - 2 * x1 + x0))
        # If the magnitude of the difference of root and the first initial
        # guess is small enough
        if np.abs(root - x0) < TOL:
            # Return the root
            return root
        # Set the initial guess as the new root
        x0 = root
    # Exits the program if the algorithm reached the max number of iterations.
    raise Exception("Steffensen's method Method failed.")


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
    # Set a counter for the number of iterations
    i = 0
    # Compute the first and second derivatives of f
    x = sym.symbols('x')
    df = sym.diff(f(x), x)
    ddf = sym.diff(f(x), x, 2)
    # Continue until you reached the max number of iterations
    while i < MAX:
        # Increment iteration counter
        i += 1
        # Compute f(x0), f'(x0), f''(x0)
        f_x = f(x0)
        df_x = df.evalf(subs={x: x0})
        ddf_x = ddf.evalf(subs={x: x0})
        # Compute the root
        root = x0 - (2 * f_x * df_x / (2 * df_x ** 2 - f_x * ddf_x))
        if np.abs(root - x0) < TOL:
            # Return the root
            return root
        # Set the initial guess as the new root
        x0 = root
    # Exits the program if the algorithm reached the max number of iterations.
    raise Exception("Halley's Method failed.")
