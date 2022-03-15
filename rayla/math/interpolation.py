"""
Author: Rayla Kurosaki

File: interpolation.py

Description:
"""

import numpy as np
import sympy as sym


def lagrange(data):
    """
    Constructs the Lagrange Interpolating Polynomial that interpolates a list
    of points.

    :param data: A list of points to interpolate.
    :return: The Lagrange Interpolating Polynomial that interpolates a list
             of points.
    """

    def lagrange_poly(k, xs):
        """
        Constructs the Lagrange basis polynomial.

        :param k: The index of the list of x values to skip.
        :param xs: A list of x values.
        :return: The Lagrange basis polynomial
        """
        poly = 1
        x_var = sym.symbols('x')
        for i, x in enumerate(xs):
            if not (i == k):
                poly *= (x_var - x) / (xs[k] - x)
                pass
            pass
        return poly

    P = 0
    xs = [x for x, y in data]
    for j, (x, y) in enumerate(data):
        P += y * lagrange_poly(j, xs)
        pass
    return P


def newton_divided_differences(data):
    """
    Computes the coefficients for the interpolating polynomial that
    interpolates the set of data points.

    :param data: The list of data points to interpolate.
    :return: The coefficients for the interpolating polynomial that
             interpolates the set of data points.
    """
    xs, ys = [x for (x, y) in data], [y for (x, y) in data]
    ndd = np.zeros((len(xs), len(ys)))
    for j, y in enumerate(ys):
        ndd[j][0] = y
        pass
    for c in range(1, len(ys)):
        for r in range(len(ys) - c):
            num = ndd[r + 1][c - 1] - ndd[r][c - 1]
            denom = xs[r + c] - xs[r]
            ndd[r][c] = sym.nsimplify(num / denom)
            pass
        pass
    return ndd[0]


def ndd_polynomial(data):
    """
    Constructs the polynomial that interpolates the set of data points.

    :param data: The list of data points to interpolate.
    :return: The polynomial that interpolates the set of data points.
    """
    xs, ys = [x for (x, y) in data], [y for (x, y) in data]
    cs = newton_divided_differences(data)
    poly = 0
    for i, c in enumerate(cs):
        x = sym.symbols('x')
        expr = 1
        for j in range(i):
            expr *= x - xs[j]
            pass
        poly += sym.nsimplify(c * expr)
        pass
    return poly


def chebyshev():
    pass


def cubic_splines():
    pass


def natural_cubic_splines():
    pass


def bezier():
    pass
