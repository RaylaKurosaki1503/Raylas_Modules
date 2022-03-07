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
        # Initialize the Lagrange basis polynomial.
        poly = 1
        # Set the variable for the Lagrange basis polynomial.
        x_var = sym.symbols('x')
        # For each x value.
        for i, x in enumerate(xs):
            # If i =/= k.
            if not (i == k):
                # Multiply the polynomial by an expression.
                poly *= (x_var - x) / (xs[k] - x)
                pass
            pass
        # Return the constructed polynomial.
        return poly

    # Initializing the Lagrange interpolating polynomial.
    P = 0
    # Get the list of x values.
    xs = [x for x, y in data]
    # For each point.
    for j, (x, y) in enumerate(data):
        # Add y_j L_j(x) to P(x).
        P += y * lagrange_poly(j, xs)
        pass
    # Return the Lagrange Interpolating Polynomial.
    return P


def newton_divided_differences(data):
    """
    Computes the coefficients for the interpolating polynomial that
    interpolates the set of data points.

    :param data: The list of data points to interpolate.
    :return: The coefficients for the interpolating polynomial that
             interpolates the set of data points.
    """
    # Get the list of x and y values from the data set.
    xs, ys = [x for (x, y) in data], [y for (x, y) in data]
    # Create a matrix for the Newton Divided Difference triangle.
    ndd = np.zeros((len(xs), len(ys)))
    # Set the first column of the ndd triangle as the list of y values.
    for j, y in enumerate(ys):
        ndd[j][0] = y
        pass
    # For each column after the first column.
    for c in range(1, len(ys)):
        # For each row in the triangle.
        for r in range(len(ys) - c):
            # Compute the new ndd value.
            num = ndd[r + 1][c - 1] - ndd[r][c - 1]
            denom = xs[r + c] - xs[r]
            ndd[r][c] = sym.nsimplify(num / denom)
            pass
        pass
    # Return the first row of the triangle, which represents the coefficient
    # for the interpolating polynomial that interpolates the set of data
    # points.
    return ndd[0]


def ndd_polynomial(data):
    """
    Constructs the polynomial that interpolates the set of data points.

    :param data: The list of data points to interpolate.
    :return: The polynomial that interpolates the set of data points.
    """
    # Get the list of x and y values from the data set.
    xs, ys = [x for (x, y) in data], [y for (x, y) in data]
    # Get the coefficients for the interpolating polynomial that interpolates
    # the set of data points.
    cs = newton_divided_differences(data)
    # Initialize the interpolating polynomial.
    poly = 0
    for i, c in enumerate(cs):
        x = sym.symbols('x')
        # Initialize the expression for the polynomial.
        expr = 1
        # For each 
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
