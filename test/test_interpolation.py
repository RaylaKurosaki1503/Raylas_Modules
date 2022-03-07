"""
Author: Rayla Kurosaki

File: test_interpolation.py

Description:
"""

import sympy as sym

from interpolation import lagrange
from interpolation import ndd_polynomial


def test_lagrange():
    # Set the variable for the Lagrange basis polynomial.
    x = sym.symbols('x')
    points = [
        [(0, 1), (2, 2), (3, 4)],
        [(0, 2), (1, 1), (2, 0)],
        [(0, 1), (2, 3), (3, 0)],
        [(-1, 0), (2, 1), (3, 1), (5, 2)],
        [(0, -2), (2, 1), (4, 4)],
        [(-1, 3), (1, 1), (2, 3), (3, 7)],
        [(0, 0), (1, 1), (2, 2), (3, 7)],
        [(0, 0), (1, 1), (2, 2), (3, 7), (4, 2)],
        [(-2, 8), (0, 4), (1, 2), (3, -2)],
    ]
    exact_expression = [
        (1 / 2) * x ** 2 - (1 / 2) * x + 1.0,
        -x + 2.0,
        (-4 / 3) * x ** 2 + (11 / 3) * x + 1.0,
        (1 / 24) * x ** 3 - (1 / 4) * x ** 2 + (11 / 24) * x + (3 / 4),
        (3 / 2) * x - 2.0,
        x ** 2 - x + 1.0,
        (2 / 3) * x ** 3 - 2 * x ** 2 + (7 / 3) * x,
        (-3 / 4) * x ** 4 + (31 / 6) * x ** 3 - (41 / 4) * x ** 2 +
        (41 / 6) * x,
        4.0 - 2 * x
    ]
    print("Testing lagrange functionality")
    if not (len(points) == len(exact_expression)):
        raise Exception(f"Sizes: (points,exact_expression) = "
                        f"({len(points)},{len(exact_expression)})")
    for index, (data, sol) in enumerate(zip(points, exact_expression)):
        poly = sym.N(sym.expand(lagrange(data)))
        sol = sym.N(sol)
        if not (poly == sol):
            raise ValueError(f"lagrange test {index + 1}. Expected "
                             f"{sol}, got {poly}.")
            pass
        pass
    print(f"All tests passed.\n")
    pass


def test_newton_divided_differences():
    # Set the variable for the Lagrange basis polynomial.
    x = sym.symbols('x')
    points = [
        [(0, 1), (2, 2), (3, 4)],
        [(0, 2), (1, 1), (2, 0)],
        [(0, 1), (2, 3), (3, 0)],
        [(-1, 0), (2, 1), (3, 1), (5, 2)],
        [(0, -2), (2, 1), (4, 4)],
        [(-1, 3), (1, 1), (2, 3), (3, 7)],
        [(0, 0), (1, 1), (2, 2), (3, 7)],
        [(0, 0), (1, 1), (2, 2), (3, 7), (4, 2)],
        [(-2, 8), (0, 4), (1, 2), (3, -2)],
    ]
    exact_expression = [
        (1 / 2) * x ** 2 - (1 / 2) * x + 1.0,
        -x + 2.0,
        (-4 / 3) * x ** 2 + (11 / 3) * x + 1.0,
        (1 / 24) * x ** 3 - (1 / 4) * x ** 2 + (11 / 24) * x + (3 / 4),
        (3 / 2) * x - 2.0,
        x ** 2 - x + 1.0,
        (2 / 3) * x ** 3 - 2 * x ** 2 + (7 / 3) * x,
        (-3 / 4) * x ** 4 + (31 / 6) * x ** 3 - (41 / 4) * x ** 2 +
        (41 / 6) * x,
        4.0 - 2 * x
    ]
    print("Testing ndd_polynomial functionality")
    if not (len(points) == len(exact_expression)):
        raise Exception(f"Sizes: (points,exact_expression) = "
                        f"({len(points)},{len(exact_expression)})")
    for index, (data, sol) in enumerate(zip(points, exact_expression)):
        poly = sym.N(sym.expand(ndd_polynomial(data)))
        sol = sym.N(sol)
        if not (poly == sol):
            raise ValueError(f"ndd_polynomial test {index + 1}. Expected "
                             f"{sol}, got {poly}.")
            pass
        pass
    print(f"All tests passed.\n")
    pass


if __name__ == '__main__':
    test_lagrange()
    test_newton_divided_differences()
    pass
