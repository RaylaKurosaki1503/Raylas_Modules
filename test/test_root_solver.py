"""
Author: Rayla Kurosaki

File: test_root_solver.py

Description: This file test the most relevant functions in root_solver.py
"""

import numpy as np
import sympy as sym

from rayla.math.root_solver import bisection
from rayla.math.root_solver import fpi
from rayla.math.root_solver import newton
from rayla.math.root_solver import secant


def test_bisection():
    fs = [
        lambda x: x ** 3 + x - 1,
        lambda x: np.cos(x) - x,
        lambda x: np.exp(x) - 3,
        lambda x: x ** 3 - 9,
        lambda x: 3 * x ** 3 + x ** 2 - x - 5,
        lambda x: (np.cos(x)) ** 2 - x + 6,
        lambda x: x ** 5 + x - 1,
        lambda x: np.sin(x) - 6 * x - 5,
        lambda x: np.log(x) + x ** 2 - 3,
        lambda x: 2 * x ** 3 - 6 * x - 1,
        lambda x: 2 * x ** 3 - 6 * x - 1,
        lambda x: 2 * x ** 3 - 6 * x - 1,
        lambda x: np.exp(x - 2) + x ** 3 - x,
        lambda x: np.exp(x - 2) + x ** 3 - x,
        lambda x: np.exp(x - 2) + x ** 3 - x,
        lambda x: 1 + 5 * x - 6 * x ** 3 - np.exp(2 * x),
        lambda x: 1 + 5 * x - 6 * x ** 3 - np.exp(2 * x),
        lambda x: 1 + 5 * x - 6 * x ** 3 - np.exp(2 * x)
    ]
    Is = [[0, 1], [0, 1], [1, 2], [2, 3], [1, 2], [6, 7], [0, 1], [-1, 0],
          [1, 2], [-2, -1], [-1, 0], [1, 2], [-2, -1], [0, 0.5], [0.5, 1],
          [-1, 0], [-0.5, 0.5], [0.5, 1]]
    exact_rs = [
        0.68232780382801933, 0.73908513321516064, 1.09861228866810970,
        2.08008382305190411, 1.16972621985372430, 6.77609231631950233,
        0.75487766624669276, -0.97089892350425586, 1.59214293705809387,
        -1.64178352745292568, -0.16825440178102742, 1.81003792923395310,
        -1.02348219485823649, 0.16382224325010850, 0.78894138905554557,
        -0.81809373448119542, 0.00000000000000000, 0.50630828634622120
    ]
    ds = [4, 4, 6, 6, 6, 6, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    print("Testing bisection functionality")
    if not (len(fs) == len(Is) == len(exact_rs) == len(ds)):
        raise Exception(f"Sizes: (fs,Is,exact_rs,ds) = "
                        f"({len(fs)},{len(Is)},{len(exact_rs)},{len(ds)})")
    for index, (f, I, r, d) in enumerate(zip(fs, Is, exact_rs, ds)):
        a, b = I
        approx = bisection(f, a, b, 10 ** (-d))
        if not (round(abs(r - approx)) < 10 ** (-d)):
            raise ValueError(f"Bisection test {index + 1}. Expected "
                             f"{round(r, d)}, got {round(approx, d)}.")
    print(f"All tests passed.\n")
    pass


def test_fpi():
    fs = [
        lambda x: (1 + 2 * x ** 3) / (1 + 3 * x ** 2),
        lambda x: np.cos(x),
        lambda x: 2.8 * x - x ** 2,
        lambda x: (2 * x + 2) ** (1 / 3),
        lambda x: np.log(7 - x),
        lambda x: np.log(4 - np.sin(x)),
        lambda x: (1 - x) ** (1 / 5),
        lambda x: (np.sin(x) - 5) / 6,
        lambda x: (3 - np.log(x)) ** (1 / 2)
    ]
    x0s = [0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    exact_rs = [
        0.68232780382801933, 0.73908513321516064, 1.80000000000000000,
        1.76929235423863142, 1.67282169862890654, 1.60928974542557943,
        0.75487766624669276, -0.9708989235042559, 1.59214293705809387
    ]
    ds = [8, 8, 8, 8, 8, 8, 8, 8, 8]
    print("Testing fpi functionality")
    if not (len(fs) == len(x0s) == len(exact_rs) == len(ds)):
        raise Exception(f"Sizes: (fs,x0s,exact_rs,ds) = "
                        f"({len(fs)},{len(x0s)},{len(exact_rs)},{len(ds)})")
    for index, (f, x0, r, d) in enumerate(zip(fs, x0s, exact_rs, ds)):
        approx = fpi(f, x0, 10 ** (-d))
        if not (round(abs(r - approx)) < 10 ** (-d)):
            raise ValueError(f"fpi test {index + 1}. Expected "
                             f"{round(r, d)}, got {round(approx, d)}.")
    print(f"All tests passed.\n")
    pass


def test_newton():
    fs = [
        lambda x: x ** 3 + x - 1,
        lambda x: x ** 3 - 2 * x - 2,
        lambda x: sym.exp(x) + x - 7,
        lambda x: sym.exp(x) + sym.sin(x) - 4,
        lambda x: x ** 5 + x - 1,
        lambda x: sym.sin(x) - 6 * x - 5,
        lambda x: sym.log(x) + x ** 2 - 3,
        lambda x: 27 * x ** 3 + 54 * x ** 2 + 36 * x + 8,
        lambda x: 36 * x ** 4 - 12 * x ** 3 + 37 * x ** 2 - 12 * x + 1,
        lambda x: 2 * sym.exp(x - 1) - x ** 2 - 1,
        lambda x: sym.log(3 - x) + x - 2
    ]
    x0s = [-0.7, 1, 0, 0, 0, 0, 0.1, 0, 0, 0.1, 0.1]
    exact_rs = [
        0.68232780382801933, 1.76929235423863142, 1.67282169862890654,
        1.12998049865083241, 0.75487766624669276, -0.9708989235042559,
        1.59214293705809387, -0.6666666666666667, 0.16666666666666667,
        1.00000000000000000, 2.00000000000000000
    ]
    ds = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    print("Testing newton functionality")
    if not (len(fs) == len(x0s) == len(exact_rs) == len(ds)):
        raise Exception(f"Sizes: (fs,x0s,exact_rs,ds) = "
                        f"({len(fs)},{len(x0s)},{len(exact_rs)},{len(ds)})")
    for index, (f, x0, r, d) in enumerate(zip(fs, x0s, exact_rs, ds)):
        approx = newton(f, x0, 10 ** (-d))
        if not (round(abs(r - approx)) < 10 ** (-d)):
            raise ValueError(f"newton test {index + 1}. Expected "
                             f"{round(r, d)}, got {round(approx, d)}.")
    print(f"All tests passed.\n")
    pass


def test_secant():
    fs = [
        lambda x: x ** 3 + x - 1,
        lambda x: x ** 3 - 2 * x - 2,
        lambda x: np.exp(x) + x - 7,
        lambda x: np.exp(x) + np.sin(x) - 4,
        lambda x: x ** 5 + x - 1,
        lambda x: np.sin(x) - 6 * x - 5,
        lambda x: np.log(x) + x ** 2 - 3,
        lambda x: 36 * x ** 4 - 12 * x ** 3 + 37 * x ** 2 - 12 * x + 1
    ]
    xis = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 2],
           [0, 1]]
    exact_rs = [
        0.68232780382801933, 1.76929235423863142, 1.67282169862890654,
        1.12998049865083241, 0.75487766624669276, -0.9708989235042559,
        1.59214293705809387, 0.16666666666666667
    ]
    ds = [8, 8, 8, 8, 8, 8, 8, 8]
    print("Testing secant functionality")
    if not (len(fs) == len(xis) == len(exact_rs) == len(ds)):
        raise Exception(f"Sizes: (fs,xis,exact_rs,ds) = "
                        f"({len(fs)},{len(xis)},{len(exact_rs)},{len(ds)})")
    for index, (f, xi, r, d) in enumerate(zip(fs, xis, exact_rs, ds)):
        x0, x1 = xi
        approx = secant(f, x0, x1, 10 ** (-d))
        if not (round(abs(r - approx)) < 10 ** (-d)):
            raise ValueError(f"secant test {index + 1}. Expected "
                             f"{round(r, d)}, got {round(approx, d)}.")
    print(f"All tests passed.\n")
    pass


def test_false_position():
    pass


def test_muller():
    pass


def test_iqi():
    pass


def test_brent():
    pass


def test_steffensen():
    pass


def test_itp():
    pass


def test_halley():
    pass


if __name__ == '__main__':
    test_bisection()
    test_fpi()
    test_newton()
    test_secant()

    test_false_position()
    test_muller()
    test_iqi()
    test_brent()
    test_steffensen()
    test_itp()
    test_halley()
    pass
