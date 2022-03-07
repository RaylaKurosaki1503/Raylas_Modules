"""
Author: Rayla Kurosaki

File: test_root_solver.py

Description:
"""
import numpy as np

from root_solver import bisection, fpi


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
        lambda x: 1 + 5 * x - 6 * x ** 3 - np.exp(2 * x),
    ]
    Is = [
        [0, 1], [0, 1], [1, 2], [2, 3], [1, 2], [6, 7], [0, 1], [-1, 0],
        [1, 2], [-2, -1], [-1, 0], [1, 2], [-2, -1], [0, 0.5], [0.5, 1],
        [-1, 0], [-0.5, 0.5], [0.5, 1]
    ]
    exact_rs = [
        0.68232780382801933, 0.73908513321516064, 1.09861228866810970,
        2.08008382305190411, 1.16972621985372430, 6.77609231631950233,
        0.75487766624669276, -0.97089892350425586, 1.59214293705809387,
        -1.64178352745292568, -0.16825440178102742, 1.81003792923395310,
        -1.02348219485823649, 0.16382224325010850, 0.78894138905554557,
        -0.81809373448119542, 0.00000000000000000, 0.50630828634622120,

    ]
    ds = [
        4, 4, 6, 6, 6, 6, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6
    ]
    print("Testing bisection functionality")
    for index, (f, I, r, d) in enumerate(zip(fs, Is, exact_rs, ds)):
        a, b = I
        approx = bisection(f, a, b, 10 ** (-d))
        if not (round(abs(r - approx)) < 10 ** (-d)):
            raise ValueError(f"Bisection test {index + 1}. Expected "
                             f"{round(r, d)}, got {round(approx, d)}.")
    print(f"All tests passed.\n")
    pass


def test_fpi():
    pass


def test_newton():
    pass


def test_secant():
    pass


def test_false_position():
    pass


def test_muller():
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
    test_steffensen()
    test_itp()
    test_halley()
    pass
