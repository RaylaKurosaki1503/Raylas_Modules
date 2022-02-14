"""
Author: Rayla Kurosaki
File: test_root_solver.py
Description: This file tests all the functions in root_solver.py.
"""

import numpy as np
import sympy as sym
import root_solver as rs


def test_bisection():
    print("Testing bisection()")
    fs = [
        lambda x: x ** 3 + x - 1,
        lambda x: x ** 3 - 9,
        lambda x: 3 * x ** 3 + x ** 2 - x - 5,
        lambda x: (np.cos(x)) ** 2 + 6 - x,
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
        [0, 1], [0, 5], [0, 5], [0, 10], [0, 1], [-1.5, -0.5], [1, 2],
        [-2, -1], [-1, 0], [1, 2], [-2, -1], [-0.5, 0.5], [0.5, 1.5],
        [-1.7, -0.7], [-0.7, 0.3], [0.3, 1.3],
    ]
    ds = [
        4, 6, 6, 6, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6
    ]
    root_exacts = [
        0.6823, 2.080084, 1.169726, 6.776092, 0.75487766, -0.97089892,
        1.59214294, -1.641783, -0.168254, 1.810038, -1.023482, 0.163822,
        0.788941, -0.818094, 0, 0.506308
    ]
    for i in range(len(fs)):
        f, a, b, d = fs[i], Is[i][0], Is[i][1], ds[i]
        TOL = 10 ** (-d)
        root_exact = root_exacts[i]
        root_approx = np.round(rs.bisection(f, a, b, TOL), d)
        if root_approx != root_exact:
            raise Exception("Test " + str(i + 1) + " of bisection() failed. "
                            + str(root_approx) + " =/= " + str(root_exact) +
                            ".")

    print("All tests for bisection() passed!\n")
    pass


def test_fpi():
    print("Testing fpi()")
    fs = [
        lambda x: x + np.cos(x) - np.sin(x),
        lambda x: (2 * x + 2) ** (1 / 3),
        lambda x: np.log(7 - x),
        lambda x: np.log(4 - np.sin(x)),
        lambda x: 1 / (1 + x ** 4),
        lambda x: (np.sin(x) - 5) / 6,
        lambda x: (3 - np.log(x)) ** (1 / 2)
    ]
    x_is = [
        0, 1, 1, 1, 1, 1, 1
    ]
    ds = [
        7, 8, 8, 8, 8, 8, 8
    ]
    root_exacts = [
        0.7853982, 1.76929235, 1.67282170, 1.12998050, 0.75487767, -0.97089892,
        1.59214294
    ]
    for i in range(len(fs)):
        f, x_i, d = fs[i], x_is[i], ds[i]
        TOL = 10 ** (-d)
        root_exact = root_exacts[i]
        root_approx = np.round(rs.fpi(f, x_i, TOL), d)
        if root_approx != root_exact:
            raise Exception("Test " + str(i + 1) + " of fpi() failed. "
                            + str(root_approx) + " =/= " + str(root_exact) +
                            ".")

    print("All tests for fpi() passed!\n")
    pass


def test_newton():
    print("Testing newton()")
    fs = [
        lambda x: x ** 3 + x - 1,
        lambda x: x ** 3 - 2 * x - 2,
        lambda x: sym.exp(x) + x - 7,
        lambda x: sym.exp(x) + sym.sin(x) - 4,
        lambda x: x ** 5 + x - 1,
        lambda x: sym.sin(x) - 6 * x - 5,
        lambda x: sym.log(x) + x ** 2 - 3
    ]
    x_is = [
        -0.7, 1, 1, 1, 1, 1, 1
    ]
    ds = [
        8, 8, 8, 8, 8, 8, 8
    ]
    root_exacts = [
        0.68232780, 1.76929235, 1.67282170, 1.12998050, 0.75487767, -0.97089892,
        1.59214294
    ]
    for i in range(len(fs)):
        f, x_i, d = fs[i], x_is[i], ds[i]
        TOL = 10 ** (-d)
        root_exact = root_exacts[i]
        root_approx = np.round(rs.newton(f, x_i, TOL), d)
        if root_approx != root_exact:
            raise Exception("Test " + str(i + 1) + " of newton() failed. "
                            + str(root_approx) + " =/= " + str(root_exact) +
                            ".")

    print("All tests for newton() passed!\n")
    pass


def test_secant():
    print("Testing secant()")
    fs = [
        lambda x: x ** 3 - 2 * x - 2,
        lambda x: np.exp(x) + x - 7,
        lambda x: np.exp(x) + np.sin(x) - 4
    ]
    x_is = [
        [1, 2], [1, 2], [1, 2]
    ]
    ds = [
        8, 8, 8
    ]
    root_exacts = [
        1.76929235, 1.67282170, 1.12998050
    ]
    for i in range(len(fs)):
        f, x0, x1, d = fs[i], x_is[i][0], x_is[i][1], ds[i]
        TOL = 10 ** (-d)
        root_exact = root_exacts[i]
        root_approx = np.round(rs.secant(f, x0, x1, TOL), d)
        if root_approx != root_exact:
            raise Exception("Test " + str(i + 1) + " of secant() failed. "
                            + str(root_approx) + " =/= " + str(root_exact) +
                            ".")

    print("All tests for secant() passed!\n")
    pass


def test_false_position():
    print("Testing false_position()")
    fs = [
        lambda x: x ** 3 - 2 * x - 2,
        lambda x: np.exp(x) + x - 7,
        lambda x: np.exp(x) + np.sin(x) - 4
    ]
    Is = [
        [1, 2], [1, 2], [1, 2]
    ]
    ds = [
        8, 8, 8
    ]
    root_exacts = [
        1.76929235, 1.67282170, 1.12998050
    ]
    for i in range(len(fs)):
        f, a, b, d = fs[i], Is[i][0], Is[i][1], ds[i]
        TOL = 10 ** (-d)
        root_exact = root_exacts[i]
        root_approx = np.round(rs.false_position(f, a, b, TOL), d)
        if root_approx != root_exact:
            raise Exception("Test " + str(i + 1) +
                            " of false_position() failed. " + str(root_approx)
                            + " =/= " + str(root_exact) + ".")

    print("All tests for false_position() passed!\n")
    pass


# def test_muller():
#     print("Testing muller()")
#     fs = [
#         lambda x: x**3-2*x**2-5,
#         # lambda x: x**3+3*x**2-1,
#         lambda x: x**3-x-1,
#         lambda x: x**4+2*x**2-x-3,
#         lambda x: x**3+4.001*x**2+4.002*x+1.101,
#         lambda x: x**5-x**4+2*x**3-3*x**2+x-4,
#     ]
#     x_is = [
#         [0,1,2],
#         # [-2,0,2],
#         [0,1,2]
#     ]
#     ds = [
#         4,4,4,4,4,4
#     ]
#     root_exacts = [
#         2.6906,
#         # [-2.8794, -0.6527, 0.5321]
#     ]
#     for i in range(len(fs)):
#         f, a, b, d = fs[i], Is[i][0], Is[i][1], ds[i]
#         TOL = 10 ** (-d)
#         root_exact = root_exacts[i]
#         root_approx = np.round(rs.muller(f, a, b, TOL), d)
#         if root_approx != root_exact:
#             raise Exception("Test " + str(i + 1) +
#                             " of false_position() failed. " + str(root_approx)
#                             + " =/= " + str(root_exact) + ".")
#
#     print("All tests for muller() passed!\n")
#     pass
#
#
# def test_steffensen():
#     pass
#
#
# def test_itp():
#     pass
#
#
# def test_halley():
#     pass


if __name__ == '__main__':
    test_bisection()
    test_fpi()
    test_newton()
    test_secant()
    test_false_position()
    # test_muller()
    # test_steffensen()
    # test_itp()
    # test_halley()
    pass
