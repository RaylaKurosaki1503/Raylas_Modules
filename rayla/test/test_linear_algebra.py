"""
Author: Rayla Kurosaki
File: test_linear_algebra.py
Description: This file tests all the functions in linear_algebra.py.
"""

import numpy as np
import linear_algebra as la


def is_equal(M, N):
    return np.array_equal(M, N)


def test_dotP():
    print("Testing dotP()")
    us = [
        np.array([[1, 2, -3]]),
        np.array([[-1, 2]]),
        np.array([[3, -2]]),
        np.array([[1, 2, 3]]),
        np.array([[3.2, -0.6, -1.4]]),
        np.array([[1, np.sqrt(2), np.sqrt(3), 0]]),
        np.array([[1.12, -3.25, 2.07, -1.83]])
    ]
    vs = [
        np.array([[-3, 5, 2]]),
        np.array([[3, 1]]),
        np.array([[4, 6]]),
        np.array([[2, 3, 1]]),
        np.array([[1.5, 4.1, -0.2]]),
        np.array([[4, -np.sqrt(2), 0, -5]]),
        np.array([[-2.29, 1.72, 4.33, -1.54]])
    ]
    res = [
        1, -1, 0, 11, 2.62, 2, 3.6265
    ]
    for i in range(len(us)):
        u, v = us[i], vs[i]
        r1 = np.round(la.dotP(u, v), 4)
        r2 = np.round(res[i], 4)
        if r1 != r2:
            raise Exception("Test " + str(i + 1) + " of dotP() failed. "
                            + str(r1) + " =/= " + str(r2) + ".")
    print("All tests for dotP() passed!\n")
    return


def test_norm():
    print("Testing norm()")
    us = [
        np.array([[1, 2, -3]]),
        np.array([[-1, 2]]),
        np.array([[3, -2]]),
        np.array([[1, 2, 3]]),
        np.array([[3.2, -0.6, -1.4]]),
        np.array([[1, np.sqrt(2), np.sqrt(3), 0]]),
        np.array([[1.12, -3.25, 2.07, -1.83]])
    ]
    res = [
        np.sqrt(14), np.sqrt(5), np.sqrt(13), np.sqrt(14), 3.5440, np.sqrt(6),
        4.4103
    ]
    for i in range(len(us)):
        u = us[i]
        r1 = np.round(la.norm(u), 4)
        r2 = np.round(res[i], 4)
        if r1 != r2:
            raise Exception("Test " + str(i + 1) + " of norm() failed. "
                            + str(r1) + " =/= " + str(r2) + ".")
    print("All tests for norm() passed!\n")
    return


def test_normalize():
    print("Testing normalize()")
    us = [
        np.array([[1, 2, -3]]),
        np.array([[-1, 2]]),
        np.array([[3, -2]]),
        np.array([[1, 2, 3]]),
        np.array([[3.2, -0.6, -1.4]]),
        np.array([[1, np.sqrt(2), np.sqrt(3), 0]]),
        np.array([[1.12, -3.25, 2.07, -1.83]])
    ]
    res = [
        np.array([[0.2673, 0.5345, -0.8018]]),
        np.array([[-0.4472, 0.8944]]),
        np.array([[0.8321, -0.5547]]),
        np.array([[0.2673, 0.5345, 0.8018]]),
        np.array([[0.9029, -0.1693, -0.3950]]),
        np.array([[0.4082, 0.5774, 0.7071, 0]]),
        np.array([[0.2540, -0.7369, 0.4694, -0.4149]])
    ]
    for i in range(len(us)):
        u = us[i]
        r1 = np.round(la.normalize(u), 4)
        r2 = np.round(res[i], 4)
        if not is_equal(r1, r2):
            raise Exception("Test " + str(i + 1) + " of normalize() failed. "
                            + str(np.round(r1, 4)) + " =/= " + str(r2) + ".")
    print("All tests for normalize() passed!\n")
    return


def test_distance():
    print("Testing distance()")
    us = [
        np.array([[1, 2, -3]]),
        np.array([[-1, 2]]),
        np.array([[3, -2]]),
        np.array([[1, 2, 3]]),
        np.array([[3.2, -0.6, -1.4]]),
        np.array([[1, np.sqrt(2), np.sqrt(3), 0]]),
        np.array([[1.12, -3.25, 2.07, -1.83]])
    ]
    vs = [
        np.array([[-3, 5, 2]]),
        np.array([[3, 1]]),
        np.array([[4, 6]]),
        np.array([[2, 3, 1]]),
        np.array([[1.5, 4.1, -0.2]]),
        np.array([[4, -np.sqrt(2), 0, -5]]),
        np.array([[-2.29, 1.72, 4.33, -1.54]])
    ]
    res = [
        5 * np.sqrt(2), np.sqrt(17), np.sqrt(65), np.sqrt(6), 5.1400,
        3 * np.sqrt(5), 6.4437
    ]
    for i in range(len(us)):
        u, v = us[i], vs[i]
        r1 = np.round(la.distance(u, v), 4)
        r2 = np.round(res[i], 4)
        if r1 != r2:
            raise Exception("Test " + str(i + 1) + " of distance() failed. "
                            + str(r1) + " =/= " + str(r2) + ".")
    print("All tests for distance() passed!\n")
    return


def test_angle_between_vectors():
    print("Testing angle_between_vectors()")
    us = [
        np.array([[1, 2, -3]]),
        np.array([[-1, 2]]),
        np.array([[3, -2]]),
        np.array([[1, 2, 3]]),
        np.array([[3.2, -0.6, -1.4]]),
        np.array([[1, np.sqrt(2), np.sqrt(3), 0]]),
        np.array([[1.12, -3.25, 2.07, -1.83]])
    ]
    vs = [
        np.array([[-3, 5, 2]]),
        np.array([[3, 1]]),
        np.array([[4, 6]]),
        np.array([[2, 3, 1]]),
        np.array([[1.5, 4.1, -0.2]]),
        np.array([[4, -np.sqrt(2), 0, -5]]),
        np.array([[-2.29, 1.72, 4.33, -1.54]])
    ]
    res = [
        1.5274, 1.7127, 1.5708, 0.6669, 1.4008, 1.4460, 1.4184
    ]
    for i in range(len(us)):
        u, v = us[i], vs[i]
        r1 = np.round(la.angle_between_vectors(u, v), 4)
        r2 = res[i]
        if r1 != r2:
            raise Exception("Test " + str(i + 1) +
                            " of angle_between_vectors() failed. " + str(r1) +
                            " =/= " + str(r2) + ".")
    print("All tests for angle_between_vectors() passed!\n")
    return


def test_is_orthogonal():
    print("Testing is_orthogonal()")
    us = [
        np.array([[1, 2, -3]]),
        np.array([[-1, 2]]),
        np.array([[3, -2]]),
        np.array([[1, 2, 3]]),
        np.array([[3.2, -0.6, -1.4]]),
        np.array([[1, np.sqrt(2), np.sqrt(3), 0]]),
        np.array([[1.12, -3.25, 2.07, -1.83]]),
        np.array([[1, 1, -2]])
    ]
    vs = [
        np.array([[-3, 5, 2]]),
        np.array([[3, 1]]),
        np.array([[4, 6]]),
        np.array([[2, 3, 1]]),
        np.array([[1.5, 4.1, -0.2]]),
        np.array([[4, -np.sqrt(2), 0, -5]]),
        np.array([[-2.29, 1.72, 4.33, -1.54]]),
        np.array([[3, 1, 2]])
    ]
    res = [
        False, False, True, False, False, False, False, True
    ]
    for i in range(len(us)):
        u, v = us[i], vs[i]
        r1 = la.is_orthogonal(u, v)
        r2 = res[i]
        if r1 != r2:
            raise Exception("Test " + str(i + 1) +
                            " of is_orthogonal() failed. " + str(r1) +
                            " =/= " + str(r2) + ".")
    print("All tests for is_orthogonal() passed!\n")
    return


def test_proj():
    print("Testing proj()")
    us = [
        np.array([[1, 2, -3]]),
        np.array([[-1, 2]]),
        np.array([[3, -2]]),
        np.array([[1, 2, 3]]),
        np.array([[3.2, -0.6, -1.4]]),
        np.array([[1, np.sqrt(2), np.sqrt(3), 0]]),
        np.array([[1.12, -3.25, 2.07, -1.83]])
    ]
    vs = [
        np.array([[-3, 5, 2]]),
        np.array([[3, 1]]),
        np.array([[4, 6]]),
        np.array([[2, 3, 1]]),
        np.array([[1.5, 4.1, -0.2]]),
        np.array([[4, -np.sqrt(2), 0, -5]]),
        np.array([[-2.29, 1.72, 4.33, -1.54]])
    ]
    res = [
        np.array([[1 / 14, 1 / 7, -3 / 14]]),
        np.array([[1 / 5, -2 / 5]]),
        np.array([[0, 0]]),
        np.array([[11 / 14, 11 / 7, 33 / 14]]),
        np.array([[0.667516, -0.125159, -0.292038]]),
        np.array([[1 / 3, np.sqrt(2) / 3, 1 / np.sqrt(3), 0]]),
        np.array([[0.208819, -0.605949, 0.385943, -0.341196]]),
    ]
    for i in range(len(us)):
        u, v = us[i], vs[i]
        r1 = np.round(la.proj(v, u), 4)
        r2 = np.round(res[i], 4)
        if not is_equal(r1, r2):
            raise Exception("Test " + str(i + 1) + " of proj() failed. "
                            + str(r1) + " =/= " + str(r2) + ".")
    print("All tests for proj() passed!\n")
    return


def test_ref():
    return


def test_rank():
    return


def test_rref():
    return


def test_is_consistent():
    return


def test_is_linear_combination_vector():
    return


def test_is_linearly_independent_vector():
    return


def test_multiply():
    return


def test_power():
    return


def test_is_symmetric():
    return


def test_is_linear_combination_matrix():
    return


def test_is_linearly_independent_matrix():
    return


def test_is_square():
    return


def test_det():
    return


def test_gauss_jordan():
    return


def test_inverse_gj():
    return


def test_LU():
    return


def test_pLU():
    return


def test_is_row_space():
    return


def test_is_col_space():
    return


def test_basis_row_space():
    return


def test_basis_col_space():
    return


def test_basis_null_space():
    return


def test_dim():
    return


def test_nullity():
    return


if __name__ == '__main__':
    test_dotP()
    test_norm()
    test_normalize()
    test_distance()
    test_angle_between_vectors()
    test_is_orthogonal()
    test_proj()
    test_ref()
    test_rank()
    test_rref()
    test_is_consistent()
    test_is_linear_combination_vector()
    test_is_linearly_independent_vector()
    test_multiply()
    test_power()
    test_is_symmetric()
    test_is_linear_combination_matrix()
    test_is_linearly_independent_matrix()
    test_is_square()
    test_det()
    test_gauss_jordan()
    test_inverse_gj()
    test_LU()
    test_pLU()
    test_is_row_space()
    test_is_col_space()
    test_basis_row_space()
    test_basis_col_space()
    test_basis_null_space()
    test_dim()
    test_nullity()
    pass
