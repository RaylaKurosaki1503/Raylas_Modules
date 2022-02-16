"""
Author: Rayla Kurosaki
File: test_linear_algebra.py
File type: Test file
Description: This file tests all the functions in linear_algebra.py.
"""

import numpy as np
import linear_algebra as la


################################################################################
# Helper functions
################################################################################
def is_equal(M, N):
    """
    A helper function to determine if two matrices are equal. This is created
    simply for readability.
    """
    return np.array_equal(M, N)


################################################################################
# Main test functions
################################################################################
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
    for i in range(len(res)):
        r1 = np.round(la.dot_product(us[i], vs[i]), 4)
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
    for i in range(len(res)):
        r1 = np.round(la.norm(us[i]), 4)
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
    for i in range(len(res)):
        r1 = np.round(la.normalize(us[i]), 4)
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
    for i in range(len(res)):
        r1 = np.round(la.distance(us[i], vs[i]), 4)
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
    for i in range(len(res)):
        r1 = np.round(la.angle_between_vectors(us[i], vs[i]), 4)
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
    for i in range(len(res)):
        r1 = la.is_orthogonal(us[i], vs[i])
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
    for i in range(len(res)):
        r1 = np.round(la.proj(vs[i], us[i]), 4)
        r2 = np.round(res[i], 4)
        if not is_equal(r1, r2):
            raise Exception("Test " + str(i + 1) + " of proj() failed. "
                            + str(r1) + " =/= " + str(r2) + ".")
    print("All tests for proj() passed!\n")
    return


def test_ref():
    print("Testing ref()")
    print("Since the row echelon form of a matrix is not unique, we will skip "
          "testing this \nfunction. We will assume that the programmer has "
          "correctly implemented the row \nechelon form algorithm.\n")
    return


def test_gaussian_elimination():
    print("Testing gaussian_elimination()")
    print("The implementation of gaussian_elimination() simply calls ref(). "
          "Therefore, there \nis no need to test this function.\n")
    return


def test_rank():
    print("Testing rank()")
    Ms = [
        np.array([
            [1, 2, -3, 9],
            [2, -1, 1, 0],
            [4, -1, 1, 4]
        ], dtype=float),
        np.array([
            [1, -1, -1, 0],
            [-1, 3, 1, 5],
            [3, 1, 7, 2]
        ], dtype=float),
        np.array([
            [1, -3, -2, 0],
            [-1, 2, 1, 0],
            [2, 4, 6, 0]
        ], dtype=float),
        np.array([
            [2, 3, -1, 4, 1],
            [3, -1, 0, 1, 1],
            [3, -4, 1, -1, 2]
        ], dtype=float),
        np.array([
            [2, 1, 3],
            [4, 1, 7],
            [2, 5, -1]
        ], dtype=float),
        np.array([
            [-1, 3, -2, 4, 0],
            [2, -6, 1, -2, -3],
            [1, -3, 4, -8, 2]
        ], dtype=float),
        np.array([
            [1 / 2, 1, -1, -6, 0, 2],
            [1 / 6, 1 / 2, 0, -3, 1, -1],
            [1 / 3, 0, -2, 0, -4, 8]
        ], dtype=float),
        np.array([
            [np.sqrt(2), 1, 2, 1],
            [0, np.sqrt(2), -3, -np.sqrt(2)],
            [0, -1, np.sqrt(2), 1]
        ], dtype=float),
        np.array([
            [1, 1, 2, 1, 1],
            [1, -1, -1, 1, 0],
            [0, 1, 1, 0, -1],
            [1, 1, 0, 1, 2]
        ], dtype=float),
        np.array([
            [1, 1, 1, 1, 4],
            [1, 2, 3, 4, 10],
            [1, 3, 6, 10, 20],
            [1, 4, 10, 20, 35]
        ], dtype=float),
    ]
    res = [
        3, 3, 2, 3, 2, 2, 2, 3, 4, 4
    ]
    for i in range(len(res)):
        r1 = la.rank(Ms[i])
        r2 = res[i]
        if not (r1 == r2):
            raise Exception("Test " + str(i + 1) + " of rank() failed. "
                            + str(r1) + " =/= " + str(r2) + ".")
    print("All tests for rank() passed!\n")
    return


def test_rref():
    print("Testing rref()")
    Ms = [
        np.array([
            [1, 2, -3, 9],
            [2, -1, 1, 0],
            [4, -1, 1, 4]
        ], dtype=float),
        np.array([
            [1, -1, -1, 0],
            [-1, 3, 1, 5],
            [3, 1, 7, 2]
        ], dtype=float),
        np.array([
            [1, -3, -2, 0],
            [-1, 2, 1, 0],
            [2, 4, 6, 0]
        ], dtype=float),
        np.array([
            [2, 3, -1, 4, 1],
            [3, -1, 0, 1, 1],
            [3, -4, 1, -1, 2]
        ], dtype=float),
        np.array([
            [2, 1, 3],
            [4, 1, 7],
            [2, 5, -1]
        ], dtype=float),
        np.array([
            [-1, 3, -2, 4, 0],
            [2, -6, 1, -2, -3],
            [1, -3, 4, -8, 2]
        ], dtype=float),
        np.array([
            [1 / 2, 1, -1, -6, 0, 2],
            [1 / 6, 1 / 2, 0, -3, 1, -1],
            [1 / 3, 0, -2, 0, -4, 8]
        ], dtype=float),
        np.array([
            [np.sqrt(2), 1, 2, 1],
            [0, np.sqrt(2), -3, -np.sqrt(2)],
            [0, -1, np.sqrt(2), 1]
        ], dtype=float),
        np.array([
            [1, 1, 2, 1, 1],
            [1, -1, -1, 1, 0],
            [0, 1, 1, 0, -1],
            [1, 1, 0, 1, 2]
        ], dtype=float),
        np.array([
            [1, 1, 1, 1, 4],
            [1, 2, 3, 4, 10],
            [1, 3, 6, 10, 20],
            [1, 4, 10, 20, 35]
        ], dtype=float),
    ]
    res = [
        np.array([
            [1, 0, 0, 2],
            [0, 1, 0, 5],
            [0, 0, 1, 1]
        ]),
        np.array([
            [1, 0, 0, 17 / 10],
            [0, 1, 0, 5 / 2],
            [0, 0, 1, -4 / 5]
        ]),
        np.array([
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ]),
        np.array([
            [1, 0, 0, 1, 1],
            [0, 1, 0, 2, 2],
            [0, 0, 1, 4, 7]
        ]),
        np.array([
            [1, 0, 2],
            [0, 1, -1],
            [0, 0, 0]
        ]),
        np.array([
            [1, -3, 0, 0, -2],
            [0, 0, 1, -2, 1],
            [0, 0, 0, 0, 0]
        ]),
        np.array([
            [1, 0, -6, 0, -12, 24],
            [0, 1, 2, -6, 6, -10],
            [0, 0, 0, 0, 0, 0]
        ]),
        np.array([
            [1, 0, 0, np.sqrt(2)],
            [0, 1, 0, -1],
            [0, 0, 1, 0]
        ]),
        np.array([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1]
        ]),
        np.array([
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1]
        ]),
    ]
    for i in range(len(res)):
        r1 = np.round(la.rref(Ms[i]), 4)
        r2 = np.round(res[i], 4)
        if not is_equal(r1, r2):
            raise Exception("Test " + str(i + 1) + " of rref() failed. "
                            + str(r1) + " =/= " + str(r2) + ".")
    print("All tests for rref() passed!\n")
    return


def test_gauss_jordan_elimination():
    print("Testing gauss_jordan_elimination()")
    print("The implementation of gauss_jordan_elimination() simply calls "
          "rref(). Therefore, \nthere is no need to test this function.\n")
    return


def test_is_consistent():
    print("Testing is_consistent()")
    Ms = [
        np.array([
            [1, 0, 0, 3],
            [0, 1, 0, 2],
            [0, 0, 1, 0]
        ], dtype=float),
        np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1]
        ], dtype=float),
        np.array([
            [1, 0, 0, 3],
            [0, 1, 0, 2],
            [0, 0, 0, 0]
        ], dtype=float),
    ]
    res = [
        True, False, True,
    ]
    for i in range(len(res)):
        M = Ms[i]
        r1 = la.is_consistent(M)
        r2 = res[i]
        if not (r1 == r2):
            raise Exception("Test " + str(i + 1) +
                            " of is_consistent() failed. " + str(r1) +
                            " =/= " + str(r2) + ".")
    print("All tests for is_consistent() passed!\n")
    return


def test_num_free_var():
    print("Testing num_free_var()")
    Ms = [
        np.array([
            [0, 2, 3, 8],
            [2, 3, 1, 5],
            [1, -1, -2, -5]
        ], dtype=float),
        np.array([
            [1, -1, -1, 2, 1],
            [2, -2, -1, 3, 3],
            [-1, 1, -1, 0, -3]
        ], dtype=float),
    ]
    res = [
        0, 2
    ]
    for i in range(len(res)):
        r1 = la.num_free_var(Ms[i])
        r2 = res[i]
        if not (r1 == r2):
            raise Exception("Test " + str(i + 1) +
                            " of num_free_var() failed. " + str(r1) +
                            " =/= " + str(r2) + ".")
    print("All tests for num_free_var() passed!\n")
    return


def test_is_linear_combination_vector():
    print("Testing is_linear_combination_vector()")
    vs_lst = [
        [np.array([[1, 0, 3]]), np.array([[-1, 1, -3]])],
        [np.array([[1, 0, 3]]), np.array([[-1, 1, -3]])],
        [np.array([[1, -1]]), np.array([[2, -1]])],
        [np.array([[4, -2]]), np.array([[-2, 1]])],
        [np.array([[1, 1, 0]]), np.array([[0, 1, 1]])],
        [np.array([[1, 1, 0]]), np.array([[0, 1, 1]])],
        [np.array([[1, 1, 0]]), np.array([[0, 1, 1]]), np.array([[1, 0, 1]])],
        [np.array([[1.0, 0.4, 4.8]]), np.array([[3.4, 1.4, -6.4]]),
         np.array([[-1.2, 0.2, -1.0]])],
    ]
    v_lst = [
        np.array([[1, 2, 3]]),
        np.array([[2, 3, 4]]),
        np.array([[1, 2]]),
        np.array([[2, 1]]),
        np.array([[1, 2, 3]]),
        np.array([[3, 2, -1]]),
        np.array([[1, 2, 3]]),
        np.array([[3.2, 2.0, -2.6]]),
    ]
    res = [
        True, False, True, False, False, True, True, True
    ]
    for i in range(len(res)):
        vs, v = vs_lst[i], v_lst[i]
        r1 = la.is_linear_combination_vector(vs, v)
        r2 = res[i]
        if not (r1 == r2):
            raise Exception("Test " + str(i + 1) +
                            " of is_linear_combination_vector() failed. "
                            + str(r1) + " =/= " + str(r2) + ".")
    print("All tests for is_linear_combination_vector() passed!\n")
    return


def test_is_linearly_independent_vector():
    print("Testing is_linearly_independent_vector()")
    vs = [
        [np.array([[1, 4]]), np.array([[-1, 2]])],
        [np.array([[1, 1, 0]]), np.array([[0, 1, 1]]),
         np.array([[1, 0, 1]])],
        [np.array([[1, -1, 0]]), np.array([[0, 1, -1]]),
         np.array([[-1, 0, 1]])],
        [np.array([[1, 2, 0]]), np.array([[1, 1, -1]]),
         np.array([[1, 4, 2]])],
        [np.array([[2, -1, 3]]), np.array([[-1, 2, 3]])],
        [np.array([[1, 1, 1]]), np.array([[1, 2, 3]]),
         np.array([[1, -1, 2]])],
        [np.array([[2, 2, 1]]), np.array([[3, 1, 2]]),
         np.array([[1, -5, 2]])],
        [np.array([[0, 1, 2]]), np.array([[2, 1, 3]]),
         np.array([[2, 0, 1]])],
        [np.array([[-2, 3, 7]]), np.array([[4, -1, 5]]),
         np.array([[3, 1, 3]]), np.array([[5, 0, 2]])],
        [np.array([[3, 4, 5]]), np.array([[6, 7, 8]]),
         np.array([[0, 0, 0]])],
        [np.array([[-1, 1, 2, 1]]), np.array([[3, 2, 2, 4]]),
         np.array([[2, 3, 1, -1]])],
        [np.array([[1, -1, 1, 0]]), np.array([[-1, 1, 0, 1]]),
         np.array([[1, 0, 1, -1]]), np.array([[0, 1, -1, 1]])],
        [np.array([[0, 0, 0, 1]]), np.array([[0, 0, 2, 1]]),
         np.array([[0, 3, 2, 1]]), np.array([[4, 3, 2, 1]])],
        [np.array([[3, -1, 1, -1]]), np.array([[-1, 3, 1, -1]]),
         np.array([[1, 1, 3, 1]]), np.array([[-1, -1, 1, 3]])],
    ]
    res = [
        True, True, False, False, True, True, False, False, False, False,
        True, True, True, False
    ]
    for i in range(len(res)):
        r1 = la.is_linearly_independent_vector(vs[i])
        r2 = res[i]
        if not (r1 == r2):
            raise Exception("Test " + str(i + 1) +
                            " of is_linearly_independent_vector() failed. "
                            + str(r1) + " =/= " + str(r2) + ".")
    print("All tests for is_linearly_independent_vector() passed!\n")


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


def test_basis():
    return


def test_dim():
    return


def test_nullity():
    return


if __name__ == '__main__':
    # test_dotP()
    # test_norm()
    # test_normalize()
    # test_distance()
    # test_angle_between_vectors()
    # test_is_orthogonal()
    # test_proj()
    # test_ref()
    # test_gaussian_elimination()
    # test_rank()
    # test_rref()
    # test_gauss_jordan_elimination()
    # test_is_consistent()
    # test_num_free_var()
    # test_is_linear_combination_vector()
    # test_is_linearly_independent_vector()
    test_multiply()
    test_power()
    test_is_symmetric()
    test_is_linear_combination_matrix()
    test_is_linearly_independent_matrix()
    test_is_square()
    test_det()
    test_inverse_gj()
    test_LU()
    test_pLU()
    test_is_row_space()
    test_is_col_space()
    test_basis()
    test_dim()
    test_nullity()
    pass
