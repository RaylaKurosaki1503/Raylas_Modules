"""
Author: Rayla Kurosaki

File: linear_algebra.py

Description: This file contains functions to solve vector/matrix based problems
             numerically. These functions are based on the formulas and
             algorithms from the following references:
             1. "Linear Algebra, A Modern Introduction" by David Poole
                (4th edition)
             2.
"""

import numpy as np

import copy

d = 8
TOL = 1 * 10 ** (-d)


################################################################################
# Helper functions
################################################################################
def separate_augmented_matrix(M_aug):
    """
    A helper function to separate an augmented matrix into its coefficient
    matrix and column vector.

    :param M_aug: An augmented matrix (numpy.array).
    :return: The coefficient matrix (numpy.array) and the column vector
             (numpy.array).
    """
    return M_aug[:, :-1], M_aug[:, -1]


################################################################################
# Main functions
################################################################################
def dot_product(u, v):
    """
    Computes dot product of 2 row vectors.

    :param u: A row vector (numpy.array).
    :param v: A row vector (numpy.array).
    :return: The dot product of u and v (number).
    """
    mu, nu = np.shape(u)
    mv, nv = np.shape(v)
    if not (mu == 1 and mv == 1):
        raise Exception("Vector(s) are not row vectors.")
    if not (nu == nv):
        raise Exception("Vectors are not the same length.")
    """
    Method 1 (using the definition of the dot product):
    """
    # res = 0
    # for i in range(nu):
    #     res += u[0][i] * v[0][i]
    # return res
    """
    Method 2 (one-line code):
    """
    return np.sum(u * v)


def norm(v):
    """
    Computes the norm (2-norm) of a vector.

    :param v: A row vector (numpy.array).
    :return: The norm (2-norm) of v (number).
    """
    m, n = np.shape(v)
    if not (m == 1):
        raise Exception("Vector is not a row vector.")
    """
    Method 1 (Using the definition of teh norm/2-norm):
    """
    # res = 0
    # for i in range(n):
    #     res += v[0][i] ** 2
    # return np.sqrt(res)
    """
    Method 2 (one-line code):
    """
    return np.sqrt(np.sum(v * v))


def normalize(v):
    """
    Normalizes a vector.

    :param v: A row vector (numpy.array).
    :return: v normalized (numpy.array).
    """
    return v / norm(v)


def distance(u, v):
    """
    Computes the distance between 2 vectors.

    :param u: A row vector (numpy.array).
    :param v: A row vector (numpy.array).
    :return: The distance between u and v (number).
    """
    return norm(u - v)


def angle_between_vectors(u, v):
    """
    Computes the angle between two vectors.

    :param u: A row vector (numpy.array).
    :param v: A row vector (numpy.array).
    :return: The angle between u and v (number).
    """
    return np.arccos(dot_product(u, v) / (norm(u) * norm(v)))


def is_orthogonal(u, v):
    """
    Determines if both vectors are orthogonal to each other.

    :param u: A row vector (numpy.array).
    :param v: A row vector (numpy.array).
    :return: True if u and v are orthogonal to each other. False otherwise
             (boolean).
    """
    return dot_product(u, v) == 0


def proj(v, u):
    """
    Computes the projection of v onto u.

    :param u: A row vector (numpy.array).
    :param v: A row vector (numpy.array).
    :return: The projection of v onto u (numpy.array).
    """
    return (dot_product(u, v) / dot_product(u, u)) * u


def ref(M):
    """
    Reduces a matrix to its row echelon form.

    :param M: A matrix (numpy.array).
    :return: The row echelon form of M (numpy.array).
    """
    rows, cols = np.shape(M)
    pivot = 0
    for r in range(rows):
        if pivot >= cols:
            return M
        i = r
        while np.abs(M[i][pivot]) < TOL:
            i += 1
            if i == rows:
                i = r
                pivot += 1
                if pivot == cols:
                    return M
        if not (i == r):
            M[[i, r]] = M[[r, i]]
        for i in range(pivot, rows):
            if not (i == r):
                M[i] = M[i] - (M[i][pivot] / M[r][pivot]) * M[r]
        pivot += 1
    return M


def gaussian_elimination(M):
    """
    Gaussian Elimination: The process of applying row reduction onto an
    augmented matrix of a system of linear equations.

    :param M: An augmented matrix.
    :return: The row echelon form of M.
    """
    return ref(M)


def rank(M):
    """
    Returns the rank of a matrix.

    :param M: A matrix (numpy.array).
    :return: The rank of M (number).
    """
    m, n = np.shape(M)
    rnk = 0
    for row in ref(M):
        if np.round(row, d).tolist().count(0) != n:
            rnk += 1
    return rnk


def rref(M):
    """
    Reduces a matrix to its reduced row echelon form.

    :param M: A matrix (numpy.array).
    :return: The reduced row echelon form of M (numpy.array).
    """
    rows, cols = np.shape(M)
    pivot = 0
    for r in range(rows):
        if pivot >= cols:
            return M
        i = r
        while np.abs(M[i][pivot]) < TOL:
            i += 1
            if i == rows:
                i = r
                pivot += 1
                if pivot == cols:
                    return M
        if not (i == r):
            M[[i, r]] = M[[r, i]]
        M[r] = M[r] / M[r][pivot]
        for i in range(rows):
            if not (i == r):
                M[i] = M[i] - M[i][pivot] * M[r]
        pivot += 1
    return M


def gauss_jordan_elimination(M):
    """
    Gauss-Jordan Elimination: The process of applying row reduction onto an
    augmented matrix of a system of linear equations, reducing the matrix as
    much as possible.

    :param M: An augmented matrix.
    :return: The reduced row echelon form of M.
    """
    return rref(M)


def is_consistent(M):
    """
    Determines if the system is consistent.

    :param M: A matrix (numpy.array).
    :return: True if M is consistent. False otherwise (boolean).
    """
    M_rref = rref(M)
    A, b = separate_augmented_matrix(M_rref)
    m_a, n_a = np.shape(A)
    for i in range(m_a):
        if np.round(A[i], d).tolist().count(0) == n_a:
            if not (b[i] == 0):
                return False
    return True


def num_free_var(M):
    """
    Computes the number of free variables in a matrix.

    :param M: A matrix (numpy.array).
    :return: The number of free variables in M.
    """
    return (np.shape(M)[1] - 1) - rank(M)


def is_linear_combination_vector(vs, v):
    """
    Determine if a vector is a linear combination of the other vectors.

    :param vs: A list of row vectors (numpy.array).
    :param v: A row vector (numpy.array).
    :return: True is v is a linear combination of the set of vectors. False
             otherwise (boolean).
    """
    for vec in vs:
        if not (np.shape(vec)[1] == np.shape(v)[1]):
            raise Exception("Vectors are not the same length")
    r, c = 1 + len(vs), len(v[0])
    M_aug_t = np.zeros((r, c))
    for i in range(r):
        if i == r - 1:
            M_aug_t[i] = v
        else:
            M_aug_t[i] = vs[i]
    return is_consistent(rref(np.transpose(M_aug_t)))


def is_linearly_independent_vector(vs):
    """
    Determines if the set of vectors are linearly independent.

    :param vs: A list of row vectors (numpy.array).
    :return: True if the set of vectors are linearly independent. False
             otherwise (boolean).
    """
    if len(vs) > len(vs[0][0]):
        return False
    r, c = len(vs), len(vs[0][0])
    M_aug_t = np.zeros((r, c))
    for i in range(r):
        M_aug_t[i] = vs[i]
    if rank(np.transpose(M_aug_t)) < r:
        return False
    return True


def multiply(A, B):
    """
    Computes the product of two matrices.

    :param A: A matrix.
    :param B: A matrix.
    :return: The product of A and B.
    """
    return np.matmul(A, B)


def power(M, n):
    """
    Raises a matrix to a power.

    :param M: A matrix.
    :param n: The power to raise a matrix to.
    :return: M raised to n.
    """
    return np.linalg.matrix_power(M, n)


def is_equal(M, N):
    """
    Determines if both matrices are equal.

    :param M: A matrix.
    :param N: A matrix.
    :return: True if both M and N are equal to each other. False otherwise.
    """
    return np.array_equal(M, N)


def is_symmetric(M):
    """
    Determines if a matrix is symmetric. A matrix is symmetric if it is equal
    to its transpose.

    :param M: A matrix.
    :return: True if M is symmetric, False otherwise.
    """
    return is_equal(M, np.transpose(M))


def is_linear_combination_matrix(Ms, M):
    """
    Determines whether a matrix is a linear combination of a set of matrices.

    :param Ms: A list of matrices.
    :param M: A matrix.
    :return: True if M is a linear combination of the set of matrices. False
             otherwise.
    """
    m, n = np.shape(M)[0] * np.shape(M)[1], 1 + len(Ms)
    M_t = np.zeros((m, n))
    for i in range(m):
        if i == n - 1:
            M_t[i] = M.flatten()
        else:
            M_t[i] = Ms[i].flatten()
    return is_consistent(rref(np.transpose(M_t)))


def is_linearly_independent_matrix(Ms):
    """
    Determines whether a set of matrices are linearly independent.

    :param Ms: A list of matrices.
    :return: True if the set of matrices are linearly independent. False
             otherwise.
    """
    M = np.zeros(np.shape(Ms[0]))
    m, n = np.shape(M)[0] * np.shape(M)[1], 1 + len(Ms)
    M_t = np.zeros((m, n))
    for i in range(m):
        if i == n - 1:
            M_t[i] = M.flatten()
        else:
            M_t[i] = Ms[i].flatten()
    return rref(np.transpose(M_t))[-1].tolist().count(0) == n


def is_square(M):
    """
    Determines if a matrix is square.

    :param M: A matrix.
    :return: True if M is square. False otherwise.
    """
    return np.shape(M)[0] == np.shape(M)[1]


def det(M):
    """
    Computes the determinant of a square matrix of size n.

    :param M: A square matrix.
    :return: The determinant of M.
    """
    if not is_square(M):
        raise Exception("Matrix is not square, so the determinant does not "
                        "exist.")
    m, n = np.shape(M)
    if m == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    else:
        res = 0
        for i in range(m):
            S = M[0][i] * det(np.delete(np.delete(M, 0, 0), i, 1))
            if not (i % 2 == 0):
                res -= S
            else:
                res += S
        return res


def inverse_gj(M):
    """
    Computes the inverse of a matrix via Gauss-Jordan Elimination.

    :param M: A matrix.
    :return: The inverse of M if it exists.
    """
    m, n = np.shape(M)
    if not is_square(M):
        raise Exception("Matrix is not square. Therefore the inverse does not "
                        "exist.")
    I = np.zeros(np.shape(M))
    for i in range(min(m, n)):
        I[i][i] = 1
    M_sam = np.append(M, I, axis=1)
    M_gj = gauss_jordan_elimination(M_sam)
    M_identity, M_inverse = np.hsplit(M_gj, 2)
    if is_equal(M_identity, np.eye(m)):
        return M_inverse
    else:
        raise Exception("The inverse of this matrix does not exist.")


def LU(M):
    """
    Computes the LU factorization of a matrix.

    :param M: A matrix.
    :return: The LU factorization of M.
    """
    m, n = np.shape(M)
    L = np.zeros((min(m, n), min(m, n)))
    for i in range(min(m, n)):
        L[i][i] = 1
    p = 0
    for col in range(n):
        if col < m:
            if M[p][col] == 0:
                swap = False
                for row in range(p + 1, m):
                    if M[row][col] != 0:
                        M[[p, row]] = M[[row, p]]
                        swap = True
                        break
            for row in range(p + 1, m):
                if M[row][col] != 0:
                    c = M[row][col] / M[p][col]
                    M[row] = M[row] - c * M[p]
                    L[row][col] = c
            if (M[p][col] != 0) or swap:
                p += 1
    return L, M


def pLU(M):
    """
    Computes the p^TLU factorization of a matrix.

    :param M: A matrix.
    :return: The p^TLU factorization of M.
    """
    A = copy.deepcopy(M)
    m, n = np.shape(M)
    P = np.zeros((n, m))
    for i in range(min(m, n)):
        P[i][i] = 1
    p = 0
    for col in range(n):
        if col < m:
            if M[p][col] == 0:
                swap = False
                for row in range(p + 1, m):
                    if M[row][col] != 0:
                        M[[p, row]] = M[[row, p]]
                        P[[p, row]] = P[[row, p]]
                        swap = True
                        break
            for row in range(p + 1, m):
                if M[row][col] != 0:
                    c = M[row][col] / M[p][col]
                    M[row] = M[row] - c * M[p]
            if (M[p][col] != 0) or swap:
                p += 1
    PA = multiply(P, A)
    L, U = LU(PA)
    return np.transpose(P), L, U


# def is_row_space(A, b):
#     """
#     Determines if a vector is in the column space of a matrix.
#
#     :param A: A matrix.
#     :param b: A vector.
#     :return: True if b is in the column space of A. False otherwise.
#     """
#     return


def is_col_space(A, b):
    """
    Determines if a vector is in the column space of a matrix.

    :param A: A matrix.
    :param b: A row vector.
    :return: True if b is in the column space of A. False otherwise.
    """
    A_aug = np.append(A, np.transpose(b), axis=1)
    return is_consistent(rref(A_aug))


def basis(M):
    """
    Determines the basis for the row space, column space, and the null space
    of a matrix.

    :param M: A matrix.
    :return: The list that makes up the row space (R), column space (C), and
             the null space (N) of M.
    """
    return


def dim(S):
    return len(S)


def nullity(M):
    return np.shape(M)[1] - rank(M)


def eigen(M):
    return


def cramer(M):
    return


def inverse_adjoint(M):
    return
