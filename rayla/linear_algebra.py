"""
Author: Rayla Kurosaki
File: linear_algebra.py
Description: This file contains functions to solve vector/matrix based problems
             and contains algorithms for different
"""
import numpy as np
import fractions

np.set_printoptions(
    formatter={
        'all': lambda x: str(fractions.Fraction(x).limit_denominator())
    }
)
TOL = 1 * 10e-4


def dotP(u, v):
    """
    Computes dot product of 2 row vectors.

    :param u: A row vector
    :param v: A row vector
    :return: The dot product of 2 row vectors.
    """
    # Get the dimensions of both vectors
    mu, nu = np.shape(u)
    mv, nv = np.shape(v)
    # Check if both vectors are row vectors
    if not (mu == 1 and mv == 1):
        raise Exception("Vector(s) are not row vectors.")
    # Check if both vectors have the same dimensions
    if not (nu == nv):
        raise Exception("Vectors are not the same length.")
    # Perform the dot product of two vectors
    # res = u_1 v_1 + u_2 v_3 + u_3 v_3 + ... + u_n v_n
    res = 0
    for i in range(nu):
        res += u[0][i] * v[0][i]
    return res


def norm(v):
    """
    Computes the norm (2-norm) of a vector.

    :param v: A row vector
    :return: The norm (2-norm) of a vector.
    """
    # Get the dimensions of the vector
    m, n = np.shape(v)
    # Check if the vector is a row vector
    if not (m == 1):
        raise Exception("Vector is not a row vector.")
    # Perform the norm of a vector
    # res = sqrt(v_1 v_1 + v_2 v_2 + ... + v_n v_n)
    res = 0
    for i in range(n):
        res += v[0][i] ** 2
    return np.sqrt(res)


def normalize(v):
    """
    Normalizes a vector.

    :param v: A row vector
    :return: The vector normalized.
    """
    return v / norm(v)


def distance(u, v):
    """
    Computes the distance between 2 vectors.

    :param u: A row vector
    :param v: A row vector
    :return: The distance between 2 vectors
    """
    return norm(u - v)


def angle_between_vectors(u, v):
    """
    Computes the angle between two vectors.

    :param u: A row vector
    :param v: A row vector
    :return: The angle between two vectors.
    """
    return np.arccos(dotP(u, v) / (norm(u) * norm(v)))


def is_orthogonal(u, v):
    """
    Determines if both vectors are orthogonal to each other.

    :param u: A row vector
    :param v: A row vector
    :return: True if both vectors are orthogonal to each other. False
             otherwise.
    """
    return dotP(u, v) == 0


def proj(v, u):
    """
    Computes the projection of v onto u.

    :param u: A row vector
    :param v: A row vector
    :return: The projection of v onto u.
    """
    return (dotP(u, v) / dotP(u, u)) * u


def ref(M):
    """
    Performs row echelon form on a matrix.

    :param M: A matrix
    :return: The row echelon form of M.
    """
    # Get the dimensions of the matrix
    m, n = np.shape(M)
    # Set the pivot counter
    p = 0
    # Iterate through each column
    for col in range(n):
        #
        if col < m:
            # If the pivot is a 0, find a row to swap
            if M[p][col] == 0:
                swap = False
                # Iterate through each row below the pivoting row
                for row in range(p + 1, m):
                    if M[row][col] != 0:
                        M[[p, row]] = M[[row, p]]
                        swap = True
                        break
            # Make all entries below the pivot a 0
            for row in range(p + 1, m):
                if M[row][col] != 0:
                    c = M[row][col] / M[p][col]
                    M[row] = M[row] - c * M[p]
            if (M[p][col] != 0) or swap:
                p += 1
    return M


def rank(M):
    """
    Returns the rank of a matrix.

    :param M:
    :return:
    """
    # Get the dimensions of the matrix
    m, n = np.shape(M)
    # Records the rank of a matrix
    rnk = 0
    # iterate through each row of the row echelon form of the matrix
    for row in ref(M):
        # Increment the counter by 1 if the row is a non-zero row
        if row.tolist().count(0) != n:
            rnk += 1
    # Return the rank of a matrix.
    return rnk


def rref(M):
    """

    :param M:
    :return:
    """
    m, n = np.shape(M)
    p = 0
    for col in range(n):
        if col < m:
            # If the pivot is a 0
            if np.abs(M[p][col]) <= TOL:
                # Find a row to swap
                swap = False
                for row in range(p + 1, m):
                    if M[row][col] > TOL:
                        M[[p, row]] = M[[row, p]]
                        swap = True
                        break
            # Make all entries below the pivot a 0
            for row in range(0, m):
                if np.abs(M[p][col]) > TOL:
                    if row == p:
                        if M[p][col] != 1:
                            M[p] = M[p] / M[p][col]
                    else:
                        if M[row][col] != 0:
                            c = M[row][col] / M[p][col]
                            M[row] = M[row] - c * M[p]
            if (M[p][col] != 0) or swap:
                p += 1
    return M


def is_linear_combination(vs, v):
    # Check if all vectors are of the same length
    for vec in vs:
        if not (np.shape(vec)[1] == np.shape(v)[1]):
            raise Exception("Vectors are not the same length")
    # Create an augmented matrix
    r, c = 1 + len(vs), len(v[0])
    M = np.zeros((r, c))
    for i in range(r):
        if i == r - 1:
            M[i] = v
        else:
            M[i] = vs[i]
    # The matrix created needs to be transposed. Then find the RREF of the
    # matrix. Then determine if the last row is a zero-row
    return rref(np.transpose(M))[-1].tolist().count(0) == len(v[0])


def is_linearly_independent(vs):
    return


if __name__ == '__main__':
    v = np.array([
        [2,3,4]
    ])
    v1 = np.array([
        [1, 0, 3]
    ])
    v2 = np.array([
        [-1, 1, -3]
    ])
    print(is_linear_combination([v1, v2], v))

    pass
