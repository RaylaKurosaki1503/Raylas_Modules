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
        'all':lambda x: str(fractions.Fraction(x).limit_denominator())
    }
)
TOL = 1*10e-4


def dotP(u, v):
    """
    Computes the dot product of 2 vectors.

    :param u:
    :param v:
    :return:
    """
    mu, nu = np.shape(u)
    mv, nv = np.shape(v)
    if not (mu == 1 and mv == 1):
        raise Exception("Vector(s) are not row vectors.")
    if not (nu == nv):
        raise Exception("Vectors are not the same length.")
    res = 0
    for i in range(nu):
        res += u[0][i] * v[0][i]
    return res


def norm(v):
    """
    Computes the norm (2-norm) of a vector.

    :param v:
    :return:
    """
    m, n = np.shape(v)
    if not (m == 1):
        raise Exception("Vector is not a row vector.")
    res = 0
    for i in range(n):
        res += v[0][i] ** 2
    return np.sqrt(res)


def normalize(v):
    """
    Normalizes a vector.

    :param v:
    :return:
    """
    return v / norm(v)


def distance(u, v):
    """
    Computes the distance between 2 vectors.

    :param u:
    :param v:
    :return:
    """
    return norm(u - v)


def angle_between_vectors(u, v):
    """
    Computes the angle between two vectors.

    :param u:
    :param v:
    :return:
    """
    return np.arccos(dotP(u, v) / (norm(u) * norm(v)))


def is_orthogonal(u, v):
    """
    Determines if both vectors are orthogonal to each other.

    :param u:
    :param v:
    :return:
    """
    return dotP(u, v) == 0


def proj(v, u):
    """
    Computes the projection of v onto u.

    :param v:
    :param u:
    :return:
    """
    return (dotP(u, v) / dotP(u, u)) * u


def ref(M):
    """
    Performs row echelon form on a matrix.

    :param M:
    :return:
    """
    m, n = np.shape(M)
    p = 0
    for col in range(n):
        if col < m:
            # If the pivot is a 0
            if M[p][col] == 0:
                # Find a row to swap
                swap = False
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
    m, n = np.shape(M)
    r = 0
    A = ref(M)
    for row in A:
        if row.tolist().count(0) != n:
            r += 1
    return r


def rref(M):
    m, n = np.shape(M)
    p = 0
    for col in range(n):
        if col < m:
            # If the pivot is a 0
            if M[p][col] <= TOL:
                # Find a row to swap
                swap = False
                for row in range(p + 1, m):
                    if M[row][col] > TOL:
                        M[[p, row]] = M[[row, p]]
                        swap = True
                        break
            # Make all entries below the pivot a 0
            for row in range(0, m):
                print(M[p][col])
                if M[p][col] > TOL:
                    if row == p:
                        M[p] = M[p] / M[p][col]
                    else:
                        c = M[row][col] / M[p][col]
                        M[row] = M[row] - c * M[p]
            if (M[p][col] != 0) or swap:
                p += 1
    return M


if __name__ == '__main__':
    pass
