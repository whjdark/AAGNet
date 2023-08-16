"""
Vector linear algebra operations for numba accelerated code..
"""

import numba as nb
import math
import numpy as np


@nb.njit(fastmath=True)
def add(vec1, vec2):
    result = np.zeros(3)
    result[0] = vec1[0] + vec2[0]
    result[1] = vec1[1] + vec2[1]
    result[2] = vec1[2] + vec2[2]

    return result


@nb.njit(fastmath=True)
def sub(vec1, vec2):
    result = np.zeros(3)
    result[0] = vec2[0] - vec1[0]
    result[1] = vec2[1] - vec1[1]
    result[2] = vec2[2] - vec1[2]

    return result


@nb.njit(fastmath=True)
def mul(a, vec):
    """ Calculate the product of a scalar and a 3d vector and store the result in the second parameter."""
    result = np.zeros(3)
    result[0] = a * vec[0]
    result[1] = a * vec[1]
    result[2] = a * vec[2]

    return result


@nb.njit(fastmath=True)
def div(a, vec):
    """ Divide a 3d vector by a scalar and store the result in the third parameter. """
    result = np.zeros(3)
    result[0] = vec[0] / a
    result[1] = vec[1] / a
    result[2] = vec[2] / a

    return result


@nb.njit(fastmath=True)
def sum(vec):
    result = 0
    for i in range(vec.shape[0]):
        result += vec[i]

    return sum


@nb.njit(fastmath=True)
def cross(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    result = np.zeros(3)
    a1, a2, a3 = nb.double(vec1[0]), nb.double(vec1[1]), nb.double(vec1[2])
    b1, b2, b3 = nb.double(vec2[0]), nb.double(vec2[1]), nb.double(vec2[2])
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1

    return result


@nb.njit(fastmath=True)
def dot(vec1, vec2):
    """ Calculate the dot product of two 3d vectors. """
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


@nb.njit(fastmath=True)
def norm(vec):
    """ Calculate the norm of a 3d vector. """
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


@nb.njit(fastmath=True)
def calc_l2_norm(v):
    s = 0
    for i in range(v.shape[0]):
        s += v[i] ** 2
    return math.sqrt(s)