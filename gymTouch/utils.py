import numpy as np

EPS = 1e-10


def mulRotT(vector, rot_matrix):
    return np.transpose(rot_matrix).dot(vector)


def mulRot(vector, rot_matrix):
    return rot_matrix.dot(vector)


def weighted_sum_vectors(vector1, vector2, weight1, weight2):
    return (vector1 * weight1 + vector2 * weight2) / (weight1 + weight2)


