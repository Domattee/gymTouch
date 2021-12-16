import numpy as np
from matplotlib import pyplot as plt

EPS = 1e-10


def mulRotT(vector, rot_matrix):
    return np.transpose(rot_matrix).dot(vector)


def mulRot(vector, rot_matrix):
    return rot_matrix.dot(vector)


def weighted_sum_vectors(vector1, vector2, weight1, weight2):
    return (vector1 * weight1 + vector2 * weight2) / (weight1 + weight2)


def plot_points(points, limit: float = 1.0, title=""):
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color="k", s=20)
    ax.set_title(title)
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()
