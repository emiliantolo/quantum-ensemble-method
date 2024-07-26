import numpy as np


def cosine_sim(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)


def euclidean_dist(a, b, norm=False):
    if norm:
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
    return np.linalg.norm(a - b)
