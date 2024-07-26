import numpy as np
from classification.helpers.distances import cosine_sim, euclidean_dist
from classification.base import BaseClassifier
import math
from classification.helpers.data import normalize

class CosineClassifier(BaseClassifier):

    def __init__(self, data):
        BaseClassifier.__init__(self, data)

    def classify(self, x):
        sum = 0
        for d in self.data:
            sum += d['y'] * cosine_sim(x, d['x'])
        conf = np.abs(sum / len(self.data))
        return np.sign(sum), conf

class SwapClassifier(BaseClassifier):

    def __init__(self, data, d_dim=1):
        BaseClassifier.__init__(self, data)
        self.d_dim = d_dim

    def classify(self, x):
        sum = 0
        for d in self.data:
            sum += d['y'] * (cosine_sim(x, d['x'])**(2*self.d_dim))
        conf = np.abs(sum / len(self.data))
        return np.sign(sum), conf

class DistanceClassifier(BaseClassifier):

    def __init__(self, data, norm=True):
        BaseClassifier.__init__(self, data)
        self.norm = norm

    def classify(self, x):
        sum = 0
        for d in self.data:
            sum += d['y'] * (1 - 1 / 4 * pow(euclidean_dist(x,
                             d['x'], norm=self.norm), 2))
        conf = np.abs(sum / len(self.data))
        return np.sign(sum), conf

class CosineIntClassifier(BaseClassifier):

    def __init__(self, data):
        self.data = data.copy()
        for d in self.data:
            d['x'] = normalize(d['x'])

    def classify(self, xx):
        p0, p1 = 0, 0
        for i, d in enumerate(self.data):
            p0 +=  3 * np.dot(xx[i], xx[i]) + 3 * np.dot(d['x'], d['x']) + 2 / math.sqrt(2) * d['y'] * np.dot(xx[i], d['x'])
            p1 += np.dot(xx[i], xx[i]) + np.dot(d['x'], d['x']) - 2 / math.sqrt(2) * d['y'] * np.dot(xx[i], d['x'])
        conf = 1
        res = np.sign(1 - 4 * p1 / (p0 + p1))
        return res, conf

class DistanceIntClassifier(BaseClassifier):

    def __init__(self, data):
        self.data = data.copy()
        for d in self.data:
            d['x'] = normalize(d['x'])

    def classify(self, xx):
        p0, p1 = 0, 0
        for i, d in enumerate(self.data):
            dd = xx[i] + d['x']
            if d['y'] == -1:
                p1 += np.dot(dd, dd)
            else:
                p0 += np.dot(dd, dd)
        conf = 1
        res = np.sign(p0 / (p0 + p1) - 0.5)
        return res, conf

class SwapIntClassifier(BaseClassifier):

    def __init__(self, data, d_dim=1):
        self.data = data.copy()
        for d in self.data:
            d['x'] = normalize(d['x'])
        self.d_dim = d_dim

    def classify(self, xx):
        p0, p1 = 0, 0
        for i, d in enumerate(self.data):
            dd = xx[i] + d['x']
            if d['y'] == -1:
                p1 += np.dot(xx[i], d['x'])**(2*self.d_dim)
            else:
                p0 += np.dot(xx[i], d['x'])**(2*self.d_dim)
        conf = 1
        res = np.sign(p0 / (p0 + p1) - 0.5)
        return res, conf
