
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit.library import XGate, MCMT
from qiskit.providers.aer import Aer
from qiskit_aer import AerSimulator
from classification.base import BaseClassifier
from classification.helpers.data import normalize
from simulation.backend import SimulatorBackend
from classification.classifiers.quantum import QuantumClassifier
from classification.classifiers.ens_weight_get_sets import *

class EnsWeightClassicalDistanceClassifier(BaseClassifier):

    def __init__(self, data, dim=2, overlap=0, method='lr', **cl_args):
        BaseClassifier.__init__(self, data)
        self.dim = dim
        self.overlap = overlap
        self.method = method
        self.cl_args = cl_args
        self.build()

    def build(self):

        ttt, sets = get_all(self.data, self.dim, self.overlap, 1)
        train_data, test_data = ttt
        classifier = TrainClassifier(train_data, sets, self.dim, self.overlap, **self.cl_args)

        all_res = []
        all_ps = []
        all_label = []
        for d in test_data:
            res, ps = classifier.classify(d['x'])
            label = d['y']
            all_res.append(res)
            all_ps.append(ps)
            all_label.append(label)
        all_res = np.array(all_res)
        all_ps = np.array(all_ps)
        all_label = np.array(all_label)

        self.weights, self.bias, self.scale = optimize(all_res, all_ps, all_label, method=self.method)

        self.test_classifier = TestClassifier(train_data, sets, self.weights, self.bias, self.dim, self.overlap, **self.cl_args)

    def classify(self, x):
        #return self.test_classifier.classify(x)
        res, conf = self.test_classifier.classify(x)
        z = self.scale * res + self.bias
        res = np.sign(1 / (1 + np.exp(-1 * z)) - 0.5)
        return res, conf

class TrainClassifier(BaseClassifier):

    def __init__(self, data, sets, dim, overlap):
        BaseClassifier.__init__(self, data)
        self.sets = sets

        self.dim = dim
        self.overlap = overlap

    def classify(self, x):

        ss = []
        ps = []
        for s in self.sets:
            idx = s['idx']
            feat = s['feat']
            p0, p1 = 0, 0
            for i in range(len(idx)):
                if self.data[idx[i]]['y'] == 1:
                    p0 += (self.data[idx[i]]['x'][feat[i]] + x[feat[i]]) ** 2
                else:
                    p1 += (self.data[idx[i]]['x'][feat[i]] + x[feat[i]]) ** 2

            if (p0 + p1) == 0:
                ss.append(0)
            else:
                ss.append(p0 / (p0 + p1))

            ps.append(p0 + p1)

        ps = np.array(ps)
        if np.sum(ps) != 0:
            ps = ps / np.sum(ps)      

        return ss, ps

class TestClassifier(BaseClassifier):

    def __init__(self, data, sets, weights, bias, dim, overlap):
        BaseClassifier.__init__(self, data)
        self.sets = sets
        self.weights = weights
        self.bias = bias

        self.dim = dim
        self.overlap = overlap

    def classify(self, x):
        x = normalize(x)
        ss0, ss1 = 0, 0
        pss = []
        ps = []
        for j, s in enumerate(self.sets):
            p0, p1 = 0, 0
            idx = s['idx']
            feat = s['feat']
            n = len(set(idx))
            for i in range(len(idx)):
                if self.data[idx[i]]['y'] == 1:
                    p0 += self.weights[j] / 1 * (self.data[idx[i]]['x'][feat[i]] + x[feat[i]]) ** 2 / n
                else:
                    p1 += self.weights[j] / 1 * (self.data[idx[i]]['x'][feat[i]] + x[feat[i]]) ** 2 / n
            ss0 += p0
            ss1 += p1

        res = ss0 / (ss0 + ss1)
        conf = 1
        return res, conf