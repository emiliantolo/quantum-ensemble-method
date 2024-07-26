
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

class EnsWeightClassicalCosineClassifier(BaseClassifier):

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
        x = normalize(x)

        ss = []
        ps = []
        for s in self.sets:
            p0b = np.zeros(2**self.overlap)
            p1b = np.zeros(2**self.overlap)
            for buck in range(2**self.overlap):
                buckets = np.array(s['bucket'])
                indexes = np.where(buckets == buck)
                idx = np.array(s['idx'])[indexes]
                feat = np.array(s['feat'])[indexes]
                ida = np.array(s['ida'])[indexes]
                n = max(ida)+1
                nxj = [0] * n
                nxi = [0] * n
                pin = [0] * n
                li = [0] * n
                for i in range(len(idx)):
                        nxj[ida[i]] += x[feat[i]]**2
                        nxi[ida[i]] += self.data[idx[i]]['x'][feat[i]]**2
                        pin[ida[i]] += (self.data[idx[i]]['x'][feat[i]] * x[feat[i]])
                        li[ida[i]] = self.data[idx[i]]['y']

                for i in list(set(ida)):
                    p0b[buck] += 3*nxj[i] + 3*nxi[i] + 2 / math.sqrt(2) * li[i] * pin[i]
                    p1b[buck] += (nxj[i] + nxi[i] - 2 / math.sqrt(2) * li[i] * pin[i])

            p0 = np.sum(p0b)
            p1 = np.sum(p1b)
            if (p0 + p1) == 0:
                ss.append(0)
            else:
                ss.append(p1 / (p0 + p1))

            ps.append(p0 + p1)

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
        p0s, p1s = 0, 0
        pss = []
        ps = []
        for j, s in enumerate(self.sets):
            p0b = np.zeros(2**self.overlap)
            p1b = np.zeros(2**self.overlap)
            for buck in range(2**self.overlap):
                buckets = np.array(s['bucket'])
                indexes = np.where(buckets == buck)
                idx = np.array(s['idx'])[indexes]
                feat = np.array(s['feat'])[indexes]
                ida = np.array(s['ida'])[indexes]
                n = max(ida)+1
                nxj = [0] * n
                nxi = [0] * n
                pin = [0] * n
                li = [0] * n
                for i in range(len(idx)):
                        nxj[ida[i]] += x[feat[i]]**2
                        nxi[ida[i]] += self.data[idx[i]]['x'][feat[i]]**2
                        pin[ida[i]] += (self.data[idx[i]]['x'][feat[i]] * x[feat[i]])
                        li[ida[i]] = self.data[idx[i]]['y']

                for i in list(set(ida)):
                    p0b[buck] += 3*nxj[i] + 3*nxi[i] + 2 / math.sqrt(2) * li[i] * pin[i]
                    p1b[buck] += nxj[i] + nxi[i] - 2 / math.sqrt(2) * li[i] * pin[i]

            p0 = np.sum(p0b)
            p1 = np.sum(p1b)

            p0s += self.weights[j] * p0
            p1s += self.weights[j] * p1
     
        res = p1s / (p0s + p1s)
        conf = 1
        return res, conf