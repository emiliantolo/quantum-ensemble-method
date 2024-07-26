
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

class EnsWeightQuantumDistanceClassifier(BaseClassifier):

    def __init__(self, data, dim=2, overlap=0, method='lr', **cl_args):
        BaseClassifier.__init__(self, data)
        self.dim = dim
        self.overlap = overlap
        self.method = method
        self.cl_args = cl_args
        self.build()

    def build(self):

        ttt, sets = get_all(self.data, self.dim, self.overlap)
        train_data, test_data = ttt
        classifier = TrainClassifier(train_data, self.dim, self.overlap, **self.cl_args)

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

        self.test_classifier = TestClassifier(train_data, self.weights, self.dim, self.overlap, **self.cl_args)

    def classify(self, x):
        #return self.test_classifier.classify(x)
        res, conf = self.test_classifier.classify(x)
        z = self.scale * res + self.bias
        res = np.sign(1 / (1 + np.exp(-1 * z)) - 0.5)
        return res, conf

class TrainClassifier(QuantumClassifier):

    def __init__(self, data, dim=2, overlap=0, execution='local', shots=1024, noise_sim_config=None, seed=123):

        for d in data:
            d['x'] = normalize(d['x'])

        QuantumClassifier.__init__(self, data, execution, shots, noise_sim_config, seed)

        feat_qubits = math.ceil(math.log2(len(data[0]['x'])))
        idx_qubits = math.ceil(math.log2(len(data)))
        ctrl_qubits = min(max(idx_qubits, feat_qubits), dim)

        self.overlap = overlap
        self.dim = ctrl_qubits + 1

    def initialize(self, x):

        N = len(self.data)
        d = len(self.data[0]['x'])

        # compute circuit size
        ancillary_qubits = 1
        index_qubits = math.ceil(math.log2(N))
        features_qubits = math.ceil(math.log2(d))
        label_qubits = 1
        control_qubits = self.dim
        qubits_num = ancillary_qubits+index_qubits+features_qubits+label_qubits+control_qubits

        # Create a Quantum Circuit acting on the q register
        qr = QuantumRegister(qubits_num, 'q')
        cr = ClassicalRegister(3+self.dim-1, 'c')
        circuit = QuantumCircuit(qr, cr)

        # Ancillary control qubit
        ancillary_control_qubit = 0

        # Initialize jointly ancillary qubit, index register and features register
        init_qubits = 1 + index_qubits + features_qubits
        amplitudes = np.zeros(2 ** init_qubits)
        amplitude_base_value = 1.0 / math.sqrt(2 * N)

        # Training data amplitudes
        for instance_index, row in enumerate(self.data):
            for feature_indx, feature_amplitude in enumerate(row['x']):
                index = 1 + 2 * instance_index + \
                    (2 ** (index_qubits+1)) * feature_indx
                amplitudes[index] = amplitude_base_value * feature_amplitude

        # Unclassified instance amplitudes
        for i in range(0, N):
            for feature_indx, feature_amplitude in enumerate(x):
                index = 0 + 2 * i + (2 ** (index_qubits+1)) * feature_indx
                amplitudes[index] = amplitude_base_value * feature_amplitude

        # Set all ancillary_qubit+index_register+features_register amplitudes
        circuit.initialize(
            amplitudes, qr[ancillary_control_qubit: ancillary_control_qubit + init_qubits])

        # Set training data labels
        for instance_index, row in enumerate(self.data):
            label = row['y']
            if label == -1:
                bin_indx = ('{0:0' + str(index_qubits) +
                            'b}').format(instance_index)
                zero_qubits_indices = [
                    ancillary_control_qubit + len(bin_indx) - i
                    for i, letter in enumerate(bin_indx) if letter == '0'
                ]

                # select the right qubits from the index register
                for qubit_indx in zero_qubits_indices:
                    circuit.x(qr[qubit_indx])

                # add multi controlled CNOT gate
                multi_controlled_cnot = MCMT(XGate(), index_qubits, 1)
                circuit.compose(multi_controlled_cnot,
                                qr[ancillary_control_qubit + 1: ancillary_control_qubit + index_qubits + 1] + [
                                    qr[qubits_num - 1 - control_qubits]],
                                inplace=True)

                # bring the index register qubits back to the original state
                for qubit_indx in zero_qubits_indices:
                    circuit.x(qr[qubit_indx])

        circuit.barrier()

        # data selection

        self.meas = qubits_num - 1
        self.label_q = ancillary_qubits+index_qubits+features_qubits

        for idx in range(control_qubits - 1):
            ctrl_idx = qubits_num - control_qubits + idx
            circuit.h(qr[ctrl_idx])

        for idx in range(control_qubits - 1):
            ctrl_idx = qubits_num - control_qubits + idx
            off = min(control_qubits - 1, index_qubits // 2)
            if idx < (index_qubits // 2):
                circuit.cswap(qr[ctrl_idx], qr[ancillary_qubits+idx], qr[ancillary_qubits+idx+off])
            if idx < index_qubits:
                circuit.cnot(qr[ctrl_idx], qr[ancillary_qubits+idx])
            off = min(control_qubits - 1, features_qubits // 2)
            if idx < (features_qubits // 2):
                circuit.cswap(qr[ctrl_idx], qr[ancillary_qubits+index_qubits+idx], qr[ancillary_qubits+index_qubits+idx+off])
            if idx < features_qubits:
                circuit.cnot(qr[ctrl_idx], qr[ancillary_qubits+index_qubits+idx])

        for idx in range(control_qubits - 1):
            for idx1 in range(control_qubits - 1):
                if idx != idx1:
                    ctrl_idx = qubits_num - control_qubits + idx
                    if idx < index_qubits and idx1 < index_qubits:
                        circuit.cswap(qr[ctrl_idx], qr[ancillary_qubits+idx], qr[ancillary_qubits+idx1])
                    if idx < features_qubits and idx1 < features_qubits:
                        circuit.cswap(qr[ctrl_idx], qr[ancillary_qubits+index_qubits+idx], qr[ancillary_qubits+index_qubits+idx1])

        circuit.ccx(qr[ancillary_qubits], qr[ancillary_qubits+index_qubits], qr[self.meas])

        if self.execution != 'statevector':
            circuit.measure([self.meas], [2])
            for idx in range(control_qubits - 1 - self.overlap):
                ctrl_idx = qubits_num - control_qubits + idx
                circuit.measure([ctrl_idx], [2 + idx + 1])

        circuit.barrier()

        # Add hadamard gate
        circuit.h(qr[0])

        #circuit.draw(output='mpl', filename='out_distance_train.png')
        circuit.barrier()

        return circuit

    def classify(self, x):
        x = normalize(x)
        circuit = self.initialize(x)
        backend = self.sim
        p0s, p1s = [], []
        if self.execution == 'statevector':
            #statevector
            circuit.save_statevector()
            job = execute(circuit, backend, seed_simulator=self.seed, seed_transpiler=self.seed)
            result = job.result()
            output_statevector = result.get_statevector(circuit, decimals=10)
            counts = {}
            for idx2 in range(int(2 ** (self.dim-1-self.overlap))):
                idx2_str = ('{0:0' + str(self.dim-1-self.overlap) + 'b}').format(idx2)
                counts[idx2_str + '000'] = 0
                counts[idx2_str + '001'] = 0
            for i, amplitude in enumerate(output_statevector):
                if i < len(output_statevector) // 2:
                    if i % 2 == 0:
                        idx2 = i // (2**(self.label_q+1))
                        idx2 = idx2 % (2**(self.dim-1-self.overlap))
                        idx2_str = ('{0:0' + str(self.dim-1-self.overlap) + 'b}').format(idx2)
                        if ((i // (2 ** self.label_q)) % 2) == 0:
                            counts[idx2_str + '000'] += (np.abs(amplitude) ** 2)
                        else:
                            counts[idx2_str + '001'] += (np.abs(amplitude) ** 2)
        else:
            #local
            circuit.measure([0, self.label_q], [1, 0])
            job = execute(circuit, backend, shots=self.shots, seed_simulator=self.seed, seed_transpiler=self.seed)
            result = job.result()
            counts = result.get_counts(circuit)
        for idx2 in range(int(2 ** (self.dim-1-self.overlap))):
            idx2_str = ('{0:0' + str(self.dim-1-self.overlap) + 'b}').format(idx2)
            p0s.append(counts.get(idx2_str + '000', 0))
            p1s.append(counts.get(idx2_str + '001', 0))
        p0s = np.array(p0s)
        p1s = np.array(p1s)
        psum = p0s + p1s
        psum[psum == 0] = 1
        ps = psum / np.sum(psum)
        p = p0s / psum
        return p, ps

class TestClassifier(BaseClassifier):

    def __init__(self, data, weights, dim=2, overlap=0, execution='local', shots=1024, noise_sim_config=None, seed=123):

        for d in data:
            d['x'] = normalize(d['x'])

        QuantumClassifier.__init__(self, data, execution, shots, noise_sim_config, seed)

        feat_qubits = math.ceil(math.log2(len(data[0]['x'])))
        idx_qubits = math.ceil(math.log2(len(data)))
        ctrl_qubits = min(max(idx_qubits, feat_qubits), dim)

        self.overlap = overlap
        self.dim = ctrl_qubits + 1

        self.weights = normalize(np.power(np.abs(weights), 0.5))

    def initialize(self, x):

        N = len(self.data)
        d = len(self.data[0]['x'])

        # compute circuit size
        ancillary_qubits = 1
        index_qubits = math.ceil(math.log2(N))
        features_qubits = math.ceil(math.log2(d))
        label_qubits = 1
        control_qubits = self.dim
        qubits_num = ancillary_qubits+index_qubits+features_qubits+label_qubits+control_qubits

        # Create a Quantum Circuit acting on the q register
        qr = QuantumRegister(qubits_num, 'q')
        cr = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qr, cr)

        # Ancillary control qubit
        ancillary_control_qubit = 0

        # Initialize jointly ancillary qubit, index register and features register
        init_qubits = 1 + index_qubits + features_qubits
        amplitudes = np.zeros(2 ** init_qubits)
        amplitude_base_value = 1.0 / math.sqrt(2 * N)

        # Training data amplitudes
        for instance_index, row in enumerate(self.data):
            for feature_indx, feature_amplitude in enumerate(row['x']):
                index = 1 + 2 * instance_index + \
                    (2 ** (index_qubits+1)) * feature_indx
                amplitudes[index] = amplitude_base_value * feature_amplitude

        # Unclassified instance amplitudes
        for i in range(0, N):
            for feature_indx, feature_amplitude in enumerate(x):
                index = 0 + 2 * i + (2 ** (index_qubits+1)) * feature_indx
                amplitudes[index] = amplitude_base_value * feature_amplitude

        # Set all ancillary_qubit+index_register+features_register amplitudes
        circuit.initialize(
            amplitudes, qr[ancillary_control_qubit: ancillary_control_qubit + init_qubits])

        ctrl_amplitudes = []
        for i in range(2**self.overlap):
            ctrl_amplitudes.append(self.weights / math.sqrt(2**self.overlap))
        ctrl_amplitudes = np.concatenate(ctrl_amplitudes)

        # Set control_register amplitudes
        circuit.initialize(
            ctrl_amplitudes, qr[ancillary_qubits+index_qubits+features_qubits+label_qubits:ancillary_qubits+index_qubits+features_qubits+label_qubits+control_qubits-1])

        # Set training data labels
        for instance_index, row in enumerate(self.data):
            label = row['y']
            if label == -1:
                bin_indx = ('{0:0' + str(index_qubits) +
                            'b}').format(instance_index)
                zero_qubits_indices = [
                    ancillary_control_qubit + len(bin_indx) - i
                    for i, letter in enumerate(bin_indx) if letter == '0'
                ]

                # select the right qubits from the index register
                for qubit_indx in zero_qubits_indices:
                    circuit.x(qr[qubit_indx])

                # add multi controlled CNOT gate
                multi_controlled_cnot = MCMT(XGate(), index_qubits, 1)
                circuit.compose(multi_controlled_cnot,
                                qr[ancillary_control_qubit + 1: ancillary_control_qubit + index_qubits + 1] + [
                                    qr[qubits_num - 1 - control_qubits]],
                                inplace=True)

                # bring the index register qubits back to the original state
                for qubit_indx in zero_qubits_indices:
                    circuit.x(qr[qubit_indx])

        circuit.barrier()

        # data selection

        self.meas = qubits_num - 1
        self.label_q = ancillary_qubits+index_qubits+features_qubits

        for idx in range(control_qubits - 1):
            ctrl_idx = qubits_num - control_qubits + idx
            off = min(control_qubits - 1, index_qubits // 2)
            if idx < (index_qubits // 2):
                circuit.cswap(qr[ctrl_idx], qr[ancillary_qubits+idx], qr[ancillary_qubits+idx+off])
            if idx < index_qubits:
                circuit.cnot(qr[ctrl_idx], qr[ancillary_qubits+idx])
            off = min(control_qubits - 1, features_qubits // 2)
            if idx < (features_qubits // 2):
                circuit.cswap(qr[ctrl_idx], qr[ancillary_qubits+index_qubits+idx], qr[ancillary_qubits+index_qubits+idx+off])
            if idx < features_qubits:
                circuit.cnot(qr[ctrl_idx], qr[ancillary_qubits+index_qubits+idx])

        for idx in range(control_qubits - 1):
            for idx1 in range(control_qubits - 1):
                if idx != idx1:
                    ctrl_idx = qubits_num - control_qubits + idx
                    if idx < index_qubits and idx1 < index_qubits:
                        circuit.cswap(qr[ctrl_idx], qr[ancillary_qubits+idx], qr[ancillary_qubits+idx1])
                    if idx < features_qubits and idx1 < features_qubits:
                        circuit.cswap(qr[ctrl_idx], qr[ancillary_qubits+index_qubits+idx], qr[ancillary_qubits+index_qubits+idx1])

        circuit.ccx(qr[ancillary_qubits], qr[ancillary_qubits+index_qubits], qr[self.meas])

        if self.execution != 'statevector':
            circuit.measure([self.meas], [2])

        circuit.barrier()

        # Add hadamard gate
        circuit.h(qr[0])

        #circuit.draw(output='mpl', filename='out_distance_test.png')
        circuit.barrier()

        return circuit

    def classify(self, x):
        x = normalize(x)
        #print(x)
        circuit = self.initialize(x)
        backend = self.sim
        p0, p1 = 0, 0
        if self.execution == 'statevector':
            #statevector
            circuit.save_statevector()
            job = execute(circuit, backend, seed_simulator=self.seed, seed_transpiler=self.seed)
            result = job.result()
            output_statevector = result.get_statevector(circuit, decimals=10)
            for i, amplitude in enumerate(output_statevector):
                if i < len(output_statevector) // 2:
                    if i % 2 == 0:
                        if ((i // (2 ** self.label_q)) % 2) == 0:
                            p0 += (np.abs(amplitude) ** 2)
                        else:
                            p1 += (np.abs(amplitude) ** 2)
        else:
            #local
            circuit.measure([0, self.label_q], [1, 0])
            job = execute(circuit, backend, shots=self.shots, seed_simulator=self.seed, seed_transpiler=self.seed)
            result = job.result()
            counts = result.get_counts(circuit)
            p0 = counts.get('000', 0)
            p1 = counts.get('001', 0)
        ok = (p0 + p1) > 0
        if ok:
            return p0 / (p0 + p1), 1
        else:
            return 1, 0