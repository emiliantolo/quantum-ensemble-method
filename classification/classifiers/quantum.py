import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit.library import XGate, MCMT
from qiskit.providers.aer import Aer
from qiskit_aer import AerSimulator
from classification.base import BaseClassifier
from classification.helpers.data import normalize
from simulation.backend import SimulatorBackend

class QuantumClassifier(BaseClassifier):
    def __init__(self, data, execution='local', shots=1024, noise_sim_config=None, seed=123):
        BaseClassifier.__init__(self, data)
        self.execution = execution
        self.shots = shots
        self.seed = seed
        if self.execution == 'local':
            self.sim = Aer.get_backend('aer_simulator')
        elif self.execution == 'local+noise':
            self.sim = SimulatorBackend().gen_simulator_from_config(noise_sim_config)
        elif self.execution == 'local+fake':
            self.sim = SimulatorBackend().gen_simulator_fake(noise_sim_config)
        elif self.execution == 'statevector':
            self.sim = Aer.get_backend('aer_simulator')
        else:
            raise ValueError('execution not recognized')

class QuantumCosineClassifier(QuantumClassifier):

    def __init__(self, data, execution='local', shots=1024, noise_sim_config=None, seed=123):

        for d in data:
            d['x'] = normalize(d['x'])

        QuantumClassifier.__init__(self, data, execution, shots, noise_sim_config, seed)

    def initialize(self, x):

        N = len(self.data)
        d = len(self.data[0]['x'])

        # compute circuit size
        swap_circuit_qubits = 3
        index_qubits = math.ceil(math.log2(N))
        features_qubits = math.ceil(math.log2(d))
        label_qubits = 1
        qubits_num = swap_circuit_qubits+index_qubits+features_qubits+label_qubits

        # Create a Quantum Circuit acting on the q register
        qr = QuantumRegister(qubits_num, 'q')
        cr = ClassicalRegister(1, 'c')
        circuit = QuantumCircuit(qr, cr)

        # Hadamard gates for swap test
        circuit.h(qr[1])

        # Ancillary control qubit
        ancillary_control_qubit = 2

        # Initialize jointly third swap test qubit, index register and features register
        init_qubits = 1 + index_qubits + features_qubits
        amplitudes = np.zeros(2 ** init_qubits)
        amplitude_base_value = 1.0 / math.sqrt(2 * N)

        # Training data amplitudes
        for instance_index, row in enumerate(self.data):
            for feature_indx, feature_amplitude in enumerate(row['x']):
                index = 2 * instance_index + \
                    (2 ** (index_qubits+1)) * feature_indx
                amplitudes[index] = amplitude_base_value * feature_amplitude

        # Unclassified instance amplitudes
        for i in range(0, N):
            for feature_indx, feature_amplitude in enumerate(x):
                index = 1 + 2 * i + (2 ** (index_qubits+1)) * feature_indx
                amplitudes[index] = amplitude_base_value * feature_amplitude

        # Set all ancillary_qubit+index_register+features_register amplitudes
        circuit.initialize(
            amplitudes, qr[ancillary_control_qubit: ancillary_control_qubit + init_qubits])

        # Set training data labels
        circuit.x(qr[ancillary_control_qubit])
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
                multi_controlled_cnot = MCMT(XGate(), index_qubits + 1, 1)
                circuit.compose(multi_controlled_cnot,
                                qr[ancillary_control_qubit: ancillary_control_qubit + index_qubits + 1] + [
                                    qr[qubits_num - 1]],
                                inplace=True)

                # bring the index register qubits back to the original state
                for qubit_indx in zero_qubits_indices:
                    circuit.x(qr[qubit_indx])

        circuit.x(qr[ancillary_control_qubit])

        # Set unclassified instance labels
        circuit.cx(qr[ancillary_control_qubit], qr[qubits_num - 1])
        circuit.ch(qr[ancillary_control_qubit], qr[qubits_num - 1])

        circuit.barrier()

        # Add swap test gates
        circuit.h(qr[0])
        circuit.cswap(qr[0], qr[1], qr[2])
        circuit.h(qr[0])

        # circuit.draw(output='mpl', filename='out.png')
        circuit.barrier()

        return circuit

    def classify(self, x):
        x = normalize(x)
        circuit = self.initialize(x)
        if self.execution == 'statevector':
            #statevector
            circuit.save_statevector()
            backend = Aer.get_backend('aer_simulator')
            job = execute(circuit, backend, seed_simulator=self.seed, seed_transpiler=self.seed)
            result = job.result()
            output_statevector = result.get_statevector(circuit, decimals=10)
            p0, p1 = 0, 0
            for i, amplitude in enumerate(output_statevector):
                if i % 2 == 0:
                    p0 += (np.abs(amplitude) ** 2)
                else:
                    p1 += (np.abs(amplitude) ** 2)
        else:
            #local
            circuit.measure([0], [0])
            #circuit.draw(output='mpl', filename='cosine.png')
            backend = self.sim
            job = execute(circuit, backend, shots=self.shots, seed_simulator=self.seed, seed_transpiler=self.seed)
            result = job.result()
            counts = result.get_counts(circuit)
            c0 = counts.get('0', 0)
            c1 = counts.get('1', 0)
            p1 = c1 / (c0 + c1)            
        return np.sign(1 - 4 * p1), np.abs(1 - 4 * p1)


class QuantumDistanceClassifier(QuantumClassifier):

    def __init__(self, data, execution='local', shots=1024, noise_sim_config=None, seed=123):

        for d in data:
            d['x'] = normalize(d['x'])

        QuantumClassifier.__init__(self, data, execution, shots, noise_sim_config, seed)

    def initialize(self, x):

        N = len(self.data)
        d = len(self.data[0]['x'])

        # compute circuit size
        ancillary_qubits = 1
        index_qubits = math.ceil(math.log2(N))
        features_qubits = math.ceil(math.log2(d))
        label_qubits = 1
        qubits_num = ancillary_qubits+index_qubits+features_qubits+label_qubits

        # Create a Quantum Circuit acting on the q register
        qr = QuantumRegister(qubits_num, 'q')
        cr = ClassicalRegister(2, 'c')
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
                                    qr[qubits_num - 1]],
                                inplace=True)

                # bring the index register qubits back to the original state
                for qubit_indx in zero_qubits_indices:
                    circuit.x(qr[qubit_indx])

        circuit.barrier()

        # Add hadamard gate
        circuit.h(qr[0])

        # circuit.draw(output='mpl', filename='out.png')
        circuit.barrier()

        return circuit

    def classify(self, x):
        x = normalize(x)
        circuit = self.initialize(x)
        if self.execution == 'statevector':
            #statevector
            circuit.save_statevector()
            backend = Aer.get_backend('aer_simulator')
            job = execute(circuit, backend, seed_simulator=self.seed, seed_transpiler=self.seed)
            result = job.result()
            output_statevector = result.get_statevector(
                circuit, decimals=10)
            p00, p01 = 0, 0
            for i, amplitude in enumerate(output_statevector):
                if i % 2 == 0:
                    if i < (len(output_statevector) / 2):
                        p00 += (np.abs(amplitude) ** 2)
                    else:
                        p01 += (np.abs(amplitude) ** 2)
        else:
            #local
            ok = False
            it = 0
            timeout = 5
            while not ok:
                circuit.measure([0, -1], [1, 0])
                #circuit.draw(output='mpl', filename='distance.png')
                backend = self.sim
                # backend.set_options(device='GPU')
                # increase shots for conditional measurement?
                job = execute(circuit, backend, shots=self.shots, seed_simulator=self.seed, seed_transpiler=self.seed)
                result = job.result()
                counts = result.get_counts(circuit)
                p00 = counts.get('00', 0)
                p01 = counts.get('01', 0)
                ok = (p00 + p01) > 0
                if not ok:
                    if it < (timeout - 1):
                        it += 1
                        print('Got 0 zero-measurements in conditional measurement, it: {}/{}, retrying...'.format(it, timeout))
                        circuit = self.initialize(x) # reinit
                    else:
                        it += 1
                        print('Got 0 zero-measurements in conditional measurement, it: {}/{}, retry timeout, returned label "0"'.format(it, timeout))
                        return 0, 0 # return 0
        return np.sign(p00 / (p00 + p01) - 0.5), np.abs(2 * (p00 / (p00 + p01) - 0.5))

class QuantumSwapClassifier(QuantumClassifier):

    def __init__(self, data, execution='local', shots=1024, noise_sim_config=None, seed=123):

        for d in data:
            d['x'] = normalize(d['x'])

        QuantumClassifier.__init__(self, data, execution, shots, noise_sim_config, seed)

    def initialize(self, x):

        N = len(self.data)
        d = len(self.data[0]['x'])

        # compute circuit size
        swap_circuit_qubits = 1
        index_qubits = math.ceil(math.log2(N))
        features_qubits = math.ceil(math.log2(d))
        label_qubits = 1
        qubits_num = swap_circuit_qubits+index_qubits+2*features_qubits+label_qubits

        # Create a Quantum Circuit acting on the q register
        qr = QuantumRegister(qubits_num, 'q')
        cr = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qr, cr)

        # Initialize index register and training features register
        init_qubits = index_qubits + features_qubits
        amplitudes = np.zeros(2 ** init_qubits)
        amplitude_base_value = 1.0 / math.sqrt(N)

        # Training data amplitudes
        for instance_index, row in enumerate(self.data):
            for feature_indx, feature_amplitude in enumerate(row['x']):
                index = instance_index + (2 ** index_qubits) * feature_indx
                amplitudes[index] = amplitude_base_value * feature_amplitude

        # Set all index_register+features_register amplitudes
        circuit.initialize(
            amplitudes, qr[swap_circuit_qubits: swap_circuit_qubits + init_qubits])

        # Initialize test features register
        init_qubits = features_qubits
        amplitudes = np.zeros(2 ** init_qubits)
        amplitude_base_value = 1.0

        # Unclassified instance amplitudes
        for feature_indx, feature_amplitude in enumerate(x):
            index = feature_indx
            amplitudes[index] = amplitude_base_value * feature_amplitude

        # Set features_register amplitudes
        circuit.initialize(amplitudes, qr[swap_circuit_qubits+index_qubits +
                           features_qubits: swap_circuit_qubits+index_qubits+features_qubits + init_qubits])

        # Set training data labels
        for instance_index, row in enumerate(self.data):
            label = row['y']
            if label == -1:
                bin_indx = ('{0:0' + str(index_qubits) +
                            'b}').format(instance_index)
                zero_qubits_indices = [
                    swap_circuit_qubits + len(bin_indx) - 1 - i
                    for i, letter in enumerate(bin_indx) if letter == '0'
                ]

                # select the right qubits from the index register
                for qubit_indx in zero_qubits_indices:
                    circuit.x(qr[qubit_indx])

                # add multi controlled CNOT gate
                multi_controlled_cnot = MCMT(XGate(), index_qubits, 1)
                circuit.compose(multi_controlled_cnot,
                                qr[swap_circuit_qubits: swap_circuit_qubits + index_qubits] + [
                                    qr[qubits_num - 1]],
                                inplace=True)

                # bring the index register qubits back to the original state
                for qubit_indx in zero_qubits_indices:
                    circuit.x(qr[qubit_indx])

        circuit.barrier()

        # Gates for swap test
        circuit.h(qr[0])
        for i in range(features_qubits):
            idx1 = i + 1 + index_qubits
            idx2 = idx1 + features_qubits
            circuit.cswap(qr[0], qr[idx1], qr[idx2])
        circuit.h(qr[0])

        #circuit.draw(output='mpl', filename='out.png')
        circuit.barrier()

        return circuit

    def classify(self, x):
        x = normalize(x)
        circuit = self.initialize(x)
        backend = self.sim
        c00, c10 = 0, 0
        c01, c11 = 0, 0
        if self.execution == 'statevector':
            #statevector
            circuit.save_statevector()
            job = execute(circuit, backend, seed_simulator=self.seed, seed_transpiler=self.seed)
            result = job.result()
            output_statevector = result.get_statevector(circuit, decimals=10)
            for i, amplitude in enumerate(output_statevector):
                if i % 2 == 0:
                    if i < (len(output_statevector) // 2):
                        c00 += (np.abs(amplitude) ** 2)
                    else:
                        c10 += (np.abs(amplitude) ** 2)
                else:
                    if i < (len(output_statevector) // 2):
                        c01 += (np.abs(amplitude) ** 2)
                    else:
                        c11 += (np.abs(amplitude) ** 2)
        else:
            #local
            circuit.measure([0, -1], [0, 1])
            #circuit.draw(output='mpl', filename='outswap.png')
            #exit()
            job = execute(circuit, backend, shots=self.shots, seed_simulator=self.seed, seed_transpiler=self.seed)
            result = job.result()
            counts = result.get_counts(circuit)
            c00 = counts.get('00', 0)
            c10 = counts.get('10', 0)
            c01 = counts.get('01', 0)
            c11 = counts.get('11', 0)
        csum = c00 + c01 + c10 + c11
        res = (c00 - c01 - c10 + c11)
        return np.sign(res), np.abs(res / csum)
