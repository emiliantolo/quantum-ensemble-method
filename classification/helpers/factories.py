from classification.classifiers.classical import CosineClassifier, SwapClassifier, DistanceClassifier, CosineIntClassifier, DistanceIntClassifier, SwapIntClassifier
from classification.classifiers.quantum import QuantumCosineClassifier, QuantumDistanceClassifier, QuantumSwapClassifier
from classification.classifiers.ens_weight_classical_cosine import EnsWeightClassicalCosineClassifier
from classification.classifiers.ens_weight_quantum_cosine import EnsWeightQuantumCosineClassifier
from classification.classifiers.ens_weight_classical_swap import EnsWeightClassicalSwapClassifier
from classification.classifiers.ens_weight_quantum_swap import EnsWeightQuantumSwapClassifier
from classification.classifiers.ens_weight_classical_distance import EnsWeightClassicalDistanceClassifier
from classification.classifiers.ens_weight_quantum_distance import EnsWeightQuantumDistanceClassifier

def gen_classifier(classifier_name, data, **cl_args):
    if classifier_name == 'cosine':
        classifier = CosineClassifier(data, **cl_args)
    elif classifier_name == 'distance':
        classifier = DistanceClassifier(data, **cl_args)
    elif classifier_name == 'swap':
        classifier = SwapClassifier(data, **cl_args)
    elif classifier_name == 'quantum_cosine':
        classifier = QuantumCosineClassifier(data, **cl_args)
    elif classifier_name == 'quantum_distance':
        classifier = QuantumDistanceClassifier(data, **cl_args)
    elif classifier_name == 'quantum_swap':
        classifier = QuantumSwapClassifier(data, **cl_args)
    elif classifier_name == 'ens_weight_classical_cosine':
        classifier = EnsWeightClassicalCosineClassifier(data, **cl_args)
    elif classifier_name == 'ens_weight_quantum_cosine':
        classifier = EnsWeightQuantumCosineClassifier(data, **cl_args)
    elif classifier_name == 'ens_weight_classical_swap':
        classifier = EnsWeightClassicalSwapClassifier(data, **cl_args)
    elif classifier_name == 'ens_weight_quantum_swap':
        classifier = EnsWeightQuantumSwapClassifier(data, **cl_args)
    elif classifier_name == 'ens_weight_classical_distance':
        classifier = EnsWeightClassicalDistanceClassifier(data, **cl_args)
    elif classifier_name == 'ens_weight_quantum_distance':
        classifier = EnsWeightQuantumDistanceClassifier(data, **cl_args)
    elif classifier_name == 'cosine_int':
        classifier = CosineIntClassifier(data, **cl_args)
    elif classifier_name == 'distance_int':
        classifier = DistanceIntClassifier(data, **cl_args)
    elif classifier_name == 'swap_int':
        classifier = SwapIntClassifier(data, **cl_args)
    else:
        raise ValueError('classifier_name not recognized')
    return classifier
