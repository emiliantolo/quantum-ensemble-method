import numpy as np
import pandas as pd
import os
import sys
import time
from classification.helpers.data import get_train_test
from helpers import *

np.random.seed(123)

# args
data_s, data_e, fold_s, fold_e, classifier_s, classifier_e = parse_args()

# folds
n_rep = 30
folds_to_execute = list(range(n_rep))

# data
dir = os.listdir('data/dataset')
dir.remove('02_transfusion.csv')

dir.sort()
test_size = 0.2
stds = ['none', 'std', 'minmax']

if not os.path.exists('results'):
    os.makedirs('results')

shots = 8192
# classifiers

classifiers = [
    {'ensemble': 'single', 'classifier': 'swap', 'args': {}},
    {'ensemble': 'single', 'classifier': 'distance', 'args': {}},
    {'ensemble': 'single', 'classifier': 'cosine', 'args': {}},

    #{'ensemble': 'single', 'classifier': 'quantum_swap', 'args': {'execution': 'statevector', 'shots': shots}},
    #{'ensemble': 'single', 'classifier': 'quantum_distance', 'args': {'execution': 'statevector', 'shots': shots}},
    #{'ensemble': 'single', 'classifier': 'quantum_cosine', 'args': {'execution': 'statevector', 'shots': shots}},

    {'ensemble': 'single', 'classifier': 'swap_int', 'args': {}},
    {'ensemble': 'single', 'classifier': 'distance_int', 'args': {}},
    {'ensemble': 'single', 'classifier': 'cosine_int', 'args': {}},

    {'ensemble': 'single', 'classifier': 'ens_weight_classical_swap', 'args': {'dim': 1, 'method': 'lr'}},
    {'ensemble': 'single', 'classifier': 'ens_weight_classical_distance', 'args': {'dim': 1, 'method': 'lr'}},
    {'ensemble': 'single', 'classifier': 'ens_weight_classical_cosine', 'args': {'dim': 1, 'method': 'lr'}},

    {'ensemble': 'single', 'classifier': 'ens_weight_classical_swap', 'args': {'dim': 3, 'method': 'lr'}},
    {'ensemble': 'single', 'classifier': 'ens_weight_classical_distance', 'args': {'dim': 3, 'method': 'lr'}},
    {'ensemble': 'single', 'classifier': 'ens_weight_classical_cosine', 'args': {'dim': 3, 'method': 'lr'}},

    {'ensemble': 'single', 'classifier': 'ens_weight_classical_swap', 'args': {'dim': 5, 'method': 'lr'}},
    {'ensemble': 'single', 'classifier': 'ens_weight_classical_distance', 'args': {'dim': 5, 'method': 'lr'}},
    {'ensemble': 'single', 'classifier': 'ens_weight_classical_cosine', 'args': {'dim': 5, 'method': 'lr'}},

    #{'ensemble': 'single', 'classifier': 'ens_weight_quantum_swap', 'args': {'dim': 2, 'method': 'lr', 'execution': 'statevector', 'shots': shots}},
    #{'ensemble': 'single', 'classifier': 'ens_weight_quantum_distance', 'args': {'dim': 2, 'method': 'lr', 'execution': 'statevector', 'shots': shots}},
    #{'ensemble': 'single', 'classifier': 'ens_weight_quantum_cosine', 'args': {'dim': 2, 'method': 'lr', 'execution': 'statevector', 'shots': shots}}
]

# select subset
dir = dir[data_s:data_e]
folds_to_execute = folds_to_execute[fold_s:fold_e]
classifiers = classifiers[classifier_s:classifier_e]

log = 'results-{}-{}.csv'.format(
    os.path.basename(sys.argv[0]).split('.')[0], time.time())
print('Running: {}'.format(log))
print('Data:\n{}'.format('\n'.join(dir)))
print('Folds: {}'.format(folds_to_execute))

b = []

results = []
for i in folds_to_execute:
    for d in dir:
        f = 'data/folds/{}/{}.txt'.format(d.split('.')[0], str(i))
        fold = np.genfromtxt(f)
        print('\nFold: {}'.format(f))
        for std in stds:
            train_data, test_data = get_train_test(fold, test_size, std=std)
            b.append(len(test_data))
            for c in classifiers:
                ensemble = c.get('ensemble')
                classifier_name = c.get('classifier')
                args = c.get('args', {})
                start = time.time()
                print('Started at: {} \t {} - {} - {} - {}'.format(time.ctime(),
                      ensemble, classifier_name, std, args))
                classifier = build_classifier(c, train_data)
                print('Classifier built, testing...')
                if '_int' in classifier_name:
                    acc, f1 = test_int(test_data, classifier, len(train_data))
                else:
                    acc, f1 = test(test_data, classifier)
                res = {'fold': f, 'standardization': std, 'ensemble': ensemble,
                       'classifier': classifier_name, 'acc': acc, 'f1': f1}
                res = {**res, **args}
                results.append(res)
                print('Finished at: {} \t Time: {}'.format(
                    time.ctime(), (time.time() - start)))
                pd.DataFrame(results).to_csv(
                    'results/{}'.format(log), sep=";", decimal=",")

df = pd.read_csv('results/{}'.format(log), sep=";", decimal=",")
single = df.groupby(['ensemble', 'classifier', 'standardization'], dropna=False).agg(median_acc=('acc', 'median'), mean_acc=('acc', 'mean'), median_f1=('f1', 'median'), mean_f1=('f1', 'mean'))

print(single)
