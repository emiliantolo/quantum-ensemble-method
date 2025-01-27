import numpy as np
import pandas as pd
import os
import sys
import time
from classification.helpers.data import get_train_test, normalize
from helpers import *
from xgboost import XGBClassifier

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

# select subset
dir = dir[data_s:data_e]
folds_to_execute = folds_to_execute[fold_s:fold_e]

log = 'results-{}-{}.csv'.format(
    os.path.basename(sys.argv[0]).split('.')[0], time.time())
print('Running: {}'.format(log))
print('Data:\n{}'.format('\n'.join(dir)))
print('Folds: {}'.format(folds_to_execute))

def get_train_test_split(fold, test_size, std):
    train_data, test_data = get_train_test(fold, test_size, std=std)
    # Prepare the data
    X_train = np.array([normalize(d['x']) for d in train_data])  # Features
    X_test = np.array([normalize(d['x']) for d in test_data])  # Features
    # Convert labels to binary (XGBoost requires 0/1 for binary classification)
    y_train = np.array([(1 - d['y']) // 2 for d in train_data])  # Labels
    y_test = np.array([(1 - d['y']) // 2 for d in test_data])  # Labels
    return X_train, X_test, y_train, y_test

def test_new(X_test, y_test, bst, verbose=True):
    # Make predictions
    y_pred_prob = bst.predict(X_test)
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)  # Convert probabilities to binary labels
    yt = np.array([(1 - 2*d) for d in y_test])  # Labels
    yp = np.array([(1 - 2*d) for d in y_pred])  # Labels
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, t in enumerate(yt):
        res = yp[i]
        if verbose:
            print('Completed {}/{}'.format(i + 1, len(yt)))
        if (res > 0) and (t > 0):
            tp += 1
        if (res > 0) and (t < 0):
            fp += 1
        if (res < 0) and (t > 0):
            fn += 1
        if (res < 0) and (t < 0):
            tn += 1
    if tp == 0:
        f1 = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc, f1

results = []
for i in folds_to_execute:
    for d in dir:
        f = 'data/folds/{}/{}.txt'.format(d.split('.')[0], str(i))
        fold = np.genfromtxt(f)
        print('\nFold: {}'.format(f))
        for std in stds:
            X_train, X_test, y_train, y_test = get_train_test_split(fold, test_size, std)
            ensemble = 'xgboost'
            classifier_name = 'binary:logistic'
            args = {}
            start = time.time()
            print('Started at: {} \t {} - {} - {} - {}'.format(time.ctime(),
                    ensemble, classifier_name, std, args))
            # Train the XGBoost model
            bst = XGBClassifier(n_estimators=100, max_depth=5, eta=0.1, objective='binary:logistic')
            bst.fit(X_train, y_train)
            print('Classifier built, testing...')
            acc, f1 = test_new(X_test, y_test, bst)
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
