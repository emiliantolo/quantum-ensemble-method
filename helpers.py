import numpy as np
import sys
from classification.helpers.factories import gen_classifier
from classification.helpers.data import normalize
import math

def test(test_data, classifier, verbose=True):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, t in enumerate(test_data):
        res, conf = classifier.classify(t['x'])
        if verbose:
            print('Completed {}/{}'.format(i + 1, len(test_data)))
        if (res > 0) and (t['y'] > 0):
            tp += 1
        if (res > 0) and (t['y'] < 0):
            fp += 1
        if (res < 0) and (t['y'] > 0):
            fn += 1
        if (res < 0) and (t['y'] < 0):
            tn += 1
    if tp == 0:
        f1 = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc, f1


def get_test_data(idx, train_size, x):
    xx = np.array([normalize(x)] * train_size)
    size_idx = 2 ** (math.ceil(math.log2(train_size)) - 1)
    size_feat = 2 ** (math.ceil(math.log2(len(xx[0]))) - 1)
    for i in range(train_size):
        if i >= size_idx:
            xx[i] = np.array(list(xx[i][:size_feat]) + [0] * (len(xx[i]) - size_feat))
    return xx


def test_int(test_data, classifier, train_size, verbose=True):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, t in enumerate(test_data):
        ttt = get_test_data(i, train_size, t['x'])
        res, conf = classifier.classify(ttt)
        if verbose:
            print('Completed {}/{}'.format(i + 1, len(test_data)))
        if (res > 0) and (t['y'] > 0):
            tp += 1
        if (res > 0) and (t['y'] < 0):
            fp += 1
        if (res < 0) and (t['y'] > 0):
            fn += 1
        if (res < 0) and (t['y'] < 0):
            tn += 1
    if tp == 0:
        f1 = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc, f1


def build_classifier(config, data):
    ensemble = config.get('ensemble')
    classifier_name = config.get('classifier')
    cl_args = config.get('args', {})
    if ensemble == 'single':
        classifier = gen_classifier(classifier_name, data, **cl_args)
    else:
        raise ValueError('classifier_name not recognized')
    return classifier


def parse_args():
    data_line = None
    fold_line = None
    classifier_line = None

    if len(sys.argv) > 1:
        line = sys.argv[1:]
        for i, l in enumerate(line):
            if (l == '--data') or (l == '-d'):
                data_line = line[i + 1]
            if (l == '--fold') or (l == '-f'):
                fold_line = line[i + 1]
            if (l == '--classifier') or (l == '-c'):
                classifier_line = line[i + 1]

    if data_line:
        s = data_line.split(':')
        if len(s) == 2:
            data_s = int(s[0]) if s[0] != '' else None
            data_e = int(s[1]) if s[1] != '' else None
        elif len(s) == 1:
            data_s = int(s[0])
            data_e = data_s + 1
        else:
            data_s = None
            data_e = None
    else:
        data_s = None
        data_e = None

    if fold_line:
        s = fold_line.split(':')
        if len(s) == 2:
            fold_s = int(s[0]) if s[0] != '' else None
            fold_e = int(s[1]) if s[1] != '' else None
        elif len(s) == 1:
            fold_s = int(s[0])
            fold_e = fold_s + 1
        else:
            fold_s = None
            fold_e = None
    else:
        fold_s = None
        fold_e = None

    if classifier_line:
        s = classifier_line.split(':')
        if len(s) == 2:
            classifier_s = int(s[0]) if s[0] != '' else None
            classifier_e = int(s[1]) if s[1] != '' else None
        elif len(s) == 1:
            classifier_s = int(s[0])
            classifier_e = classifier_s + 1
        else:
            classifier_s = None
            classifier_e = None
    else:
        classifier_s = None
        classifier_e = None

    return data_s, data_e, fold_s, fold_e, classifier_s, classifier_e


# usage
# python test_ex.py
# python test_ex.py --data 0:1 --fold 0:1 --classifier 0:1
# python test_ex.py -d 0:1
# python test_ex.py -f :1
# python test_ex.py -c 0:
# python test_ex.py -d 0