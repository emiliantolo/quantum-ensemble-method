
import math
import numpy as np
from classification.base import BaseClassifier
from classification.helpers.data import get_samples
from classification.helpers.data import normalize
from scipy.optimize import minimize

def get_sets(train_data, dim, overlap=0):

    idx_qubits = math.ceil(math.log2(len(train_data)))
    feat_qubits = math.ceil(math.log2(len(train_data[0]['x'])))
    ctrl_qubits = min(max(idx_qubits, feat_qubits), dim)

    def bin_arr(decimal, size):
        l = [0] * size
        i = 0
        while decimal >= 1:
            if (decimal % 2) == 1:
                l[i] = 1
            decimal //= 2
            i += 1
        return l

    def decimal_from_bin_arr(l):
        sum = 0
        for i in range(len(l)):
            sum += l[i] * (2 ** i)
        return sum
    
    conf_ctrl = [bin_arr(i, ctrl_qubits) for i in range(2**ctrl_qubits)]
    conf_idx = [bin_arr(i, idx_qubits) for i in range(len(train_data))]
    conf_feat = [bin_arr(i, feat_qubits) for i in range(len(train_data[0]['x']))]
    all_states = []
    for cc in conf_ctrl:
        for ci in conf_idx:
            for cf in conf_feat:
                all_states.append({'ctrl': cc.copy(), 'idx': ci.copy(), 'feat': cf.copy(), 'i_dec': decimal_from_bin_arr(ci), 'f_dec': decimal_from_bin_arr(cf)})

    for state in all_states:
        cc = state['ctrl']
        ci = state['idx']
        cf = state['feat']
        for i in range(ctrl_qubits):

            if cc[i] == 1:
                off = min(ctrl_qubits, idx_qubits // 2)
                if i < (idx_qubits // 2):
                    #cswap
                    ci[i], ci[i + off] = ci[i + off], ci[i]
                if i < idx_qubits:
                    #cnot
                    ci[i] = int(1 - ci[i])
                off = min(ctrl_qubits, feat_qubits // 2)
                if i < (feat_qubits // 2):
                    #cswap
                    cf[i], cf[i + off] = cf[i + off], cf[i]
                if i < feat_qubits:
                    #cnot
                    cf[i] = int(1 - cf[i])

    for state in all_states:
        cc = state['ctrl']
        ci = state['idx']
        cf = state['feat']
        for i in range(ctrl_qubits):
            if cc[i] == 1:
                for j in range(ctrl_qubits):
                    if i != j:
                        #cswap
                        if i < idx_qubits and j < idx_qubits:
                            ci[i], ci[j] = ci[j], ci[i]
                        if i < feat_qubits and j < feat_qubits:
                            cf[i], cf[j] = cf[j], cf[i]

    sets = [0] * 2**(ctrl_qubits-overlap)
    for i in range(2**(ctrl_qubits-overlap)):
        sets[i] = {'idx': [], 'feat': [], 'bucket': [], 'ida': []}
    for state in all_states:
        cc = state['ctrl']
        ci = state['idx']
        cf = state['feat']
        idec = state['i_dec']
        fdec = state['f_dec']
        # select outer set
        li = 1 if idx_qubits > 1 else 1
        lf = 1 if feat_qubits > 1 else 1
        if ((sum(ci[:li])) + sum(cf[:lf])) != (li + lf):
            sets[decimal_from_bin_arr(cc[:ctrl_qubits-overlap])]['idx'].append(idec)
            sets[decimal_from_bin_arr(cc[:ctrl_qubits-overlap])]['feat'].append(fdec)
            sets[decimal_from_bin_arr(cc[:ctrl_qubits-overlap])]['bucket'].append(decimal_from_bin_arr(cc[dim-overlap:]))
            sets[decimal_from_bin_arr(cc[:ctrl_qubits-overlap])]['ida'].append(decimal_from_bin_arr(ci))

    return sets 

def get_train_test(data):
    data = data #get_samples(data) #shuffle
    for d in data:
        d['x'] = normalize(d['x'])
    frac = 1#0.8 #frac
    train_size = int(frac * len(data)) 
    train_data = data[:train_size]
    test_data = train_data#data[train_size:]
    return train_data, test_data


def get_all(data, dim, overlap=0, num=1):
    train_data, test_data = get_train_test(data)
    if num > 1:
        train_data = np.concatenate([np.random.permutation(train_data) for i in range(num)])
    sets = get_sets(train_data, dim, overlap)
    return (train_data, test_data), sets


def optimize(all_res, all_ps, all_lab, method='lr', yrange=0):

    if method == 'lr':

        def objective(w, all_res, all_ps, all_lab):
            nums = []
            dens = []
            for i in range(len(all_lab)):
                num = 0
                den = 0
                for j in range(len(w) - 2):
                    num += w[j + 1] * all_ps[i,j] * all_res[i,j]
                    den += w[j + 1] * all_ps[i,j]
                nums.append(num)
                dens.append(den)
            nums = np.array(nums)
            dens = np.array(dens)
            y = (all_lab + 1) // 2
            z = (w[-1] * nums / dens + w[0])
            y_pred_prob = 1 / (1 + np.exp(-1 * z))
            epsilon = 1e-20 # small value to avoid log(0)
            loss = -np.mean(y * np.log(y_pred_prob + epsilon) + (1 - y) * np.log(1 - y_pred_prob + epsilon))
            return loss

        initial_guess = np.concatenate([[-5], np.ones(all_res.shape[1]), [10]])
        bounds = [(-np.inf, np.inf)] + [(0, np.inf)] * (all_res.shape[1]) + [(-np.inf, np.inf)] # bounds for weights

        result = minimize(objective, initial_guess, args=(all_res, all_ps, all_lab), bounds=bounds, constraints=None, method=None)
        weights = result.x

        return weights[1:-1], weights[0], weights[-1]

    else:
        raise ValueError('method not recognized')
