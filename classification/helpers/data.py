import numpy as np

def get_samples(data, n_samples=-1, replace=False, balanced=False, p=None):
    if balanced:
        pos = []
        neg = []
        for d in data:
            if d['y'] > 0:
                pos.append(d)
            else:
                neg.append(d)
        if n_samples == -1:
            samples = min(len(pos), len(neg))
        else:
            samples = int(n_samples / 2)
        if p == None:
            pp = np.random.choice(pos, samples, replace=replace)
            nn = np.random.choice(neg, samples, replace=replace)
        else:
            p_pos = []
            p_neg = []
            for i, d in enumerate(data):
                if d['y'] > 0:
                    p_pos.append(p[i])
                else:
                    p_neg.append(p[i])
            p_pos = p_pos / np.sum(p_pos)
            p_neg = p_neg / np.sum(p_neg)
            pp = np.random.choice(pos, samples, replace=replace, p=p_pos)
            nn = np.random.choice(neg, samples, replace=replace, p=p_neg)
        # returns even number of samples 
        return np.concatenate([nn, pp])
    else:
        if n_samples == -1:
            return np.random.choice(data, len(data), replace=replace, p=p)
        else:
            return np.random.choice(data, n_samples, replace=replace, p=p)

def get_train_test(permuted, test_size=0.1, std='none', balance_train=False):
    size = int(test_size * len(permuted))
    # over train set
    if std == 'std':
        train = permuted[size:]
        mean_train = np.mean(train, axis=0)
        mean_train[-1] = 0
        std_train = np.std(train, axis=0)
        std_train[-1] = 1
        std_train[std_train == 0] = 1
        permuted = (permuted - mean_train) / std_train
    elif std == 'minmax':
        train = permuted[size:]
        labels = permuted[:, -1]
        min_train = np.min(train, axis=0)
        range_train = np.max(train, axis=0) - min_train
        range_train[range_train == 0] = 1
        permuted = (permuted - min_train) / range_train
        permuted.clip(0, 1)
        permuted[:, -1] = labels
    data = np.array([{'x': d[:-1], 'y': d[-1]} for d in permuted])
    return get_samples(data[size:], replace=False, balanced=balance_train), data[:size]

def normalize(x):
    return x / np.linalg.norm(x)
