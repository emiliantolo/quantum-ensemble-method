import numpy as np
import os


def get_folds(dataset, n=30):
    folds = []
    for i in range(n):
        folds.append(np.random.permutation(dataset))
    return np.array(folds)


def save_folds(folds):
    dir = 'folds/' + d.split('.')[0] + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in range(len(folds)):
        np.savetxt(dir + str(i) + '.txt', folds[i])


np.random.seed(123)
n = 30

dir = os.listdir('dataset')
dir.remove('02_transfusion.csv')

dir.sort()
#dir = dir[15:]
print(dir)
#exit()

for d in dir:
    data = np.genfromtxt('dataset/' + d, delimiter=',', skip_header=True)
    folds = get_folds(data, n)
    save_folds(folds)

'''
dir = os.listdir('synthetic')

dir.sort()
print(dir)

for d in dir:
    data = np.genfromtxt('synthetic/' + d, delimiter=',', skip_header=True)
    folds = get_folds(data, n)
    save_folds(folds)
'''