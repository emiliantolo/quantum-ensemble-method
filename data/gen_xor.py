import pandas as pd
import numpy as np
import random
import os

if not os.path.exists('synthetic'):
    os.makedirs('synthetic')

dim = 2
size = 100

data = []
for i in range(size):
    r = random.randint(0, 2**dim)
    x = []
    temp = r
    y = 0
    for d in range(dim):
        s = int((temp % 2))
        x.append((1 if s == 0 else -1) * random.random())
        y += s
        temp //= 2
    y = 1 if (y % 2) == 0 else -1
    data.append(x + [y])

header = ['x_{}'.format(i) for i in range(dim)] + ['class']

df = pd.DataFrame(data)
#print(df)

df.to_csv('synthetic/xor_{}d.csv'.format(dim), index=False, header=header)