import os.path as osp
import numpy as np
from numpy.random import randint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

path = 'complete.csv'
source = np.loadtxt(path, dtype=str, delimiter=', ')
source = shuffle(source, random_state=10_123)
size = len(source)
data = []
sampled_indices = []
exp_count = np.zeros(5)
icm_count = np.zeros(4)
teq_count = np.zeros(4)

collecting_data = True
target_path = 'testset_filenames.csv'
with open(target_path, 'w+') as anno:
    while (collecting_data):
        index = randint(low=0, high=size)
        if index in sampled_indices: continue
        file, exp, icm, teq = source[index]
        if icm == '2': continue
        if teq == '2': continue
        data.append(file)
        exp_count[int(exp)] += 1
        icm_count[int(icm)] += 1
        teq_count[int(teq)] += 1
        sampled_indices.append(index)
        if len(data) >= 268:
            collecting_data = False
        anno.write(file + ', ' + exp + ', ' + icm + ', ' + teq + '\n')

    # get all c images
    for count, (file, exp, icm, teq) in enumerate(source):
        if icm == '2' and icm_count[2] < 7:
            anno.write(file + ', ' + exp + ', ' + icm + ', ' + teq + '\n')
            data.append(file)
            exp_count[int(exp)] += 1
            icm_count[int(icm)] += 1
            teq_count[int(teq)] += 1
        if teq == '2' and teq_count[2] < 25:
            anno.write(file + ', ' + exp + ', ' + icm + ', ' + teq + '\n')
            data.append(file)
            exp_count[int(exp)] += 1
            icm_count[int(icm)] += 1
            teq_count[int(teq)] += 1
            sampled_indices.append(index)
    anno.close()
# do some tests
assert np.sum(exp_count) == np.sum(icm_count) == np.sum(teq_count) == len(np.unique(data)) == 300
del data
# load finished anno file
data = np.loadtxt(target_path, dtype=str, delimiter=', ')
print(len(np.unique(data[:, 0])))
print(*exp_count)
print(*icm_count)
print(*teq_count)

# create train_val csv file

train_val_data = [list(element) for element in source if element[0] not in data]

save_path = ''
tasks = ['train', 'val']

train_val_data = train_test_split(train_val_data, test_size=1. / 5., shuffle=True, random_state=10_123)

for c, task in enumerate(tasks):
    with open(osp.join(save_path, task + '.csv'), 'w+') as file:
        for e in train_val_data[c]:
            file.write(e[0] + ', ' + e[1] + ', ' + e[2] + ', ' + e[3] + ', ' + '\n')
