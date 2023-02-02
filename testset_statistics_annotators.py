import numpy as np
import pandas as pd
from glob import glob

# get list of all annotation files from international annotators
anno_list = glob('model_predictions/*/pred.csv')
#anno_list = glob('annotations/*.csv')
#anno_list.remove('annotations/test.csv')
#anno_list.remove('annotations/test_rev.csv')
print(anno_list)

test_anno = np.loadtxt('annotations/test_rev.csv', dtype=str, delimiter=',', usecols=(0, 1, 2, 3))

annotations = {image: [int(float(exp)), int(icm), int(teq)] for image, exp, icm, teq in test_anno}

acc_exp_list = []
acc_icm_list = []
acc_teq_list = []
annotator_list = []
# add to annotations dict
for file in anno_list:
    annotator = file.split('/')[-1].split('.')[0]
    #if annotator not in ['Gardner_Expert', 'Annotator_0', 'Annotator_1', 'Annotator_2', 'Annotator_5', 'Annotator_7',
    #                     'Annotator_8']: continue
    print(annotator)
    annotator_list.append(annotator)
    wrong_labels_exp = 0
    wrong_labels_teq = 0
    wrong_labels_icm = 0
    conf_mat = np.zeros(shape=(3, 5, 5,))
    count = np.zeros(shape=(3,))

    anno = pd.read_csv(file, header=None).fillna(-1)
    for idx, row in anno.iterrows():
        filename, exp, icm, teq = row
        exp, icm, teq = int(exp), int(icm), int(teq)

        exp_gt, icm_gt, teq_gt = annotations[filename]

        if exp in [0, 1]:
            icm = teq = 3
        if exp in [2, 3, 4]:
            if icm == 3:
                icm = -1
            if teq == 3:
                teq = -1

        if exp != -1 and exp_gt != -1:
            conf_mat[0, exp_gt, exp] += 1
            count[0] += 1
            if exp != exp_gt: wrong_labels_exp += 1
        if icm != 3 and icm_gt not in [-1, 3]:
            conf_mat[1, icm_gt, icm] += 1
            count[1] += 1
            if icm != icm_gt: wrong_labels_icm += 1
        if teq != 3 and teq_gt not in [-1, 3]:
            conf_mat[2, teq_gt, teq] += 1
            count[2] += 1
            if teq != teq_gt: wrong_labels_teq += 1

    acc_exp = (count[0] - wrong_labels_exp) / count[0]
    acc_icm = (count[1] - wrong_labels_icm) / count[1]
    acc_teq = (count[2] - wrong_labels_teq) / count[2]

    print(acc_exp)
    print(acc_icm)
    print(acc_teq)

    acc_exp_list.append(acc_exp)
    acc_icm_list.append(acc_icm)
    acc_teq_list.append(acc_teq)

    # for e in conf_mat:
    # print(e)
    print('--------------------------------')

acc_exp_list = np.asarray(acc_exp_list)
acc_icm_list = np.asarray(acc_icm_list)
acc_teq_list = np.asarray(acc_teq_list)

mean_exp = np.mean(acc_exp_list)
mean_icm = np.mean(acc_icm_list)
mean_teq = np.mean(acc_teq_list)

median_exp = np.median(acc_exp_list)
median_icm = np.median(acc_icm_list)
median_teq = np.median(acc_teq_list)

std_exp = np.std(acc_exp_list)
std_icm = np.std(acc_icm_list)
std_teq = np.std(acc_teq_list)

cutoff_exp = mean_exp - std_exp
cutoff_icm = mean_icm - std_icm
cutoff_teq = mean_teq - std_teq

print('median:', median_exp, median_icm, median_teq)
print('mean:', mean_exp, mean_icm, mean_teq)
print('std:', std_exp, std_icm, std_teq)
print(cutoff_exp, cutoff_icm, cutoff_teq)

for i in range(len(annotator_list)):
    if acc_exp_list[i] < cutoff_exp:
        print(annotator_list[i])
        continue

    if acc_icm_list[i] < cutoff_icm:
        print(annotator_list[i])
        continue

    if acc_teq_list[i] < cutoff_teq:
        print(annotator_list[i])
        continue
