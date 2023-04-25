import numpy as np
import pandas as pd
from glob import glob

filenames = np.loadtxt('annotations/Gardner_Expert.csv', dtype=str, usecols=(0,), delimiter=',')
annotations = {filename: [[], [], [], 0] for filename in filenames}

# get list of all annotation files from international annotators
anno_list = glob('annotations/*.csv')

anno_list.reverse()  # orders Garner Expert first to take their annotation in case of 50/50
mj_icm = mj_exp = mj_teq = 0
# add to annotations dict
for file in anno_list:
    annotator = file.split('/')[-1].split('.')[0]
    if annotator not in ['Gardner_Expert', 'Annotator_0', 'Annotator_1', 'Annotator_2', 'Annotator_5', 'Annotator_7',
                         'Annotator_8']:
        continue
    anno = pd.read_csv(file, header=None).fillna(-1)
    for idx, row in anno.iterrows():
        filename, exp, icm, teq = row
        exp, icm, teq = int(exp), int(icm), int(teq)
        annotations[filename][3] += 1
        if exp != -1:
            annotations[filename][0].append(exp)

        if exp in [2, 3, 4] and icm in [0, 1, 2]:
            annotations[filename][1].append(icm)
        if exp in [0, 1] and icm == 3:
            annotations[filename][1].append(icm)
        if exp == -1 and icm != -1:
            annotations[filename][1].append(icm)
        if exp in [2, 3, 4] and teq in [0, 1, 2]:
            annotations[filename][2].append(teq)
        if exp in [0, 1] and teq == 3:
            annotations[filename][2].append(teq)
        if exp == -1 and teq != -1:
            annotations[filename][2].append(teq)

with open('annotations/test_rev.csv', 'w+') as file:
    for k, v in annotations.items():

        # majority vote on expansion
        exp, icm, teq, count = v
        exp_vote = max(exp, key=exp.count)
        if exp_vote in [0, 1]:
            icm_vote = teq_vote = 3  # not defined
            conf_icm = conf_teq = 1
            icm_count = teq_count = ''

        else:
            # remove all not defined from icm and teq list
            icm = [e for e in icm if e != 3]
            teq = [e for e in teq if e != 3]
            icm_vote = max(icm, key=icm.count)
            teq_vote = max(teq, key=teq.count)
            icm_count = icm.count(icm_vote)
            teq_count = teq.count(teq_vote)
            conf_icm = icm_count / len(icm)
            conf_teq = teq_count / len(teq)

        exp_count = exp.count(exp_vote)
        conf_exp = exp_count / len(exp)
        # print(k, v, conf_exp, conf_icm, conf_teq)
        file.write(
            k + ',' + str(exp_vote) + ',' + str(icm_vote) + ',' + str(teq_vote) + ','
            + str(conf_exp)[0:4] + ',' + str(conf_icm)[0:4] + ',' + str(conf_teq)[0:4] + ',' +
            str(exp_count) + '/' + str(len(exp)) + ',' + str(icm_count) + '/' + str(len(icm)) + ',' + str(
                teq_count) + '/' + str(len(teq)) + ',' +
            str(count) +
            '\n')

        if conf_exp <= 0.5: mj_exp += 1
        if conf_icm <= 0.5: mj_icm += 1
        if conf_teq <= 0.5: mj_teq += 1

print(mj_exp, mj_icm, mj_teq)
