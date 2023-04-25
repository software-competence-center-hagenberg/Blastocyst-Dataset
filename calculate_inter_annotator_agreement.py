# Florian Kromp
# 20.02.2023
# Calculate inter-annotator agreement including the consensus votes using Cohen's Kappa score

import numpy as np
import pandas as pd
from glob import glob
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score

# get list of all annotation files from international annotators upon expert filtering

anno_list = ['annotations\\Annotator_0.csv','annotations\\Annotator_1.csv','annotations\\Annotator_2.csv','annotations\\Annotator_5.csv','annotations\\Annotator_7.csv','annotations\\Annotator_8.csv','annotations\\Annotator_Expert.csv']
print(anno_list)
test_list = ['annotations\\test_rev.csv']

acc_exp_list = []
acc_icm_list = []
acc_teq_list = []
annotator_list = []

# add to annotations dict
for file in test_list:
    current_annotator = file.split('\\')[-1].split('.')[0]
    test_anno = np.loadtxt(file, dtype=str, delimiter=',', usecols=(0, 1, 2, 3), skiprows=0)
    annotations = {image: [str(exp), str(icm), str(teq)] for image, exp, icm, teq in test_anno}
    for file_to_compare in anno_list:
        annotator = file_to_compare.split('\\')[-1].split('.')[0]

        print("Comparing " + current_annotator + " with " + annotator)
        annotator_list.append(annotator)
        wrong_labels_exp = 0
        wrong_labels_teq = 0
        wrong_labels_icm = 0
        conf_mat = np.zeros(shape=(3, 6, 6,))
        count = np.zeros(shape=(3,))

        anno = pd.read_csv(file_to_compare, header=None).fillna(-1)
        label_list_exp = []
        label_list_exp_gt = []
        label_list_icm = []
        label_list_icm_gt = []
        label_list_teq = []
        label_list_teq_gt = []

        for idx, row in anno.iterrows():
            filename, exp, icm, teq = row


            try:
                exp_gt, icm_gt, teq_gt = annotations[filename]
            except:
                continue
            if exp_gt == '':
                exp_gt = -1
            if icm_gt == '':
                icm_gt = -1
            if teq_gt == '':
                teq_gt = -1
            if exp == '':
                exp = -1
            if icm == '':
                icm = -1
            if teq == '':
                teq = -1
            exp, icm, teq = int(exp), int(icm), int(teq)
            exp_gt, icm_gt, teq_gt = int(exp_gt), int(icm_gt), int(teq_gt)
            consider_teq_icm = 1

            #Set correct labels for evaluation
            if icm not in [0, 1, 2]:
                icm = 3  # 3=not assessible, other value set by mistake
            if teq not in [0, 1, 2]:
                teq = 3  # 3=not assessible, other value set by mistake
            if icm_gt == -1:
                icm_gt = 3 # -1 is not assessible, set it to 3 for the confusion matrix
            if teq_gt == -1:
                teq_gt = 3 # -1 is not assessible, set it to 3 for the confusion matrix
            if exp not in [0,1,2,3,4]:
                exp = 5 # 5=Not assessible
            if exp_gt not in [0,1,2,3,4]:
                exp_gt = 5 # 5=not assessible
            label_list_exp.append(exp)
            label_list_exp_gt.append(exp_gt)

            if exp in [0, 1] or exp_gt in [0,1]:
                consider_teq_icm = 0 # ICM or TE not defined only depends on the expansion, so do not consider these cases
            else:
                label_list_icm.append(icm)
                label_list_icm_gt.append(icm_gt)
                label_list_teq.append(teq)
                label_list_teq_gt.append(teq_gt)
                
        # Calculate Kappa scores
        print(cohen_kappa_score(label_list_exp_gt, label_list_exp))
        print(cohen_kappa_score(label_list_icm_gt, label_list_icm))
        print(cohen_kappa_score(label_list_teq_gt, label_list_teq))
        
        # Break here and collect metrics
        e=1
