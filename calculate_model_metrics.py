# Florian Kromp
# 20.02.2023
# Example evaluation of an algorithm prediction (XCeption: prediction_xception.csv) with respect to the consensus vote (Gardner_test_gold_onlyGardnerScores.csv)

import numpy as np
import pandas as pd
from glob import glob
from sklearn import metrics

# get list of prediction files from architectures
prediction_list = [r'model_predictions\xception\pred.csv']
consensus_list = [r'annotations\test_rev.csv']
acc_exp_list = []
acc_icm_list = []
acc_teq_list = []
annotator_list = []

for file in consensus_list:
    current_consensus = file.split('\\')[-1].split('.')[0]
    test_anno = np.loadtxt(file, dtype=str, delimiter=',', usecols=(0, 1, 2, 3), skiprows=0)
    consensus_annotations = {image: [str(exp), str(icm), str(teq)] for image, exp, icm, teq in test_anno}
    for file_to_compare in prediction_list:
        model = file_to_compare.split('\\')[-1].split('.')[0]

        print("Comparing " + model + " to " + current_consensus)
        annotator_list.append(model)

        conf_mat = np.zeros(shape=(3, 6, 6,))
        count = np.zeros(shape=(3,))

        pred = pd.read_csv(file_to_compare, header=None).fillna(-1)
        label_list_exp = []
        label_list_exp_gt = []
        label_list_icm = []
        label_list_icm_gt = []
        label_list_teq = []
        label_list_teq_gt = []

        for idx, row in pred.iterrows():
            filename, exp, icm, teq = row

            try:
                exp_gt, icm_gt, teq_gt = consensus_annotations[filename]
            except:
                continue
            # Treat missing values as not assessable
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
            exp_gt, icm_gt, teq_gt = int(exp_gt.replace('NA','-1').replace('ND','3')), int(icm_gt.replace('NA','-1').replace('ND','3')), int(teq_gt.replace('NA','-1').replace('ND','3'))
            consider_teq_icm = 1

            #Set correct labels for evaluation
            if icm not in [0, 1, 2]:
                icm = 3  # 3=ICM not assessable, other value set by mistake
            if teq not in [0, 1, 2]:
                teq = 3  # 3=TE not assessable, other value set by mistake
            if icm_gt == -1:
                icm_gt = 3 # -1 is not assessable, set it to 3 (ICM not assessable) for the confusion matrix
            if teq_gt == -1:
                teq_gt = 3 # -1 is not assessable, set it to 3 (TE not assessable) for the confusion matrix
            if exp not in [0,1,2,3,4]:
                exp = 5 # 5=Expansion Not assessable
            if exp_gt not in [0,1,2,3,4]:
                exp_gt = 5 # 5=Eexpansion not assessable
            label_list_exp.append(exp)
            label_list_exp_gt.append(exp_gt)

            if exp in [0, 1] or exp_gt in [0,1]:
                consider_teq_icm = 0 # ICM or TE not defined only depends on the expansion, so do not consider these cases
            else:
                label_list_icm.append(icm)
                label_list_icm_gt.append(icm_gt)
                label_list_teq.append(teq)
                label_list_teq_gt.append(teq_gt)
        # Calculate evaluation metrics (including accuracy, class-weighted average of precision, recall and F1 score)
        print (metrics.classification_report(label_list_exp_gt, label_list_exp))
        print (metrics.classification_report(label_list_icm_gt, label_list_icm))
        print (metrics.classification_report(label_list_teq_gt, label_list_teq))
        
        # Break here and collect metrics
        e=1
