import os
import pandas as pd
import numpy as np

from glob import glob

paths = glob('model_predictions/*/')
category = ['exp', 'icm', 'teq']

for path in paths:
    df_list = []
    for c in category:
        df = pd.read_csv(os.path.join(path, c + '.csv'), header=None).fillna(-1)
        df[c] = np.argmax(df.iloc[:, [1, 3, 5, 7, 9]].to_numpy(), axis=1)
        df_list.append(df[[0, c]])

    df_combined = df_list[0].merge(df_list[1]).merge(df_list[2])
    df_combined.to_csv(os.path.join(path, 'pred.csv'), header=False, sep=',', index=False)