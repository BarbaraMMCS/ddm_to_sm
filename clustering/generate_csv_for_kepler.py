import glob
import os

import numpy as np
import pandas as pd

from clustering.add_class_to_data import CLUSTERS_HEADERS_ADDED
from preprocessing.config import CYNGNSS_FEATURES_HEADER


def reformat_date(date):
    return date[:4] + '-' + date[4:6] + '-' + date[6:8] + 'T00:00:00'


def main():
    headers_list = (CYNGNSS_FEATURES_HEADER + CLUSTERS_HEADERS_ADDED).split(',')
    columns = ['sp_lat', 'sp_lon', 'nb_cluster_2', 'nb_cluster_3', 'nb_cluster_5', 'nb_cluster_7', 'nb_cluster_10', 'nb_cluster_15', 'nb_cluster_20']
    idx = [headers_list.index(c) for c in columns]
    dfs = []
    for filename in glob.glob("../data/train_label_with_cluster/202210*.npy"):
        data = np.load(filename)
        df = pd.DataFrame(data[:, idx], columns=columns)
        df['date'] = reformat_date(os.path.basename(filename).split('.')[0])
        dfs.append(df)

    pd.concat(dfs, axis=0).to_csv("../data/kepler/clusters.csv", index=False)

def main2():
    headers_list = (CYNGNSS_FEATURES_HEADER + CLUSTERS_HEADERS_ADDED).split(',')
    columns = ['sp_lat', 'sp_lon']
    idx = [headers_list.index(c) for c in columns]
    dfs = []
    for filename in glob.glob("../data/train_label/2022011*.npy"):
        data = np.load(filename)
        df = pd.DataFrame(data[:, idx], columns=columns)
        df['date'] = reformat_date(os.path.basename(filename).split('.')[0])
        df['label'] = 'label'
        dfs.append(df)
    for filename in glob.glob("../data/train_no_label/2022011*.npy"):
        data = np.load(filename)
        df = pd.DataFrame(data[:, idx], columns=columns)
        df['date'] = reformat_date(os.path.basename(filename).split('.')[0])
        df['label'] = 'no-label'
        dfs.append(df)
    pd.concat(dfs, axis=0).to_csv("../data/kepler/test.csv", index=False)

if __name__ == "__main__":
    main2()