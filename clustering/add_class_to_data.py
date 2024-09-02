import glob
import json
import os
import pickle

import numpy as np

from clustering.kmeans import normalize_ddms, order_labels
from preprocessing.config import CYNGNSS_FEATURES_HEADER, _ddm_names

N_CLUSTERS = [2,3,5,7,10,15,20]
CLUSTERS_ADDED = list(map(str, N_CLUSTERS))

CYNGNSS_FEATURES_WITH_CLUSTERS_HEADER = ','.join(CYNGNSS_FEATURES_HEADER.split(',')[:-len(_ddm_names)] + CLUSTERS_ADDED)


def get_labels(nb_clusters, ddms):
    with open(f'models/kmeans/kmean_more_days_{nb_clusters}.pck', 'rb') as f:
        kmean = pickle.load(f)

    with open(f'models/kmeans/kmean_more_days_{nb_clusters}.json', 'rb') as f:
        label_to_ordered_label = json.load(f)
        label_to_ordered_label = {int(k): int(v) for k, v in label_to_ordered_label.items()}

    labels = kmean.predict(ddms)
    ordered_labels = order_labels(labels, label_to_ordered_label)
    return np.array(ordered_labels).reshape(len(ordered_labels), 1)


def main(dataset_filename):
    try:
        data = np.load(f'../data/train_label/{dataset_filename}')
        ddms = data[:, data.shape[1] - (17 * 11):]
        ddms = normalize_ddms(ddms)
        data_no_ddms = data[:,:data.shape[1] - (17 * 11)]

        np.save(
            # f'../data/train_label_with_cluster/{dataset_filename}',
            f'../data/no_label_with_cluster/{dataset_filename}',
            np.concat([data_no_ddms] + [get_labels(n_cluster, ddms) for n_cluster in N_CLUSTERS], axis=1),
            allow_pickle=False
        )
    except:
        pass

def main_no_label(dataset_filename):
    try:
        data = np.load(f'../data/train_no_label/{dataset_filename}')
        ddms = data[:, data.shape[1] - (17 * 11):]
        ddms = normalize_ddms(ddms)
        data_no_ddms = data[:,:data.shape[1] - (17 * 11)]

        np.save(
            # f'../data/train_label_with_cluster/{dataset_filename}',
            f'../data/no_label_with_cluster/{dataset_filename}',
            np.concat([data_no_ddms] + [get_labels(n_cluster, ddms) for n_cluster in N_CLUSTERS], axis=1),
            allow_pickle=False
        )
    except:
        pass

if __name__ == '__main__':
    # for filename in map(os.path.basename, glob.glob("../data/train_label/*.npy")):
    #     main(
    #         filename,
    #     )
    for filename in map(os.path.basename, glob.glob("../data/train_no_label/*.npy")):
        main_no_label(
            filename,
        )