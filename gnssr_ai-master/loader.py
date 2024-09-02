import os.path

import pandas as pd
from datetime import timedelta, datetime
import numpy as np

from clustering.add_class_to_data import CYNGNSS_FEATURES_WITH_CLUSTERS_HEADER


def load_as_df(start_date: object, end_date: object, folder:str='train_label_with_cluster') -> object:
    start_date = datetime.strptime(start_date, '%Y%m%d').date()
    end_date = datetime.strptime(end_date, '%Y%m%d').date()
    n_days = (end_date - start_date).days + 1
    data = np.concat([
        np.load(f'../data/{folder}/{(start_date + timedelta(days=i)).strftime("%Y%m%d")}.npy')
        for i in range(n_days) if os.path.exists(f'../data/{folder}/{(start_date + timedelta(days=i)).strftime("%Y%m%d")}.npy')], axis=0)

    return pd.DataFrame(data=data, columns=CYNGNSS_FEATURES_WITH_CLUSTERS_HEADER.split(','))


if __name__ == '__main__':
    load_as_df('20220801', '20220930')