from time import time

import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor

from clustering.add_class_to_data import CLUSTERS_ADDED
from loader import load_as_df


def train_random_forest(df: DataFrame):
    # DF_TRAIN = df.sample(n=280000, random_state=1)

    print(f'Train samples: {len(df)}')

    X = df.drop(columns=['soil_moisture'])
    y = df['soil_moisture']

    # Random Forest model

    TREES = 50

    print(f"Random Forest ({TREES} Trees):")
    start = time()

    rf_model = RandomForestRegressor(n_estimators=TREES, random_state=0, n_jobs=7, verbose=0,
                                     min_samples_split=4, min_samples_leaf=2, max_depth=25, max_features=12)
    rf_model.fit(X, y)
    joblib.dump(rf_model, f'models/rf_no_cluster_2022_{TREES}.pkl')

    end = time()
    result = end - start
    print('%.3f seconds' % result)

    return


if __name__ == '__main__':
    start = time()
    df = load_as_df('20220101', '20221231')
    df.landcover = df.landcover.astype(str)
    df = df.drop(columns=['timestamp_lst', 'rounded_timestamps', '7'] + [elem for elem in CLUSTERS_ADDED if elem != '7'])
    print('%.3f seconds' % (time() - start))
    n = len(df)
    df = df.dropna()
    print(f'Number of rows removed: {n - len(df)}')
    train_random_forest(df)
