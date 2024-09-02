import json
from time import time

import joblib
import pandas as pd
from pandas import DataFrame
from xgboost import XGBRegressor

from clustering.add_class_to_data import CLUSTERS_ADDED
from loader import load_as_df


def train_xgb(df: DataFrame):
    # DF_TRAIN = df.sample(n=280000, random_state=1)

    print(f'Train samples: {len(df)}')

    X = df.drop(columns=['soil_moisture'])
    y = df['soil_moisture']

    # Random Forest model

    TREES = 1000

    print(f"Random Forest ({TREES} Trees):")
    start = time()

    rf_model = XGBRegressor(n_estimators=TREES, random_state=0, max_depth=10, tree_method='hist')
    rf_model.fit(X, y)
    joblib.dump(rf_model, f'models/xgb_balanced_no_vwc_2022_{TREES}.pkl')
    with open(f'models/xgb_balanced_no_vwc_2022_{TREES}.json', 'w') as out:
        json.dump(list(X.columns), out)

    end = time()
    result = end - start
    print('%.3f seconds' % result)

    return


if __name__ == '__main__':
    start = time()
    df = load_as_df('20220101', '20221231')
    df = df.groupby(['7']).head(10_000)
    df.landcover = df.landcover.astype(str)
    df = pd.concat([df, pd.get_dummies(df.landcover)], axis=1)
    df = df.drop(columns=['timestamp_lst', 'rounded_timestamps', 'landcover', 'vegetation_water_content'] + [elem for elem in CLUSTERS_ADDED if elem != '7'])
    print('%.3f seconds' % (time() - start))
    n = len(df)
    df = df.dropna()
    print(f'Number of rows removed: {n - len(df)}')
    train_xgb(df)
