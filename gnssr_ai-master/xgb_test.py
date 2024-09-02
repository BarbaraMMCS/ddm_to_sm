import json
import math
from time import time

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from clustering.add_class_to_data import CLUSTERS_ADDED
from loader import load_as_df


def test_random_forest(df: DataFrame, rf_model):
    print(f'Test samples: {len(df)}')

    X = df.drop(columns=['soil_moisture'])
    y = df['soil_moisture']

    print(f'Predict...')
    start = time()
    rf_predictions = rf_model.predict(X)
    print('%.3f seconds' % (time() - start))
    rf_error = y - rf_predictions

    # Metrics
    rf_std_dev = np.std(rf_predictions)
    rf_mae = mean_absolute_error(y, rf_predictions)
    rf_r2 = r2_score(y, rf_predictions)
    # rf_r = math.sqrt(rf_r2)
    rf_mse = mean_squared_error(y, rf_predictions)
    rf_rmse = math.sqrt(rf_mse)
    rf_bias = np.mean(rf_error)
    rf_ubrmse = math.sqrt(rf_rmse ** 2 - rf_bias ** 2)

    print(f"stdDEV:\t{rf_std_dev:.4f}")
    print(f"MAE:\t{rf_mae:.4f}")
    print(f"R2: \t{rf_r2:.4f}")
    # print(f"R:  \t{rf_r:.4f}")
    print(f"MSE:\t{rf_mse:.4f}")
    print(f"RMSE:\t{rf_rmse:.4f}")
    print(f"BIAS:\t{rf_bias:.4f}")
    print(f"ubRMSE:\t{rf_ubrmse:.4f}")
    return


if __name__ == '__main__':
    start = time()
    with open('models/xgb_balced_no_vwc_2022_1000.json') as f:
        features = json.load(f)
    df = load_as_df('20230101', '20231231')
    df.landcover = df.landcover.astype(str)
    df = pd.concat([df, pd.get_dummies(df.landcover)], axis=1)
    df = df.filter(features + ['soil_moisture'])

    print('%.3f seconds' % (time() - start))

    model_path = 'models/xgb_balced_no_vwc_2022_1000.pkl'
    print(f'Load: {model_path}')
    start = time()
    rf = joblib.load(model_path)
    print('%.3f seconds' % (time() - start))
    test_random_forest(df, rf)
