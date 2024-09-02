import json
from time import time

import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor

from clustering.add_class_to_data import CLUSTERS_ADDED
from loader import load_as_df

def predict_random_forest(df: DataFrame, xgb_model, filename: str) -> None:
    # DF_TRAIN = df.sample(n=280000, random_state=1)

    print(f'Test samples: {len(df)}')

    X = df.drop(columns=['soil_moisture'])
    y = df['soil_moisture']

    print(f'Predict...')
    start = time()
    rf_predictions = xgb_model.predict(X)
    print('%.3f seconds' % (time() - start))

    (
        pd.DataFrame({'lat': df.sp_lat, 'lon': df.sp_lon, 'soil_moisture': y, 'pred_soil_moisture': rf_predictions})
        .to_csv(f'predictions/{filename}', index=False)
    )
    return

def predict_no_label(df: DataFrame, xgb_model, filename: str) -> None:
    # DF_TRAIN = df.sample(n=280000, random_state=1)

    print(f'Test samples: {len(df)}')

    print(f'Predict...')
    start = time()
    xgb_predictions = xgb_model.predict(df.drop(columns=['soil_moisture']))
    print('%.3f seconds' % (time() - start))
    (
        pd.DataFrame({'lat': df.sp_lat, 'lon': df.sp_lon,'pred_soil_moisture': xgb_predictions, 'soil_moisture': df.soil_moisture})
        .to_csv(f'../data/predictions/xgb/{filename}', index=False)
    )


if __name__ == '__main__':
    # start = time()
    with open('models/xgb_2022_1000.json') as f:
        features = json.load(f)
    # df = load_as_df('20220101', '20221231')
    # df.landcover = df.landcover.astype(str)
    # df = pd.concat([df, pd.get_dummies(df.landcover)], axis=1)
    # df = df.filter(features + ['soil_moisture'])
    #
    # print('%.3f seconds' % (time() - start))
    #
    # model_path = 'models/xgb_2022_1000.pkl'
    # print(f'Load: {model_path}')
    # start = time()
    # xgb = joblib.load(model_path)
    # predict_random_forest(df, xgb, f'xgb_2022.csv')

    first_last_days = [
        ('20220101', '20220331'),
        ('20220401', '20220630'),
        ('20220701', '20220930'),
        ('20221001', '20221231'),
        ('20230101', '20230331'),
        ('20230401', '20230630'),
        ('20230701', '20230930'),
        ('20231001', '20231231')
    ]

    model_path = 'models/xgb_2022_1000.pkl'
    print(f'Load: {model_path}')
    start = time()
    xgb = joblib.load(model_path)
    print('%.3f seconds' % (time() - start))
    # Print the results
    for first_day, last_day in first_last_days:
        print(first_day, last_day)
        start = time()
        df = pd.concat([load_as_df(first_day, last_day),
                        load_as_df(first_day, last_day, 'no_label_with_cluster')], axis=0)

        df.landcover = df.landcover.astype(str)
        df = pd.concat([df, pd.get_dummies(df.landcover)], axis=1)
        df = df.filter(features + ['soil_moisture'])

        print('%.3f seconds' % (time() - start))



        predict_no_label(df, xgb, f'xgb_{first_day}_{last_day}.csv')
