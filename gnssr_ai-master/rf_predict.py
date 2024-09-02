from time import time

import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor

from clustering.add_class_to_data import CLUSTERS_ADDED
from loader import load_as_df

def predict_random_forest(df: DataFrame, rf_model: RandomForestRegressor, filename: str) -> None:
    # DF_TRAIN = df.sample(n=280000, random_state=1)

    print(f'Test samples: {len(df)}')

    X = df.drop(columns=['soil_moisture'])
    y = df['soil_moisture']

    print(f'Predict...')
    start = time()
    rf_predictions = rf_model.predict(X)
    print('%.3f seconds' % (time() - start))

    (
        pd.DataFrame({'lat': df.sp_lat, 'lon': df.sp_lon, 'soil_moisture': y, 'pred_soil_moisture': rf_predictions})
        .to_csv(f'../data/predictions/{filename}', index=False)
    )
    return


def predict_no_label(df: DataFrame, rf_model: RandomForestRegressor, filename: str) -> None:
    # DF_TRAIN = df.sample(n=280000, random_state=1)

    print(f'Test samples: {len(df)}')

    print(f'Predict...')
    start = time()
    df['pred_soil_moisture'] = rf_model.predict(df.drop(columns=['soil_moisture']))
    print('%.3f seconds' % (time() - start))
    df.to_csv(f'../data/predictions/rf/{filename}', index=False)


if __name__ == '__main__':
    start = time()
    df = load_as_df('20230101', '20231231')


    df.landcover = df.landcover.astype(str)
    df = df.drop(columns=['timestamp_lst', 'rounded_timestamps', 'vegetation_water_content'] + [elem for elem in CLUSTERS_ADDED if elem != '7'])
    print('%.3f seconds' % (time() - start))

    model_path = 'models/rf_balanced_no_vwc_2022_50.pkl'
    print(f'Load: {model_path}')
    start = time()
    rf = joblib.load(model_path)
    print('%.3f seconds' % (time() - start))
    predict_random_forest(df, rf, f'rf_balanced_no_vwc_2023.csv')



    # first_last_days = [
    #     ('20220101', '20220331'),
    #     ('20220401', '20220630'),
    #     ('20220701', '20220930'),
    #     ('20221001', '20221231'),
    #     ('20230101', '20230331'),
    #     ('20230401', '20230630'),
    #     ('20230701', '20230930'),
    #     ('20231001', '20231231')
    # ]
    #
    # model_path = 'models/rf_balanced_no_vwc_2022_50.pkl'
    # print(f'Load: {model_path}')
    # start = time()
    # rf = joblib.load(model_path)
    # print('%.3f seconds' % (time() - start))
    # # Print the results
    # for first_day, last_day in first_last_days:
    #     print(first_day, last_day)
    #     start = time()
    #     df = pd.concat([load_as_df(first_day, last_day),
    #                load_as_df(first_day, last_day, 'no_label_with_cluster')], axis=0)
    #
    #     df.landcover = df.landcover.astype(str)
    #     df = df.drop(columns=['timestamp_lst', 'rounded_timestamps'] + [elem for elem in CLUSTERS_ADDED if elem != '7'])
    #
    #     print('%.3f seconds' % (time() - start))
    #     predict_no_label(df, rf, f'rf_{first_day}_{last_day}.csv')

