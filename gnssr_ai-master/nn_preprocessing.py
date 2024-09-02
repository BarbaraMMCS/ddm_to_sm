import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def nn_preprocess(df, year, filename):
    features = ['smap_vwc', 'ddm_snr', 'sp_inc_angle', 'sp_lon', 'sp_lat', 'gps_eirp',
           'sp_rx_gain', 'rx_to_sp_range', 'tx_to_sp_range',
           'pp5_3x3', 'reflectivity', 'day_sin', 'day_cos']
    if year == 2020:
        scaler = StandardScaler()
        scaler.fit(df.filter(features))
        joblib.dump(scaler, 'models/standard_scaler.pkl')

        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(df.lccs_class.drop_duplicates().to_numpy().reshape(-1, 1))
        joblib.dump(ohe, 'models/ohe.pkl')
    else:
        scaler = joblib.load('models/standard_scaler.pkl')
        ohe = joblib.load('models/ohe.pkl')


    (
        pd.DataFrame(scaler.transform(df.filter(features)), columns=features)
        .join(pd.DataFrame(ohe.transform(df.lccs_class.to_numpy().reshape(-1, 1)).toarray()))
        .join(df['smap_sm_d'])
    ).to_csv(f'dataset/nn/{filename}')


if __name__ == '__main__':
    year = 2021
    filename = f'sm_known_{year}_9.0km.csv'
    path = f'dataset/{filename}'
    df = pd.read_csv(path)
    nn_preprocess(df, year, filename)