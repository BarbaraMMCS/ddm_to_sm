import numpy as np
from datetime import datetime, timezone
import h5py
import glob
import re
import os

from preprocessing.config import PATH_SMAP_FOLDER, BOUNDING_BOX, SMAP_FEATURES_NAMES

pattern = (
    r'.*SMAP_L3_SM_P_E_(?P<day>\d{8})_'
    r'.*\.h5'
)

# Compile the regex
SMAP_DATE_REGEX = re.compile(pattern)


def get_filter_mask(smap, bbox, suffix=''):
    flag_exclude = (1 + 2 + 4)
    min_lon, min_lat, max_lon, max_lat = bbox
    flag_mask = ((np.bitwise_and(smap[f'retrieval_qual_flag{suffix}'][:, :], flag_exclude) == 0)).flatten()

    lon_mask = ((smap[f'longitude{suffix}'][:, :] >= min_lon) & (smap[f'longitude{suffix}'][:, :] < max_lon)).flatten()
    lat_mask = ((smap[f'latitude{suffix}'][:, :] >= min_lat) & (smap[f'latitude{suffix}'][:, :] < max_lat)).flatten()

    return flag_mask & lon_mask & lat_mask


def get_features(smap, filter_mask, features_names, d_timestamp, suffix=''):
    features = []
    n_selected_sample = filter_mask.sum()
    for feature_name in features_names:
        features.append(smap[feature_name + suffix][:, :].flatten()[filter_mask].reshape(n_selected_sample, 1))
    features.append(
        np.repeat(
            d_timestamp,
            n_selected_sample
        ).reshape(n_selected_sample, 1)
    )
    return np.concatenate(features, axis=1)


def process_on_file(filename, bbox, features_names, day):
    smap_file = h5py.File(filename)
    features = []
    for period in ['PM', 'AM']:
        smap = smap_file[f'Soil_Moisture_Retrieval_Data_{period}']
        suffix = '_pm' if period == 'PM' else ''
        d_timestamp = datetime.strptime(f"{day}T{'18' if period == 'PM' else '06'}", "%Y%m%dT%H").replace(tzinfo=timezone.utc).timestamp()
        filter_mask = get_filter_mask(smap, bbox, suffix)
        features.append(get_features(smap, filter_mask, features_names, d_timestamp, suffix))

    return np.concatenate(features, axis=0)


def main(folder, bbox, features_names, delete_file=False):
    prev_day = None
    current_day_data = []
    for filename in sorted(glob.glob(os.path.join(folder, '*.h5'))):
        print(filename)
        match = SMAP_DATE_REGEX.match(filename)
        if match:
            day = match.group('day')
        else:
            raise Exception('filename not correct')
        if prev_day is None:
            prev_day = day
        elif prev_day != day and prev_day is not None:
            np.save(os.path.join(folder, f'{prev_day}.npy'), np.concatenate(current_day_data, axis=0), allow_pickle=False)
            current_day_data = []
            prev_day = day
        current_day_data.append(process_on_file(filename, bbox, features_names, day))
        if delete_file:
            os.remove(filename)
    np.save(os.path.join(folder, f'{prev_day}.npy'), np.concatenate(current_day_data, axis=0), allow_pickle=False)


if __name__ == '__main__':
    main(PATH_SMAP_FOLDER, BOUNDING_BOX, SMAP_FEATURES_NAMES)

