import datetime
import glob

import netCDF4
import numpy as np
import re
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import os
from pathlib import Path

from preprocessing.config import BOUNDING_BOX, PATH_WATER_MAP_FILE, PATH_LAND_COVER_MAP_FILE, PATH_SMAP_FOLDER, \
    CYNGNSS_FEATURES_NAMES, PATH_CYGNSS_FOLDER, PATH_TRAIN_WITHOUT_LABEL_FOLDER, PATH_TRAIN_WITH_LABEL_FOLDER, \
    SMAP_FEATURES_HEADER

pattern = (
    r'.*lon_(?P<min_lon>-?\d+)_'
    r'(?P<max_lon>-?\d+)_'
    r'lat_(?P<min_lat>-?\d+)_'
    r'(?P<max_lat>-?\d+)_'
    r'(?P<precision_lon>-?\d+)x'
    r'(?P<precision_lat>-?\d+).npy'
)

# Compile the regex
MAP_REGEX = re.compile(pattern)

pattern = (
    r'.*cyg[0-9]{2}\.ddmi\.s(?P<day>\d{8})-'
    r'.*\.nc'
)

# Compile the regex
CYGNSS_DATE_REGEX = re.compile(pattern)

WATER_MAPS = {}
LANDCOVER_MAPS = {}
SMAP_DATA = {}


def get_water_map(water_map_filename):
    if water_map_filename not in WATER_MAPS:
        WATER_MAPS[water_map_filename] = np.load(water_map_filename).astype(bool)
    return WATER_MAPS[water_map_filename]


def get_landcover_map(land_cover_map_filename):
    if land_cover_map_filename not in LANDCOVER_MAPS:
        LANDCOVER_MAPS[land_cover_map_filename] = np.load(land_cover_map_filename)
    return LANDCOVER_MAPS[land_cover_map_filename]


def extract_min_step_precision_from_filename(filename):
    match = MAP_REGEX.match(filename)
    if match:
        min_lon = int(match.group('min_lon'))
        max_lon = int(match.group('max_lon'))
        min_lat = int(match.group('min_lat'))
        max_lat = int(match.group('max_lat'))
        precision_lon = int(match.group('precision_lon'))
        precision_lat = int(match.group('precision_lat'))
        step_lon = (max_lon - min_lon) / precision_lon
        step_lat = (max_lat - min_lat) / precision_lat
        return min_lon, min_lat, step_lon, step_lat, precision_lon, precision_lat
    else:
        raise Exception(f"{filename} does not match the right format")


def get_water_mask(water_map_filename, nc):
    min_lon, min_lat, step_lon, step_lat, precision_lon, precision_lat = extract_min_step_precision_from_filename(
        water_map_filename)

    water_map = get_water_map(water_map_filename)

    lat_index = (precision_lat - 1 - ((nc['sp_lat'][:, :] - min_lat) // step_lat)).astype(int).flatten()
    lat_index[(lat_index < 0) | (lat_index >= precision_lat)] = 0
    lon_index = ((nc['sp_lon'][:, :] - min_lon) // step_lon).astype(int).flatten()
    lon_index[(lon_index < 0) | (lon_index >= precision_lon)] = 0
    return water_map[lat_index, lon_index]


def get_landcover_feature(land_cover_map_filename, nc):
    min_lon, min_lat, step_lon, step_lat, precision_lon, precision_lat = extract_min_step_precision_from_filename(
        land_cover_map_filename)
    landcover_map = get_landcover_map(land_cover_map_filename)
    lat_index = (precision_lat - 1 - ((nc['sp_lat'][:, :] - min_lat) // step_lat)).astype(int).flatten()
    lat_index[(lat_index < 0) | (lat_index >= precision_lat)] = 0
    lon_index = ((nc['sp_lon'][:, :] - min_lon) // step_lon).astype(int).flatten()
    lon_index[(lon_index < 0) | (lon_index >= precision_lon)] = 0
    return landcover_map[lat_index, lon_index]


def get_filter_mask(nc, water_map_filename, bbox, landcover_map):
    min_lon, min_lat, max_lon, max_lat = bbox
    flag_include = 1024
    flag_exclude = (2 + 8 + 16 + 128 + 32768 + 65536)

    flag_mask = ((np.bitwise_and(nc['quality_flags'][:, :], flag_include) == flag_include) & (
            np.bitwise_and(nc['quality_flags'][:, :], flag_exclude) == 0)).flatten()

    water_mask = get_water_mask(water_map_filename, nc)
    lon_mask = ((nc['sp_lon'][:, :] >= min_lon) & (nc['sp_lon'][:, :] < max_lon)).flatten()
    lat_mask = ((nc['sp_lat'][:, :] >= min_lat) & (nc['sp_lat'][:, :] < max_lat)).flatten()
    ddm_snr_mask = (nc['ddm_snr'][:, :] >= 0.5).flatten()
    sp_inc_angle_mask = (nc['sp_inc_angle'][:, :] < 65).flatten()
    landcover_mask = landcover_map != 210
    return ~water_mask & lon_mask & lat_mask & flag_mask & ddm_snr_mask & sp_inc_angle_mask & landcover_mask


def compute_peak_power(power_analog, brcs_ddm_peak_bin_delay_row, brcs_ddm_peak_bin_dopp_col):
    sample_size, delay, dopp = power_analog.shape

    # Create a padded version of the matrix to handle boundaries
    padded_matrix = np.pad(power_analog, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    # Offsets for the neighborhood (including the center point)
    offsets = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 0], [0, 1],
        [1, -1], [1, 0], [1, 1]
    ])

    delay_indices_fill_0 = np.where(brcs_ddm_peak_bin_delay_row.mask, 0, brcs_ddm_peak_bin_delay_row)
    dopp_indices_fill_0 = np.where(brcs_ddm_peak_bin_dopp_col.mask, 0, brcs_ddm_peak_bin_dopp_col)

    # Calculate padded indices for y and z
    padded_delay_indices = delay_indices_fill_0[:, None] + offsets[:, 0] + 1
    padded_dopp_indices = dopp_indices_fill_0[:, None] + offsets[:, 1] + 1

    # Gather neighborhood values for each (y, z) coordinate for the corresponding x slice
    neighbors = padded_matrix[
        np.arange(sample_size)[:, None, None], padded_delay_indices[:, :, None], padded_dopp_indices[:, :,
                                                                                 None]]  # Shape: (x_dim, 9, 9)

    # Compute the average of the neighbors for each x slice
    result = neighbors.mean(axis=(1, 2))  # Shape: (x_dim,)
    result = np.where(brcs_ddm_peak_bin_delay_row.mask | brcs_ddm_peak_bin_dopp_col.mask, np.nan, result)
    return result


def surface_reflectivity(PtrGt, Gr, Rsr, Rts, Prrl):
    """
    PtrGt = gps_eirp, Gr = sp_rx_gain, Rsr = rx_to_sp_range, Rts = tx_to_sp_range, Prrl = peak_power.
    pp unit [dB]
    """
    l = 299792458 / (1575.42 * 10 ** 6)
    return 10 * np.log10(Prrl) - 10 * np.log10(PtrGt) - Gr + 20 * np.log10(Rsr + Rts) - 20 * np.log10(
        l) + 20 * np.log10(4 * np.pi)


def round_timestamp_to_nearest_6am_6pm(timestamp):
    timestamp_day = (timestamp) // (24 * 60 * 60)
    timestamp_day_seconds = (timestamp) % (24 * 60 * 60)
    six_am = 6 * 3600
    six_pm = 18 * 3600
    to_six_am = np.abs(timestamp_day_seconds - six_am)
    to_six_pm = np.abs(timestamp_day_seconds - six_pm)
    nearest_time_mask = to_six_am < to_six_pm
    return np.where(nearest_time_mask, timestamp_day * (24 * 60 * 60) + six_am, timestamp_day * (24 * 60 * 60) + six_pm)


def load_smap_gdf(day, smap_folder):
    if day not in SMAP_DATA:
        try:
            day_datetime = datetime.strptime(day, "%Y%m%d")
            dates_to_load = [(day_datetime - timedelta(days=1)).strftime("%Y%m%d"), day,
                             (day_datetime + timedelta(days=1)).strftime("%Y%m%d")]
            dfs = []
            for date in dates_to_load:
                try:
                    dfs.append(pd.DataFrame(np.load(os.path.join(smap_folder, f'{date}.npy')), columns=SMAP_FEATURES_HEADER))
                except Exception:
                    pass

            df = pd.concat(dfs)
            SMAP_DATA[day] = gpd.GeoDataFrame(
                df.filter(['soil_moisture', 'vegetation_water_content', 'timestamp_lst']),
                geometry=gpd.points_from_xy(df.longitude, df.latitude)
            ).set_crs(4326).to_crs(6933)
        except Exception:
            SMAP_DATA[day] = None
    return SMAP_DATA[day]


def get_smap_features(lat, lon, rounded_timestamps, day, smap_folder):
    smap_gdf = load_smap_gdf(day, smap_folder)
    if smap_gdf is None:
        return np.array([np.nan] * len(lat)), np.array([np.nan] * len(lat))

    cygnss_gdf = gpd.GeoDataFrame(
        {'rounded_timestamps': rounded_timestamps}, geometry=gpd.points_from_xy(lon, lat)
    ).set_crs(4326).to_crs(6933)

    merged_gdf = (
        cygnss_gdf.sjoin_nearest(smap_gdf, how='left', max_distance=9000)
        .reset_index()
        .eval("time_diff = abs(timestamp_lst - rounded_timestamps)")
        .sort_values('time_diff')
        .groupby('index')
        .first()
        .sort_values('index')
    )
    # H12_in_sec = 60 * 60 * 12
    H12_in_sec = 0
    return np.where(merged_gdf.time_diff > H12_in_sec, np.nan, merged_gdf.soil_moisture), np.where(
        merged_gdf.time_diff > H12_in_sec, np.nan, merged_gdf.vegetation_water_content)



def get_features(nc, filter_mask, simple_features, landcover, day, smap_folder):
    n_sample = len(filter_mask)
    n_selected_sample = filter_mask.sum()
    second_in_a_day = 60 * 60 * 24
    features = []

    sp_lon = nc['sp_lon'][:, :].flatten()[filter_mask]
    sp_lat = nc['sp_lat'][:, :].flatten()[filter_mask]

    brcs_ddm_peak_bin_delay_row = nc['brcs_ddm_peak_bin_delay_row'][:, :].flatten()[filter_mask]
    brcs_ddm_peak_bin_dopp_col = nc['brcs_ddm_peak_bin_dopp_col'][:, :].flatten()[filter_mask]
    power_analog = nc['power_analog'][:, :, :, :].reshape(n_sample, 17, 11)[filter_mask, :, :]
    peak_power = compute_peak_power(power_analog, brcs_ddm_peak_bin_delay_row, brcs_ddm_peak_bin_dopp_col)
    timestamp = np.repeat(
        nc['ddm_timestamp_gps_week'][:] * second_in_a_day * 7 + nc['ddm_timestamp_gps_sec'][:] + 315961200 - 18,
        4)[filter_mask]
    timestamp_lst = timestamp + sp_lon * 12 * 60 * 60 / 180
    rounded_timestamps = round_timestamp_to_nearest_6am_6pm(timestamp_lst)

    soil_moisture, vegetation_water_content = get_smap_features(sp_lat, sp_lon, rounded_timestamps, day, smap_folder)

    cos_yearly = np.cos(2 * np.pi * (timestamp_lst % (second_in_a_day * 365.25) / (second_in_a_day * 365.25))).reshape(
        n_selected_sample, 1)
    sin_yearly = np.sin(2 * np.pi * (timestamp_lst % (second_in_a_day * 365.25) / (second_in_a_day * 365.25))).reshape(
        n_selected_sample, 1)
    cos_daily = np.cos(2 * np.pi * (timestamp_lst % (second_in_a_day) / (second_in_a_day))).reshape(n_selected_sample,
                                                                                                    1)
    sin_daily = np.sin(2 * np.pi * (timestamp_lst % (second_in_a_day) / (second_in_a_day))).reshape(n_selected_sample,
                                                                                                    1)

    reflectivity = surface_reflectivity(
        nc['gps_eirp'][:, :].flatten()[filter_mask],
        nc['sp_rx_gain'][:, :].flatten()[filter_mask],
        nc['rx_to_sp_range'][:, :].flatten()[filter_mask],
        nc['tx_to_sp_range'][:, :].flatten()[filter_mask],
        peak_power
    ).reshape(n_selected_sample, 1)

    features += [soil_moisture.reshape(n_selected_sample, 1),
                 vegetation_water_content.reshape(n_selected_sample, 1),
                 landcover[filter_mask].reshape(n_selected_sample, 1), reflectivity,
                 peak_power.reshape(n_selected_sample, 1),
                 cos_yearly, sin_yearly, cos_daily, sin_daily,
                 timestamp_lst.reshape(n_selected_sample, 1),
                 rounded_timestamps.reshape(n_selected_sample, 1),
                 ]
    for feature_name in simple_features:
        features.append(nc[feature_name][:, :].flatten()[filter_mask].reshape(n_selected_sample, 1))
    features.append(power_analog.reshape(n_selected_sample, 17 * 11))
    return np.concatenate(features, axis=1)

def process_one_file(nc_file, water_map_filename, bbox, simple_features, land_cover_map_filename, smap_folder, day):

    nc = netCDF4.Dataset(nc_file)
    landcover = get_landcover_feature(land_cover_map_filename, nc)
    filter_mask = get_filter_mask(nc, water_map_filename, bbox, landcover)
    features = get_features(nc, filter_mask, simple_features, landcover, day, smap_folder)
    nc.close()
    return features


def extract_day_from_nc_filename(nc_filename):
    match = CYGNSS_DATE_REGEX.match(nc_filename)
    if match:
        day = match.group('day')
    else:
        raise Exception('filename not correct')
    return day


def save_cygnss_features(train_with_label_folder, train_without_label_folder, filename, features):
    np.save(
        os.path.join(train_with_label_folder, filename),
        np.array(features[~np.isnan(features[:, 0])]),
        allow_pickle=False
    )
    np.save(
        os.path.join(train_without_label_folder, filename),
        np.array(features[np.isnan(features[:, 0])]),
        allow_pickle=False
    )


def main(cygnss_folder, water_map_filename, bbox, simple_features, land_cover_map_filename, smap_folder,
         train_with_label_folder, train_without_label_folder, delete_file=False):
    Path(train_with_label_folder).mkdir(parents=True, exist_ok=True)
    Path(train_without_label_folder).mkdir(parents=True, exist_ok=True)

    prev_day = None
    current_day_data = []

    for filename in sorted(glob.glob(os.path.join(cygnss_folder, '*.nc')), key=extract_day_from_nc_filename):
        print(filename)
        day = extract_day_from_nc_filename(filename)
        if prev_day is None:
            prev_day = day
        elif prev_day != day and prev_day is not None:
            save_cygnss_features(train_with_label_folder, train_without_label_folder, f'{prev_day}.npy', np.concatenate(current_day_data, axis=0))
            current_day_data = []
            del SMAP_DATA[prev_day]
            prev_day = day
        current_day_data.append(process_one_file(filename, water_map_filename, bbox, simple_features, land_cover_map_filename,
                                    smap_folder, day))
        if delete_file:
            os.remove(filename)
    save_cygnss_features(train_with_label_folder, train_without_label_folder, f'{day}.npy', np.concatenate(current_day_data, axis=0))
    del SMAP_DATA[day]


if __name__ == '__main__':
    process_one_file('../data/data_cygnss_l1_v3.2/cyg04.ddmi.s20220823-000000-e20220823-235959.l1.power-brcs.a32.d33.nc',
                     PATH_WATER_MAP_FILE, BOUNDING_BOX, CYNGNSS_FEATURES_NAMES, PATH_LAND_COVER_MAP_FILE,
         PATH_SMAP_FOLDER, '20220823')
