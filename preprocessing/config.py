import os

# WATER
# -----------------
# PATH_WATER_FOLDER = '/Users/barbara.symeon/PycharmProjects/cygnss_ddm_to_sm/data/water'

WATER_TILE_PRECISION = 4000, 4000
# -----------------------------------------------------------------------------------------
# CYGNSS & SMAP & WATER
# ----------------------
BOUNDING_BOX = 112.9211, -43.7405, 153.6388, -10.9332  # Australia
WIDER_BOUNDING_BOX = 110, -50, 160, -10
# -----------------------------------------------------------------------------------------
# CYGNSS & SMAP
# --------------
# FOLDER = '/scratch/users/bsymeon/cygnss_ddm_to_sm/data/'
# SAVE_FOLDER = '/home/users/bsymeon/cygnss_ddm_to_sm/data/'
FOLDER = '../data/'
SAVE_FOLDER = '../data/'

# -----------------------------------------------------------------------------------------
PATH_WATER_FOLDER = os.path.join(FOLDER, 'water')
PATH_SMAP_FOLDER = os.path.join(FOLDER, 'SPL3SMP_E')
PATH_LAND_COVER_FOLDER = os.path.join(FOLDER, 'landcover')
PATH_CYGNSS_FOLDER = os.path.join(FOLDER, 'data_cygnss_l1_v3.2')
PATH_TRAIN_WITH_LABEL_FOLDER = os.path.join(SAVE_FOLDER, 'train_label')
PATH_TRAIN_WITHOUT_LABEL_FOLDER = os.path.join(SAVE_FOLDER, 'train_no_label')

PATH_LAND_COVER_FILE = os.path.join(PATH_LAND_COVER_FOLDER, 'C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc')
PATH_WATER_MAP_FILE = os.path.join(PATH_WATER_FOLDER, 'lon_110_160_lat_-50_-10_20000x16000.npy')
PATH_LAND_COVER_MAP_FILE = os.path.join(PATH_LAND_COVER_FOLDER, 'lon_110_160_lat_-50_-10_18000x14400.npy')
PATH_CYGNSS_HEADER_FILE = os.path.join(PATH_TRAIN_WITH_LABEL_FOLDER, 'columns_names.txt')

SMAP_FEATURES_NAMES = ['latitude', 'longitude', 'soil_moisture', 'vegetation_water_content']
SMAP_FEATURES_HEADER = SMAP_FEATURES_NAMES + ['timestamp_lst']

CYNGNSS_FEATURES_NAMES = ['sp_lat',
                          'sp_lon',
                          'ddm_snr',
                          'sp_inc_angle',
                            'gps_eirp',
                            'sp_rx_gain',
                            'rx_to_sp_range',
                            'tx_to_sp_range',
                          ]
_ddm_names = [f'delay{delay:02}_dopp{dopp:02}' for delay in range(17) for dopp in range(11)]

CYNGNSS_FEATURES_HEADER = ','.join(
    ['soil_moisture', 'vegetation_water_content', 'landcover', 'reflectivity', 'peak_power',
     'cos_yearly', 'sin_yearly', 'cos_daily', 'sin_daily', 'timestamp_lst', 'rounded_timestamps'
     ] + CYNGNSS_FEATURES_NAMES + _ddm_names)
