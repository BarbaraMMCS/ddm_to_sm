import os
# WATER
# -----------------
PATH_PY_FILE = '/scratch/users/bsymeon/cygnss_ddm_to_sm/download/download_water_data.py'
# PATH_PY_FILE = '/Users/barbara.symeon/PycharmProjects/cygnss_ddm_to_sm/download/download_water_data.py'

DATASET_NAME = 'seasonality'
# -----------------------------------------------------------------------------------------
# CYGNSS & SMAP & WATER
# ----------------------
BOUNDING_BOX = 112.9211, -43.7405, 153.6388, -10.9332  # Australia
# -----------------------------------------------------------------------------------------
# CYGNSS & SMAP
# --------------
TEMPORAL_S = "2023-10-01"
TEMPORAL_E = "2023-12-31"

TEMPORAL_S = "2021-12-31"
TEMPORAL_E = "2024-01-01"



# FOLDER = '/scratch/users/bsymeon/cygnss_ddm_to_sm/data/'
FOLDER = '../data'
# -----------------------------------------------------------------------------------------
PATH_WATER_FOLDER = os.path.join(FOLDER, 'water')
PATH_SMAP_FOLDER = os.path.join(FOLDER, 'SPL3SMP_E')
PATH_LAND_COVER_FOLDER = os.path.join(FOLDER, 'landcover')
PATH_CYGNSS_FOLDER = os.path.join(FOLDER, 'data_cygnss_l1_v3.2')

SMAP_URLS_FILE = 'download/smap_urls.txt'
