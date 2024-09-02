import os
import pathlib
from datetime import datetime, timedelta

from download import download_cygnss_data
# from download.config import TEMPORAL_S, TEMPORAL_E
from preprocessing import cygnss
from preprocessing.config import PATH_TRAIN_WITH_LABEL_FOLDER, PATH_CYGNSS_FOLDER, BOUNDING_BOX, PATH_WATER_MAP_FILE, \
    CYNGNSS_FEATURES_NAMES, PATH_LAND_COVER_MAP_FILE, PATH_SMAP_FOLDER, PATH_TRAIN_WITHOUT_LABEL_FOLDER

if __name__ == '__main__':
    TEMPORAL_S = '2022-01-01'
    TEMPORAL_E = '2022-01-03'
    start_date = datetime.strptime(TEMPORAL_S, '%Y-%m-%d').date()
    n_days = (datetime.strptime(TEMPORAL_E, '%Y-%m-%d').date() - start_date).days + 1
    for i in range(n_days):
        day = start_date + timedelta(days=i)
        day_str = day.strftime('%Y-%m-%d')
        print(day_str)
        # if pathlib.Path(os.path.join(PATH_TRAIN_WITH_LABEL_FOLDER, f"{day.strftime('%Y%m%d')}.npy")).exists():
        #     continue

        download_cygnss_data.main(day_str, day_str, PATH_CYGNSS_FOLDER, BOUNDING_BOX)

        # cygnss.main(PATH_CYGNSS_FOLDER, PATH_WATER_MAP_FILE, BOUNDING_BOX, CYNGNSS_FEATURES_NAMES,
        #             PATH_LAND_COVER_MAP_FILE,
        #             PATH_SMAP_FOLDER, PATH_TRAIN_WITH_LABEL_FOLDER, PATH_TRAIN_WITHOUT_LABEL_FOLDER, delete_file=True)
