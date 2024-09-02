import os
import pathlib

import earthaccess

from download.config import BOUNDING_BOX, TEMPORAL_S, TEMPORAL_E, PATH_SMAP_FOLDER
from datetime import datetime, timedelta

from preprocessing import smap
from preprocessing.config import SMAP_FEATURES_NAMES

if __name__ == '__main__':
    earthaccess.login()
    start_date = datetime.strptime('2022-01-01', '%Y-%m-%d').date()
    n_days = (datetime.strptime('2022-01-14', '%Y-%m-%d').date() - start_date).days + 1
    for i in range(n_days):
        day = start_date + timedelta(days=i)
        day_str = day.strftime('%Y-%m-%d')
        print(day_str)
        # if pathlib.Path(os.path.join(PATH_SMAP_FOLDER, f"{day.strftime('%Y%m%d')}.npy")).exists():
        #     continue

        results = earthaccess.search_data(
            short_name="SPL3SMP_E",
            bounding_box=BOUNDING_BOX,
            temporal=(day_str, day_str)
        )
        if len(results) > 0:
            earthaccess.download(results, PATH_SMAP_FOLDER)

            # smap.main(PATH_SMAP_FOLDER, BOUNDING_BOX, SMAP_FEATURES_NAMES, delete_file=True)

