import earthaccess

from download.config import BOUNDING_BOX, TEMPORAL_S, TEMPORAL_E, PATH_SMAP_FOLDER

if __name__ == '__main__':
    earthaccess.login()

    results = earthaccess.search_data(
        short_name="SPL3SMP_E",
        bounding_box=BOUNDING_BOX,
        temporal=(TEMPORAL_S, TEMPORAL_E)
    )

    earthaccess.download(results, PATH_SMAP_FOLDER)
