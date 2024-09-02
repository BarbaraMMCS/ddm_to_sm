import netCDF4
import numpy as np
import os

from preprocessing.config import WIDER_BOUNDING_BOX, PATH_LAND_COVER_FILE, PATH_LAND_COVER_FOLDER


def main(filename, bbox):
    landcover = netCDF4.Dataset(filename)
    min_lon, min_lat, max_lon, max_lat = bbox
    min_lon_idx = int((min_lon + 180) // (360 / len(landcover['lon'])))
    max_lon_idx = int((max_lon + 180) // (360 / len(landcover['lon'])))
    max_lat_idx = -int((min_lat - 90) // (180 / len(landcover['lat'])))
    min_lat_idx = -int((max_lat - 90) // (180 / len(landcover['lat'])))
    save_filename = os.path.join(PATH_LAND_COVER_FOLDER, f'lon_{min_lon}_{max_lon}_lat_{min_lat}_{max_lat}_{max_lon_idx-min_lon_idx}x{max_lat_idx-min_lat_idx}.npy')
    print(f"{save_filename} ->", (max_lat_idx-min_lat_idx, max_lon_idx-min_lon_idx))
    np.save(save_filename, np.array(landcover['lccs_class'][:, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx].reshape(max_lat_idx-min_lat_idx, max_lon_idx-min_lon_idx)))


if __name__ == '__main__':
    main(PATH_LAND_COVER_FILE, WIDER_BOUNDING_BOX)