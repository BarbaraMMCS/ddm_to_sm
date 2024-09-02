from typing import Tuple
from PIL import Image
import numpy as np
from config import BOUNDING_BOX, WATER_TILE_PRECISION, PATH_WATER_FOLDER
import os

Image.MAX_IMAGE_PIXELS = 40000 * 40000


def process_one_file(filename: str, precision: Tuple[int, int], lat: int, lon: int):
    with Image.open(filename) as water_tif:
        values = np.array(water_tif.resize(precision))
        values[values >= 1] = 1
        return values


def main(precision: Tuple[int, int], bbox: Tuple[float, float, float, float], folder: str):
    min_lon, min_lat, max_lon, max_lat = bbox
    longs = [lon for lon in range(int((min_lon // 10) * 10), int((max_lon // 10) * 10 + 1), 10)]
    lats = [lat for lat in range(int((min_lat // 10) * 10 + 10), int((max_lat // 10) * 10 + 11), 10)]
    masks = []
    for lon in longs:
        masks_lat = []
        for lat in lats:
            filename = "seasonality_" + (str(lon * -1) + "W" if lon < 0 else str(lon) + "E") + "_" + (
                str(lat * -1) + "S" if lat < 0 else str(lat) + "N") + "v1_4_2021.tif"
            masks_lat.insert(0, process_one_file(os.path.join(folder, filename), precision, lat, lon))
        masks.append(np.concatenate(masks_lat, axis=0))
    mask = np.concatenate(masks, axis=1)
    export_filename = f"lon_{longs[0]}_{longs[-1] + 10}_lat_{lats[0] - 10}_{lats[-1]}_{precision[1] * len(longs)}x{precision[0] * len(lats)}.npy"
    print(f"{export_filename} ->", mask.shape)
    np.save(os.path.join(folder, export_filename), mask, allow_pickle=False)


if __name__ == '__main__':
    main(WATER_TILE_PRECISION, BOUNDING_BOX, PATH_WATER_FOLDER)
