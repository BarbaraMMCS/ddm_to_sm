import urllib.request
import os
from config import PATH_WATER_FOLDER, DATASET_NAME, BOUNDING_BOX


def main(destination_folder, dataset_name, bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    if destination_folder[-1:] != "/":
        destination_folder = destination_folder + "/"

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    longs = [str(lon * -1) + "W" if lon < 0 else str(lon) + "E" for lon in
             range(int((min_lon // 10) * 10), int((max_lon // 10) * 10 + 1), 10)]
    lats = [str(lat * -1) + "S" if lat < 0 else str(lat) + "N" for lat in
            range(int((min_lat // 10) * 10 + 10), int((max_lat // 10) * 10 + 11), 10)]

    file_count = len(longs) * len(lats)
    counter = 1
    for lng in longs:
        for lat in lats:
            filename = dataset_name + "_" + str(lng) + "_" + str(lat) + "v1_4_2021.tif"
            if os.path.exists(destination_folder + filename):
                print(destination_folder + filename + " already exists - skipping")
            else:
                url = "http://storage.googleapis.com/global-surface-water/downloads2021/" + dataset_name + "/" + filename
                print(url)
                code = urllib.request.urlopen(url).getcode()
                if code != 404:
                    print("Downloading " + url + " (" + str(counter) + "/" + str(file_count) + ")")
                    urllib.request.urlretrieve(url, destination_folder + filename)
                else:
                    print(url + " not found")
            counter += 1


if __name__ == "__main__":
    main(PATH_WATER_FOLDER, DATASET_NAME, BOUNDING_BOX)
