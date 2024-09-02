import os

from preprocessing.config import CYNGNSS_FEATURES_HEADER, PATH_CYGNSS_HEADER_FILE


def main(filename, header):
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            file.write(header)


if __name__ == '__main__':
    main(PATH_CYGNSS_HEADER_FILE, CYNGNSS_FEATURES_HEADER)
