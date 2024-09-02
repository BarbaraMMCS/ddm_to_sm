import subprocess
from datetime import datetime, timedelta
from download.config import BOUNDING_BOX, TEMPORAL_S, TEMPORAL_E, PATH_CYGNSS_FOLDER


def download_data(start_date, end_date, dest_folder, bbox):
    print(dest_folder)
    command = [
        'podaac-data-downloader',
        '-c', 'CYGNSS_L1_V3.2',
        '-d', dest_folder,
        '-sd', f'{start_date}T00:00:00Z',
        '-ed', f'{end_date}T23:59:59Z',
        '-e', '.nc',
        '-b', ','.join(map(str, bbox)),
        '--verbose',
    ]
    result = subprocess.run(command)
    print(result.stdout)
    print(result.stderr)


def split_time_range(start_date, end_date, chunk_size_days):
    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=chunk_size_days - 1), end_date)
        yield current_start, current_end
        current_start = current_end + timedelta(days=1)


def main(temp_s, temp_e, path_cygnss, bbox):
    temporal_s = datetime.strptime(temp_s, "%Y-%m-%d")
    temporal_e = datetime.strptime(temp_e, "%Y-%m-%d")
    for i, (chunk_start, chunk_end) in enumerate(split_time_range(temporal_s, temporal_e, chunk_size_days=180)):
        download_data(chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), path_cygnss, bbox)


if __name__ == '__main__':
    main(TEMPORAL_S, TEMPORAL_E, PATH_CYGNSS_FOLDER, BOUNDING_BOX)