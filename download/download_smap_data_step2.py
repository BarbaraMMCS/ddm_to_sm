import requests
from netrc import netrc
import os

from config import PATH_SMAP_FOLDER, SMAP_URLS_FILE


def get_auth():
    home = os.path.expanduser("~")
    netrc_path = os.path.join(home, '.netrc')
    auth_data = netrc(netrc_path).authenticators('urs.earthdata.nasa.gov')
    if not auth_data:
        raise FileNotFoundError("Check that $HOME/.netrc has credentials for 'urs.earthdata.nasa.gov'")
    return auth_data[0], auth_data[2]


def download_file(url, session):
    response = session.get(url, allow_redirects=True)
    if response.status_code == 200:
        filename = os.path.join(PATH_SMAP_FOLDER, os.path.basename(url))

        if os.path.exists(filename):
            print(f"{filename} already exists.")
            return

        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename} successfully.")
    else:
        print(f"Failed to download {url}: HTTP {response.status_code}")


def main():
    if not os.path.exists(PATH_SMAP_FOLDER):
        os.makedirs(PATH_SMAP_FOLDER)

    with open(SMAP_URLS_FILE, 'r') as f:
        urls = [line.strip() for line in f.readlines()]

    username, password = get_auth()
    session = requests.Session()
    session.auth = (username, password)

    for url in urls:
        download_file(url, session)
        break

    session.close()


if __name__ == '__main__':
    main()
