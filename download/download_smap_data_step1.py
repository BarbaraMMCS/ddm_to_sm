import earthaccess # need python >3.8
from download.config import TEMPORAL_S, TEMPORAL_E, BOUNDING_BOX

if __name__ == '__main__':

    results = earthaccess.search_data(
        short_name="SPL3SMP_E",
        bounding_box=BOUNDING_BOX,
        temporal=(TEMPORAL_S, TEMPORAL_E)
    )

    with open('smap_urls.txt', 'w') as f:
        for g in results:
            for url in g.data_links(access='on_prem'):
                f.write(url + '\n')


