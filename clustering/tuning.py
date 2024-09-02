import numpy as np

from clustering.kmeans import get_ddm, get_non_normalize_ddm, get_pca, get_kmeans
import matplotlib.pyplot as plt


def score(kmeans, ddms, pca):
    if pca is not None:
        return kmeans.score(pca.transform(ddms))
    else:
        return kmeans.score(ddms)


def train_distortion(train_ddms, pca, n_clusters):
    kmeans, _ = get_kmeans(n_clusters, train_ddms, pca)
    return kmeans.inertia_


def plot(ks, distortion, filename):
    # Example data

    # Plotting
    plt.plot(ks, distortion, marker='o')

    # Labels and title
    plt.xlabel('k')
    plt.ylabel('distortion')
    plt.title('Distortion Score Elbow vs Number of Cluster (K-means)')

    # Show plot
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    for normalize in [True, False]:
        for pca_n_components in [None, 2, 8, 64]:
            train_ddms = np.concat(
                [get_ddm(train_dataset_path) if normalize else get_non_normalize_ddm(train_dataset_path) for train_dataset_path
                 in [f'../data/train_label/2022{i:02d}{j}.npy' for i in range(1, 13) for j in ['01','15']]], axis=0)

            pca = get_pca(train_ddms, pca_n_components)
            Ks = range(1, 20)

            distortions = [train_distortion(train_ddms, pca, n_clusters) for n_clusters in Ks]
            plot(Ks, distortions, f'../figures/elbow/normalized_{normalize}_pca_{pca_n_components}.png')
