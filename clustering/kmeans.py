import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from random import sample

from PIL import ImageOps
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


def normalize_ddms(ddms):
    maxes = np.amax(ddms, axis=1).reshape(len(ddms), 1)
    mines = np.amin(ddms, axis=1).reshape(len(ddms), 1)
    return (ddms - mines) / (maxes - mines)  # normalized ddms

def get_ddm(dataset_path):
    data = np.load(dataset_path)
    ddms = data[:, data.shape[1] - (17 * 11):]
    return normalize_ddms(ddms)


def get_non_normalize_ddm(dataset_path):
    data = np.load(dataset_path)
    ddms = data[:, data.shape[1] - (17 * 11):]
    return ddms  # normalized ddm


def get_pca(ddms, pca_n_components):
    if pca_n_components is not None:
        pca = PCA(n_components=pca_n_components)
        pca.fit(ddms)
        return pca
    return None


def get_kmeans(n_clusters, ddms, pca):
    features = pca.transform(ddms) if pca is not None else ddms
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    return kmeans, features


def predict_cluster(kmeans, ddms, pca):
    if pca is not None:
        return kmeans.predict(pca.transform(ddms))
    else:
        return kmeans.predict(ddms)


def order_labels(labels, label_to_ordered_label):
    return [label_to_ordered_label[label] for label in labels]


def save_plot_clusters(ddms, ordered_labels, pca_2, filename, alpha=0.3):
    cmap = plt.get_cmap('viridis')
    n_labels = max(ordered_labels) + 1
    colors = [cmap(i / max(ordered_labels)) for i in range(n_labels)]
    ordered_labels = np.array(ordered_labels)

    x, y = zip(*pca_2.transform(ddms))
    x = np.array(x)
    y = np.array(y)

    fig, ax = plt.subplots()
    fig.set_size_inches(17, 15)
    for i, color in enumerate(colors):
        ax.scatter(x[ordered_labels == i], y[ordered_labels == i], c=[color], label=i, edgecolors='none', alpha=alpha)
    ax.legend()
    ax.grid(True)
    # legend1 = ax.legend(handles, list(range(max(ordered_labels) + 1)), loc="upper right")
    # ax.add_artist(legend1)
    fig.savefig(filename)
    plt.close(fig)


def resize_image_by_x_pixels(original_image, factor):
    original_width, original_height = original_image.size

    new_width = original_width * factor
    new_height = original_height * factor

    resized_image = Image.new('RGB', (new_width, new_height))

    original_pixels = original_image.load()
    resized_pixels = resized_image.load()

    for y in range(original_height):
        for x in range(original_width):
            pixel = original_pixels[x, y]
            for _y in range(factor * y, factor * y + factor):
                for _x in range(factor * x, factor * x + factor):
                    resized_pixels[_x, _y] = pixel
    return resized_image


def concatenate_images(images, direction='horizontal'):
    if direction == 'horizontal':
        total_width = sum(image.width for image in images)
        max_height = max(image.height for image in images)
        concatenated_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for image in images:
            concatenated_image.paste(image, (x_offset, 0))
            x_offset += image.width

    elif direction == 'vertical':
        max_width = max(image.width for image in images)
        total_height = sum(image.height for image in images)
        concatenated_image = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for image in images:
            concatenated_image.paste(image, (0, y_offset))
            y_offset += image.height

    # Save the concatenated image
    return concatenated_image


def plot_images(filename, ordered_labels, ddms, pca, order_label_to_label, kmeans, closests, row_per_label=20,
                nb_of_col=75, colormap='viridis', normalized=True):
    cm = plt.get_cmap(colormap)

    label_list_image = defaultdict(list)
    for i, label in enumerate(ordered_labels):
        if len(label_list_image[label]) >= 5 * row_per_label * nb_of_col:
            continue
        arr = ddms[i].reshape((17, 11))
        if normalized:
            im = arr
        else:
            _max = np.max(arr)
            _min = np.min(arr)
            im = (arr - _min) / (_max - _min)
        im = cm(im)
        im = np.uint8(im * 255)
        im = Image.fromarray(im)  # .resize((17*10,11*10),resample=Image.NEAREST)
        im = ImageOps.expand(im, border=1, fill='white')
        label_list_image[label].append(im)

    list_image = []

    for label in range(len(label_list_image)):
        n_images = row_per_label * nb_of_col
        max_n_images = len(label_list_image[label])
        selected_images = []
        while n_images > max_n_images:
            selected_images += list(sample(label_list_image[label], max_n_images))
            n_images -= max_n_images
        selected_images += list(sample(label_list_image[label], n_images))
        tmp_list = []
        for i in range(row_per_label):
            tmp_list.append(concatenate_images(selected_images[nb_of_col * i: nb_of_col * (i + 1)]))
        concat_image = concatenate_images(tmp_list, 'vertical')
        concat_image = ImageOps.expand(concat_image, border=1, fill='black')
        list_image.append(concat_image)
    samples_images = concatenate_images(list_image, 'vertical')

    if pca is None:
        centers_images = []
        for i in range(len(order_label_to_label)):
            center = kmeans.cluster_centers_[order_label_to_label[i]]
            arr = center.reshape((17, 11))
            if normalized:
                im = arr
            else:
                _max = np.max(arr)
                _min = np.min(arr)
                im = (arr - _min) / (_max - _min)
            im = cm(im)
            im = np.uint8(im * 255)
            im = Image.fromarray(im)
            im = resize_image_by_x_pixels(im, row_per_label)
            im = ImageOps.expand(im, border=row_per_label, fill='white')
            im = ImageOps.expand(im, border=1, fill='black')
            centers_images.append(im)
        centers_image = concatenate_images(centers_images, 'vertical')

    closest_images = []
    for i in range(len(order_label_to_label)):
        closest = ddms[closests[order_label_to_label[i]]]
        arr = closest.reshape((17, 11))
        if normalized:
            im = arr
        else:
            _max = np.max(arr)
            _min = np.min(arr)
            im = (arr - _min) / (_max - _min)
        im = cm(im)
        im = np.uint8(im * 255)
        im = Image.fromarray(im)
        im = resize_image_by_x_pixels(im, row_per_label)
        im = ImageOps.expand(im, border=row_per_label, fill='white')
        im = ImageOps.expand(im, border=1, fill='black')
        closest_images.append(im)
    closest_image = concatenate_images(closest_images, 'vertical')

    images = [centers_image, closest_image, samples_images] if pca is None else [closest_image, samples_images]
    concatenate_images(images).save(filename)


def main(train_dataset_path, test_dataset_path, pca_n_components=None, n_clusters=5, row_per_label=3, nb_of_col=75,
         colormap='viridis', normalize=True, folder='', model_folder='models/kmeans/'):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    train_ddms = get_ddm(train_dataset_path) if normalize else get_non_normalize_ddm(train_dataset_path)
    test_ddms = get_ddm(test_dataset_path) if normalize else get_non_normalize_ddm(test_dataset_path)

    pca_2 = get_pca(train_ddms, pca_n_components=2)
    pca = get_pca(train_ddms, pca_n_components)

    kmeans, train_features = get_kmeans(n_clusters, train_ddms, pca)
    test_features = pca.transform(test_ddms) if pca is not None else test_ddms

    train_labels = predict_cluster(kmeans, train_ddms, pca)
    test_labels = predict_cluster(kmeans, test_ddms, pca)

    train_closests, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, train_features)
    test_closests, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, test_features)
    label_to_ordered_label = {k: i for i, k in enumerate(np.argsort(pca_2.transform(train_ddms[train_closests])[:, 0]))}
    order_label_to_label = {v: k for k, v in label_to_ordered_label.items()}

    train_ordered_labels = order_labels(train_labels, label_to_ordered_label)
    test_ordered_labels = order_labels(test_labels, label_to_ordered_label)

    if pca_n_components is None and normalize:
        with open(Path.joinpath(Path(model_folder), f'kmean_{n_clusters}.pck'), 'wb') as out:
            pickle.dump(kmeans, out, protocol=pickle.HIGHEST_PROTOCOL)

        with open(Path.joinpath(Path(model_folder), f'kmean_{n_clusters}.json'), 'w') as out:
            json.dump({int(k): int(v) for k, v in label_to_ordered_label.items()}, out)

    save_plot_clusters(train_ddms, train_ordered_labels, pca_2,
                       Path.joinpath(Path(folder), f"pca_{pca_n_components}_n_clusters_{n_clusters}_train.png"), alpha=0.01)
    save_plot_clusters(test_ddms, test_ordered_labels, pca_2,
                       Path.joinpath(Path(folder), f"pca_{pca_n_components}_n_clusters_{n_clusters}_test.png"), alpha=0.01)

    plot_images(Path.joinpath(Path(folder), f"pca_{pca_n_components}_n_clusters_{n_clusters}_train_images.png"),
                train_ordered_labels, train_ddms, pca, order_label_to_label, kmeans, train_closests, row_per_label,
                nb_of_col, colormap, normalize)
    plot_images(Path.joinpath(Path(folder), f"pca_{pca_n_components}_n_clusters_{n_clusters}_test_images.png"),
                test_ordered_labels, test_ddms, pca, order_label_to_label, kmeans, test_closests, row_per_label,
                nb_of_col, colormap, normalize)

def main_more_days(train_dataset_paths, test_dataset_paths, pca_n_components=None, n_clusters=5, row_per_label=3, nb_of_col=75,
         colormap='viridis', normalize=True, folder='', model_folder='models/kmeans/'):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    train_ddms = np.concat([get_ddm(train_dataset_path) if normalize else get_non_normalize_ddm(train_dataset_path) for train_dataset_path in train_dataset_paths], axis =0)
    test_ddms = np.concat([get_ddm(test_dataset_path) if normalize else get_non_normalize_ddm(test_dataset_path)  for test_dataset_path in test_dataset_paths], axis =0)

    pca_2 = get_pca(train_ddms, pca_n_components=2)
    pca = get_pca(train_ddms, pca_n_components)

    kmeans, train_features = get_kmeans(n_clusters, train_ddms, pca)
    test_features = pca.transform(test_ddms) if pca is not None else test_ddms

    train_labels = predict_cluster(kmeans, train_ddms, pca)
    test_labels = predict_cluster(kmeans, test_ddms, pca)

    train_closests, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, train_features)
    test_closests, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, test_features)
    label_to_ordered_label = {k: i for i, k in enumerate(np.argsort(pca_2.transform(train_ddms[train_closests])[:, 0]))}
    order_label_to_label = {v: k for k, v in label_to_ordered_label.items()}

    train_ordered_labels = order_labels(train_labels, label_to_ordered_label)
    test_ordered_labels = order_labels(test_labels, label_to_ordered_label)
    if pca_n_components is None and normalize:
        with open(Path.joinpath(Path(model_folder), f'kmean_more_days_{n_clusters}.pck'), 'wb') as out:
            pickle.dump(kmeans, out, protocol=pickle.HIGHEST_PROTOCOL)

        with open(Path.joinpath(Path(model_folder), f'kmean_more_days_{n_clusters}.json'), 'w') as out:
            json.dump({int(k): int(v) for k, v in label_to_ordered_label.items()}, out)

    save_plot_clusters(train_ddms, train_ordered_labels, pca_2,
                       Path.joinpath(Path(folder), f"pca_{pca_n_components}_n_clusters_{n_clusters}_train.png"))
    save_plot_clusters(test_ddms, test_ordered_labels, pca_2,
                       Path.joinpath(Path(folder), f"pca_{pca_n_components}_n_clusters_{n_clusters}_test.png"))

    plot_images(Path.joinpath(Path(folder), f"pca_{pca_n_components}_n_clusters_{n_clusters}_train_images.png"),
                train_ordered_labels, train_ddms, pca, order_label_to_label, kmeans, train_closests, row_per_label,
                nb_of_col, colormap, normalize)
    plot_images(Path.joinpath(Path(folder), f"pca_{pca_n_components}_n_clusters_{n_clusters}_test_images.png"),
                test_ordered_labels, test_ddms, pca, order_label_to_label, kmeans, test_closests, row_per_label,
                nb_of_col, colormap, normalize)

if __name__ == "__main__":
    # for pca_n_components in [None, 2, 4, 8, 16, 32, 64]:
    #     for n_clusters in [2, 3, 5, 7, 10, 15, 20]:
    #         main('../data/train_label/20220101.npy', '../data/train_label/20220601.npy', pca_n_components, n_clusters, normalize=True, folder='../figures/pca_kmean/')
    # for pca_n_components in [None, 2, 10]:
    #     for n_clusters in [2, 3, 5, 7, 10, 15, 20]:
    #         main('../data/train_label/20220101.npy', '../data/train_label/20220601.npy', pca_n_components, n_clusters, normalize=False, folder='../figures/non_normalized_pca_kmean/')

    # for pca_n_components in [None, 2, 4, 8, 16, 32, 64]:
    #     for n_clusters in [2, 3, 5, 7, 10, 15, 20]:
    #         main_more_days([f'../data/train_label/2022{i:02d}{j}.npy' for i in range(1, 13) for j in ['01', '15']],[f'../data/train_label/2022{i:02d}{j}.npy' for i in range(1, 13) for j in ['08', '22']], pca_n_components, n_clusters, normalize=True, folder='../figures/pca_kmean_more_days/')
    for pca_n_components in [None, 2, 10]:
        for n_clusters in [2, 3, 5, 7, 10, 15, 20]:
            main_more_days([f'../data/train_label/2022{i:02d}{j}.npy' for i in range(1, 13) for j in ['01', '15']],[f'../data/train_label/2022{i:02d}{j}.npy' for i in range(1, 13) for j in ['08', '22']], pca_n_components, n_clusters, normalize=False, folder='../figures/non_normalized_pca_kmean_more_days/')


