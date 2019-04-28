import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys
import numpy as np
import pickle

DATA_EXTRACT_PATH = './data_set/'
CIFAR_10_DATA_DIR = 'cifar_10/'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
NUM_CLASSES = 10


def get_train_data():
    maybe_download_and_extract()
    batch_data = [get_batch(i) for i in range(5)]
    return prepare_data(batch_data)


def get_test_data():
    maybe_download_and_extract()
    with open(CIFAR_10_DATA_DIR + '/test_batch', 'rb') as file:
        datadict = pickle.load(file, encoding='latin1')
        return prepare_data([[datadict['data'], datadict['labels']]])


def prepare_data(data_and_labels):
    data = np.array([x for X, _ in data_and_labels for x in X], dtype=float)
    labels = np.array([y for _, Y in data_and_labels for y in Y])
    return normalize_data(data), create_one_hot_labels(labels)


def normalize_data(data):
    data /= 255.0
    data = data.reshape([-1, 3, 32, 32])
    data = data.transpose([0, 2, 3, 1])
    data = data.reshape(-1, 32 * 32 * 3)
    return data


def create_one_hot_labels(labels):
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, NUM_CLASSES))
    one_hot[np.arange(num_labels), labels] = 1
    return one_hot


def get_batch(num_batch):
    with open(CIFAR_10_DATA_DIR + '/data_batch_' + str(num_batch + 1), 'rb') as file:
        datadict = pickle.load(file, encoding='latin1')
        return [datadict['data'], datadict['labels']]


def maybe_download_and_extract():
    def _print_download_progress(count, block_size, total_size):
        pct_complete = float(count * block_size) / total_size
        msg = "\r- Download progress: {0:.1%}".format(pct_complete)
        sys.stdout.write(msg)
        sys.stdout.flush()

    if not os.path.exists(DATA_EXTRACT_PATH):
        os.makedirs(DATA_EXTRACT_PATH)
        file_path = os.path.join(DATA_EXTRACT_PATH, DATA_URL.split('/')[-1])
        file_path, _ = urlretrieve(url=DATA_URL, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(DATA_EXTRACT_PATH)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(DATA_EXTRACT_PATH)
        print("Done.")

        os.rename(DATA_EXTRACT_PATH + "./cifar-10-batches-py", CIFAR_10_DATA_DIR)
        os.remove(file_path)
