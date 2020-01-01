import os
import gzip
import struct
from urllib.request import urlretrieve
from PIL import Image
import numpy as np


class MnistHelper:
    URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAINING_SET_IMAGES = 'train-images-idx3-ubyte'
    TRAINING_SET_LABELS = 'train-labels-idx1-ubyte'
    TEST_SET_IMAGES = 't10k-images-idx3-ubyte'
    TEST_SET_LABELS = 't10k-labels-idx1-ubyte'

    def __init__(self, local_path) -> None:
        if not os.path.isdir(local_path):
            os.mkdir(local_path)
        self.local_path = local_path
        self.__retrieve_data()

    # download files if not already there
    def __retrieve_data(self) -> None:
        for file in [self.TRAINING_SET_IMAGES,
                     self.TRAINING_SET_LABELS,
                     self.TEST_SET_IMAGES,
                     self.TEST_SET_LABELS]:
            gz_path = os.path.join(self.local_path, file + '.gz')
            if not os.path.isfile(gz_path):
                urlretrieve(self.URL + file + '.gz', gz_path)

    #  ungzip & read data
    def load_data(self, kind='train', one_hot=True, scaling_features=True):

        if kind == 'train':
            labels_file = self.TRAINING_SET_LABELS
            images_file = self.TRAINING_SET_IMAGES
        else:
            labels_file = self.TEST_SET_LABELS
            images_file = self.TEST_SET_IMAGES

        # labels
        labels_path = os.path.join(self.local_path, labels_file + '.gz')
        with gzip.open(labels_path, 'rb') as buffer:
            magic, n = struct.unpack('>II', buffer.read(8))
            labels = np.frombuffer(buffer.read(n), dtype=np.uint8)
            if one_hot:
                labels = self.__to_one_hot(labels, 10)

        # features
        images_path = os.path.join(self.local_path, images_file + '.gz')
        with gzip.open(images_path, 'rb') as buffer:
            magic, num, rows, cols = struct.unpack(">IIII", buffer.read(16))
            images = np.frombuffer(buffer.read(rows * cols * num), dtype=np.uint8)
            images = images.reshape(num, rows * cols)
            if scaling_features:
                images = images / 255

        return images, labels

    # convert class labels from scalars to one-hot vectors
    @staticmethod
    def __to_one_hot(labels, num_classes):
        num_labels = labels.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1
        return labels_one_hot

    @staticmethod
    def display_images(data, indices):
        stack = np.zeros([1, 28])
        for i in indices:
            stack = np.vstack((stack, data[i, :].reshape(28, 28)))
        img = Image.fromarray(stack)
        img.show()
