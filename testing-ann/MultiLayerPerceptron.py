# todo:
#   - decreasing rate
#   - other cost/activation functions
#   - validation set

import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

from ActivationFunctions import ActivationFunction, Sigmoid
from CostFunctions import CostFunctionWithActivation
from RegularizationFunctions import RegularizationFunction


class DataSet(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALIDATION = 'validation'


class MultiLayerPerceptron:
    # static parameters
    LOG_FREQ = 5  #  elapsed time between two logs

    def __add_layer(self, size: int, activation: ActivationFunction = Sigmoid) -> None:
        self.layer_sizes.append(size)
        self.activ_fns.append(activation)
        self.activations.append(None)
        self.nets.append(None)
        self.weights.append(None)
        self.velocities.append(None)
        self.biases.append(None)
        self.deltas.append(None)

    def __init__(self) -> None:

        # layer dependent properties
        self.layer_sizes = []
        self.activ_fns = []
        self.activations = []
        self.nets = []
        self.weights = []
        self.velocities = []
        self.biases = []
        self.deltas = []
        self.hidden_layers = []

        # training properties
        self.batch_idx = 0
        self.lr = 1
        self.mu = 1
        self.plot_epochs = []
        self.plot_costs = {DataSet.TRAIN: [], DataSet.TEST: []}
        self.plot_accuracies = {DataSet.TRAIN: [], DataSet.TEST: []}
        self.cost_fn = None
        self.regul_fn = None
        self.batch_size = 0
        self.epochs = 0
        self.dropout = 0

        np.random.seed(1)
        self.start_time = dt.datetime.now()

    # init weights & biases with Xavier initialization
    def __init_weights(self):
        L = len(self.layer_sizes)
        for l in range(1, L):
            a = self.layer_sizes[l - 1]
            b = self.layer_sizes[l]
            self.weights[l] = np.random.normal(
                0, 1 / np.sqrt(a / 2), [a, b])       # 2 => ReLU
            self.biases[l] = np.random.normal(0, 1, b)
            self.velocities[l] = np.zeros([a, b])

    # forward feed
    def __forward_feed(self, features: np.ndarray, dataset: DataSet) -> None:
        L = len(self.layer_sizes)
        self.activations[0] = features
        for l in range(L - 1):
            self.nets[l + 1] = self.activations[l].dot(
                self.weights[l + 1]) + self.biases[l + 1]
            self.activations[l + 1] = self.activ_fns[l +
                                                     1].value(self.nets[l + 1])

            # apply dropout strategy on hidden layers
            if self.dropout > 0 and l < L - 2:
                if dataset == DataSet.TRAIN:
                    p = 1 - self.dropout  # keeping probability
                    mask = np.random.binomial(
                        1, p, size=self.activations[l + 1].shape) / p
                    self.activations[l + 1] *= mask

    # backpropagation
    def __backprogation(self, labels: np.ndarray, training_data_size: int) -> None:

        L = len(self.layer_sizes)

        # compute deltas
        self.deltas[L - 1] = self.cost_fn.delta(
            self.activations[L - 1], labels, self.nets[L - 1])
        for l in range(L - 2, 0, -1):
            self.deltas[l] = self.deltas[l +
                                         1].dot(self.weights[l + 1].T) * self.activ_fns[l].delta(self.nets[l])

        # update weights & biases
        for l in range(1, L):
            self.biases[l] -= self.lr * np.sum(self.deltas[l], axis=0)
            self.velocities[l] = self.mu * self.velocities[l] - \
                self.lr * self.activations[l - 1].T.dot(self.deltas[l])
            self.weights[l] += self.velocities[l]
            if self.regul_fn is not None:
                self.weights[l] -= self.lr * \
                    self.regul_fn.delta(labels, self.weights[l])

    # build
    def build(self,
              hidden_layers: list,
              cost_function: CostFunctionWithActivation,
              epochs: int,
              batch_size: int,
              learning_rate: float,
              learning_rate_decay: float,
              momentum: float,
              regularization_function: RegularizationFunction = None,
              dropout: float = 0):

        # set features/labels-linked properties
        self.cost_fn = cost_function
        self.regul_fn = regularization_function
        self.lr = learning_rate
        self.lrd = learning_rate_decay
        self.mu = momentum
        self.epochs = epochs
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.batch_size = batch_size

    #  train
    def train(self,
              train_features: np.ndarray,
              train_labels: np.ndarray,
              test_features: np.ndarray,
              test_labels: np.ndarray) -> None:

        # build layers
        self.__build_network(
            train_features.shape[1], train_labels.shape[1], self.hidden_layers)

        # init weights & biases
        self.__init_weights()

        # training loop
        batches = int(train_features.shape[0] / self.batch_size)
        start_time = dt.datetime.now()
        for epoch in range(self.epochs):
            # randomize sample
            train_features, train_labels = self.__unison_shuffled_copies(
                train_features, train_labels)

            for batch in range(batches):
                #  extract batch elements
                sub_features, sub_labels = self.__next_batch(
                    train_features, train_labels)

                self.__forward_feed(sub_features, DataSet.TRAIN)
                self.__backprogation(sub_labels, len(train_labels))

                #  eval cost & accuracy every x seconds
                if (dt.datetime.now() - start_time).seconds > self.LOG_FREQ:
                    start_time = self.print_log(batch, batches, epoch, start_time, test_features, test_labels,
                                                train_features, train_labels)

            # lr decay
            #self.lr *= (1- self.lrd)

            # log epoch
            start_time = self.print_log(batch, batches, epoch, start_time, test_features, test_labels,
                                        train_features, train_labels)

    def print_log(self, batch, batches, epoch, start_time, test_features, test_labels, train_features, train_labels):
        start_time = dt.datetime.now()
        self.plot_epochs.append(epoch + batch / batches)
        self.display_cost_and_accuracy(
            epoch, DataSet.TRAIN, start_time, train_features, train_labels)
        self.display_cost_and_accuracy(
            epoch, DataSet.TEST, start_time, test_features, test_labels)
        return start_time

    def display_cost_and_accuracy(self, epoch, dataset: DataSet, start_time, features, labels):
        self.__forward_feed(features, dataset)
        c = self.__eval_cost(labels)
        a = self.__eval_accuracy(labels)
        self.plot_costs[dataset].append(c)
        self.plot_accuracies[dataset].append(a)
        print('{}: {}, epoch={}/{}, cost={:.4f}, accur={:.4f}'.format(dataset.value, start_time, epoch, self.epochs, c,
                                                                      a))

    def __eval_accuracy(self, labels):
        y_ = self.activations[-1]
        a = np.average(np.equal(np.argmax(labels, axis=1),
                                np.argmax(y_, axis=1)) * 1) * 100
        return a

    def __eval_cost(self, labels):
        c = self.cost_fn.value(self.activations[-1], labels)
        if self.regul_fn is not None:
            c += self.regul_fn.value(labels, self.weights[1:])
        return c

    def save_results(self, path: str) -> None:

        fig, axes = plt.subplots(2, 1)
        fig.tight_layout(pad=3)

        axes[0].plot(self.plot_epochs, self.plot_costs[DataSet.TRAIN])
        axes[0].plot(self.plot_epochs, self.plot_costs[DataSet.TEST])
        axes[0].set_ylabel('cost')
        axes[0].legend([DataSet.TRAIN.value, DataSet.TEST.value])

        axes[1].plot(self.plot_epochs, self.plot_accuracies[DataSet.TRAIN])
        axes[1].plot(self.plot_epochs, self.plot_accuracies[DataSet.TEST])
        axes[1].set_ylabel('accuracy')
        axes[1].legend([DataSet.TRAIN.value, DataSet.TEST.value])

        # legends
        xy = [self.plot_epochs[-1], self.plot_accuracies[DataSet.TEST][-1]]
        axes[1].annotate('({})'.format(xy[1]), xy=xy, textcoords='data')
        xy = [self.plot_epochs[-1], self.plot_accuracies[DataSet.TRAIN][-1]]
        axes[1].annotate('({})'.format(xy[1]), xy=xy, textcoords='data')

        dtime = dt.timedelta(
            seconds=(dt.datetime.now() - self.start_time).seconds)
        cmt = 'time={}, lr={}, mu={}, epochs={}, batch={}, layers={}, regul={}, \ndropout={}, cost_fn={}' \
            .format(str(dtime),
                    self.lr,
                    self.mu,
                    self.epochs,
                    self.batch_size,
                    str(self.layer_sizes),
                    self.regul_fn.lmbda if self.regul_fn is not None else 0,
                    self.dropout,
                    type(self.cost_fn))
        plt.title(cmt, fontdict={'size': 7}, ha='center')

        plt.savefig(os.path.join(
            path, dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'))

    @staticmethod
    def __unison_shuffled_copies(a, b):
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def __next_batch(self, features: np.ndarray, labels: np.ndarray) -> tuple:
        if (self.batch_idx + 1) * self.batch_size > features.shape[0]:
            self.batch_idx = 0
        self.batch_idx += 1
        a = (self.batch_idx - 1) * self.batch_size
        b = self.batch_idx * self.batch_size
        return features[a:b, :], labels[a:b, :]

    def __build_network(self, input_layer_size: int, output_layer_size: int, hidden_layers) -> None:
        self.__add_layer(size=input_layer_size)
        for layer in hidden_layers:
            self.__add_layer(size=layer[0], activation=layer[1])
        self.__add_layer(size=output_layer_size,
                         activation=self.cost_fn.activ_fn())
