import numpy as np


class ActivationFunction:
    @staticmethod
    def value(x):
        pass

    @staticmethod
    def delta(x):
        pass


class Linear(ActivationFunction):
    @staticmethod
    def value(x):
        return x

    @staticmethod
    def delta(x):
        return 1


class Sigmoid(ActivationFunction):
    @staticmethod
    def value(x):
        return .5 * (1 + np.tanh(.5 * x))  # more stable

    @staticmethod
    def delta(x):
        y = Sigmoid.value(x)
        return y * (1 - y)


class LeakyReLu(ActivationFunction):
    @staticmethod
    def value(x):
        return (x > 0) * x + (x < 0) * x * 0.01     # slow

    @staticmethod
    def delta(x):
        return (x > 0) * 1. + (x < 0) * 0.01


class ReLu(ActivationFunction):
    @staticmethod
    def value(x):
        return (x > 0) * x

    @staticmethod
    def delta(x):
        return (x > 0) * 1.


class Softmax(ActivationFunction):
    @staticmethod
    def value(x):
        # compute the softmax in a numerically stable way
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        smax = exps / np.sum(exps, axis=1, keepdims=True)
        return smax

    @staticmethod
    def delta(x):
        # compute the softmax in a numerically stable way
        pass
