import numpy as np


class RegularizationFunction:
    def __init__(self, lmbda):
        self.lmbda = lmbda


class L2Regularization(RegularizationFunction):
    def __init__(self, lmbda):
        super().__init__(lmbda)

    def value(self, labels, weights):
        return self.lmbda * 0.5 * sum(np.sum(w ** 2) for w in weights)

    def delta(self, labels, weights):
        return self.lmbda * weights
