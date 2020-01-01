import numpy as np
from ActivationFunctions import Sigmoid, Linear, Softmax


class CostFunctionWithActivation:
    @staticmethod
    def value(estimations, labels):
        pass

    @staticmethod
    def delta(estimations, labels, nets):
        pass

    @staticmethod
    def activ_fn():
        pass


class QuadracticCostWithLinear(CostFunctionWithActivation):
    @staticmethod
    def value(estimations, labels):
        # sum of squared errors, averaged over the samples
        return 0.5 * np.linalg.norm(estimations - labels) ** 2 / estimations.shape[0]

    @staticmethod
    def delta(estimations, labels, nets):
        return (estimations - labels) / estimations.shape[0]

    @staticmethod
    def activ_fn():
        return Linear


class QuadracticCostWithSigmoid(CostFunctionWithActivation):
    @staticmethod
    def value(estimations, labels):
        # sum of squared errors, averaged over the samples
        return 0.5 * np.linalg.norm(estimations - labels) ** 2 / estimations.shape[0]

    @staticmethod
    def delta(estimations, labels, nets):
        return (estimations - labels) * Sigmoid.delta(nets) / estimations.shape[0]

    @staticmethod
    def activ_fn():
        return Sigmoid


class CrossEntropyCostWithSoftmax(CostFunctionWithActivation):
    @staticmethod
    def value(estimations, labels):
        #  labels are supposed to be one-hot
        x = -np.mean(np.log(estimations[labels == 1]))
        return x

    @staticmethod
    def delta(estimations, labels, nets):
        return (estimations - labels) / estimations.shape[0]

    @staticmethod
    def activ_fn():
        return Softmax
