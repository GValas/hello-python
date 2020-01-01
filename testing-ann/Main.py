from ActivationFunctions import Sigmoid, ReLu
from CostFunctions import CrossEntropyCostWithSoftmax
from MnistHelper import MnistHelper
from RegularizationFunctions import L2Regularization
from MultiLayerPerceptron import MultiLayerPerceptron

if __name__ == '__main__':
    # local path
    local_path = '/home/gege/Development/pyTest/pyLab/NeuralNetworks/data/'

    # data
    mnist = MnistHelper(local_path)
    X_train, Y_train = mnist.load_data(
        'train', one_hot=True, scaling_features=True)
    X_test, Y_test = mnist.load_data(
        'test', one_hot=True, scaling_features=True)
    # mnist.display_images(X, [0,1])
    # sys.exit()

    #  mlp
    mlp = MultiLayerPerceptron()
    mlp.build(hidden_layers=[(784, ReLu)] * 1,
              cost_function=CrossEntropyCostWithSoftmax(),
              epochs=10,
              batch_size=500,
              learning_rate=0.10,
              learning_rate_decay=0.05,     # not ised
              momentum=0.9,
              regularization_function=L2Regularization(0.00001),
              dropout=0)
    mlp.train(train_features=X_train,
              train_labels=Y_train,
              test_features=X_test,
              test_labels=Y_test)
    mlp.save_results(local_path)
