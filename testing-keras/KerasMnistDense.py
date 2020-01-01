# http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

import numpy
from keras import optimizers
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.regularizers import l2

# fix random seed for reproducibility
numpy.random.seed(0)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels) / 255  # normalization
X_test = X_test.reshape(X_test.shape[0], num_pixels) / 255  # normalization
y_train = np_utils.to_categorical(y_train)  #  one hot
y_test = np_utils.to_categorical(y_test)  # one hot
num_classes = y_test.shape[1]
hidden_pixels = 784

# define baseline model
sgd = optimizers.SGD(lr=0.1, momentum=0.9)
model = Sequential()
model.add(Dense(hidden_pixels, input_dim=num_pixels, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(num_classes, kernel_initializer='glorot_normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=500, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=2)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
