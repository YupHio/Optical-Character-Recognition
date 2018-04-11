import tensorflow as tf
import tensorflow.contrib.keras as keras
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

# MNIST handwritten, 60k 28*28 grayscale images of the 10 digits,
# along with a test set of 10k images
# http://yann.lecun.com/exdb/mnist/

# load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print('Training set: {} and Training Targets: {}'.format(x_train.shape, y_train.shape))
print('Test set: {} and test targets: {}'.format(x_test.shape, y_test.shape))
print('First training data: {}. \n Its size is: {}'.format(x_train[0], x_train[0].shape))

#show first 16 images
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(x_train[i], cmap = 'Greys_r')
plt.show()

# set seeds
np.random.seed(9987)
tf.set_random_seed(9987)

#generate one-hot labels
y_train_onehot = keras.utils.to_categorical(y_train)
print('First 10 labels: ', y_train[:10])
print('First 10 one-hot labels: ', y_train_onehot[:10])
print('Units in last layer: ', y_train_onehot.shape[1])

# preprocessing data
#1 reshape images to be row vectors
x_train_1 = np.reshape(x_train, [x_train.shape[0], x_train.shape[1] * x_train.shape[2]])
x_test_1 = np.reshape(x_test, [x_test.shape[0], x_test.shape[1] * x_test.shape[2]])
plt.imshow(np.reshape(x_train_1[0], [28, 28]), cmap = 'Greys_r')
plt.show()

#implement a feedforward NN: 2 hidden layers each have 50 hidden unites with tanh activation,
# and one output layer with 10 units for the 10 classes
model = keras.models.Sequential()
model.add(keras.layers.Dense(
    units=200,
    input_dim=x_train_1.shape[1],
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(
    units=200,
    input_dim=50,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    activation='relu'))
model.add(keras.layers.Dense(
    units=200,
    input_dim=50,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    activation='sigmoid'))

model.add(keras.layers.Dense(
    units=y_train_onehot.shape[1],
    input_dim=50,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    activation='softmax'))
sgd_optimizer = keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=.9)
rms_optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=rms_optimizer, loss='categorical_crossentropy')

history = model.fit(x_train_1, y_train_onehot,
                    batch_size=64, epochs=50,
                    verbose=1,
                    validation_split=0.1)

#predict the class labels
y_train_pred = model.predict_classes(x_train_1, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds / y_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

y_test_pred = model.predict_classes(x_test_1, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))

print('end')
