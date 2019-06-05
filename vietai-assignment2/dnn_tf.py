"""dnn_tf_sol.py
Solution of deep neural network implementation using tensorflow
Author: Kien Huynh

---------------------------------------
Editor: Trung Ng
Date: 05/06/2019
VietAi Ha Noi - Course 04 - 2019
Result:
    1. Bat classification
        Confusion matrix:
        [[0.99 0.01 0.  ]
         [0.04 0.95 0.02]
         [0.   0.1  0.9 ]]
        Diagonal values:
        [0.99 0.95 0.9 ] with learning_rate=0.001, steps=200000

        Confusion matrix:
        [[0.97 0.03 0.  ]
         [0.07 0.91 0.02]
         [0.   0.08 0.92]]
        Diagonal values:
        [0.97 0.91 0.92] with learning_rate=0.001, steps=20000

    2. Minist classification
        Confusion matrix:
        [[0.84 0.   0.02 0.02 0.   0.   0.11 0.   0.01 0.  ]
         [0.   0.97 0.   0.01 0.   0.   0.01 0.   0.   0.  ]
         [0.02 0.   0.83 0.01 0.05 0.   0.08 0.   0.   0.  ]
         [0.03 0.01 0.02 0.89 0.02 0.   0.03 0.   0.   0.  ]
         [0.   0.   0.11 0.04 0.77 0.   0.07 0.   0.   0.  ]
         [0.   0.   0.   0.   0.   0.97 0.   0.02 0.   0.01]
         [0.12 0.   0.08 0.03 0.04 0.   0.72 0.   0.01 0.  ]
         [0.   0.   0.   0.   0.   0.03 0.   0.95 0.   0.02]
         [0.01 0.   0.01 0.   0.   0.01 0.01 0.   0.96 0.  ]
         [0.   0.   0.   0.   0.   0.05 0.   0.04 0.   0.91]]
        Diagonal values:
        [0.84 0.97 0.83 0.89 0.77 0.97 0.72 0.95 0.96 0.91]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from util import *
from dnn_np import test
import pdb

debug = True
def test(y_hat, test_y):
    """test
    Compute the confusion matrix based on labels and predicted values

    :param y_hat: predicted probabilites, output of classifier.feed_forward
    :param test_y: test labels
    """
    # if s.ndim == 2:
    #     y_hat = np.argmax(s, axis=1)
    num_class = np.unique(test_y).size
    confusion_mat = np.zeros((num_class, num_class))

    for i in range(num_class):
        class_i_idx = test_y == i
        num_class_i = np.sum(class_i_idx)
        y_hat_i = y_hat[class_i_idx]
        for j in range(num_class):
            confusion_mat[i, j] = 1.0 * np.sum(y_hat_i == j) / num_class_i

    np.set_printoptions(precision=2)
    print('Confusion matrix:')
    print(confusion_mat)
    print('Diagonal values:')
    print(confusion_mat.flatten()[0::(num_class + 1)])

def bat_classification():
    # Load data from file
    train_x, train_y, test_x, test_y = get_bat_data()
    train_x, _, test_x = normalize(train_x, train_x, test_x)
    test_y  = test_y.flatten().astype(np.int32)
    train_y = train_y.flatten().astype(np.int32)
    num_class = (np.unique(train_y)).shape[0]
    if debug:
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # DNN parameters
    hidden_layers = [100, 100, 100]
    learning_rate = 0.001
    batch_size = 100
    steps = 20000

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[train_x.shape[1]])]


    # Available activition functions
    # https://www.tensorflow.org/api_guides/python/nn#Activation_Functions
    # tf.nn.relu
    # tf.nn.elu
    # tf.nn.sigmoid
    # tf.nn.tanh
    activation = tf.nn.relu

    # [TODO 1.7] Create a neural network and train it using estimator
    #           - Doing by Trung Ng - 05/06/2019
    #           - State: Done

    # Some available gradient descent optimization algorithms
    # https://www.tensorflow.org/api_docs/python/tf/train#classes
    # tf.train.GradientDescentOptimizer
    # tf.train.AdadeltaOptimizer
    # tf.train.AdagradOptimizer
    # tf.train.AdagradDAOptimizer
    # tf.train.MomentumOptimizer
    # tf.train.AdamOptimizer
    # tf.train.FtrlOptimizer
    # tf.train.ProximalGradientDescentOptimizer
    # tf.train.ProximalAdagradOptimizer
    # tf.train.RMSPropOptimizer
    # Create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    # optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate, l2_regularization_strength=1e-3)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.005)

    # build a deep neural network
    # https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier
    classifier = tf.estimator.DNNClassifier(
        feature_columns = feature_columns,
        hidden_units = hidden_layers,
        optimizer = optimizer,
        n_classes = num_class,
        activation_fn = activation
    )

    # Define the training inputs
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x  = {"x" : train_x},
        y = train_y,
        num_epochs = None,
        batch_size = batch_size,
        shuffle = True
    )

    # Train model.
    classifier.train(
        input_fn=train_input_fn,
        steps=steps)

    # Evaluate accuracy.
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_x},
      num_epochs=1,
      shuffle=False)
    y_hat = classifier.predict(input_fn=predict_input_fn)
    y_hat = list(y_hat)
    y_hat = np.asarray([int(x['classes'][0]) for x in y_hat])
    test(y_hat, test_y)


def mnist_classification():
    # Load data from file
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data(1)
    train_x, val_x, test_x = normalize(train_x, train_x, test_x)
    train_y = train_y.flatten().astype(np.int32)
    val_y = val_y.flatten().astype(np.int32)
    test_y = test_y.flatten().astype(np.int32)
    num_class = (np.unique(train_y)).shape[0]
    # pdb.set_trace()

    # DNN parameters
    hidden_layers = [100, 100, 100]
    learning_rate = 0.001
    batch_size = 200
    steps = 50000

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[train_x.shape[1]])]

    # Choose activation function
    activation = tf.nn.relu

    # Some available gradient descent optimization algorithms
    # TODO: [YC1.7] Create optimizer
    #       - Doing by Trung Ng - 05/06/2019
    #       - State: Done
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # build a deep neural network
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_layers,
        optimizer=optimizer,
        n_classes=num_class,
        activation_fn=activation
    )

    # Define the training inputs
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : train_x},
        y = train_y,
        num_epochs = None,
        batch_size = batch_size,
        shuffle = True
    )

    # Train model.
    classifier.train(
        input_fn=train_input_fn,
        steps=steps)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
                                    x={"x": test_x},
                                    y=test_y,
                                    num_epochs=1,
                                    shuffle=False)

    # Evaluate accuracy.
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_x},
      num_epochs=1,
      shuffle=False)
    y_hat = classifier.predict(input_fn=predict_input_fn)
    y_hat = list(y_hat)
    y_hat = np.asarray([int(x['classes'][0]) for x in y_hat])
    test(y_hat, test_y)


if __name__ == '__main__':
    np.random.seed(2017) 

    plt.ion()
    bat_classification()
    mnist_classification()
